"""
Fast BERT BIO Tagger Training (No Optuna)

Uses bert-base-uncased with frozen layers + 6 binary classifiers
for B-SUBJ, I-SUBJ, B-PRED, I-PRED, B-OBJ, I-OBJ tags.

Data format: msgpack with 'tokens' (list[str]) and 'labels' (dict[str, list[int]])
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import msgpack
import numpy as np
from transformers import BertTokenizerFast, BertModel
from typing import Dict, List
import time


class BIODataset(Dataset):
    def __init__(self, examples: List[Dict]):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, tokenizer):
    """Convert pre-split BERT subtokens to IDs with 1:1 label alignment"""
    precomputed_labels = [item['labels'] for item in batch]
    token_lists = [item['tokens'] for item in batch]
    
    label_keys = ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']
    
    # Build input_ids: [CLS] + tokens + [SEP]
    all_ids = []
    all_lengths = []
    for tokens in token_lists:
        full_tokens = ['[CLS]'] + tokens + ['[SEP]']
        ids = tokenizer.convert_tokens_to_ids(full_tokens)
        all_ids.append(ids)
        all_lengths.append(len(ids))
    
    max_len = min(max(all_lengths), 512)
    batch_size = len(batch)
    
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels_batch = torch.zeros(batch_size, max_len, 6)
    
    pad_id = tokenizer.pad_token_id or 0
    
    for i in range(batch_size):
        ids = all_ids[i]
        seq_len = min(len(ids), max_len)
        
        input_ids[i, :seq_len] = torch.tensor(ids[:seq_len], dtype=torch.long)
        input_ids[i, seq_len:] = pad_id
        attention_mask[i, :seq_len] = 1
        
        # Labels: position 0=[CLS], 1..N=tokens, N+1=[SEP]
        labels_dict = precomputed_labels[i]
        n_tokens = len(labels_dict['B-SUBJ'])
        n_fill = min(n_tokens, max_len - 2)
        
        for label_idx, label_name in enumerate(label_keys):
            label_vals = labels_dict[label_name][:n_fill]
            labels_batch[i, 1:1+n_fill, label_idx] = torch.tensor(label_vals, dtype=torch.float32)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels_batch
    }


class BIOTagger(nn.Module):
    """BERT + 6 binary classifiers with dropout"""
    
    def __init__(self, dropout=0.1, freeze_bert=True):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        self.classifiers = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(6)])
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        logits = torch.stack([clf(sequence_output).squeeze(-1) for clf in self.classifiers], dim=-1)
        probs = self.sigmoid(logits)
        return probs, logits


def compute_f1_macro(predictions, targets, threshold=0.3):
    """F1 macro for 6 BIO labels"""
    pred_binary = (predictions > threshold).float()
    
    pred_reshaped = pred_binary.view(-1, 6)
    target_reshaped = targets.view(-1, 6)
    
    label_f1_scores = []
    for label_idx in range(6):
        pred_label = pred_reshaped[:, label_idx]
        target_label = target_reshaped[:, label_idx]
        
        tp = (pred_label * target_label).sum().item()
        fp = (pred_label * (1 - target_label)).sum().item()
        fn = ((1 - pred_label) * target_label).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        label_f1_scores.append(f1)
    
    return sum(label_f1_scores) / len(label_f1_scores)


def evaluate(model, dataloader, device, threshold=0.3):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            probs, _ = model(input_ids, attention_mask)
            
            mask = attention_mask.unsqueeze(-1).expand_as(probs)
            all_predictions.append((probs * mask).view(-1, 6))
            all_targets.append((labels * mask).view(-1, 6))
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return compute_f1_macro(predictions, targets, threshold)


def train_epoch(model, dataloader, optimizer, device, o_weight=0.01):
    model.train()
    epoch_losses = []
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        probs, logits = model(input_ids, attention_mask)
        
        # O-token downweighting
        pad_mask = attention_mask.unsqueeze(-1).expand_as(logits)
        has_positive = labels.sum(dim=-1) > 0
        token_weights = torch.where(
            has_positive,
            torch.ones_like(has_positive, dtype=torch.float32),
            torch.full_like(has_positive, o_weight, dtype=torch.float32)
        ).unsqueeze(-1).expand_as(logits)
        combined_mask = pad_mask.float() * token_weights
        
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        weighted_loss = (bce_loss * combined_mask).sum() / combined_mask.sum()
        
        weighted_loss.backward()
        optimizer.step()
        epoch_losses.append(weighted_loss.item())
    
    return np.mean(epoch_losses)


def main():
    print("=" * 70)
    print("FAST BERT BIO TAGGER TRAINING")
    print("=" * 70)
    
    # Config
    DATA_FILE = 'data/bio_training_250chunks_complete.msgpack'
    BATCH_SIZE = 8
    LR = 1e-5
    DROPOUT = 0.1
    O_WEIGHT = 0.005
    THRESHOLD = 0.4
    MAX_EPOCHS = 50
    PATIENCE = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print(f"\nLoading {DATA_FILE}...")
    with open(DATA_FILE, 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
    
    all_examples = data['training_data']
    print(f"  {len(all_examples)} examples")
    
    # Split 80/20
    np.random.seed(42)
    indices = np.random.permutation(len(all_examples))
    eval_size = max(1, int(0.2 * len(all_examples)))
    train_examples = [all_examples[i] for i in indices[eval_size:]]
    eval_examples = [all_examples[i] for i in indices[:eval_size]]
    
    print(f"  Train: {len(train_examples)} | Eval: {len(eval_examples)}")
    
    # Create datasets
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    train_dataset = BIODataset(train_examples)
    eval_dataset = BIODataset(eval_examples)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=16,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    
    # Model
    print("\nInitializing model...")
    model = BIOTagger(dropout=DROPOUT, freeze_bert=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    print(f"  Dropout: {DROPOUT}")
    print(f"  LR: {LR}")
    print(f"  O-weight: {O_WEIGHT}")
    print(f"  Threshold: {THRESHOLD}")
    print(f"  BERT layers: FROZEN")
    
    # Training loop
    print(f"\nTraining (max {MAX_EPOCHS} epochs, patience {PATIENCE})...")
    best_f1 = 0.0
    no_improve = 0
    
    start_time = time.time()
    
    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, device, o_weight=O_WEIGHT)
        eval_f1 = evaluate(model, eval_loader, device, threshold=THRESHOLD)
        
        print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Eval F1: {eval_f1:.4f}")
        
        if eval_f1 > best_f1:
            best_f1 = eval_f1
            no_improve = 0
            torch.save(model.state_dict(), 'bio_tagger_best.pt')
            print(f"    → New best! Saved.")
        else:
            no_improve += 1
        
        if no_improve >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break
    
    elapsed = time.time() - start_time
    
    # Final eval
    model.load_state_dict(torch.load('bio_tagger_best.pt'))
    final_f1 = evaluate(model, eval_loader, device, threshold=THRESHOLD)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Best Eval F1: {best_f1:.4f}")
    print(f"  Final Eval F1: {final_f1:.4f}")
    print(f"  Training time: {elapsed/60:.1f} minutes")
    print(f"\n  Model saved: bio_tagger_best.pt")


if __name__ == '__main__':
    main()
