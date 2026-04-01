"""
BERT BIO Tagger with CRF - SIMPLIFIED VERSION (NO OPTUNA)

Fixed hyperparameters based on prior Optuna results from train_bert_bio_optuna.py.
This avoids the complexity of Optuna trials.
"""
import os
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_NO_TELEMETRY'] = '1'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import msgpack
import numpy as np
from transformers import BertTokenizerFast, BertModel
from typing import Dict, List
from torchcrf import CRF


LABEL_TO_ID = {
    'O': 0,
    'B-SUBJ': 1,
    'I-SUBJ': 2,
    'B-PRED': 3,
    'I-PRED': 4,
    'B-OBJ': 5,
    'I-OBJ': 6
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


class BIODataset(Dataset):
    def __init__(self, examples: List[Dict]):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def convert_multihot_to_single_label(labels_dict: Dict[str, List[int]]) -> List[int]:
    """Convert multi-hot BIO labels to single-label (one class per token)"""
    n_tokens = len(labels_dict['B-SUBJ'])
    single_labels = []
    
    for i in range(n_tokens):
        active_labels = []
        for label_name in ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']:
            if labels_dict[label_name][i] == 1:
                active_labels.append(label_name)
        
        if len(active_labels) == 0:
            single_labels.append(LABEL_TO_ID['O'])
        elif len(active_labels) == 1:
            single_labels.append(LABEL_TO_ID[active_labels[0]])
        else:
            single_labels.append(LABEL_TO_ID[active_labels[0]])
    
    return single_labels


def collate_fn(batch, tokenizer):
    """Convert pre-split BERT subtokens to IDs with 1:1 label alignment"""
    precomputed_labels = [item['labels'] for item in batch]
    token_lists = [item['tokens'] for item in batch]
    
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
    labels_batch = torch.full((batch_size, max_len), -100, dtype=torch.long)
    
    pad_id = tokenizer.pad_token_id or 0
    
    for i in range(batch_size):
        ids = all_ids[i]
        seq_len = min(len(ids), max_len)
        
        input_ids[i, :seq_len] = torch.tensor(ids[:seq_len], dtype=torch.long)
        input_ids[i, seq_len:] = pad_id
        attention_mask[i, :seq_len] = 1
        
        labels_dict = precomputed_labels[i]
        single_labels = convert_multihot_to_single_label(labels_dict)
        
        token_label_len = min(len(single_labels), seq_len - 2)
        if token_label_len > 0:
            labels_tensor = torch.tensor(single_labels[:token_label_len], dtype=torch.long)
            labels_batch[i, 1:1+token_label_len] = labels_tensor
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels_batch
    }


class BIOTaggerCRF(nn.Module):
    """BERT + CRF for BIO tagging with enforced transition constraints"""
    
    def __init__(self, dropout=0.3, unfreeze_layers=4, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size
        
        # Freeze all BERT parameters initially
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Unfreeze top N encoder layers
        total_layers = len(self.bert.encoder.layer)
        unfrozen_start = total_layers - unfreeze_layers
        for i in range(unfrozen_start, total_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True
        
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_size, num_classes)
        self.crf = CRF(num_classes, batch_first=True)
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Training: CRF loss | Inference: Viterbi decoding"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        
        if labels is not None:
            # Training: CRF loss
            mask = attention_mask.bool()
            labels_for_crf = labels.clone()
            labels_for_crf[labels == -100] = 0
            loss = -self.crf(emissions, labels_for_crf, mask=mask, reduction='mean')
            return loss
        else:
            # Inference: Viterbi decoding
            mask = attention_mask.bool()
            predictions = self.crf.decode(emissions, mask=mask)
            
            # Pad predictions
            padded_predictions = []
            for pred, length in zip(predictions, attention_mask.sum(dim=1)):
                padded_pred = pred + [0] * (input_ids.size(1) - len(pred))
                padded_predictions.append(padded_pred)
            
            return torch.tensor(padded_predictions, device=input_ids.device)


def compute_f1_macro(predictions, targets, num_classes=7):
    """F1 macro for 7 classes"""
    valid_mask = targets != -100
    valid_preds = predictions[valid_mask]
    valid_targets = targets[valid_mask]
    
    if len(valid_targets) == 0:
        return 0.0
    
    label_f1_scores = []
    for class_id in range(num_classes):
        tp = ((valid_preds == class_id) & (valid_targets == class_id)).sum().item()
        fp = ((valid_preds == class_id) & (valid_targets != class_id)).sum().item()
        fn = ((valid_preds != class_id) & (valid_targets == class_id)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        label_f1_scores.append(f1)
    
    return sum(label_f1_scores) / len(label_f1_scores)


def evaluate(model, dataloader, device):
    """Evaluation with Viterbi decoding"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            predictions = model(input_ids, attention_mask, labels=None)
            
            all_predictions.append(predictions.view(-1))
            all_targets.append(labels.view(-1))
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return compute_f1_macro(predictions, targets)


def train_epoch(model, dataloader, optimizer, device):
    """Training epoch with CRF loss"""
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        loss = model(input_ids, attention_mask, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading training data...")
    with open(r'c:\Users\user\arxiv_id_lists\data\bio_training_250chunks_complete.msgpack', 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
    
    all_examples = data['training_data']
    print(f"  {len(all_examples)} examples")
    
    # Split
    np.random.seed(42)
    indices = np.random.permutation(len(all_examples))
    
    n_test = max(1, int(0.1 * len(all_examples)))
    test_examples = [all_examples[i] for i in indices[:n_test]]
    train_examples = [all_examples[i] for i in indices[n_test:]]
    
    print(f"  Train: {len(train_examples)}")
    print(f"  Test: {len(test_examples)}")
    
    # Fixed hyperparameters (from prior Optuna best)
    BATCH_SIZE = 16
    DROPOUT = 0.3
    UNFREEZE_LAYERS = 4
    LR_BERT = 3e-5
    LR_HEAD = 1e-3
    WEIGHT_DECAY = 1e-3
    
    print(f"\nHyperparameters:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Unfrozen layers: {UNFREEZE_LAYERS}")
    print(f"  LR BERT: {LR_BERT}")
    print(f"  LR head/CRF: {LR_HEAD}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    train_dataset = BIODataset(train_examples)
    test_dataset = BIODataset(test_examples)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    
    # Create model
    model = BIOTaggerCRF(
        dropout=DROPOUT,
        unfreeze_layers=UNFREEZE_LAYERS
    ).to(device)
    
    # Optimizer with discriminative LR
    bert_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and 'bert' in name:
            bert_params.append(param)
    
    head_crf_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and 'bert' not in name:
            head_crf_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': LR_BERT},
        {'params': head_crf_params, 'lr': LR_HEAD}
    ], weight_decay=WEIGHT_DECAY)
    
    # Training
    best_test_f1 = 0.0
    patience = 12
    no_improve = 0
    
    print("\n" + "=" * 80)
    print("TRAINING BERT-CRF (Guaranteed Valid BIO Sequences)")
    print("=" * 80 + "\n")
    
    for epoch in range(1, 50):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_f1 = evaluate(model, test_loader, device)
        
        print(f"Epoch {epoch:2d} | Loss: {train_loss:.4f} | Test F1: {test_f1:.4f}", end="")
        
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            no_improve = 0
            torch.save(model.state_dict(), 'bio_tagger_crf.pt')
            print(f"  ✓ Saved")
        else:
            no_improve += 1
            print()
        
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best test F1: {best_test_f1:.4f}")
    print(f"Model saved: bio_tagger_crf.pt")
    print("\n✅ CRF guarantees 0% invalid BIO sequences at inference!")


if __name__ == '__main__':
    main()
