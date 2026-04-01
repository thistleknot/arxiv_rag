"""Quick training test with proper GLiNER-formatted BIO labels"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import msgpack
import numpy as np
from transformers import BertTokenizerFast, BertModel
from typing import Dict, List
import time

print("="*80)
print("TRAINING BIO TAGGER WITH PROPER GLINER-FORMATTED LABELS")
print("="*80)

# Load corrected training data
print("\n[1/4] Loading corrected training data...")
with open('data/bio_training_416_gliner_fixed.msgpack', 'rb') as f:
    data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)

all_examples = data['training_data']
print(f"  Total examples: {len(all_examples)}")

# Show example
ex = all_examples[0]
print(f"\n  Example 1:")
print(f"    Sentence: {ex['sentence'][:80]}...")
print(f"    Tokens: {ex['tokens'][:10]}...")
print(f"    Labels (B-SUBJ): {ex['labels']['B-SUBJ'][:25]}...")
print(f"    Labels (I-SUBJ): {ex['labels']['I-SUBJ'][:25]}...")
print(f"    Labels (B-PRED): {ex['labels']['B-PRED'][:25]}...")

# Verify we have B-I labels
b_count = sum(1 for ex in all_examples if any(ex['labels'].get(f'B-{role}', []) for role in ['SUBJ', 'PRED', 'OBJ']))
i_count = sum(1 for ex in all_examples if any(ex['labels'].get(f'I-{role}', []) for role in ['SUBJ', 'PRED', 'OBJ']))
print(f"\n  Examples with B- labels: {b_count}/{len(all_examples)}")
print(f"  Examples with I- labels: {i_count}/{len(all_examples)}")

# Split: 80% train, 20% eval
np.random.seed(42)
indices = np.random.permutation(len(all_examples))
split = int(0.8 * len(all_examples))

train_examples = [all_examples[i] for i in indices[:split]]
eval_examples = [all_examples[i] for i in indices[split:]]

print(f"\n  Train: {len(train_examples)}")
print(f"  Eval: {len(eval_examples)}")


class BIODataset(Dataset):
    """Dataset with BERT tokenization"""
    
    def __init__(self, examples: List[Dict], tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_names = ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Build token sequence: [CLS] + tokens + [SEP]
        # Tokens are already BERT subtokens (contain ## subwords)
        # So convert_tokens_to_ids gives exact 1:1 mapping
        full_tokens = ['[CLS]'] + ex['tokens'] + ['[SEP]']
        input_ids_list = self.tokenizer.convert_tokens_to_ids(full_tokens)
        
        seq_len = len(input_ids_list)
        
        # Pad or truncate to max_length=256
        max_len = 256
        pad_id = self.tokenizer.pad_token_id or 0
        
        if seq_len > max_len:
            input_ids_list = input_ids_list[:max_len]
            seq_len = max_len
        
        # Build tensors
        input_ids = torch.zeros(max_len, dtype=torch.long)
        attention_mask = torch.zeros(max_len, dtype=torch.long)
        input_ids[:seq_len] = torch.tensor(input_ids_list, dtype=torch.long)
        input_ids[seq_len:] = pad_id
        attention_mask[:seq_len] = 1
        
        # Convert labels to tensor (6 labels per token)
        # Labels aligned: position 0 = [CLS] (0), positions 1..N = original tokens, position N+1 = [SEP] (0)
        labels = torch.zeros((max_len, len(self.label_names)))
        for i, label_name in enumerate(self.label_names):
            label_values = ex['labels'].get(label_name, [0]*len(ex['tokens']))
            n_fill = min(len(label_values), max_len - 2)  # -2 for CLS and SEP
            labels[1:1+n_fill, i] = torch.tensor(label_values[:n_fill], dtype=torch.float32)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class BIOTagger(nn.Module):
    """BERT-based BIO tagger"""
    
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze BERT
        for param in self.bert.parameters():
            param.requires_grad = False
        
        hidden_size = self.bert.config.hidden_size
        num_labels = 6  # B-SUBJ, I-SUBJ, B-PRED, I-PRED, B-OBJ, I-OBJ
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
    
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state  # [batch, seq_len, hidden]
        
        logits = self.classifier(sequence_output)  # [batch, seq_len, num_labels]
        probs = torch.sigmoid(logits)
        
        return probs, logits


def compute_f1_score(predictions, targets, threshold=0.3):
    """Compute macro-averaged F1 across 6 label types"""
    pred = (predictions > threshold).float()
    
    num_labels = pred.shape[-1]
    label_f1_scores = []
    
    for label_idx in range(num_labels):
        pred_label = pred[:, label_idx]
        target_label = targets[:, label_idx]
        
        tp = (pred_label * target_label).sum().item()
        fp = (pred_label * (1 - target_label)).sum().item()
        fn = ((1 - pred_label) * target_label).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        label_f1_scores.append(f1)
    
    return sum(label_f1_scores) / len(label_f1_scores)


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    epoch_losses = []
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        probs, logits = model(input_ids, attention_mask)
        
        # Binary cross-entropy loss
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())
    
    return np.mean(epoch_losses)


def evaluate(model, dataloader, device, threshold=0.3):
    """Evaluate on eval set with attention mask applied"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            probs, _ = model(input_ids, attention_mask)
            
            # Apply attention mask to exclude padding from F1
            mask = attention_mask.unsqueeze(-1).expand_as(probs)
            probs_masked = probs * mask
            labels_masked = labels * mask
            
            batch_size, seq_len, num_labels = probs_masked.shape
            all_predictions.append(probs_masked.view(-1, num_labels))
            all_targets.append(labels_masked.view(-1, num_labels))
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    return compute_f1_score(predictions, targets, threshold)


# Setup
print("\n[2/4] Setting up model and optimizer...")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BIOTagger().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

print(f"  Device: {device}")
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create dataloaders
print("\n[3/4] Creating dataloaders...")
train_dataset = BIODataset(train_examples, tokenizer)
eval_dataset = BIODataset(eval_examples, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=16)

print(f"  Train batches: {len(train_loader)}")
print(f"  Eval batches: {len(eval_loader)}")

# Training loop
print("\n[4/4] Training...")
max_epochs = 30
best_f1 = 0.0
patience_count = 0
patience = 5

for epoch in range(1, max_epochs + 1):
    start = time.time()
    
    train_loss = train_epoch(model, train_loader, optimizer, device)
    eval_f1 = evaluate(model, eval_loader, device, threshold=0.3)
    
    elapsed = time.time() - start
    
    print(f"  Epoch {epoch:2d} | Loss: {train_loss:.4f} | Eval F1: {eval_f1:.4f} | Time: {elapsed:.1f}s")
    
    if eval_f1 > best_f1:
        best_f1 = eval_f1
        patience_count = 0
        # Save best model
        torch.save(model.state_dict(), 'bio_tagger_gliner_best.pt')
        print(f"           -> New best F1: {best_f1:.4f} [OK]")
    else:
        patience_count += 1
        if patience_count >= patience:
            print(f"  Early stopping (patience {patience})")
            break

print(f"\n{'='*80}")
print(f"FINAL RESULT: F1 = {best_f1:.4f}")
print(f"{'='*80}")

# Save training summary
summary = f"""
TRAINING SUMMARY
================

Data:
  Total examples: {len(all_examples)}
  Train: {len(train_examples)} (80%)
  Eval: {len(eval_examples)} (20%)

Model:
  Architecture: Frozen BERT + Linear classifier
  Device: {device}

Hyperparameters:
  Learning rate: 1e-5
  Weight decay: 0.01
  Batch size: 8
  Threshold: 0.3
  Max epochs: {max_epochs}
  Patience: {patience}

Results:
  Best F1: {best_f1:.4f}
  Model saved: bio_tagger_gliner_best.pt

Notes:
- All 416 examples converted from GLiNER span format to token-level BIO labels
- "deep learning models" correctly labeled B-SUBJ, I-SUBJ, I-SUBJ (not separate B-S tokens)
- Training with proper contiguous entity annotations should improve over 0.1764 baseline
"""

with open('training_summary_gliner.txt', 'w') as f:
    f.write(summary)

print(summary)
