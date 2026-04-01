"""
Train BIO Tagger with UNFROZEN BERT layers
Uses BIOTaggerMultiClass model (same as dashboard)
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import msgpack
import numpy as np
from transformers import BertTokenizer, BertModel
from typing import Dict, List
import argparse
import json
from pathlib import Path


class BIOTaggerMultiClass(nn.Module):
    """Multi-class BIO tagger with unfrozen BERT layers"""
    
    def __init__(self, dropout=0.1, unfreeze_layers=12):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size
        total_layers = len(self.bert.encoder.layer)
        
        # Unfreeze top N layers (-1 or >= total means all)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        if unfreeze_layers == -1 or unfreeze_layers >= total_layers:
            # Unfreeze ALL BERT layers
            for param in self.bert.parameters():
                param.requires_grad = True
            self.unfrozen_count = total_layers
        elif unfreeze_layers > 0:
            # Unfreeze top N layers
            for layer in self.bert.encoder.layer[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            self.unfrozen_count = unfreeze_layers
        else:
            self.unfrozen_count = 0
        
        # Classifier head with dropout
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 13)  # 13 classes (O + 6 B/I pairs)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class BIODataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    texts = [' '.join(ex['tokens']) for ex in batch]
    encodings = tokenizer(texts, padding=True, truncation=True, 
                         max_length=512, return_tensors='pt')
    
    # Convert multi-hot BIO to class indices
    label_map = {
        'O': 0,
        'B-SUBJ': 1, 'I-SUBJ': 2,
        'B-PRED': 3, 'I-PRED': 4,
        'B-OBJ': 5, 'I-OBJ': 6,
        'B-SUBJ|B-PRED': 7, 'B-SUBJ|B-OBJ': 8, 'B-PRED|B-OBJ': 9,
        'I-SUBJ|I-PRED': 10, 'I-SUBJ|I-OBJ': 11, 'I-PRED|I-OBJ': 12
    }
    
    max_len = encodings['input_ids'].size(1)
    labels_batch = []
    
    for ex in batch:
        # Convert multi-hot to class labels
        labels = [label_map['O']] * len(ex['tokens'])
        
        # Assign BIO labels
        for i, token_idx in enumerate(range(len(ex['tokens']))):
            label_str = 'O'
            if ex['labels']['B-SUBJ'][token_idx]:
                label_str = 'B-SUBJ'
            elif ex['labels']['I-SUBJ'][token_idx]:
                label_str = 'I-SUBJ'
            elif ex['labels']['B-PRED'][token_idx]:
                label_str = 'B-PRED'
            elif ex['labels']['I-PRED'][token_idx]:
                label_str = 'I-PRED'
            elif ex['labels']['B-OBJ'][token_idx]:
                label_str = 'B-OBJ'
            elif ex['labels']['I-OBJ'][token_idx]:
                label_str = 'I-OBJ'
            
            labels[i] = label_map.get(label_str, 0)
        
        # Pad to match tokenized length
        labels = [label_map['O']] + labels + [label_map['O']]  # CLS, SEP
        labels = labels[:max_len] + [label_map['O']] * (max_len - len(labels))
        labels_batch.append(labels)
    
    labels_tensor = torch.tensor(labels_batch, dtype=torch.long)
    
    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels_tensor
    }


def compute_f1(preds, labels, attention_mask):
    """Compute token-level F1 (ignoring padding)"""
    preds = preds[attention_mask.bool()]
    labels = labels[attention_mask.bool()]
    
    # Exclude O class (index 0)
    mask = labels > 0
    if mask.sum() == 0:
        return 0.0
    
    preds = preds[mask]
    labels = labels[mask]
    
    correct = (preds == labels).sum().item()
    total = len(labels)
    
    return correct / total if total > 0 else 0.0


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_f1 = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        
        loss = nn.functional.cross_entropy(
            logits.view(-1, 13), 
            labels.view(-1),
            ignore_index=-100
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = logits.argmax(dim=-1)
        f1 = compute_f1(preds, labels, attention_mask)
        total_f1 += f1
    
    return total_loss / len(dataloader), total_f1 / len(dataloader)


def eval_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_f1 = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            
            loss = nn.functional.cross_entropy(
                logits.view(-1, 13), 
                labels.view(-1),
                ignore_index=-100
            )
            
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            f1 = compute_f1(preds, labels, attention_mask)
            total_f1 += f1
    
    return total_loss / len(dataloader), total_f1 / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--unfreeze_layers', type=int, default=12, 
                       help='Number of top BERT layers to unfreeze (-1 or 12 = all layers)')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output', type=str, default='bio_tagger_unfrozen_all.pt')
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}...")
    with open(args.data, 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False)
    
    examples = data['training_data']
    print(f"Total examples: {len(examples)}")
    
    # Split 80/20
    np.random.seed(42)
    np.random.shuffle(examples)
    split = int(0.8 * len(examples))
    train_data = examples[:split]
    eval_data = examples[split:]
    
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")
    
    # Create dataloaders
    train_dataset = BIODataset(train_data)
    eval_dataset = BIODataset(eval_data)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size,
                            collate_fn=collate_fn)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = BIOTaggerMultiClass(dropout=args.dropout, 
                                unfreeze_layers=args.unfreeze_layers)
    
    # Report unfrozen layers
    if args.unfreeze_layers == -1 or args.unfreeze_layers >= 12:
        print(f"Unfreezing ALL {model.unfrozen_count} BERT layers (full fine-tuning)")
    else:
        print(f"Unfreezing top {model.unfrozen_count} BERT layers (partial fine-tuning)")
    
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    best_f1 = 0
    patience = 0
    max_patience = 8
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, device)
        eval_loss, eval_f1 = eval_epoch(model, eval_loader, device)
        
        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
              f"Eval Loss: {eval_loss:.4f} F1: {eval_f1:.4f}")
        
        if eval_f1 > best_f1:
            best_f1 = eval_f1
            patience = 0
            torch.save(model.state_dict(), args.output)
            print(f"  ✓ Saved checkpoint (F1: {best_f1:.4f})")
        else:
            patience += 1
        
        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\nBest F1: {best_f1:.4f}")
    print(f"Model saved to: {args.output}")


if __name__ == '__main__':
    main()
