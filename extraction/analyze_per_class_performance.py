"""
Analyze per-class performance of trained BIO tagger.

Shows which entity types (B-SUBJ, I-SUBJ, etc.) are performing well
and which ones are dragging down the overall F1 score.
"""

import torch
import torch.nn as nn
import msgpack
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel
import numpy as np


class BIODataset(Dataset):
    """Dataset for BIO tagging"""
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class BIOTaggerWithDropout(nn.Module):
    """BIO tagger with configurable BERT unfreezing"""
    def __init__(self, dropout_rate=0.1, freeze_bert=True, unfreeze_layers=0):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze all initially
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Unfreeze based on unfreeze_layers argument
        if unfreeze_layers == -1 or unfreeze_layers >= 12:
            # ALL layers
            for param in self.bert.parameters():
                param.requires_grad = True
            self.unfrozen_count = 12
        elif unfreeze_layers > 0:
            # Unfreeze top N layers
            layers_to_unfreeze = list(self.bert.encoder.layer[-unfreeze_layers:])
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
            # Also unfreeze pooler
            if hasattr(self.bert, 'pooler') and self.bert.pooler is not None:
                for param in self.bert.pooler.parameters():
                    param.requires_grad = True
            self.unfrozen_count = unfreeze_layers
        else:
            self.unfrozen_count = 0
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # 6 classifiers for BIO with I-tags
        self.classifiers = nn.ModuleList([
            nn.Linear(768, 1) for _ in range(6)
        ])
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        sequence_output = self.dropout(sequence_output)
        
        logits = []
        for classifier in self.classifiers:
            logit = classifier(sequence_output).squeeze(-1)
            logits.append(logit)
        
        logits = torch.stack(logits, dim=-1)
        probs = self.sigmoid(logits)
        
        return probs, logits


def collate_fn(batch, vocab, validate=False):
    """Convert pre-split BERT subtokens to IDs with 1:1 label alignment"""
    batch_tokens = [ex['tokens'] for ex in batch]
    batch_labels = [ex['labels'] for ex in batch]
    
    # Convert tokens to IDs directly (no re-tokenization)
    batch_input_ids = []
    for tokens in batch_tokens:
        token_ids = [vocab.get(tok, vocab['[UNK]']) for tok in tokens]
        batch_input_ids.append(token_ids)
    
    max_len = max(len(ids) for ids in batch_input_ids)
    
    padded_input_ids = []
    padded_labels = []
    attention_masks = []
    
    for input_ids, labels in zip(batch_input_ids, batch_labels):
        seq_len = len(input_ids)
        pad_len = max_len - seq_len
        
        # Pad input_ids
        padded_ids = input_ids + [vocab['[PAD]']] * pad_len
        padded_input_ids.append(padded_ids)
        
        # Pad labels
        label_array = np.zeros((max_len, 6), dtype=np.float32)
        for i in range(seq_len):
            label_array[i, 0] = labels['B-SUBJ'][i]
            label_array[i, 1] = labels['I-SUBJ'][i]
            label_array[i, 2] = labels['B-PRED'][i]
            label_array[i, 3] = labels['I-PRED'][i]
            label_array[i, 4] = labels['B-OBJ'][i]
            label_array[i, 5] = labels['I-OBJ'][i]
        
        padded_labels.append(label_array)
        
        # Attention mask
        mask = [1] * seq_len + [0] * pad_len
        attention_masks.append(mask)
    
    return {
        'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
        'labels': torch.tensor(np.stack(padded_labels), dtype=torch.float32),
        'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
    }


def compute_per_class_metrics(predictions, targets, threshold=0.3):
    """Compute detailed metrics for each BIO class"""
    pred_binary = (predictions > threshold).float()
    
    label_names = ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']
    
    pred_reshaped = pred_binary.view(-1, 6)
    target_reshaped = targets.view(-1, 6)
    
    metrics = []
    
    for label_idx in range(6):
        pred_label = pred_reshaped[:, label_idx]
        target_label = target_reshaped[:, label_idx]
        
        tp = (pred_label * target_label).sum().item()
        fp = (pred_label * (1 - target_label)).sum().item()
        fn = ((1 - pred_label) * target_label).sum().item()
        tn = ((1 - pred_label) * (1 - target_label)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        support = int(target_label.sum().item())
        
        metrics.append({
            'label': label_names[label_idx],
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        })
    
    # Compute macro F1
    macro_f1 = sum(m['f1'] for m in metrics) / len(metrics)
    
    return macro_f1, metrics


def main():
    print("Loading data and model...")
    
    # Load data
    with open('bio_training_250chunks_complete_FIXED.msgpack', 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
    
    # Build vocab from bert-base-uncased tokenizer
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.get_vocab()
    
    all_examples = data['training_data']
    
    # Split into train/eval (same as training)
    split_idx = int(0.8 * len(all_examples))
    eval_examples = all_examples[split_idx:]
    
    print(f"Loaded {len(eval_examples)} eval examples")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = BIOTaggerWithDropout(dropout_rate=0.1, unfreeze_layers=12)
    model.load_state_dict(torch.load('bio_tagger_atomic.pt', map_location=device))
    model = model.to(device)
    model.eval()
    
    # Create dataloader
    eval_dataset = BIODataset(eval_examples)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=16,
        collate_fn=lambda b: collate_fn(b, vocab)
    )
    
    # Evaluate
    print("\nEvaluating...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            probs, _ = model(input_ids, attention_mask)
            
            mask = attention_mask.unsqueeze(-1).expand_as(probs)
            probs_masked = probs * mask
            labels_masked = labels * mask
            
            batch_size, seq_len, num_labels = probs_masked.shape
            all_predictions.append(probs_masked.view(-1, num_labels))
            all_targets.append(labels_masked.view(-1, num_labels))
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    macro_f1, metrics = compute_per_class_metrics(predictions, targets, threshold=0.5)
    
    # Print results
    print("\n" + "=" * 80)
    print("PER-CLASS PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"\nOverall Macro F1: {macro_f1:.4f}\n")
    
    print(f"{'Label':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10} {'TP':>8} {'FP':>8} {'FN':>8}")
    print("-" * 84)
    
    for m in metrics:
        print(f"{m['label']:<10} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['support']:>10} {m['tp']:>8} {m['fp']:>8} {m['fn']:>8}")
    
    # Identify weak spots
    print("\n" + "=" * 80)
    print("WEAK SPOTS (F1 < 0.65):")
    print("=" * 80)
    
    weak_classes = [m for m in metrics if m['f1'] < 0.65]
    if weak_classes:
        for m in weak_classes:
            print(f"\n  {m['label']}:")
            print(f"    F1: {m['f1']:.4f}")
            print(f"    Issue: ", end="")
            if m['precision'] < 0.6:
                print(f"Low precision ({m['precision']:.2f}) - too many false positives")
            elif m['recall'] < 0.6:
                print(f"Low recall ({m['recall']:.2f}) - missing true entities")
            else:
                print(f"Both precision ({m['precision']:.2f}) and recall ({m['recall']:.2f}) need improvement")
            print(f"    Support: {m['support']} instances")
    else:
        print("  No classes below F1=0.65 threshold. All performing well!")
    
    print("\n" + "=" * 80)
    print("STRONGEST CLASSES (F1 > 0.70):")
    print("=" * 80)
    
    strong_classes = [m for m in metrics if m['f1'] > 0.70]
    if strong_classes:
        for m in strong_classes:
            print(f"  {m['label']}: F1={m['f1']:.4f} (P={m['precision']:.2f}, R={m['recall']:.2f})")
    else:
        print("  No classes above F1=0.70 threshold.")


if __name__ == '__main__':
    main()
