"""
BIO Tagger Training: Iterative Curriculum Learning
===================================================

Training strategy:
1. Reserve holdout set (20% of data)
2. Split remaining into train/eval (80/20)
3. Train in batches of 50 examples (with replacement)
4. Evaluate F1 score after each batch
5. Stop when F1 stops improving (early stopping)
6. Report final score on holdout set

NO SPECULATION. ACTUAL MEASUREMENTS.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import msgpack
import numpy as np
from transformers import BertTokenizer, BertModel
from typing import Dict, List
import time
from scipy.stats import boxcox
from collections import Counter


class BIODataset(Dataset):
    """Dataset for BIO-tagged training examples"""
    
    def __init__(self, examples: List[Dict]):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Convert tokens to input_ids
        tokens = ['[CLS]'] + example['tokens'] + ['[SEP]']
        
        # Get labels (multi-hot BIO)
        labels = example['labels']
        
        # Pad labels for [CLS] and [SEP]
        label_names = ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']
        padded_labels = []
        for name in label_names:
            padded = [0] + labels[name] + [0]
            padded_labels.append(padded)
        
        return {
            'tokens': tokens,
            'labels': torch.tensor(padded_labels, dtype=torch.float32).T,
            'length': len(tokens)
        }


def collate_fn(batch, tokenizer, max_seq_len=512):
    """Collate with padding and truncation to max_seq_len"""
    max_len = min(max(item['length'] for item in batch), max_seq_len)
    
    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []
    
    for item in batch:
        input_ids = tokenizer.convert_tokens_to_ids(item['tokens'])
        # Truncate to max_seq_len if needed
        input_ids = input_ids[:max_seq_len]
        attention_mask = [1] * len(input_ids)
        
        padding_length = max_len - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        
        labels = item['labels'][:max_seq_len]  # truncate labels too
        label_padding = torch.zeros(padding_length, 6)
        labels = torch.cat([labels, label_padding], dim=0)
        
        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)
        labels_batch.append(labels)
    
    return {
        'input_ids': torch.tensor(input_ids_batch, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask_batch, dtype=torch.long),
        'labels': torch.stack(labels_batch)
    }


class BIOTagger(nn.Module):
    """BERT + 6 independent binary classifiers"""
    
    def __init__(self, bert_model_name='bert-base-uncased', num_labels=6):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_labels)
        ])
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        logits = []
        for classifier in self.classifiers:
            logit = classifier(sequence_output).squeeze(-1)
            logits.append(logit)
        
        logits = torch.stack(logits, dim=-1)
        probs = self.sigmoid(logits)
        
        return probs, logits


def compute_f1_score(predictions, targets, threshold=0.3):
    """
    Compute F1 Macro for multi-hot BIO labels.
    
    F1 Macro: Compute F1 per label, then average (gives equal weight to all labels).
    """
    pred_binary = (predictions > threshold).float()
    
    # Reshape to [total_tokens, 6_labels]
    # predictions/targets shape: [batch * seq_len * 6]
    num_labels = 6
    pred_reshaped = pred_binary.view(-1, num_labels)
    target_reshaped = targets.view(-1, num_labels)
    
    # Compute per-label metrics
    label_f1_scores = []
    label_precisions = []
    label_recalls = []
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for label_idx in range(num_labels):
        pred_label = pred_reshaped[:, label_idx]
        target_label = target_reshaped[:, label_idx]
        
        tp = (pred_label * target_label).sum().item()
        fp = (pred_label * (1 - target_label)).sum().item()
        fn = ((1 - pred_label) * target_label).sum().item()
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Per-label metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        label_f1_scores.append(f1)
        label_precisions.append(precision)
        label_recalls.append(recall)
    
    # F1 Macro: Average of per-label F1 scores
    f1_macro = sum(label_f1_scores) / len(label_f1_scores)
    precision_macro = sum(label_precisions) / len(label_precisions)
    recall_macro = sum(label_recalls) / len(label_recalls)
    
    return {
        'f1': f1_macro,
        'precision': precision_macro,
        'recall': recall_macro,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }


def evaluate(model, dataloader, device):
    """Evaluate on eval/holdout set"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            probs, _ = model(input_ids, attention_mask)
            
            # Apply attention mask
            mask = attention_mask.unsqueeze(-1).expand_as(probs)
            probs_masked = probs * mask
            labels_masked = labels * mask
            
            # Keep shape [batch, seq_len, 6] for per-label F1 computation
            batch_size, seq_len, num_labels = probs_masked.shape
            all_predictions.append(probs_masked.view(-1, num_labels))
            all_targets.append(labels_masked.view(-1, num_labels))
    
    predictions = torch.cat(all_predictions, dim=0)  # [total_tokens, 6]
    targets = torch.cat(all_targets, dim=0)  # [total_tokens, 6]
    
    metrics = compute_f1_score(predictions, targets)
    
    return metrics


def train_batch(model, batch, optimizer, device, o_weight=0.01):
    """
    Train on one batch with O-token downweighting.
    
    Key insight: 71% of tokens are 'O' (all labels = 0).
    We downweight them to 1% so model focuses on actual BIO labels.
    """
    model.train()
    
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    optimizer.zero_grad()
    
    probs, logits = model(input_ids, attention_mask)
    
    # Padding mask
    pad_mask = attention_mask.unsqueeze(-1).expand_as(logits)
    
    # O-token downweighting:
    # - Tokens with ANY positive label → weight 1.0
    # - Tokens with ALL zeros ('O') → weight 0.01
    has_positive = labels.sum(dim=-1) > 0  # [batch, seq_len]
    token_weights = torch.where(
        has_positive, 
        torch.ones_like(has_positive, dtype=torch.float32),
        torch.full_like(has_positive, o_weight, dtype=torch.float32)
    )
    token_weights = token_weights.unsqueeze(-1).expand_as(logits)
    
    # Combined mask
    combined_mask = pad_mask.float() * token_weights
    
    # Manual weighted BCE loss
    bce_loss = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction='none'
    )
    
    # Weighted mean
    weighted_loss = (bce_loss * combined_mask).sum() / combined_mask.sum()
    
    weighted_loss.backward()
    optimizer.step()
    
    return weighted_loss.item()


def main(data_file='data/bio_training_data.msgpack'):
    print("=" * 70)
    print("BIO TAGGER TRAINING: ITERATIVE CURRICULUM LEARNING")
    print("=" * 70)
    
    print(f"\n[1/7] Loading training data from {data_file}...")
    with open(data_file, 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
    
    examples = data['training_data']
    print(f"  Total examples: {len(examples)}")
    
    # Split: 20% holdout, remaining 80/20 train/eval
    np.random.seed(42)
    indices = np.random.permutation(len(examples))
    
    holdout_size = int(0.2 * len(examples))
    remaining_size = len(examples) - holdout_size
    eval_size = int(0.2 * remaining_size)
    
    holdout_indices = indices[:holdout_size]
    eval_indices = indices[holdout_size:holdout_size + eval_size]
    train_indices = indices[holdout_size + eval_size:]
    
    holdout_examples = [examples[i] for i in holdout_indices]
    eval_examples = [examples[i] for i in eval_indices]
    train_examples = [examples[i] for i in train_indices]
    
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Eval: {len(eval_examples)} examples")
    print(f"  Holdout: {len(holdout_examples)} examples")
    
    print("\n[2/7] Initializing model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BIOTagger()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    model = model.to(device)
    
    print("\n[3/7] Setting up training...")
    
    # Build class index: which examples have which BIO labels
    print("  Building class stratification index...")
    label_names = ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']
    
    # For each example, track which classes are present
    example_classes = []  # List of sets: which classes each example has
    class_to_examples = {label: [] for label in label_names}  # Which examples have each class
    
    for idx, example in enumerate(train_examples):
        present_classes = set()
        for label_name in label_names:
            if sum(example['labels'][label_name]) > 0:
                present_classes.add(label_name)
                class_to_examples[label_name].append(idx)
        example_classes.append(present_classes)
    
    # Compute sampling weights using log + box-cox
    class_counts = {label: len(indices) for label, indices in class_to_examples.items()}
    print(f"  Class counts: {class_counts}")
    
    # Log transform + box-cox + round
    log_counts = np.array([np.log(max(1, class_counts[label])) for label in label_names])
    
    # Box-cox (add small constant to ensure positive)
    log_counts_positive = log_counts + 1e-6
    transformed, lambda_param = boxcox(log_counts_positive)
    
    # Round and ensure minimum of 1
    sample_weights = np.maximum(1, np.round(transformed - transformed.min() + 1)).astype(int)
    class_sample_weights = dict(zip(label_names, sample_weights))
    print(f"  Sample weights: {class_sample_weights}")
    
    # O-token downweighting: 71% of tokens are 'O', we weight them at 1%
    o_weight = 0.01
    print(f"  O-token weight: {o_weight} (97.6% of loss focuses on BIO labels)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    eval_dataset = BIODataset(eval_examples)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=16,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    
    print("\n[4/7] Training with epoch-based learning...")
    print("  Strategy: Full pass through training data per epoch (without replacement)")
    print("  Early stopping: when eval F1 improvement < 1%")
    
    best_f1 = 0.0
    no_improve_count = 0
    patience = 5
    epoch = 0
    batch_size = 8
    improvement_threshold = 0.01  # 1% improvement required
    
    history = {
        'epochs': [],
        'train_losses': [],
        'eval_f1': [],
        'eval_precision': [],
        'eval_recall': []
    }
    
    start_time = time.time()
    
    while True:
        epoch += 1
        
        # Full epoch: shuffle all training examples, process in batches (without replacement)
        epoch_indices = list(range(len(train_examples)))
        np.random.shuffle(epoch_indices)
        
        epoch_dataset = BIODataset([train_examples[i] for i in epoch_indices])
        epoch_loader = DataLoader(
            epoch_dataset,
            batch_size=batch_size,
            shuffle=False,  # Already shuffled
            collate_fn=lambda b: collate_fn(b, tokenizer)
        )
        
        # Train on all batches in epoch
        epoch_losses = []
        for batch in epoch_loader:
            loss = train_batch(model, batch, optimizer, device, o_weight=o_weight)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        
        # Evaluate
        eval_metrics = evaluate(model, eval_loader, device)
        eval_f1 = eval_metrics['f1']
        
        history['epochs'].append(epoch)
        history['train_losses'].append(avg_loss)
        history['eval_f1'].append(eval_f1)
        history['eval_precision'].append(eval_metrics['precision'])
        history['eval_recall'].append(eval_metrics['recall'])
        
        print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
              f"Eval F1: {eval_f1:.4f} | P: {eval_metrics['precision']:.4f} | R: {eval_metrics['recall']:.4f}")
        
        # Check improvement (>1% required)
        improvement = eval_f1 - best_f1
        if improvement > improvement_threshold:
            best_f1 = eval_f1
            no_improve_count = 0
            torch.save(model.state_dict(), 'bio_tagger_best.pt')
        else:
            no_improve_count += 1
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"\n  ⏹️  Early stopping (F1 improvement < {improvement_threshold:.1%} for {patience} epochs)")
            break
        
        if epoch >= 100:
            print(f"\n  ⏹️  Max epochs reached")
            break
    
    train_time = time.time() - start_time
    
    print(f"\n[5/7] Training complete")
    print(f"  Epochs: {epoch}")
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Best eval F1: {best_f1:.4f}")
    
    print("\n[6/7] Loading best model...")
    model.load_state_dict(torch.load('bio_tagger_best.pt'))
    
    print("\n[7/7] Final evaluation on holdout set...")
    holdout_dataset = BIODataset(holdout_examples)
    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=16,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    
    holdout_metrics = evaluate(model, holdout_loader, device)
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS (HOLDOUT SET)")
    print("=" * 70)
    print(f"  F1 Score: {holdout_metrics['f1']:.4f}")
    print(f"  Precision: {holdout_metrics['precision']:.4f}")
    print(f"  Recall: {holdout_metrics['recall']:.4f}")
    print(f"  True Positives: {holdout_metrics['tp']:.0f}")
    print(f"  False Positives: {holdout_metrics['fp']:.0f}")
    print(f"  False Negatives: {holdout_metrics['fn']:.0f}")
    
    results = {
        'history': history,
        'final_holdout_metrics': holdout_metrics,
        'training_time_seconds': train_time,
        'epochs': epoch,
        'train_size': len(train_examples),
        'eval_size': len(eval_examples),
        'holdout_size': len(holdout_examples)
    }
    
    with open('data/training_results.msgpack', 'wb') as f:
        f.write(msgpack.packb(results, use_bin_type=True))
    
    print("\n✅ Model saved: bio_tagger_best.pt")
    print("✅ Results saved: data/training_results.msgpack")
    

if __name__ == '__main__':
    import sys
    
    data_file = 'data/bio_training_data.msgpack'
    if len(sys.argv) > 1 and sys.argv[1] == '--data':
        data_file = sys.argv[2]
    
    main(data_file)
