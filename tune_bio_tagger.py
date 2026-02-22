"""
BIO Tagger Hyperparameter Tuning with Optuna
============================================

Strategy:
1. Sample 50 examples for quick trials
2. Define hyperparameter search space (dropout, lr, optimizer, etc.)
3. Run Optuna with pruning for early stopping
4. Use warmup period: min 5 epochs before patience kicks in
5. Patience of 2 after warmup
6. Apply best hyperparams to full dataset

Hyperparameters to tune:
- Learning rate (1e-6 to 1e-3)
- Dropout rate (0.0 to 0.5)
- O-token weight (0.001 to 0.1)
- Batch size (4, 8, 16)
- Optimizer (AdamW, Adam, SGD)
- Weight decay (0 to 0.1)
- Threshold for predictions (0.1 to 0.5)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import msgpack
import numpy as np
from transformers import BertTokenizerFast, BertModel
from typing import Dict, List, Optional
import time
from tqdm import tqdm
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import optuna
from optuna.trial import Trial
import argparse
import warnings
from scipy import stats
from scipy.stats import boxcox
import json
import os
from collections import Counter
warnings.filterwarnings('ignore')

# Module-level BERT weight cache: load once, copy into every trial
# Avoids 21x BertModel.from_pretrained() disk reads during Optuna search
_BERT_STATE_DICT_CACHE = None
_BERT_CONFIG_CACHE = None


class BIODataset(Dataset):
    """Dataset that stores sentences and triplets - tokenization done in collate_fn"""
    
    def __init__(self, examples: List[Dict]):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'tokens': example['tokens'],
            'labels': example['labels'],  # Pre-computed BIO labels {B-SUBJ: [...], ...}
            'ner_spans': example.get('ner_spans', []),
        }


def compute_entity_sampling_weights(examples: List[Dict], alpha: float = 1.0):
    """
    Compute adaptive sampling weights based on entity rarity.
    
    Works with both 'triplets' and 'ner_spans' data formats.
    
    Args:
        examples: Training examples with triplets or ner_spans
        alpha: Upweighting factor for rare entities (higher = more upweighting)
    
    Returns:
        weights: Per-example sampling weights
        entity_counts: Counter of all entities for monitoring
    """
    entity_counts = Counter()
    example_entities = []
    
    for ex in examples:
        ex_entities = set()
        # Try triplets first, fall back to ner_spans
        if 'triplets' in ex:
            for triplet in ex['triplets']:
                ex_entities.add(triplet['subject'])
                ex_entities.add(triplet['predicate'])
                ex_entities.add(triplet['object'])
                entity_counts[triplet['subject']] += 1
                entity_counts[triplet['predicate']] += 1
                entity_counts[triplet['object']] += 1
        elif 'ner_spans' in ex:
            for span in ex['ner_spans']:
                # GLiNER format: [start, end, label] or dict with 'text'/'label' keys
                if isinstance(span, list):
                    start, end, label = span
                    text = ' '.join(ex['tokens'][start:end]) if 'tokens' in ex else str(span)
                    role = label
                else:
                    text = span.get('text', span.get('entity', ''))
                    role = span.get('label', span.get('role', 'UNK'))
                key = f"{role}:{text}"
                ex_entities.add(key)
                entity_counts[key] += 1
        else:
            # Fallback: use label density as weight proxy
            n_positive = sum(sum(v) for v in ex['labels'].values())
            ex_entities.add(f"density:{n_positive}")
            entity_counts[f"density:{n_positive}"] += 1
        example_entities.append(ex_entities)
    
    # Step 2: Transform counts to sampling weights
    entities = list(entity_counts.keys())
    counts = np.array([entity_counts[e] for e in entities])
    
    # Apply log
    log_counts = np.log(counts + 1)  # +1 to avoid log(0)
    
    # Apply Box-Cox transform (requires positive values)
    # Box-Cox finds optimal lambda for normalization
    try:
        transformed_counts, lambda_param = stats.boxcox(log_counts + 1)  # +1 ensures positive
    except:
        # Fallback if Box-Cox fails
        transformed_counts = log_counts
    
    # Convert to inverse frequency weights (rare entities get higher weight)
    # Normalize to [0, 1] range first
    min_val = transformed_counts.min()
    max_val = transformed_counts.max()
    if max_val > min_val:
        normalized = (transformed_counts - min_val) / (max_val - min_val)
    else:
        normalized = np.ones_like(transformed_counts)
    
    # Inverse weighting: low count = high weight
    entity_weights = 1.0 / (normalized + 0.01)  # +0.01 to avoid division by zero
    entity_weights = entity_weights ** alpha  # Apply upweighting factor
    
    # Create entity -> weight mapping
    entity_weight_map = {entity: weight for entity, weight in zip(entities, entity_weights)}
    
    # Step 3: Assign weight to each example (based on rarest entity)
    example_weights = []
    for ex_entities in example_entities:
        if len(ex_entities) > 0:
            # Use MAXIMUM weight (rarest entity determines example weight)
            ex_weight = max(entity_weight_map[e] for e in ex_entities)
        else:
            ex_weight = 1.0  # Default for examples with no triplets
        example_weights.append(ex_weight)
    
    # Normalize weights to sum to 1 (proper probability distribution)
    example_weights = np.array(example_weights)
    example_weights = example_weights / example_weights.sum()
    
    return example_weights, entity_counts


def update_sampling_weights(example_weights: np.ndarray, examples: List[Dict], 
                            sampled_indices: List[int], entity_counts: Counter, 
                            alpha: float = 1.2):
    """
    Bayesian-style update of sampling weights after each epoch.
    
    Strategy:
    1. Track which entities were seen in sampled examples
    2. Increase weights for examples containing undersampled entities
    3. Mimics Bayesian posterior update based on sampling history
    
    Args:
        example_weights: Current per-example sampling weights
        examples: All training examples
        sampled_indices: Indices of examples sampled in last epoch
        entity_counts: Total entity counts in dataset
        alpha: Upweighting factor for undersampled classes
    
    Returns:
        updated_weights: Adjusted sampling weights
    """
    # Count which entities were sampled
    sampled_entity_counts = Counter()
    for idx in sampled_indices:
        ex = examples[idx]
        if 'triplets' in ex:
            for triplet in ex['triplets']:
                sampled_entity_counts[triplet['subject']] += 1
                sampled_entity_counts[triplet['predicate']] += 1
                sampled_entity_counts[triplet['object']] += 1
        elif 'ner_spans' in ex:
            for span in ex['ner_spans']:
                # GLiNER format: [start, end, label] or dict with 'text'/'label' keys
                if isinstance(span, list):
                    start, end, label = span
                    text = ' '.join(ex['tokens'][start:end]) if 'tokens' in ex else str(span)
                    role = label
                else:
                    text = span.get('text', span.get('entity', ''))
                    role = span.get('label', span.get('role', 'UNK'))
                sampled_entity_counts[f"{role}:{text}"] += 1
    
    # Compute undersampling ratio for each entity
    # (expected proportion - actual proportion)
    total_entities = sum(entity_counts.values())
    total_sampled = sum(sampled_entity_counts.values())
    
    if total_sampled == 0:
        return example_weights  # No update if nothing sampled
    
    entity_undersampling = {}
    for entity, count in entity_counts.items():
        expected_proportion = count / total_entities
        actual_count = sampled_entity_counts.get(entity, 0)
        actual_proportion = actual_count / total_sampled if total_sampled > 0 else 0
        
        # Undersampling ratio: how much more we should sample this entity
        # Positive = undersampled, Negative = oversampled
        undersampling = expected_proportion - actual_proportion
        entity_undersampling[entity] = max(0, undersampling)  # Only boost undersampled
    
    # Update example weights based on undersampling
    updated_weights = example_weights.copy()
    for i, ex in enumerate(examples):
        # Collect entities in this example
        ex_entities = set()
        if 'triplets' in ex:
            for triplet in ex['triplets']:
                ex_entities.add(triplet['subject'])
                ex_entities.add(triplet['predicate'])
                ex_entities.add(triplet['object'])
        elif 'ner_spans' in ex:
            for span in ex['ner_spans']:
                # GLiNER format: [start, end, label] or dict with 'text'/'label' keys
                if isinstance(span, list):
                    start, end, label = span
                    text = ' '.join(ex['tokens'][start:end]) if 'tokens' in ex else str(span)
                    role = label
                else:
                    text = span.get('text', span.get('entity', ''))
                    role = span.get('label', span.get('role', 'UNK'))
                ex_entities.add(f"{role}:{text}")
        
        # Boost weight if example contains undersampled entities
        if len(ex_entities) > 0:
            max_undersampling = max(entity_undersampling.get(e, 0) for e in ex_entities)
            boost_factor = 1.0 + (alpha * max_undersampling)
            updated_weights[i] *= boost_factor
    
    # Re-normalize to probability distribution
    updated_weights = updated_weights / updated_weights.sum()
    
    return updated_weights


def detect_nonexistent_labels(examples: List[Dict], label_names: List[str]) -> List[int]:
    """
    Auto-detect labels with 0 tokens in training data (conditional removal).
    
    User requirement: "only remove it if we detect this condition (doesn't exist in class)"
    This ensures the code adapts to the data rather than hardcoding assumptions.
    
    Args:
        examples: Training examples with multi-hot label encoding
        label_names: Full list of label names (e.g., ['B-SUBJ', 'I-SUBJ', ...])
    
    Returns:
        List of label indices that should be removed (have 0 tokens)
    """
    num_labels = len(label_names)
    label_counts = {i: 0 for i in range(num_labels)}
    
    # Count tokens for each label across all examples
    for ex in examples:
        labels_dict = ex['labels']
        
        for label_idx in range(num_labels):
            label_name = label_names[label_idx]
            if label_name in labels_dict:
                # Check if any token has this label (multi-hot encoding)
                if sum(labels_dict[label_name]) > 0:
                    label_counts[label_idx] += 1
    
    # Find labels with 0 tokens
    nonexistent = [idx for idx, count in label_counts.items() if count == 0]
    
    if nonexistent:
        removed_names = [label_names[i] for i in nonexistent]
        print(f"\n[AUTO-DETECT] Found non-existent labels (0 tokens in training data):")
        print(f"    Labels to remove: {removed_names} (indices: {nonexistent})")
        print(f"    Label token counts: {label_counts}")
        print(f"    Reason: User specified 'only remove it if we detect this condition'")
        print(f"    This avoids the problem of impossible classes dragging down macro F1")
    else:
        print(f"\n[AUTO-DETECT] All labels present in training data")
        print(f"    Label token counts: {label_counts}")
    
    return nonexistent


def collate_fn(batch, tokenizer, nlp, validate=False, active_label_indices=None):
    """Convert pre-split BERT subtokens to IDs with 1:1 label alignment.
    
    Since tokens in the msgpack are already BERT wordpieces (contain ## subwords),
    we use convert_tokens_to_ids for exact 1:1 mapping instead of re-tokenizing.
    Labels are padded with 0 for [CLS] and [SEP] tokens.
    
    Args:
        active_label_indices: Optional list of label indices to keep (for conditional removal)
    """
    precomputed_labels = [item['labels'] for item in batch]
    token_lists = [item['tokens'] for item in batch]
    
    # Full label set (before detection)
    full_label_keys = ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']
    
    # Use only active labels if specified
    if active_label_indices is not None:
        label_keys = [full_label_keys[i] for i in active_label_indices]
    else:
        label_keys = full_label_keys
    
    # Build input_ids manually: [CLS] + tokens + [SEP] + [PAD...]
    all_ids = []
    all_lengths = []  # Track actual length (with CLS/SEP) for padding
    for tokens in token_lists:
        full_tokens = ['[CLS]'] + tokens + ['[SEP]']
        ids = tokenizer.convert_tokens_to_ids(full_tokens)
        all_ids.append(ids)
        all_lengths.append(len(ids))
    
    # Determine max_len (capped at 512)
    max_len = min(max(all_lengths), 512)
    batch_size = len(batch)
    
    # Dynamic num_labels based on active labels
    num_labels = len(label_keys)
    
    # Build tensors
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels_batch = torch.zeros(batch_size, max_len, num_labels)
    
    pad_id = tokenizer.pad_token_id or 0
    
    for i in range(batch_size):
        ids = all_ids[i]
        seq_len = min(len(ids), max_len)
        
        # Fill input_ids and attention_mask
        input_ids[i, :seq_len] = torch.tensor(ids[:seq_len], dtype=torch.long)
        input_ids[i, seq_len:] = pad_id
        attention_mask[i, :seq_len] = 1
        
        # Fill labels: position 0 = [CLS] (label=0), positions 1..N = tokens, position N+1 = [SEP] (label=0)
        labels_dict = precomputed_labels[i]
        n_tokens = len(labels_dict['B-SUBJ'])
        # Offset by 1 for [CLS]; cap at max_len-1 to leave room for [SEP]
        n_fill = min(n_tokens, max_len - 2)  # -2 for CLS and SEP
        
        for label_idx, label_name in enumerate(label_keys):
            label_vals = labels_dict[label_name][:n_fill]
            labels_batch[i, 1:1+n_fill, label_idx] = torch.tensor(label_vals, dtype=torch.float32)
        
        # Validation assertions (enabled with --validate flag)
        if validate:
            assert len(ids) == len(token_lists[i]) + 2, \
                f"ID count {len(ids)} != tokens {len(token_lists[i])} + 2 (CLS/SEP)"
            unk_id = tokenizer.unk_token_id
            # Only flag UNKs that weren't already [UNK] in the source tokens
            spurious_unks = sum(1 for j, x in enumerate(ids[1:-1]) 
                               if x == unk_id and token_lists[i][j] != '[UNK]')
            assert spurious_unks == 0, \
                f"Found {spurious_unks} spurious UNK tokens (not from source data)"
            assert n_tokens == len(token_lists[i]), \
                f"Label length {n_tokens} != token length {len(token_lists[i])}"
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels_batch
    }


class BIOTaggerWithDropout(nn.Module):
    """BERT + N independent binary classifiers with configurable dropout
    
    N is dynamic based on detected active labels (conditional removal of non-existent labels).
    """
    
    def __init__(self, num_labels: int = 6, dropout_rate: float = 0.1, freeze_bert: bool = True, unfreeze_layers: int = 0):
        super().__init__()
        global _BERT_STATE_DICT_CACHE, _BERT_CONFIG_CACHE
        if _BERT_STATE_DICT_CACHE is None:
            # First trial: load from pretrained (hits HuggingFace cache on disk)
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            _BERT_CONFIG_CACHE = self.bert.config
            _BERT_STATE_DICT_CACHE = {k: v.cpu().clone() for k, v in self.bert.state_dict().items()}
        else:
            # Subsequent trials: instantiate from config + copy weights from RAM
            self.bert = BertModel(_BERT_CONFIG_CACHE)
            self.bert.load_state_dict(_BERT_STATE_DICT_CACHE)
        hidden_size = self.bert.config.hidden_size
        total_layers = len(self.bert.encoder.layer)
        self.num_labels = num_labels
        
        # Freeze all BERT layers initially
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Unfreeze top N layers if specified (overrides freeze_bert)
        if unfreeze_layers == -1 or unfreeze_layers >= total_layers:
            # Unfreeze ALL layers
            for param in self.bert.parameters():
                param.requires_grad = True
            self.unfrozen_count = total_layers
        elif unfreeze_layers > 0:
            # Unfreeze top N layers
            for layer in self.bert.encoder.layer[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            self.unfrozen_count = unfreeze_layers
        elif not freeze_bert:
            # Legacy: unfreeze all if freeze_bert=False
            for param in self.bert.parameters():
                param.requires_grad = True
            self.unfrozen_count = total_layers
        else:
            self.unfrozen_count = 0
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Dynamic number of classifiers based on active labels
        # Will be 5 if I-PRED is detected as non-existent
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_labels)
        ])
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        
        logits = []
        for classifier in self.classifiers:
            logit = classifier(sequence_output).squeeze(-1)
            logits.append(logit)
        
        logits = torch.stack(logits, dim=-1)
        probs = self.sigmoid(logits)
        
        return probs, logits


def compute_f1_score(predictions, targets, threshold=0.3, return_detailed=False, active_label_indices=None, full_label_names=None):
    """Compute F1 Macro for multi-hot BIO labels with optional per-class breakdown
    
    Args:
        active_label_indices: Optional list of active label indices (for conditional removal)
        full_label_names: Full label names list (before removal)
    """
    pred_binary = (predictions > threshold).float()
    
    # Dynamic num_labels based on active labels
    num_labels = predictions.shape[-1]
    
    # Use active label names if provided
    if active_label_indices is not None and full_label_names is not None:
        label_names = [full_label_names[i] for i in active_label_indices]
    else:
        # Fallback to full label set
        label_names = ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ'][:num_labels]
    
    pred_reshaped = pred_binary.view(-1, num_labels)
    target_reshaped = targets.view(-1, num_labels)
    
    label_f1_scores = []
    label_metrics = []  # Store detailed metrics if requested
    
    for label_idx in range(num_labels):
        pred_label = pred_reshaped[:, label_idx]
        target_label = target_reshaped[:, label_idx]
        
        tp = (pred_label * target_label).sum().item()
        fp = (pred_label * (1 - target_label)).sum().item()
        fn = ((1 - pred_label) * target_label).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        label_f1_scores.append(f1)
        
        if return_detailed:
            label_metrics.append({
                'label': label_names[label_idx],
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': int(target_label.sum().item())
            })
    
    f1_macro = sum(label_f1_scores) / len(label_f1_scores)
    
    if return_detailed:
        return f1_macro, label_metrics
    return f1_macro


def evaluate(model, dataloader, device, threshold=0.3, return_detailed=False, active_label_indices=None, full_label_names=None):
    """Evaluate on eval set with optional per-class breakdown
    
    Args:
        active_label_indices: Optional list of active label indices (for conditional removal)
        full_label_names: Full label names list (before removal)
    """
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
            probs_masked = probs * mask
            labels_masked = labels * mask
            
            batch_size, seq_len, num_labels = probs_masked.shape
            all_predictions.append(probs_masked.view(-1, num_labels))
            all_targets.append(labels_masked.view(-1, num_labels))
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    return compute_f1_score(predictions, targets, threshold, return_detailed=return_detailed, 
                           active_label_indices=active_label_indices, full_label_names=full_label_names)


def train_epoch(model, dataloader, optimizer, device, o_weight=0.01, desc=""):
    """Train for one epoch"""
    model.train()
    
    epoch_losses = []
    pbar = tqdm(dataloader, desc=desc, leave=True, ncols=90, unit="batch")
    for batch in pbar:
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
        )
        token_weights = token_weights.unsqueeze(-1).expand_as(logits)
        combined_mask = pad_mask.float() * token_weights
        
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction='none'
        )
        weighted_loss = (bce_loss * combined_mask).sum() / combined_mask.sum()
        
        weighted_loss.backward()
        optimizer.step()
        
        epoch_losses.append(weighted_loss.item())
        pbar.set_postfix(loss=f"{weighted_loss.item():.4f}")
    
    return np.mean(epoch_losses)


def compute_boxcox_thresholds(label_counts: Dict[int, int], 
                             min_target: int = 2, max_target: int = 5) -> Dict[int, float]:
    """
    Compute class importance weights using Box-Cox transformation.
    
    Strategy (User spec Phase 43):
    1. Box-Cox transform class frequencies -> natural importance weights
    2. Scale transformed values to [min_target, max_target] range
    3. Convert to percentages (sum to 1.0)
    
    Args:
        label_counts: dict {label_idx: count} e.g., {0: 21, 1: 8, 2: 24, 4: 23, 5: 13}
        min_target: Minimum threshold (default 2)
        max_target: Maximum threshold (default 5)
    
    Returns:
        thresholds: dict {label_idx: threshold_percentage}
                   e.g., {0: 0.22, 1: 0.18, 2: 0.24, 4: 0.23, 5: 0.13}
    
    Example:
        >>> label_counts = {0: 21, 1: 8, 2: 24, 4: 23, 5: 13}
        >>> compute_boxcox_thresholds(label_counts, min_target=2, max_target=5)
        {0: 0.22, 1: 0.18, 2: 0.24, 4: 0.23, 5: 0.13}  # Box-Cox derived
    """
    if not label_counts:
        return {}
    
    labels = list(label_counts.keys())
    counts = np.array([label_counts[label] for label in labels], dtype=float)
    
    # Filter zero counts
    active_mask = counts > 0
    if active_mask.sum() == 0:
        return {label: 0.0 for label in labels}
    
    active_counts = counts[active_mask]
    active_labels = [labels[i] for i in range(len(labels)) if active_mask[i]]
    
    # Box-Cox transformation (find optimal lambda)
    # Add small constant to avoid log(0)
    transformed, lambda_param = boxcox(active_counts + 1e-10)
    
    # Scale to [min_target, max_target]
    t_min, t_max = transformed.min(), transformed.max()
    if t_max > t_min:
        scaled = min_target + (transformed - t_min) / (t_max - t_min) * (max_target - min_target)
    else:
        # All equal, use midpoint
        scaled = np.full_like(transformed, (min_target + max_target) / 2)
    
    # Convert to percentages (sum to 1.0)
    percentages = scaled / scaled.sum()
    
    # Map back to full label set
    thresholds = {}
    active_idx = 0
    for i, label in enumerate(labels):
        if active_mask[i]:
            thresholds[label] = float(percentages[active_idx])
            active_idx += 1
        else:
            thresholds[label] = 0.0
    
    return thresholds


def greedy_select_by_label_ratio(examples: List[Dict], 
                                 label_thresholds: Dict[int, float],
                                 seed: int = 42,
                                 max_fraction: float = 1.0) -> List[int]:
    """
    Greedy selection: pick examples until all label ratios meet/exceed thresholds.
    
    Strategy (User spec Phase 43):
    1. Count tokens per label in each example
    2. Track current ratios (tokens per label / total tokens)
    3. Repeatedly select example that helps lowest-ratio label
    4. Stop when all ratios >= thresholds
    5. Use seed for reproducibility
    
    Args:
        examples: list of training examples with 'labels' dict
        label_thresholds: dict {label_idx: threshold_percentage}
                         e.g., {0: 0.22, 1: 0.18, 2: 0.24, 4: 0.23, 5: 0.13}
        seed: Random seed (use epoch number for training, 42 for eval)
        max_fraction: Maximum fraction of examples to select (default 1.0 = no cap).
                      Set to e.g. 0.35 for eval splits to guarantee training examples remain.
    
    Returns:
        selected_indices: list of example indices
    
    Example:
        >>> thresholds = {0: 0.22, 1: 0.18, 2: 0.24, 4: 0.23, 5: 0.13}
        >>> indices = greedy_select_by_label_ratio(train_examples, thresholds, seed=1)
        >>> len(indices)  # Variable, until all thresholds met
        47
    """
    np.random.seed(seed)
    max_count = int(len(examples) * max_fraction) if max_fraction < 1.0 else len(examples)
    
    # Count tokens per label in each example
    example_label_tokens = {}  # {example_idx: {label_idx: token_count}}
    for idx, ex in enumerate(examples):
        example_label_tokens[idx] = {}
        for label_type_idx, token_labels in enumerate(ex['labels'].values()):
            token_count = sum(token_labels)
            if token_count > 0:
                example_label_tokens[idx][label_type_idx] = token_count
    
    # Active labels (those with thresholds)
    active_labels = [label for label, thresh in label_thresholds.items() if thresh > 0]
    
    # Track current state
    selected_indices = set()
    current_tokens = {label: 0 for label in active_labels}
    
    # Shuffle examples for randomness in ties
    all_indices = list(range(len(examples)))
    np.random.shuffle(all_indices)
    
    # Greedy selection loop
    while True:
        total_tokens = sum(current_tokens.values())
        if total_tokens == 0:
            # First selection, pick any example with active labels
            for idx in all_indices:
                if any(label in example_label_tokens[idx] for label in active_labels):
                    selected_indices.add(idx)
                    for label, count in example_label_tokens[idx].items():
                        if label in current_tokens:
                            current_tokens[label] += count
                    break
            continue
        
        # Compute current ratios
        current_ratios = {label: current_tokens[label] / total_tokens for label in active_labels}
        
        # Find label with lowest ratio relative to threshold
        min_ratio_label = None
        min_relative_ratio = float('inf')
        for label in active_labels:
            if label_thresholds[label] > 0:
                relative_ratio = current_ratios[label] / label_thresholds[label]
                if relative_ratio < min_relative_ratio:
                    min_relative_ratio = relative_ratio
                    min_ratio_label = label
        
        # Check if all thresholds met
        all_met = all(
            current_ratios[label] >= label_thresholds[label]
            for label in active_labels
            if label_thresholds[label] > 0
        )
        
        if all_met:
            break
        
        # Find best example to add (helps lowest-ratio label most)
        best_idx = None
        best_contribution = 0
        
        for idx in all_indices:
            if idx in selected_indices:
                continue
            
            # Calculate contribution to min_ratio_label
            contribution = example_label_tokens[idx].get(min_ratio_label, 0)
            
            if contribution > best_contribution:
                best_contribution = contribution
                best_idx = idx
        
        if best_idx is None:
            # No more examples can help, stop
            break
        
        if len(selected_indices) >= max_count:
            # Hit the cap — stop to preserve examples for training
            break
        
        # Add best example
        selected_indices.add(best_idx)
        for label, count in example_label_tokens[best_idx].items():
            if label in current_tokens:
                current_tokens[label] += count
    
    return list(selected_indices)


def objective(trial: Trial, train_examples: List[Dict], eval_examples: List[Dict], 
              tokenizer, device, min_warmup_epochs: int = 5, patience: int = 2,
              samples_per_epoch: int = 50, validate: bool = False,
              active_label_indices: List[int] = None, full_label_names: List[str] = None,
              num_labels: int = 6):
    """
    Optuna objective function with adaptive entity-based sampling.
    
    Training strategy:
    - Min warmup epochs: 5 (no early stopping during warmup)
    - Patience: 2 (stop if no improvement for 2 epochs after warmup)
    - Samples per epoch: 50 (adaptive sampling with entity balancing)
    
    Args:
        active_label_indices: Optional list of active label indices (after conditional removal)
        full_label_names: Full label names list (before removal)
        num_labels: Number of active labels (dynamic, 5 or 6 depending on detection)
    """
    
    # Hyperparameters to tune
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'Adam', 'Lion'])
    
    # Optimizer-specific learning rates (Lion needs ~10x lower)
    if optimizer_name == 'Lion':
        lr = trial.suggest_float('learning_rate', 5e-7, 5e-6, log=True)
    else:  # Adam/AdamW
        lr = trial.suggest_float('learning_rate', 5e-6, 5e-5, log=True)
    
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    o_weight = trial.suggest_float('o_weight', 0.001, 0.01, log=True)
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
    threshold = trial.suggest_float('threshold', 0.3, 0.6)
    
    # Fixed structural parameters (not tuned)
    batch_size = 8
    unfreeze_layers = trial.study.user_attrs.get('unfreeze_layers', 12)
    
    # Create model with dynamic num_labels (5 or 6 depending on detection)
    model = BIOTaggerWithDropout(num_labels=num_labels, dropout_rate=dropout, unfreeze_layers=unfreeze_layers)
    model = model.to(device)
    
    # Create optimizer
    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # Lion
        from lion_pytorch import Lion
        optimizer = Lion(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Compute class distribution for resampling (only active labels)
    # Use active_label_indices if provided, otherwise all 6 labels
    labels_to_count = active_label_indices if active_label_indices is not None else list(range(6))
    
    label_counts = {i: 0 for i in labels_to_count}
    for example in train_examples:
        labels_dict = example['labels']
        label_keys = list(labels_dict.keys())
        
        for label_idx in labels_to_count:
            if label_idx < len(label_keys):
                label_name = label_keys[label_idx]
                # Check if any token has this label
                if sum(labels_dict[label_name]) > 0:
                    label_counts[label_idx] += 1
    
    # Compute Box-Cox thresholds (min=2, max=5)
    label_thresholds = compute_boxcox_thresholds(label_counts, min_target=2, max_target=5)
    
    if trial.number == 0:  # Print resampling info once
        tqdm.write(f"\n  [RESAMPLING - Box-Cox] Class distribution: {label_counts}")
        tqdm.write(f"  [RESAMPLING - Box-Cox] Thresholds (percentages): {label_thresholds}")
        tqdm.write(f"  [RESAMPLING - Box-Cox] Strategy: Greedy select until ratios meet thresholds")
    
    # Create datasets
    train_dataset = BIODataset(train_examples)
    eval_dataset = BIODataset(eval_examples)
    
    # Eval loader (no sampling, use all data)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=16,
        collate_fn=lambda b: collate_fn(b, tokenizer, None, validate=validate, active_label_indices=active_label_indices)
    )
    
    best_f1 = 0.0
    no_improve_count = 0
    max_epochs = 20
    
    # FAST tuning: hardcoded low warmup/patience for trials
    trial_warmup = 5
    trial_patience = 8
    
    for epoch in range(1, max_epochs + 1):
        # Greedy resample: select until label ratios meet Box-Cox thresholds
        # Use epoch as seed for reproducibility (Phase 43 user spec)
        epoch_indices = greedy_select_by_label_ratio(train_examples, label_thresholds, seed=epoch)
        resampled_data = [train_examples[i] for i in epoch_indices]
        
        # Recreate dataset and loader for this epoch
        epoch_train_dataset = BIODataset(resampled_data)
        train_loader = DataLoader(
            epoch_train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Still shuffle resampled data
            collate_fn=lambda b: collate_fn(b, tokenizer, None, validate=validate, active_label_indices=active_label_indices)
        )
        
        # Train
        loss = train_epoch(model, train_loader, optimizer, device, o_weight=o_weight, desc=f"T{trial.number} E{epoch}/{max_epochs}")
        
        # Evaluate
        eval_f1, per_class = evaluate(model, eval_loader, device, threshold=threshold, return_detailed=True,
                                     active_label_indices=active_label_indices, full_label_names=full_label_names)
        
        # Print epoch progress
        tqdm.write(f"    Trial {trial.number} Epoch {epoch:2d} | Loss: {loss:.4f} | Macro F1: {eval_f1:.4f}")
        tqdm.write(f"      {'Label':<10} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Supp':>6}")
        for m in per_class:
            tqdm.write(f"      {m['label']:<10} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>6}")
        
        # Report to Optuna for pruning
        trial.report(eval_f1, epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Check improvement
        if eval_f1 > best_f1:
            best_f1 = eval_f1
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Early stopping (only after warmup) - use hardcoded fast params for trials
        if epoch >= trial_warmup and no_improve_count > trial_patience:
            break
    
    return best_f1


def main():
    parser = argparse.ArgumentParser(description='Tune BIO tagger hyperparameters with Optuna')
    parser.add_argument('--data', type=str, default='data/bio_training_250chunks_complete.msgpack',
                       help='Training data file')
    parser.add_argument('--tune-samples', type=int, default=50,
                       help='Number of examples for tuning (default: 50)')
    parser.add_argument('--samples-per-epoch', type=int, default=50,
                       help='Examples sampled per epoch with adaptive weighting (default: 50)')
    parser.add_argument('--n-trials', type=int, default=20,
                       help='Number of Optuna trials (default: 20)')
    parser.add_argument('--min-warmup', type=int, default=5,
                       help='Minimum epochs before early stopping (default: 5)')
    parser.add_argument('--patience', type=int, default=8,
                       help='Patience after warmup (default: 8)')
    parser.add_argument('--unfreeze-layers', type=int, default=12,
                       help='Number of top BERT layers to unfreeze (-1 or 12 = all layers, 0 = frozen, default: 12)')
    parser.add_argument('--validate', action='store_true',
                       help='Enable data pipeline validation assertions at each step')
    args = parser.parse_args()
    
    print("=" * 70)
    print("BIO TAGGER HYPERPARAMETER TUNING WITH OPTUNA")
    print("=" * 70)
    
    print(f"\n[1/5] Loading training data from {args.data}...")
    with open(args.data, 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
    
    all_examples = data['training_data']
    print(f"  Total examples: {len(all_examples)}")
    
    # Auto-detect non-existent labels (conditional removal per user requirement)
    full_label_names = ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']
    nonexistent_label_indices = detect_nonexistent_labels(all_examples, full_label_names)
    
    # Create active label list (excluding non-existent)
    active_label_indices = [i for i in range(6) if i not in nonexistent_label_indices]
    num_active_labels = len(active_label_indices)
    
    if nonexistent_label_indices:
        print(f"    [OK] Proceeding with {num_active_labels} active labels (removed {len(nonexistent_label_indices)})")
        print(f"    Active labels: {[full_label_names[i] for i in active_label_indices]}")
    else:
        print(f"    [OK] All 6 labels active (no removal needed)")
    
    # Validate data structure if --validate flag is set
    if args.validate:
        print("\n  [VALIDATE] Checking data structure...")
        for i, ex in enumerate(all_examples):
            assert 'tokens' in ex, f"Example {i} missing 'tokens'"
            assert 'labels' in ex, f"Example {i} missing 'labels'"
            for key in ['B-SUBJ','I-SUBJ','B-PRED','I-PRED','B-OBJ','I-OBJ']:
                assert key in ex['labels'], f"Example {i} missing label key '{key}'"
                assert len(ex['labels'][key]) == len(ex['tokens']), \
                    f"Example {i}: label '{key}' length {len(ex['labels'][key])} != tokens {len(ex['tokens'])}"
        print(f"  [VALIDATE] All {len(all_examples)} examples pass structure checks")
    
    # Sample for tuning
    np.random.seed(42)
    indices = np.random.permutation(len(all_examples))
    
    tune_size = min(args.tune_samples, len(all_examples))
    tune_indices = indices[:tune_size]
    tune_examples = [all_examples[i] for i in tune_indices]
    
    # Stratified split ensuring min 4 examples per BIO label
    from collections import defaultdict
    
    label_to_examples = defaultdict(list)
    # Count tokens per label per example (for token-based eval selection)
    example_token_counts = {}  # {example_idx: {label_idx: token_count}}
    for idx, ex in enumerate(tune_examples):
        example_token_counts[idx] = {}
        for label_type_idx, token_labels in enumerate(ex['labels'].values()):
            token_count = sum(token_labels)
            if token_count > 0:
                example_token_counts[idx][label_type_idx] = token_count
        
        # Also track which examples have each label
        for label_idx in example_token_counts[idx].keys():
            label_to_examples[label_idx].append(idx)
    
    # Select eval examples: Box-Cox thresholds (same algorithm as training)
    print("  [Stratified Eval Split - Box-Cox]")
    
    # Count total tokens per label across all tune examples (for Box-Cox)
    label_token_counts = {}
    for idx in range(len(tune_examples)):
        for label_idx, token_count in example_token_counts[idx].items():
            label_token_counts[label_idx] = label_token_counts.get(label_idx, 0) + token_count
    
    # Compute Box-Cox thresholds (min=2, max=5)
    eval_thresholds = compute_boxcox_thresholds(label_token_counts, min_target=2, max_target=5)
    
    print(f"  Box-Cox thresholds (percentages): {eval_thresholds}")
    print(f"  Strategy: Greedy select until label ratios meet thresholds")
    
    # Greedy selection using Box-Cox thresholds (fixed seed for reproducibility)
    # max_fraction=0.35 caps eval at 35% of tune pool so training keeps ≥65%
    eval_indices = set(greedy_select_by_label_ratio(tune_examples, eval_thresholds, seed=42, max_fraction=0.35))
    
    # Count actual token distribution in selected eval set
    actual_eval_tokens = {}
    for idx in eval_indices:
        for label_idx, token_count in example_token_counts[idx].items():
            actual_eval_tokens[label_idx] = actual_eval_tokens.get(label_idx, 0) + token_count
    
    total_eval_tokens = sum(actual_eval_tokens.values())
    eval_ratios = {label: count / total_eval_tokens for label, count in actual_eval_tokens.items()} if total_eval_tokens > 0 else {}
    
    print(f"  Actual token distribution: {actual_eval_tokens}")
    print(f"  Actual ratios: {eval_ratios}")
    print(f"  Total eval tokens: {total_eval_tokens}")
    
    train_indices = set(range(len(tune_examples))) - eval_indices
    tune_eval = [tune_examples[i] for i in sorted(eval_indices)]
    tune_train = [tune_examples[i] for i in sorted(train_indices)]
    
    print(f"  Tuning train: {len(tune_train)} examples")
    print(f"  Tuning eval: {len(tune_eval)} examples ({100*len(tune_eval)/len(tune_examples):.1f}%)")
    print("  Eval label counts:")
    for label_idx in range(6):
        count = sum(1 for i in eval_indices if i in label_to_examples[label_idx])
        print(f"    Label {label_idx}: {count} examples")
    
    print("\n[2/5] Initializing...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    print(f"\n[3/5] Running Optuna hyperparameter search...")
    print(f"  Trials: {args.n_trials}")
    print(f"  Samples per epoch: {args.samples_per_epoch} (adaptive entity-based)")
    print(f"  Warmup epochs: {args.min_warmup}")
    print(f"  Patience after warmup: {args.patience}")
    
    # Create study with median pruner
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=args.min_warmup
        )
    )
    
    # Store unfreeze_layers in study user_attrs so objective function can access it
    study.set_user_attr('unfreeze_layers', args.unfreeze_layers)
    
    # WARM START: Load previous best hyperparameters if available
    best_params_file = 'best_hyperparams.json'
    if os.path.exists(best_params_file):
        print(f"\n  [WARM START] Loading previous best hyperparameters from {best_params_file}")
        with open(best_params_file, 'r') as f:
            prev_best = json.load(f)
        
        print(f"     Previous best F1: {prev_best.get('best_f1', 'unknown')}")
        print(f"     Enqueuing as Trial 0...")
        
        # Enqueue the previous best as the first trial
        study.enqueue_trial(prev_best['params'])
    else:
        print(f"\n  No previous hyperparameters found ({best_params_file} does not exist)")
        print(f"  Starting fresh exploration...")
    
    # Report unfreezing configuration
    if args.unfreeze_layers == -1 or args.unfreeze_layers >= 12:
        print(f"  Unfreezing: ALL 12 BERT layers (full fine-tuning)")
    elif args.unfreeze_layers > 0:
        print(f"  Unfreezing: top {args.unfreeze_layers} BERT layers (partial fine-tuning)")
    else:
        print(f"  Unfreezing: NONE (classifier-only training)")
    
    # Optimize
    study.optimize(
        lambda trial: objective(
            trial, tune_train, tune_eval, tokenizer, device,
            min_warmup_epochs=args.min_warmup,
            patience=args.patience,
            samples_per_epoch=args.samples_per_epoch,
            validate=args.validate,
            active_label_indices=active_label_indices,
            full_label_names=full_label_names,
            num_labels=num_active_labels
        ),
        n_trials=args.n_trials,
        show_progress_bar=True
    )
    
    print(f"\n[4/5] Best trial results:")
    best_trial = study.best_trial
    print(f"  Best F1: {best_trial.value:.4f}")
    print(f"  Best params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best params for next run (warm start)
    best_params = best_trial.params
    best_params_with_metadata = {
        'params': best_params,
        'best_f1': best_trial.value,
        'trial_number': best_trial.number,
        'n_trials_run': len(study.trials),
        'unfreeze_layers': args.unfreeze_layers
    }
    
    with open('best_hyperparams.json', 'w') as f:
        json.dump(best_params_with_metadata, f, indent=2)
    print(f"\n  Saved best hyperparameters to best_hyperparams.json for warm-start")
    
    print(f"\n[5/5] Training on full dataset with adaptive entity-based sampling...")
    
    # Stratified split ensuring min 4 examples per BIO label
    from collections import defaultdict
    
    label_to_examples = defaultdict(list)
    for idx, ex in enumerate(all_examples):
        labels_present = set()
        for label_type_idx, token_labels in enumerate(ex['labels'].values()):
            if sum(token_labels) > 0:
                labels_present.add(label_type_idx)
        for label_idx in labels_present:
            label_to_examples[label_idx].append(idx)
    
    # Select eval examples: max(4, 20% per label)
    min_per_label = 4
    eval_indices = set()
    np.random.seed(42)
    
    print("  [Stratified Eval Split]")
    for label_idx in range(6):
        indices_for_label = list(label_to_examples[label_idx])
        if len(indices_for_label) == 0:
            print(f"    WARNING: Label {label_idx} has 0 examples")
            continue
        
        eval_count = max(min_per_label, int(0.2 * len(indices_for_label)))
        np.random.shuffle(indices_for_label)
        eval_indices.update(indices_for_label[:eval_count])
    
    train_indices = set(range(len(all_examples))) - eval_indices
    eval_examples = [all_examples[i] for i in sorted(eval_indices)]
    train_examples = [all_examples[i] for i in sorted(train_indices)]
    
    # Use ALL training data per epoch (no subsampling for full training)
    full_samples_per_epoch = len(train_examples)
    
    print(f"  Train pool: {len(train_examples)} ({100*len(train_examples)/len(all_examples):.1f}%)")
    print(f"  Eval: {len(eval_examples)} ({100*len(eval_examples)/len(all_examples):.1f}%)")
    print("  Eval label counts:")
    for label_idx in range(6):
        count = sum(1 for i in eval_indices if i in label_to_examples[label_idx])
        print(f"    Label {label_idx}: {count} examples")
    print(f"  Samples per epoch: {full_samples_per_epoch} (full pass, entity-weighted)")
    
    # Fixed structural parameters
    batch_size = 8
    
    # Compute initial entity-based sampling weights
    train_weights, entity_counts = compute_entity_sampling_weights(train_examples, alpha=1.0)
    
    print(f"  Unique entities: {len(entity_counts)}")
    print(f"  Total entity occurrences: {sum(entity_counts.values())}")
    
    # Create model with best params and dynamic num_labels
    model = BIOTaggerWithDropout(
        num_labels=num_active_labels,
        dropout_rate=best_params['dropout'],
        unfreeze_layers=args.unfreeze_layers
    )
    
    # Report unfrozen layers
    if args.unfreeze_layers == -1 or args.unfreeze_layers >= 12:
        print(f"  Unfreezing ALL {model.unfrozen_count} BERT layers (full fine-tuning)")
    elif args.unfreeze_layers > 0:
        print(f"  Unfreezing top {model.unfrozen_count} BERT layers (partial fine-tuning)")
    else:
        print(f"  BERT layers FROZEN (classifier-only training)")
    
    model = model.to(device)
    
    # Create optimizer
    if best_params['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=best_params['learning_rate'],
            weight_decay=best_params['weight_decay']
        )
    elif best_params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=best_params['learning_rate'],
            weight_decay=best_params['weight_decay']
        )
    else:  # Lion
        from lion_pytorch import Lion
        optimizer = Lion(
            model.parameters(),
            lr=best_params['learning_rate'],
            weight_decay=best_params['weight_decay']
        )
    
    # Create dataloaders
    train_dataset = BIODataset(train_examples)
    eval_dataset = BIODataset(eval_examples)
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=16,
        collate_fn=lambda b: collate_fn(b, tokenizer, None, validate=args.validate, active_label_indices=active_label_indices)
    )
    
    # Training loop - FULL TRAINING with generous patience and adaptive sampling
    best_f1 = 0.0
    no_improve_count = 0
    max_epochs = 100
    final_warmup = 20
    final_patience = 20
    
    print("\n  Training with optimized hyperparameters + adaptive entity sampling...")
    for epoch in range(1, max_epochs + 1):
        # Create weighted sampler for this epoch (without replacement)
        # Cap samples at dataset size to avoid sampling error
        actual_samples = min(full_samples_per_epoch, len(train_dataset))
        sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=actual_samples,
            replacement=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=lambda b: collate_fn(b, tokenizer, None, active_label_indices=active_label_indices)
        )
        
        # Track sampled indices for weight updates
        sampled_indices = list(sampler)
        
        # Train
        loss = train_epoch(model, train_loader, optimizer, device, o_weight=best_params['o_weight'], desc=f"Final E{epoch}/{max_epochs}")
        eval_f1, per_class = evaluate(model, eval_loader, device, threshold=best_params['threshold'], return_detailed=True,
                                     active_label_indices=active_label_indices, full_label_names=full_label_names)
        
        # Update sampling weights (Bayesian-style adaptive reweighting)
        train_weights = update_sampling_weights(
            train_weights, train_examples, sampled_indices, entity_counts, alpha=1.2
        )
        
        print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Macro F1: {eval_f1:.4f}")
        print(f"    {'Label':<10} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Supp':>6}")
        for m in per_class:
            print(f"    {m['label']:<10} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>6}")
        
        if eval_f1 > best_f1:
            best_f1 = eval_f1
            no_improve_count = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'active_labels': [full_label_names[i] for i in active_label_indices],
            }, 'bio_tagger_atomic.pt')
        else:
            no_improve_count += 1
        
        if epoch >= final_warmup and no_improve_count >= final_patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break
    
    # Final evaluation on eval set with per-class breakdown
    _ckpt = torch.load('bio_tagger_atomic.pt')
    model.load_state_dict(_ckpt['model_state_dict'] if 'model_state_dict' in _ckpt else _ckpt)
    final_eval_f1, per_class_metrics = evaluate(
        model, eval_loader, device, 
        threshold=best_params['threshold'],
        return_detailed=True,
        active_label_indices=active_label_indices,
        full_label_names=full_label_names
    )
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Training completed on {len(train_examples)} examples ({100*len(train_examples)/len(all_examples):.1f}% of data)")
    print(f"  Best Eval F1: {best_f1:.4f}")
    print(f"  Final Eval F1 (Macro): {final_eval_f1:.4f}")
    
    print(f"\n  Per-Class Performance:")
    print(f"  {'Label':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*54}")
    for metric in per_class_metrics:
        print(f"  {metric['label']:<10} {metric['precision']:>10.4f} {metric['recall']:>10.4f} "
              f"{metric['f1']:>10.4f} {metric['support']:>10}")
    
    print(f"\n  Best hyperparameters:")
    for key, value in best_params.items():
        print(f"    {key}: {value}")
    
    # Save results
    results = {
        'best_params': best_params,
        'best_eval_f1': best_f1,
        'final_eval_f1': final_eval_f1,
        'n_trials': args.n_trials,
        'train_examples': len(train_examples),
        'total_examples': len(all_examples)
    }
    
    with open('data/tuning_results.msgpack', 'wb') as f:
        f.write(msgpack.packb(results, use_bin_type=True))
    
    print("\nTuned model saved: bio_tagger_atomic.pt")
    print("Results saved: data/tuning_results.msgpack")


if __name__ == '__main__':
    main()
