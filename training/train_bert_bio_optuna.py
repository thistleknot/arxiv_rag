"""
BERT BIO Tagger with Optuna - MULTI-CLASS (7 classes)

Correct architecture: Each token belongs to ONE of 7 classes:
  O, B-SUBJ, I-SUBJ, B-PRED, I-PRED, B-OBJ, I-OBJ

Uses cross-entropy loss + softmax (not binary classifiers).
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import msgpack
import numpy as np
from transformers import BertTokenizerFast, BertModel
from typing import Dict, List
import time
import optuna
from optuna.trial import TrialState


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
        # Check which label is active (should be at most one)
        active_labels = []
        for label_name in ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ']:
            if labels_dict[label_name][i] == 1:
                active_labels.append(label_name)
        
        if len(active_labels) == 0:
            single_labels.append(LABEL_TO_ID['O'])
        elif len(active_labels) == 1:
            single_labels.append(LABEL_TO_ID[active_labels[0]])
        else:
            # Multiple labels active - take first one (data error)
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
    labels_batch = torch.full((batch_size, max_len), -100, dtype=torch.long)  # -100 = ignore
    
    pad_id = tokenizer.pad_token_id or 0
    
    for i in range(batch_size):
        ids = all_ids[i]
        seq_len = min(len(ids), max_len)
        
        input_ids[i, :seq_len] = torch.tensor(ids[:seq_len], dtype=torch.long)
        input_ids[i, seq_len:] = pad_id
        attention_mask[i, :seq_len] = 1
        
        # Labels: position 0=[CLS] (ignore), 1..N=tokens, N+1=[SEP] (ignore)
        labels_dict = precomputed_labels[i]
        single_labels = convert_multihot_to_single_label(labels_dict)
        
        n_fill = min(len(single_labels), max_len - 2)
        labels_batch[i, 1:1+n_fill] = torch.tensor(single_labels[:n_fill], dtype=torch.long)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels_batch
    }


class BIOTaggerMultiClass(nn.Module):
    """BERT + single multi-class classifier (7 classes) with unfrozen top layers"""
    
    def __init__(self, dropout=0.1, unfreeze_layers=2, num_classes=7):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size
        
        # Freeze all BERT parameters initially
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Unfreeze top N encoder layers
        total_layers = len(self.bert.encoder.layer)  # 12 for bert-base
        unfrozen_start = total_layers - unfreeze_layers
        for i in range(unfrozen_start, total_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def get_param_groups(self, lr_bert, lr_head):
        """Return parameter groups for discriminative LR"""
        bert_params = [p for p in self.bert.parameters() if p.requires_grad]
        head_params = list(self.classifier.parameters())
        
        return [
            {'params': bert_params, 'lr': lr_bert},
            {'params': head_params, 'lr': lr_head}
        ]
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


def compute_f1_macro(predictions, targets, num_classes=7):
    """F1 macro for 7 classes, ignoring -100 (padding/special tokens)"""
    # predictions: [batch*seq_len] (class IDs)
    # targets: [batch*seq_len] (class IDs, with -100 for ignore)
    
    valid_mask = targets != -100
    valid_preds = predictions[valid_mask]
    valid_targets = targets[valid_mask]
    
    if len(valid_targets) == 0:
        return 0.0
    
    # Compute F1 per class
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
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.append(predictions.view(-1))
            all_targets.append(labels.view(-1))
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return compute_f1_macro(predictions, targets)


def train_epoch(model, dataloader, optimizer, device, class_weights=None):
    model.train()
    epoch_losses = []
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        
        # logits: [batch, seq_len, num_classes]
        # labels: [batch, seq_len]
        loss = criterion(logits.view(-1, 7), labels.view(-1))
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        epoch_losses.append(loss.item())
    
    return sum(epoch_losses) / len(epoch_losses)


def objective(trial, train_examples, eval_examples, device, tokenizer,
              trial_frac=0.1, trial_epochs=20, trial_patience=5):
    """Optuna objective function — trains on trial_frac of data, fast early stopping"""
    
    # Subsample training data for speed
    n_trial = max(1, int(trial_frac * len(train_examples)))
    rng = np.random.RandomState(trial.number)
    trial_indices = rng.choice(len(train_examples), size=n_trial, replace=False)
    train_examples = [train_examples[i] for i in trial_indices]

    # Hyperparameters to tune
    lr_bert = trial.suggest_float('lr_bert', 1e-6, 5e-5, log=True)
    lr_head = trial.suggest_float('lr_head', 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
    o_weight = trial.suggest_float('o_weight', 0.001, 0.05)
    unfreeze_layers = trial.suggest_int('unfreeze_layers', 2, 6)
    
    # Create datasets
    train_dataset = BIODataset(train_examples)
    eval_dataset = BIODataset(eval_examples)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=16,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    
    # Create model
    model = BIOTaggerMultiClass(dropout=dropout, unfreeze_layers=unfreeze_layers).to(device)
    
    # Create optimizer with discriminative LR
    param_groups = model.get_param_groups(lr_bert, lr_head)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    
    # Class weights (downweight O tokens)
    class_weights = torch.ones(7, device=device)
    class_weights[0] = o_weight  # O class
    
    # Training loop
    best_f1 = 0.0
    no_improve = 0
    
    for epoch in range(1, trial_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device, class_weights)
        eval_f1 = evaluate(model, eval_loader, device)
        
        if eval_f1 > best_f1:
            best_f1 = eval_f1
            no_improve = 0
        else:
            no_improve += 1
        
        # Report intermediate values for pruning
        trial.report(eval_f1, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if no_improve >= trial_patience:
            break
    
    return best_f1


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train BERT BIO Tagger with Optuna')
    parser.add_argument('--train', type=str, default='data/bio_training_250chunks_complete.msgpack',
                       help='Training data msgpack file')
    parser.add_argument('--output', type=str, default='bio_tagger_multiclass.pt',
                       help='Output model path')
    parser.add_argument('--trials', type=int, default=20,
                       help='Number of Optuna trials')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience for full training')
    parser.add_argument('--max-epochs', type=int, default=100,
                       help='Maximum epochs for full training')
    parser.add_argument('--trial-frac', type=float, default=0.1,
                       help='Fraction of training data used per Optuna trial (default 0.1)')
    parser.add_argument('--trial-epochs', type=int, default=20,
                       help='Max epochs per Optuna trial (default 20)')
    parser.add_argument('--trial-patience', type=int, default=5,
                       help='Early stopping patience per Optuna trial (default 5)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("BERT BIO TAGGER - MULTI-CLASS WITH OPTUNA")
    print("=" * 70)
    
    # Config
    DATA_FILE = args.train
    N_TRIALS = args.trials
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Training data: {DATA_FILE}")
    print(f"Output model: {args.output}")
    print(f"Optuna trials: {N_TRIALS}")
    print(f"Trial frac: {args.trial_frac} | Trial epochs: {args.trial_epochs} | Trial patience: {args.trial_patience}")
    print(f"Full training max epochs: {args.max_epochs} | Patience: {args.patience}")
    
    # Load data
    print(f"\nLoading {DATA_FILE}...")
    with open(DATA_FILE, 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
    
    all_examples = data['training_data']
    print(f"  {len(all_examples)} examples")
    
    # Split: 80% train, 10% tune-eval, 10% final-test
    np.random.seed(42)
    indices = np.random.permutation(len(all_examples))
    
    n_test = max(1, int(0.1 * len(all_examples)))
    n_tune_eval = max(1, int(0.1 * len(all_examples)))
    
    tune_train_examples = [all_examples[i] for i in indices[n_test+n_tune_eval:]]
    tune_eval_examples = [all_examples[i] for i in indices[n_test:n_test+n_tune_eval]]
    test_examples = [all_examples[i] for i in indices[:n_test]]
    
    print(f"  Tuning train: {len(tune_train_examples)}")
    print(f"  Tuning eval: {len(tune_eval_examples)}")
    print(f"  Final test: {len(test_examples)}")
    
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    # Optuna study
    print(f"\nRunning Optuna ({N_TRIALS} trials)...")
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    study.optimize(
        lambda trial: objective(
            trial, tune_train_examples, tune_eval_examples, device, tokenizer,
            trial_frac=args.trial_frac,
            trial_epochs=args.trial_epochs,
            trial_patience=args.trial_patience,
        ),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )
    
    # Best hyperparameters
    print("\n" + "=" * 70)
    print("BEST HYPERPARAMETERS")
    print("=" * 70)
    best_params = study.best_params
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\n  Best tuning F1: {study.best_value:.4f}")
    
    # Full training with best params
    print("\n" + "=" * 70)
    print("FULL TRAINING WITH BEST PARAMS")
    print("=" * 70)
    
    full_train = tune_train_examples + tune_eval_examples
    print(f"  Train: {len(full_train)} | Test: {len(test_examples)}")
    
    train_dataset = BIODataset(full_train)
    test_dataset = BIODataset(test_examples)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=best_params['batch_size'],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    
    model = BIOTaggerMultiClass(
        dropout=best_params['dropout'], 
        unfreeze_layers=best_params['unfreeze_layers']
    ).to(device)
    
    param_groups = model.get_param_groups(best_params['lr_bert'], best_params['lr_head'])
    optimizer = torch.optim.AdamW(param_groups, weight_decay=best_params['weight_decay'])
    
    class_weights = torch.ones(7, device=device)
    class_weights[0] = best_params['o_weight']
    
    best_f1 = 0.0
    patience = args.patience
    no_improve = 0
    
    print("\nTraining...")
    start_time = time.time()
    
    for epoch in range(1, args.max_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device, class_weights)
        test_f1 = evaluate(model, test_loader, device)
        
        print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Test F1: {test_f1:.4f}")
        
        if test_f1 > best_f1:
            best_f1 = test_f1
            no_improve = 0
            torch.save(model.state_dict(), args.output)
            print(f"    -> New best! Saved.")
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break
    
    elapsed = time.time() - start_time
    
    # Final results
    model.load_state_dict(torch.load(args.output))
    final_f1 = evaluate(model, test_loader, device)
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Best Test F1: {best_f1:.4f}")
    print(f"  Final Test F1: {final_f1:.4f}")
    print(f"  Training time: {elapsed/60:.1f} minutes")
    print(f"\n  Model saved: {args.output}")


if __name__ == '__main__':
    main()
