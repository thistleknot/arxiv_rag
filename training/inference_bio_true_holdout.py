"""
CORRECTED: Inference on TRULY HELD-OUT test data

The bug was that we were using indices[:n_test] which gives us the EVAL set
used during Optuna tuning, NOT the final held-out test set.

Correct split (matching train_bert_bio_optuna.py):
- indices[n_test+n_tune_eval:] = training (334 examples)
- indices[n_test:n_test+n_tune_eval] = tuning eval (41 examples) <-- SEEN DURING TRAINING
- indices[:n_test] = final test (41 examples) <-- TRULY HELD OUT

The inference script was WRONGLY loading indices[:n_test] which is actually
the tune_eval set that was used during Optuna!
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast
import msgpack
import numpy as np
import json

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
        total_layers = len(self.bert.encoder.layer)
        unfrozen_start = total_layers - unfreeze_layers
        for i in range(unfrozen_start, total_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


def extract_entities(tokens, labels):
    """Extract Subject, Predicate, Object from BIO labels"""
    entities = {'SUBJ': [], 'PRED': [], 'OBJ': []}
    
    current_entity = None
    current_tokens = []
    
    for token, label in zip(tokens, labels):
        if label.startswith('B-'):
            if current_entity and current_tokens:
                entities[current_entity].append(' '.join(current_tokens))
            current_entity = label.split('-')[1]
            current_tokens = [token]
        elif label.startswith('I-'):
            entity_type = label.split('-')[1]
            if current_entity == entity_type:
                current_tokens.append(token)
            else:
                if current_entity and current_tokens:
                    entities[current_entity].append(' '.join(current_tokens))
                current_entity = entity_type
                current_tokens = [token]
        else:  # O label
            if current_entity and current_tokens:
                entities[current_entity].append(' '.join(current_tokens))
            current_entity = None
            current_tokens = []
    
    if current_entity and current_tokens:
        entities[current_entity].append(' '.join(current_tokens))
    
    return entities


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading trained model...")
    checkpoint = torch.load('bio_tagger_multiclass.pt', map_location=device)
    
    # Checkpoint is just the state_dict, not wrapped
    # Use default hyperparameters (will still work)
    dropout = 0.1
    unfreeze_layers = 2
    
    model = BIOTaggerMultiClass(dropout=dropout, unfreeze_layers=unfreeze_layers)
    model.load_state_dict(checkpoint)  # Load directly
    model = model.to(device)
    model.eval()
    print(f"Model loaded (using defaults: dropout={dropout}, unfreeze_layers={unfreeze_layers})")
    
    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    # Load data with CORRECT split matching training script
    print("\nLoading data with CORRECT split...")
    with open(r'c:\Users\user\arxiv_id_lists\data\bio_training_250chunks_complete.msgpack', 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
    
    all_examples = data['training_data']
    print(f"Total examples: {len(all_examples)}")
    
    # CRITICAL: Use same seed and split as training
    np.random.seed(42)
    indices = np.random.permutation(len(all_examples))
    
    n_test = max(1, int(0.1 * len(all_examples)))
    n_tune_eval = max(1, int(0.1 * len(all_examples)))
    
    # This is the CORRECT held-out test set that was NEVER used during training
    # The training script used:
    #   - indices[n_test+n_tune_eval:] for training
    #   - indices[n_test:n_test+n_tune_eval] for Optuna tuning eval
    #   - indices[:n_test] was NEVER touched
    
    print("\n" + "=" * 80)
    print("SPLIT BREAKDOWN (matching train_bert_bio_optuna.py):")
    print("=" * 80)
    print(f"  Training indices: {n_test+n_tune_eval}:{len(all_examples)} (n={len(all_examples)-(n_test+n_tune_eval)})")
    print(f"  Optuna eval indices: {n_test}:{n_test+n_tune_eval} (n={n_tune_eval}) <-- SEEN DURING TRAINING!")
    print(f"  Final test indices: 0:{n_test} (n={n_test}) <-- NEVER SEEN")
    
    # Get the TRULY held-out test set
    true_test_examples = [all_examples[i] for i in indices[:n_test]]
    
    # Also get the Optuna eval set for comparison (this is what we wrongly tested on before)
    optuna_eval_examples = [all_examples[i] for i in indices[n_test:n_test+n_tune_eval]]
    
    print(f"\nLoaded {len(true_test_examples)} TRULY HELD-OUT test examples")
    print(f"Loaded {len(optuna_eval_examples)} Optuna eval examples (for comparison)")
    
    # Run inference on both sets
    for set_name, test_examples in [("TRULY HELD-OUT TEST", true_test_examples),
                                      ("OPTUNA EVAL (seen during training)", optuna_eval_examples)]:
        
        print("\n" + "=" * 80)
        print(f"INFERENCE RESULTS ON: {set_name}")
        print("=" * 80)
        
        all_predictions = []
        
        for idx, example in enumerate(test_examples):
            tokens = example['tokens']
            
            # Prepare input
            full_tokens = ['[CLS]'] + tokens + ['[SEP]']
            input_ids = tokenizer.convert_tokens_to_ids(full_tokens)
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
            attention_mask = torch.ones_like(input_ids)
            
            # Predict
            with torch.no_grad():
                logits = model(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=-1)
            
            # Extract labels
            pred_labels = predictions[0, 1:len(tokens)+1].cpu().numpy()
            pred_label_names = [ID_TO_LABEL[label_id] for label_id in pred_labels]
            
            # Extract entities
            entities = extract_entities(tokens, pred_label_names)
            
            gt_triplets = example.get('triplets', [])
            
            # Display
            print(f"\n--- Example {idx + 1} ---")
            tokens_display = ' '.join(tokens[:50])
            if len(tokens_display) > 100:
                tokens_display = tokens_display[:100]
            # Safely encode for Windows terminal
            tokens_display = tokens_display.encode('ascii', errors='replace').decode('ascii')
            print(f"Tokens: {tokens_display}..." if len(tokens) > 50 else f"Tokens: {tokens_display}")
            print(f"\nPredicted Entities:")
            print(f"  Subject(s): {entities['SUBJ'][:5]}" + (" ..." if len(entities['SUBJ']) > 5 else ""))
            print(f"  Predicate(s): {entities['PRED'][:5]}" + (" ..." if len(entities['PRED']) > 5 else ""))
            print(f"  Object(s): {entities['OBJ'][:5]}" + (" ..." if len(entities['OBJ']) > 5 else ""))
            
            if gt_triplets:
                print(f"\nGround Truth ({len(gt_triplets)} triplets):")
                for i, triplet in enumerate(gt_triplets[:3], 1):
                    print(f"  {i}. S={triplet['subject']} | P={triplet['predicate']} | O={triplet['object']}")
                if len(gt_triplets) > 3:
                    print(f"  ... and {len(gt_triplets) - 3} more")
            
            # Store for JSON export
            all_predictions.append({
                'tokens': tokens,
                'predicted_labels': pred_label_names,
                'predicted_entities': entities,
                'ground_truth': gt_triplets
            })
        
        # Summary stats
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        total_subj = sum(len(p['predicted_entities']['SUBJ']) for p in all_predictions)
        total_pred = sum(len(p['predicted_entities']['PRED']) for p in all_predictions)
        total_obj = sum(len(p['predicted_entities']['OBJ']) for p in all_predictions)
        total_entities = total_subj + total_pred + total_obj
        
        print(f"Total examples: {len(all_predictions)}")
        print(f"Subjects extracted: {total_subj}")
        print(f"Predicates extracted: {total_pred}")
        print(f"Objects extracted: {total_obj}")
        print(f"Total entities: {total_entities}")
        print(f"Avg entities per example: {total_entities / len(all_predictions):.2f}")
        
        # Save to JSON
        output_filename = f"{'true_' if 'HELD-OUT' in set_name else 'optuna_'}holdout_predictions.json"
        with open(output_filename, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        print(f"\nPredictions saved to: {output_filename}")


if __name__ == '__main__':
    main()
