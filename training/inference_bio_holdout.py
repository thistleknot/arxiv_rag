"""
Inference on holdout data with trained BIO tagger
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast
import msgpack
import random

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
        total_layers = len(self.bert.encoder.layer)  # 12 for bert-base
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
            # Save previous entity if exists
            if current_entity and current_tokens:
                entities[current_entity].append(' '.join(current_tokens))
            
            # Start new entity
            current_entity = label.split('-')[1]
            current_tokens = [token]
        
        elif label.startswith('I-'):
            # Continue current entity
            entity_type = label.split('-')[1]
            if current_entity == entity_type:
                current_tokens.append(token)
            else:
                # Mismatched I- tag, treat as new entity
                if current_entity and current_tokens:
                    entities[current_entity].append(' '.join(current_tokens))
                current_entity = entity_type
                current_tokens = [token]
        
        else:  # 'O' tag
            # Save previous entity if exists
            if current_entity and current_tokens:
                entities[current_entity].append(' '.join(current_tokens))
            current_entity = None
            current_tokens = []
    
    # Save final entity
    if current_entity and current_tokens:
        entities[current_entity].append(' '.join(current_tokens))
    
    return entities


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    # Load trained model
    print("\nLoading trained model...")
    model = BIOTaggerMultiClass(dropout=0.1, unfreeze_layers=4).to(device)
    model.load_state_dict(torch.load('bio_tagger_multiclass.pt', map_location=device))
    model.eval()
    print("Model loaded!")
    
    # Load holdout data (41 test examples)
    print("\nLoading holdout data...")
    with open('data/bio_training_250chunks_complete.msgpack', 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
    
    all_examples = data['training_data']
    random.seed(42)
    indices = list(range(len(all_examples)))
    random.shuffle(indices)
    
    n_test = 41
    test_examples = [all_examples[i] for i in indices[:n_test]]
    
    print(f"Loaded {len(test_examples)} test examples")
    
    # Run inference
    print("\n" + "=" * 80)
    print("INFERENCE RESULTS ON HOLDOUT DATA")
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
        
        # Extract labels (skip [CLS] and [SEP])
        pred_labels = predictions[0, 1:len(tokens)+1].cpu().numpy()
        pred_label_names = [ID_TO_LABEL[label_id] for label_id in pred_labels]
        
        # Extract entities
        entities = extract_entities(tokens, pred_label_names)
        
        # Get ground truth triplets for comparison
        gt_triplets = example.get('triplets', [])
        
        # Display
        print(f"\n--- Example {idx + 1} ---")
        print(f"Tokens: {' '.join(tokens[:50])}..." if len(tokens) > 50 else f"Tokens: {' '.join(tokens)}")
        print(f"\nPredicted Entities:")
        print(f"  Subject(s): {entities['SUBJ']}")
        print(f"  Predicate(s): {entities['PRED']}")
        print(f"  Object(s): {entities['OBJ']}")
        
        if gt_triplets:
            print(f"\nGround Truth Triplets:")
            for triplet in gt_triplets:
                if isinstance(triplet, dict):
                    print(f"  ({triplet.get('subject', '')}, {triplet.get('predicate', '')}, {triplet.get('object', '')})")
                elif isinstance(triplet, (list, tuple)) and len(triplet) >= 3:
                    print(f"  ({triplet[0]}, {triplet[1]}, {triplet[2]})")
                else:
                    print(f"  {triplet}")
        
        all_predictions.append({
            'tokens': tokens,
            'predicted_labels': pred_label_names,
            'predicted_entities': entities,
            'ground_truth': gt_triplets
        })
        
        if idx < 5:  # Show first 5 in detail
            print(f"\nBIO Labels: {' '.join(pred_label_names[:50])}..." if len(pred_label_names) > 50 else f"\nBIO Labels: {' '.join(pred_label_names)}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_subj = sum(len(p['predicted_entities']['SUBJ']) for p in all_predictions)
    total_pred = sum(len(p['predicted_entities']['PRED']) for p in all_predictions)
    total_obj = sum(len(p['predicted_entities']['OBJ']) for p in all_predictions)
    
    print(f"Total examples: {len(test_examples)}")
    print(f"Total Subjects extracted: {total_subj}")
    print(f"Total Predicates extracted: {total_pred}")
    print(f"Total Objects extracted: {total_obj}")
    print(f"Average entities per example: {(total_subj + total_pred + total_obj) / len(test_examples):.2f}")
    
    # Save predictions
    import json
    with open('holdout_predictions.json', 'w') as f:
        json.dump(all_predictions, f, indent=2)
    print(f"\nPredictions saved to: holdout_predictions.json")


if __name__ == '__main__':
    main()
