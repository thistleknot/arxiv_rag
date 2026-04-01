"""
Extract S-P-O triplets from BIO-tagged predictions using cartesian products.

Key Logic:
1. Consecutive B-X tags of same type → Multiple spans (compound entity)
2. B-X followed by I-X tags → Single continuous span  
3. Cartesian product: SUBJ spans × PRED spans × OBJ spans

Example:
  "The Eiffel Tower is in Paris"
  Tokens: [The, Eiffel, Tower, is, in, Paris]
  Labels: [O, B-SUBJ, B-SUBJ, B-PRED, B-OBJ, B-OBJ]
  
  SUBJ spans: ["Eiffel", "Tower"]
  PRED spans: ["is"]  
  OBJ spans: ["in", "Paris"]
  
  Triplets (2×1×2=4):
    (Eiffel, is, in)
    (Eiffel, is, Paris)
    (Tower, is, in)
    (Tower, is, Paris)
"""

import json
import msgpack
from collections import defaultdict
from itertools import product

try:
    from inference_bio_tagger import clean_span_tokens, SPAN_STOPWORDS
except ImportError:
    # Fallback inline if inference module not available
    SPAN_STOPWORDS = {'a', 'an', 'the', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with',
                      'from', 'into', 'through', 'and', 'or', 'but', 'nor',
                      'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'that', 'which', 'who', 'this', 'these', 'those', 'it', 'its'}
    def clean_span_tokens(tokens, label_type):
        cleaned = [t for t in tokens if t.lower() not in SPAN_STOPWORDS]
        return cleaned


def extract_spans_from_bio(tokens, labels):
    """
    Extract entity spans from BIO labels, grouping continuous I-tags.
    
    Returns:
        dict: {entity_type: [span1, span2, ...]}
              where each span is a string (joined tokens)
    
    Logic:
        - B-X starts a new span
        - I-X continues current span
        - O ends current span
        - Consecutive B-X of same type → separate spans
    """
    spans_by_type = defaultdict(list)
    current_span_tokens = []
    current_entity_type = None
    
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label == 'O':
            # End current span if any
            if current_span_tokens and current_entity_type:
                cleaned = clean_span_tokens(current_span_tokens, current_entity_type)
                if cleaned:
                    span_text = ' '.join(cleaned)
                    spans_by_type[current_entity_type].append(span_text)
                current_span_tokens = []
                current_entity_type = None
        
        elif label.startswith('B-'):
            # Start new span
            entity_type = label[2:]  # Remove "B-"
            
            # Save previous span if exists
            if current_span_tokens and current_entity_type:
                cleaned = clean_span_tokens(current_span_tokens, current_entity_type)
                if cleaned:
                    span_text = ' '.join(cleaned)
                    spans_by_type[current_entity_type].append(span_text)
            
            # Start new span
            current_span_tokens = [token]
            current_entity_type = entity_type
        
        elif label.startswith('I-'):
            # Continue current span
            entity_type = label[2:]  # Remove "I-"
            
            # Should match current entity type
            if entity_type == current_entity_type:
                current_span_tokens.append(token)
            else:
                # Malformed: I-X without matching B-X
                # Treat as new B-X
                if current_span_tokens and current_entity_type:
                    cleaned = clean_span_tokens(current_span_tokens, current_entity_type)
                    span_text = ' '.join(cleaned)
                    spans_by_type[current_entity_type].append(span_text)
                
                current_span_tokens = [token]
                current_entity_type = entity_type
    
    # Save final span
    if current_span_tokens and current_entity_type:
        cleaned = clean_span_tokens(current_span_tokens, current_entity_type)
        if cleaned:
            span_text = ' '.join(cleaned)
            spans_by_type[current_entity_type].append(span_text)
    
    return dict(spans_by_type)


def extract_triplets(tokens, labels):
    """
    Extract S-P-O triplets using cartesian product.
    
    Returns:
        list: [(subject, predicate, object), ...]
    """
    spans = extract_spans_from_bio(tokens, labels)
    
    subjects = spans.get('SUBJ', [])
    predicates = spans.get('PRED', [])
    objects = spans.get('OBJ', [])
    
    # Cartesian product
    triplets = list(product(subjects, predicates, objects))
    
    return triplets


def analyze_predictions_file(json_file):
    """
    Analyze predictions from JSON file, extract triplets using cartesian product.
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING: {json_file}")
    print(f"{'='*80}\n")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_triplets = 0
    total_subjects = 0
    total_predicates = 0
    total_objects = 0
    
    # Analyze first 5 examples in detail
    print("DETAILED EXAMPLES (first 5):\n")
    
    for i, example in enumerate(data[:5]):
        tokens = example['tokens']
        labels = example.get('labels', example.get('predicted_labels', []))
        
        spans = extract_spans_from_bio(tokens, labels)
        triplets = extract_triplets(tokens, labels)
        
        print(f"Example {i+1}:")
        print(f"  Tokens: {' '.join(tokens)}")
        print(f"  Labels: {' '.join(labels)}")
        print(f"\n  Extracted Spans:")
        for entity_type, span_list in spans.items():
            print(f"    {entity_type}: {span_list}")
        
        print(f"\n  Triplets ({len(triplets)} total):")
        for s, p, o in triplets:
            print(f"    ({s}, {p}, {o})")
        
        print()
        
        total_subjects += len(spans.get('SUBJ', []))
        total_predicates += len(spans.get('PRED', []))
        total_objects += len(spans.get('OBJ', []))
        total_triplets += len(triplets)
    
    # Aggregate statistics
    print(f"\n{'='*80}")
    print("AGGREGATE STATISTICS (all examples):\n")
    
    all_triplets = 0
    all_subjects = 0
    all_predicates = 0
    all_objects = 0
    
    compound_subjects = 0
    compound_predicates = 0
    compound_objects = 0
    
    for example in data:
        tokens = example['tokens']
        labels = example['labels']
        
        spans = extract_spans_from_bio(tokens, labels)
        triplets = extract_triplets(tokens, labels)
        
        subj_count = len(spans.get('SUBJ', []))
        pred_count = len(spans.get('PRED', []))
        obj_count = len(spans.get('OBJ', []))
        
        all_subjects += subj_count
        all_predicates += pred_count
        all_objects += obj_count
        all_triplets += len(triplets)
        
        # Track compound entities (multiple spans of same type)
        if subj_count > 1:
            compound_subjects += 1
        if pred_count > 1:
            compound_predicates += 1
        if obj_count > 1:
            compound_objects += 1
    
    print(f"Total Examples: {len(data)}")
    print(f"\nEntity Spans:")
    print(f"  Subjects: {all_subjects} ({compound_subjects} examples with multiple spans)")
    print(f"  Predicates: {all_predicates} ({compound_predicates} examples with multiple spans)")
    print(f"  Objects: {all_objects} ({compound_objects} examples with multiple spans)")
    
    print(f"\nTriplets (via cartesian product):")
    print(f"  Total: {all_triplets}")
    print(f"  Average per example: {all_triplets / len(data):.2f}")
    
    # Distribution of triplet counts
    triplet_counts = []
    for example in data:
        tokens = example['tokens']
        labels = example['labels']
        triplets = extract_triplets(tokens, labels)
        triplet_counts.append(len(triplets))
    
    from collections import Counter
    count_dist = Counter(triplet_counts)
    
    print(f"\nTriplet Count Distribution:")
    for count in sorted(count_dist.keys()):
        print(f"  {count} triplets: {count_dist[count]} examples")
    
    print(f"\n{'='*80}\n")


def compare_with_gold_labels(predictions_json, gold_msgpack):
    """
    Compare predicted triplets vs gold label triplets.
    """
    print(f"\n{'='*80}")
    print(f"COMPARING PREDICTIONS VS GOLD LABELS")
    print(f"{'='*80}\n")
    
    # Load predictions
    with open(predictions_json, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # Load gold labels
    with open(gold_msgpack, 'rb') as f:
        gold_data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
    
    training_data = gold_data['training_data']
    
    # Match by indices
    # Predictions should be from true holdout: indices 0:41
    indices = gold_data['random_indices']
    true_test_indices = indices[:41]
    
    matches = []
    for pred_idx, pred_example in enumerate(predictions):
        gold_idx = true_test_indices[pred_idx]
        gold_example = training_data[gold_idx]
        matches.append((pred_example, gold_example))
    
    # Compare triplets
    exact_match = 0
    span_match = 0
    total = len(matches)
    
    print("DETAILED COMPARISON (first 3):\n")
    
    for i, (pred, gold) in enumerate(matches[:3]):
        pred_triplets = extract_triplets(pred['tokens'], pred['labels'])
        
        # Extract gold triplets from gold labels dict
        gold_tokens = gold['tokens']
        gold_labels_dict = gold['labels']
        
        # Reconstruct gold BIO labels
        gold_labels = ['O'] * len(gold_tokens)
        for label_type, token_indices in gold_labels_dict.items():
            for idx in token_indices:
                gold_labels[idx] = label_type
        
        gold_triplets = extract_triplets(gold_tokens, gold_labels)
        
        print(f"Example {i+1}:")
        print(f"  Tokens: {' '.join(pred['tokens'])}")
        print(f"\n  Predicted triplets ({len(pred_triplets)}):")
        for s, p, o in pred_triplets:
            print(f"    ({s}, {p}, {o})")
        
        print(f"\n  Gold triplets ({len(gold_triplets)}):")
        for s, p, o in gold_triplets:
            print(f"    ({s}, {p}, {o})")
        
        # Check if sets match
        pred_set = set(pred_triplets)
        gold_set = set(gold_triplets)
        
        if pred_set == gold_set:
            exact_match += 1
            print(f"\n  ✅ EXACT MATCH")
        else:
            print(f"\n  ❌ MISMATCH")
            print(f"     Precision: {len(pred_set & gold_set)} / {len(pred_set)} correct")
            print(f"     Recall: {len(pred_set & gold_set)} / {len(gold_set)} found")
        
        print()
    
    print(f"\n{'='*80}")
    print("AGGREGATE COMPARISON:\n")
    print(f"Total examples: {total}")
    print(f"Exact matches: {exact_match} ({exact_match/total*100:.1f}%)")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    # Analyze CRF predictions (when available)
    import os
    
    if os.path.exists('crf_true_holdout_predictions.json'):
        analyze_predictions_file('crf_true_holdout_predictions.json')
    elif os.path.exists('true_holdout_predictions.json'):
        print("⚠️  Using old CrossEntropyLoss predictions")
        print("    Run inference with CRF model for proper results\n")
        analyze_predictions_file('true_holdout_predictions.json')
    else:
        print("❌ No predictions file found")
        print("   Run inference first to generate predictions\n")
    
    # Compare with gold if available
    if os.path.exists('true_holdout_predictions.json') and \
       os.path.exists('bio_training_250chunks_complete.msgpack'):
        compare_with_gold_labels(
            'true_holdout_predictions.json',
            'bio_training_250chunks_complete.msgpack'
        )
