"""
Fix Invalid BIO Sequences

Problem: Model predicts consecutive B-X B-X of same type, which is invalid.
Valid: B-X I-X I-X O B-X
Invalid: B-X B-X (same type with no separator)

Solution: Post-process predictions to enforce valid BIO constraints.
"""

def fix_bio_sequence(labels):
    """
    Fix invalid BIO sequences by converting invalid B- tags to I- tags.
    
    Rules:
    - B-X can be followed by: I-X, O, or B-Y (different type)
    - B-X followed by B-X (same type) is INVALID → convert second to I-X
    
    Args:
        labels: List of BIO label strings
    
    Returns:
        Fixed list of BIO labels
    """
    if not labels:
        return labels
    
    fixed = [labels[0]]  # First label is always valid
    
    for i in range(1, len(labels)):
        prev_label = fixed[-1]
        curr_label = labels[i]
        
        # If previous was B-X or I-X and current is B-X (same type), convert to I-X
        if prev_label.startswith('B-') or prev_label.startswith('I-'):
            prev_entity = prev_label.split('-')[1]  # SUBJ, PRED, or OBJ
            
            if curr_label.startswith('B-'):
                curr_entity = curr_label.split('-')[1]
                
                # INVALID: B-X followed by B-X (same type)
                if prev_entity == curr_entity:
                    # Fix: Convert B-X to I-X to continue the entity
                    fixed.append(f'I-{curr_entity}')
                    continue
        
        # Otherwise keep the label as-is
        fixed.append(curr_label)
    
    return fixed


def validate_bio_sequence(labels):
    """
    Validate if BIO sequence is correct.
    
    Returns: (is_valid, error_indices)
    """
    errors = []
    
    for i in range(1, len(labels)):
        prev_label = labels[i-1]
        curr_label = labels[i]
        
        if prev_label.startswith('B-') or prev_label.startswith('I-'):
            prev_entity = prev_label.split('-')[1]
            
            if curr_label.startswith('B-'):
                curr_entity = curr_label.split('-')[1]
                
                # Check for consecutive B-X B-X of same type
                if prev_entity == curr_entity:
                    errors.append(i)
    
    return len(errors) == 0, errors


def analyze_predictions_file(json_file):
    """Analyze prediction file for invalid BIO sequences."""
    import json
    
    data = json.load(open(json_file))
    
    print(f"Analyzing {len(data)} predictions...")
    print("=" * 80)
    
    total_sequences = len(data)
    invalid_sequences = 0
    total_errors = 0
    
    for idx, example in enumerate(data):
        labels = example['predicted_labels']
        is_valid, errors = validate_bio_sequence(labels)
        
        if not is_valid:
            invalid_sequences += 1
            total_errors += len(errors)
            
            if invalid_sequences <= 5:  # Show first 5 examples
                print(f"\n❌ Example {idx + 1}: {len(errors)} errors")
                print(f"Tokens: {' '.join(example['tokens'][:30])}...")
                print(f"\nInvalid transitions at positions: {errors[:10]}")
                
                for err_idx in errors[:3]:
                    print(f"  Position {err_idx}: "
                          f"{labels[err_idx-1]} → {labels[err_idx]}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total sequences: {total_sequences}")
    print(f"Invalid sequences: {invalid_sequences} ({invalid_sequences/total_sequences*100:.1f}%)")
    print(f"Total invalid transitions: {total_errors}")
    print(f"Avg errors per invalid sequence: {total_errors/invalid_sequences:.1f}" if invalid_sequences > 0 else "")


def fix_predictions_file(input_file, output_file):
    """Fix all predictions in a JSON file."""
    import json
    
    data = json.load(open(input_file))
    
    print(f"Fixing {len(data)} predictions...")
    
    fixed_count = 0
    total_fixes = 0
    
    for example in data:
        labels = example['predicted_labels']
        is_valid, errors = validate_bio_sequence(labels)
        
        if not is_valid:
            fixed_labels = fix_bio_sequence(labels)
            example['predicted_labels'] = fixed_labels
            fixed_count += 1
            total_fixes += len(errors)
    
    # Save fixed predictions
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Fixed {fixed_count} sequences ({total_fixes} invalid transitions)")
    print(f"Saved to: {output_file}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Analyze: python fix_invalid_bio.py analyze <file.json>")
        print("  Fix: python fix_invalid_bio.py fix <input.json> <output.json>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'analyze':
        analyze_predictions_file(sys.argv[2])
    elif command == 'fix':
        fix_predictions_file(sys.argv[2], sys.argv[3])
    else:
        print(f"Unknown command: {command}")
