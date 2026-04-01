"""Class-balanced sampler using iterative token-level sampling with log/Box-Cox transform."""

import numpy as np
import msgpack
from collections import Counter
from scipy import stats

def compute_token_class_distribution(examples):
    """Count tokens in each BIO class across all examples."""
    class_counts = Counter()
    
    for example in examples:
        labels = example['labels']
        num_tokens = len(labels['B-SUBJ'])
        
        # Count each class
        for i in range(num_tokens):
            # Check which label is active
            if labels['B-SUBJ'][i] > 0:
                class_counts['B-SUBJ'] += 1
            elif labels['I-SUBJ'][i] > 0:
                class_counts['I-SUBJ'] += 1
            elif labels['B-PRED'][i] > 0:
                class_counts['B-PRED'] += 1
            elif labels['I-PRED'][i] > 0:
                class_counts['I-PRED'] += 1
            elif labels['B-OBJ'][i] > 0:
                class_counts['B-OBJ'] += 1
            elif labels['I-OBJ'][i] > 0:
                class_counts['I-OBJ'] += 1
            else:
                class_counts['O'] += 1
    
    return class_counts


def compute_example_class_counts(example):
    """Count tokens in each class for a single example."""
    counts = Counter()
    labels = example['labels']
    num_tokens = len(labels['B-SUBJ'])
    
    for i in range(num_tokens):
        if labels['B-SUBJ'][i] > 0:
            counts['B-SUBJ'] += 1
        elif labels['I-SUBJ'][i] > 0:
            counts['I-SUBJ'] += 1
        elif labels['B-PRED'][i] > 0:
            counts['B-PRED'] += 1
        elif labels['I-PRED'][i] > 0:
            counts['I-PRED'] += 1
        elif labels['B-OBJ'][i] > 0:
            counts['B-OBJ'] += 1
        elif labels['I-OBJ'][i] > 0:
            counts['I-OBJ'] += 1
        else:
            counts['O'] += 1
    
    return counts


def iterative_class_balanced_sampling(examples, target_samples, target_distribution=None):
    """
    Iteratively sample examples to achieve target class distribution.
    
    Algorithm:
    1. Identify current class percentages
    2. Find class with lowest percentage vs target
    3. Sample one example that maximally increases that class
    4. Update percentages
    5. Repeat until target_samples reached or all targets met
    
    Args:
        examples: List of training examples with 'labels' key
        target_samples: Number of examples to sample
        target_distribution: Dict of class -> target percentage (defaults to uniform)
    
    Returns:
        List of sampled example indices
    """
    num_classes = 7  # B-SUBJ, I-SUBJ, B-PRED, I-PRED, B-OBJ, I-OBJ, O
    class_names = ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ', 'O']
    
    # Default target: uniform distribution
    if target_distribution is None:
        target_distribution = {c: 1.0/num_classes for c in class_names}
    
    # Pre-compute class counts for each example
    example_class_counts = [compute_example_class_counts(ex) for ex in examples]
    
    # Track current class counts
    current_counts = Counter()
    sampled_indices = []
    available_indices = list(range(len(examples)))
    
    for _ in range(target_samples):
        if not available_indices:
            break
        
        # Compute current percentages
        total_tokens = sum(current_counts.values())
        
        if total_tokens == 0:
            # First sample - pick example with most balanced distribution
            # Use example with highest entropy
            best_idx = available_indices[0]
            best_score = -float('inf')
            
            for idx in available_indices:
                ex_counts = example_class_counts[idx]
                ex_total = sum(ex_counts.values())
                if ex_total == 0:
                    continue
                
                # Compute entropy
                entropy = 0
                for c in class_names:
                    p = ex_counts[c] / ex_total
                    if p > 0:
                        entropy -= p * np.log(p)
                
                if entropy > best_score:
                    best_score = entropy
                    best_idx = idx
        else:
            # Find class with largest deficit
            current_pct = {c: current_counts[c] / total_tokens for c in class_names}
            deficits = {c: target_distribution[c] - current_pct.get(c, 0) for c in class_names}
            
            # Find class with max deficit
            max_deficit_class = max(deficits, key=deficits.get)
            max_deficit = deficits[max_deficit_class]
            
            if max_deficit <= 0:
                # All targets met - sample randomly from remaining
                best_idx = np.random.choice(available_indices)
            else:
                # Find example that maximally increases deficit class
                best_idx = None
                best_increase = 0
                
                for idx in available_indices:
                    ex_counts = example_class_counts[idx]
                    ex_total = sum(ex_counts.values())
                    
                    if ex_total == 0:
                        continue
                    
                    # How much would this example increase the deficit class percentage?
                    new_total = total_tokens + ex_total
                    new_count = current_counts[max_deficit_class] + ex_counts[max_deficit_class]
                    new_pct = new_count / new_total
                    increase = new_pct - current_pct.get(max_deficit_class, 0)
                    
                    if increase > best_increase:
                        best_increase = increase
                        best_idx = idx
                
                # Fallback if no example increases deficit class
                if best_idx is None:
                    best_idx = np.random.choice(available_indices)
        
        # Sample this example
        sampled_indices.append(best_idx)
        available_indices.remove(best_idx)
        
        # Update current counts
        for c in class_names:
            current_counts[c] += example_class_counts[best_idx][c]
    
    return sampled_indices


def log_boxcox_balanced_sampling(examples, target_samples):
    """
    Class-balanced sampling using log + Box-Cox transform on class counts.
    
    Algorithm:
    1. Compute class counts across dataset
    2. Apply log transform to reduce range
    3. Apply Box-Cox normalization
    4. Compute inverse frequency weights (rare classes get high weight)
    5. Assign weight to each example based on rarest class it contains
    6. Sample with replacement using these weights
    
    Args:
        examples: List of training examples
        target_samples: Number of examples to sample
    
    Returns:
        List of sampled example indices (with replacement)
    """
    # Compute dataset-wide class counts
    class_counts = compute_token_class_distribution(examples)
    class_names = ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ', 'O']
    
    counts_array = np.array([class_counts[c] for c in class_names], dtype=float)
    
    # Log transform
    log_counts = np.log1p(counts_array)  # log(1 + x) to handle zeros
    
    # Box-Cox transform (requires positive values)
    try:
        transformed_counts, lambda_param = stats.boxcox(log_counts + 1)
    except:
        transformed_counts = log_counts
    
    # Normalize to [0, 1]
    min_val = transformed_counts.min()
    max_val = transformed_counts.max()
    if max_val > min_val:
        normalized = (transformed_counts - min_val) / (max_val - min_val)
    else:
        normalized = np.ones_like(transformed_counts)
    
    # Inverse frequency weights
    class_weights = 1.0 / (normalized + 0.01)
    class_weight_map = {c: w for c, w in zip(class_names, class_weights)}
    
    # Assign weight to each example (based on rarest class)
    example_weights = []
    for example in examples:
        ex_counts = compute_example_class_counts(example)
        
        # Find rarest class in this example
        max_weight = 0
        for c in class_names:
            if ex_counts[c] > 0:
                max_weight = max(max_weight, class_weight_map[c])
        
        example_weights.append(max_weight if max_weight > 0 else 1.0)
    
    # Normalize to probability distribution
    example_weights = np.array(example_weights)
    example_weights = example_weights / example_weights.sum()
    
    # Sample with replacement
    sampled_indices = np.random.choice(
        len(examples),
        size=target_samples,
        replace=True,
        p=example_weights
    )
    
    return sampled_indices.tolist()


if __name__ == '__main__':
    # Test both methods
    print("Loading data...")
    with open('data/bio_training_250chunks_complete.msgpack', 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False)
    
    examples = data['training_data']
    print(f"Loaded {len(examples)} examples")
    
    # Compute baseline distribution
    print("\n" + "="*80)
    print("BASELINE CLASS DISTRIBUTION")
    print("="*80)
    baseline_counts = compute_token_class_distribution(examples)
    total = sum(baseline_counts.values())
    for c in ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ', 'O']:
        print(f"{c:8}: {baseline_counts[c]:6} ({baseline_counts[c]/total*100:5.2f}%)")
    
    # Test iterative sampling
    print("\n" + "="*80)
    print("ITERATIVE BALANCED SAMPLING (50 examples)")
    print("="*80)
    sampled_idx = iterative_class_balanced_sampling(examples, 50)
    
    sampled_counts = Counter()
    for idx in sampled_idx:
        ex_counts = compute_example_class_counts(examples[idx])
        for c in ex_counts:
            sampled_counts[c] += ex_counts[c]
    
    total_sampled = sum(sampled_counts.values())
    for c in ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ', 'O']:
        baseline_pct = baseline_counts[c]/total*100
        sampled_pct = sampled_counts[c]/total_sampled*100
        print(f"{c:8}: {sampled_counts[c]:6} ({sampled_pct:5.2f}%) vs baseline {baseline_pct:5.2f}% ({sampled_pct - baseline_pct:+5.2f}%)")
    
    # Test log/Box-Cox sampling
    print("\n" + "="*80)
    print("LOG/BOX-COX BALANCED SAMPLING (50 examples)")
    print("="*80)
    sampled_idx2 = log_boxcox_balanced_sampling(examples, 50)
    
    sampled_counts2 = Counter()
    for idx in sampled_idx2:
        ex_counts = compute_example_class_counts(examples[idx])
        for c in ex_counts:
            sampled_counts2[c] += ex_counts[c]
    
    total_sampled2 = sum(sampled_counts2.values())
    for c in ['B-SUBJ', 'I-SUBJ', 'B-PRED', 'I-PRED', 'B-OBJ', 'I-OBJ', 'O']:
        baseline_pct = baseline_counts[c]/total*100
        sampled_pct = sampled_counts2[c]/total_sampled2*100
        print(f"{c:8}: {sampled_counts2[c]:6} ({sampled_pct:5.2f}%) vs baseline {baseline_pct:5.2f}% ({sampled_pct - baseline_pct:+5.2f}%)")
