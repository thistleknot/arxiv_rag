"""
Empirical Benchmarking: BIO Tagger vs OpenIE
==============================================

STOP MAKING CLAIMS WITHOUT DATA.

This script measures:
1. Inference time per chunk (BIO vs OpenIE)
2. Quality metrics (precision, recall, F1)
3. Triplet-level comparison
4. Training time (when model exists)

Run after training the BIO model.
"""

import time
import psycopg2
import msgpack
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import numpy as np

# Will be used when BIO model is trained
# from train_bio_tagger import load_trained_model, inference


def normalize_triplet(s, p, o):
    """Normalize for fair comparison"""
    return (
        s.lower().strip(),
        p.lower().strip(),
        o.lower().strip()
    )


def extract_openie_triplets(content: str) -> List[Tuple[str, str, str]]:
    """Extract with OpenIE (ground truth / teacher)"""
    from triplet_extract import extract
    
    triplets = []
    try:
        triplet_objects = extract(content)
        if triplet_objects:
            for tobj in triplet_objects:
                try:
                    subj = str(tobj.subject).strip() if tobj.subject else None
                    pred = str(tobj.relation).strip() if tobj.relation else None
                    obj = str(tobj.object).strip() if tobj.object else None
                    
                    if subj and pred and obj:
                        triplets.append(normalize_triplet(subj, pred, obj))
                except:
                    continue
    except:
        pass
    
    return triplets


def compute_metrics(predicted: Set[Tuple], ground_truth: Set[Tuple]) -> Dict[str, float]:
    """Compute precision, recall, F1"""
    if not predicted and not ground_truth:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    
    if not predicted:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    if not ground_truth:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    true_positives = len(predicted & ground_truth)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'predicted_count': len(predicted),
        'ground_truth_count': len(ground_truth)
    }


def benchmark_inference_speed(test_chunks: List[str], method: str = 'openie', model=None) -> Dict:
    """Measure actual inference time"""
    times = []
    all_triplets = []
    
    print(f"\n⏱️  Benchmarking {method.upper()} inference speed...")
    print(f"  Testing on {len(test_chunks)} chunks")
    
    for i, content in enumerate(test_chunks):
        start = time.perf_counter()
        
        if method == 'openie':
            triplets = extract_openie_triplets(content)
        else:
            # BIO model inference (when trained)
            # triplets = inference(model, content)
            raise NotImplementedError("BIO model not trained yet")
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        all_triplets.append(triplets)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(test_chunks)} - Avg: {np.mean(times)*1000:.1f}ms")
    
    return {
        'method': method,
        'num_chunks': len(test_chunks),
        'times_ms': [t * 1000 for t in times],
        'mean_ms': np.mean(times) * 1000,
        'median_ms': np.median(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
        'total_seconds': np.sum(times),
        'triplets': all_triplets
    }


def benchmark_quality(test_chunks: List[str], bio_model=None) -> Dict:
    """Measure quality: BIO predictions vs OpenIE ground truth"""
    
    print("\n📊 Benchmarking quality (BIO vs OpenIE)...")
    
    all_metrics = []
    
    for i, content in enumerate(test_chunks):
        # Ground truth (OpenIE)
        openie_triplets = set(extract_openie_triplets(content))
        
        # BIO predictions (when trained)
        # bio_triplets = set(inference(bio_model, content))
        bio_triplets = set()  # Placeholder
        
        metrics = compute_metrics(bio_triplets, openie_triplets)
        all_metrics.append(metrics)
        
        if (i + 1) % 10 == 0:
            avg_f1 = np.mean([m['f1'] for m in all_metrics])
            print(f"  Processed {i+1}/{len(test_chunks)} - Avg F1: {avg_f1:.3f}")
    
    return {
        'per_chunk_metrics': all_metrics,
        'avg_precision': np.mean([m['precision'] for m in all_metrics]),
        'avg_recall': np.mean([m['recall'] for m in all_metrics]),
        'avg_f1': np.mean([m['f1'] for m in all_metrics]),
        'total_true_positives': sum(m['true_positives'] for m in all_metrics),
        'total_predicted': sum(m['predicted_count'] for m in all_metrics),
        'total_ground_truth': sum(m['ground_truth_count'] for m in all_metrics)
    }


def main():
    print("=" * 70)
    print("EMPIRICAL BENCHMARK: BIO TAGGER VS OPENIE")
    print("=" * 70)
    
    # Connect to database
    print("\n[1/4] Connecting to database...")
    conn = psycopg2.connect(
        dbname='postgres',
        user='postgres',
        password='postgres',
        host='localhost',
        port=5432
    )
    cur = conn.cursor()
    
    # Fetch test chunks (different from training data)
    print("\n[2/4] Fetching test chunks...")
    test_size = 20
    
    cur.execute("""
        SELECT id, content 
        FROM arxiv_papers_lemma_fullembed 
        WHERE length(content) > 100 
        ORDER BY id 
        LIMIT 1000 OFFSET 50
    """)
    
    rows = cur.fetchall()
    test_chunks = [row[1] for row in rows[:test_size]]
    
    print(f"  Fetched {len(test_chunks)} test chunks")
    
    # Benchmark OpenIE speed (baseline)
    print("\n[3/4] Benchmarking OpenIE inference speed...")
    openie_results = benchmark_inference_speed(test_chunks, method='openie')
    
    print("\n" + "=" * 70)
    print("OPENIE BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Chunks tested: {openie_results['num_chunks']}")
    print(f"  Mean time: {openie_results['mean_ms']:.1f}ms per chunk")
    print(f"  Median time: {openie_results['median_ms']:.1f}ms per chunk")
    print(f"  Std dev: {openie_results['std_ms']:.1f}ms")
    print(f"  Min time: {openie_results['min_ms']:.1f}ms")
    print(f"  Max time: {openie_results['max_ms']:.1f}ms")
    print(f"  Total time: {openie_results['total_seconds']:.1f}s")
    print(f"  Throughput: {openie_results['num_chunks'] / openie_results['total_seconds']:.2f} chunks/sec")
    
    # Calculate projection for 3000 chunks
    projection_3000 = (openie_results['mean_ms'] / 1000) * 3000
    print(f"\n  📈 Projection for 3000 chunks: {projection_3000/3600:.1f} hours")
    
    print("\n[4/4] BIO model benchmarking...")
    print("  ⚠️  BIO model not trained yet - cannot benchmark")
    print("  Run after training: python train_bio_tagger.py")
    
    # Save results
    results = {
        'timestamp': time.time(),
        'test_chunks': test_size,
        'openie_benchmark': openie_results,
        'bio_benchmark': None  # Fill after training
    }
    
    with open('benchmark_results.msgpack', 'wb') as f:
        f.write(msgpack.packb(results, use_bin_type=True))
    
    print("\n" + "=" * 70)
    print("SAVED: benchmark_results.msgpack")
    print("=" * 70)
    print("\n✅ Baseline (OpenIE) benchmarked")
    print("⏳ Train BIO model, then re-run to compare")
    
    cur.close()
    conn.close()


if __name__ == '__main__':
    main()
