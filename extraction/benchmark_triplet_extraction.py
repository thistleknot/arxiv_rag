"""
Benchmark triplet extraction speed to verify assumptions.

Testing:
1. spaCy parsing time
2. Triplet extraction time
3. Total time per sentence
4. Comparison to theoretical predictions
"""

import time
import spacy
from triplet_extract import extract  # Correct function name

# Test sentences of varying complexity
TEST_SENTENCES = [
    "John ate food.",
    "The red car was running towards station very fast.",
    "The Ferrari and the Pajero were running towards airport.",
    "Self-attention mechanisms enable transformers to capture long-range dependencies efficiently.",
    "Recent work on agentic memory systems has shown that combining episodic and semantic memory improves task performance across multiple domains.",
]

def benchmark_extraction():
    """Benchmark actual extraction speed."""
    print("=" * 70)
    print("TRIPLET EXTRACTION BENCHMARK")
    print("=" * 70)
    
    # Load spaCy model once
    print("\nLoading spaCy model...")
    start = time.time()
    nlp = spacy.load("en_core_web_sm")
    load_time = time.time() - start
    print(f"Model load time: {load_time:.3f}s")
    
    print("\n" + "=" * 70)
    print("PER-SENTENCE TIMING")
    print("=" * 70)
    
    total_parse_time = 0
    total_extract_time = 0
    total_sentences = len(TEST_SENTENCES)
    
    for i, sentence in enumerate(TEST_SENTENCES, 1):
        print(f"\n[{i}] Sentence: {sentence[:60]}...")
        print(f"    Length: {len(sentence)} chars")
        
        # Time spaCy parsing
        start = time.time()
        doc = nlp(sentence)
        parse_time = time.time() - start
        total_parse_time += parse_time
        
        print(f"    spaCy parse: {parse_time*1000:.1f}ms")
        
        # Time extraction (using library's internal method)
        start = time.time()
        triplets = extract(sentence)
        extract_time = time.time() - start
        total_extract_time += extract_time
        
        print(f"    Extraction:  {extract_time*1000:.1f}ms")
        print(f"    Total:       {(parse_time + extract_time)*1000:.1f}ms")
        print(f"    Triplets:    {len(triplets)} found")
        
        # Show triplets (Triplet objects have .subject, .relation, .object)
        for triplet in triplets[:3]:  # Show first 3
            print(f"      - ({triplet.subject}, {triplet.relation}, {triplet.object})")
        if len(triplets) > 3:
            print(f"      ... and {len(triplets) - 3} more")
    
    # Summary statistics
    avg_parse = (total_parse_time / total_sentences) * 1000
    avg_extract = (total_extract_time / total_sentences) * 1000
    avg_total = avg_parse + avg_extract
    
    print("\n" + "=" * 70)
    print("AVERAGE TIMING")
    print("=" * 70)
    print(f"spaCy parsing:   {avg_parse:.1f}ms ({avg_parse/avg_total*100:.0f}%)")
    print(f"Extraction:      {avg_extract:.1f}ms ({avg_extract/avg_total*100:.0f}%)")
    print(f"Total:           {avg_total:.1f}ms per sentence")
    
    print("\n" + "=" * 70)
    print("THEORETICAL vs ACTUAL")
    print("=" * 70)
    print(f"Predicted spaCy:     50ms")
    print(f"Actual spaCy:        {avg_parse:.1f}ms")
    print(f"Predicted extraction: 10ms")
    print(f"Actual extraction:   {avg_extract:.1f}ms")
    print(f"Predicted total:     60ms")
    print(f"Actual total:        {avg_total:.1f}ms")
    
    slowdown = avg_total / 60.0
    print(f"\nActual is {slowdown:.1f}x slower than predicted")
    
    print("\n" + "=" * 70)
    print("EXTRAPOLATION TO 161,389 CHUNKS")
    print("=" * 70)
    print(f"At {avg_total:.1f}ms per chunk:")
    total_hours = (161389 * avg_total / 1000) / 3600
    print(f"  Total time: {total_hours:.1f} hours")
    
    print(f"\nAt theoretical 60ms per chunk:")
    theoretical_hours = (161389 * 60 / 1000) / 3600
    print(f"  Total time: {theoretical_hours:.1f} hours")
    
    print(f"\nOverhead cost: {total_hours - theoretical_hours:.1f} hours")
    
    return avg_parse, avg_extract, avg_total

if __name__ == "__main__":
    try:
        avg_parse, avg_extract, avg_total = benchmark_extraction()
        
        print("\n" + "=" * 70)
        print("CONCLUSION")
        print("=" * 70)
        
        if avg_extract < 20:
            print("✅ Extraction is fast (<20ms) - implementation is efficient")
        elif avg_extract < 50:
            print("⚠️  Extraction is moderate (20-50ms) - some overhead but acceptable")
        else:
            print("❌ Extraction is slow (>50ms) - implementation has bloat")
        
        if avg_total < 100:
            print("✅ Total time is reasonable (<100ms)")
        else:
            print("⚠️  Total time exceeds 100ms - consider optimizations")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPossible issues:")
        print("- triplet_extract library not installed")
        print("- spaCy model 'en_core_web_sm' not downloaded")
        print("- Import path issues")
