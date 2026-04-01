"""
Benchmark: BIO Tagger vs OpenIE vs Stanza Speed
Compare inference speed of trained BIO model vs OpenIE triplet extraction vs Stanza dependency parsing
"""

import torch
import time
import msgpack
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# Load BIO model
class BIOTagger(torch.nn.Module):
    def __init__(self, num_labels=6):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.1)
        self.classifiers = torch.nn.ModuleList([
            torch.nn.Linear(768, 1) for _ in range(num_labels)
        ])
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        logits_list = [classifier(sequence_output) for classifier in self.classifiers]
        logits = torch.cat(logits_list, dim=-1)
        probs = torch.sigmoid(logits)
        return probs, logits

def benchmark_bio_tagger(chunks, num_runs=50):
    """Benchmark BIO tagger inference speed"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = BIOTagger(num_labels=6).to(device)
    checkpoint = torch.load('bio_tagger_best.pt', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    times = []
    
    with torch.no_grad():
        for i, chunk_text in enumerate(chunks[:num_runs]):
            # Split into sentences (simple split)
            sentences = [s.strip() for s in chunk_text.split('.') if s.strip()]
            
            start = time.perf_counter()
            
            for sentence in sentences:
                # Tokenize
                inputs = tokenizer(
                    sentence,
                    padding='max_length',
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                
                # Inference
                probs, _ = model(input_ids, attention_mask)
                
                # Apply threshold (0.3)
                predictions = (probs > 0.3).cpu().numpy()
            
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{num_runs} chunks...")
    
    return times

def benchmark_openie(chunks, num_runs=50):
    """Benchmark OpenIE (your actual triplet extraction) speed"""
    try:
        from triplet_extract import OpenIEExtractor
        
        print("  Initializing OpenIEExtractor...")
        extractor = OpenIEExtractor(
            fast=True,
            speed_preset="fast",
            use_gpu=True
        )
        
        times = []
        
        for i, chunk_text in enumerate(chunks[:num_runs]):
            start = time.perf_counter()
            
            # Extract triplets with OpenIE
            triplets = extractor.extract_triplet_objects(chunk_text)
            
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{num_runs} chunks...")
        
        return times
        
    except ImportError:
        print("⚠️  triplet_extract not found")
        return None

def benchmark_stanza(chunks, num_runs=50):
    """Benchmark Stanza dependency parsing speed"""
    try:
        import stanza
        
        print("  Initializing Stanza pipeline...")
        try:
            nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', use_gpu=True)
        except:
            print("  Downloading Stanza model...")
            stanza.download('en')
            nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', use_gpu=True)
        
        times = []
        
        for i, chunk_text in enumerate(chunks[:num_runs]):
            start = time.perf_counter()
            
            # Process with Stanza
            doc = nlp(chunk_text)
            
            # Extract dependencies (triplet-like structures)
            for sentence in doc.sentences:
                for word in sentence.words:
                    _ = word.text, word.lemma, word.upos, word.head, word.deprel
            
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{num_runs} chunks...")
        
        return times
        
    except ImportError:
        print("⚠️  Stanza not installed. Install with: pip install stanza")
        return None

def load_chunks(num_chunks=50):
    """Load sample chunks from msgpack"""
    chunks_file = Path('checkpoints/chunks.msgpack')
    
    if not chunks_file.exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        return []
    
    with open(chunks_file, 'rb') as f:
        data = msgpack.unpackb(f.read(), raw=False)
    
    # Extract text from chunks
    chunk_texts = []
    for chunk in data[:num_chunks]:
        if isinstance(chunk, dict) and 'text' in chunk:
            chunk_texts.append(chunk['text'])
        elif isinstance(chunk, str):
            chunk_texts.append(chunk)
    
    return chunk_texts

def main():
    print("=" * 60)
    print("SPEED BENCHMARK: BIO Tagger vs OpenIE vs Stanza")
    print("=" * 60)
    
    # Load test chunks
    print("\n[1/3] Loading test chunks...")
    chunks = load_chunks(num_chunks=50)
    print(f"  Loaded {len(chunks)} chunks")
    
    if not chunks:
        print("❌ No chunks available for testing")
        return
    
    # Check model exists
    if not Path('bio_tagger_best.pt').exists():
        print("❌ Trained model not found: bio_tagger_best.pt")
        print("   Run training first: python train_bio_tagger.py")
        return
    
    # Benchmark BIO tagger
    print("\n[2/3] Benchmarking BIO Tagger...")
    bio_times = benchmark_bio_tagger(chunks, num_runs=50)
    
    bio_mean = np.mean(bio_times)
    bio_std = np.std(bio_times)
    bio_median = np.median(bio_times)
    
    print(f"\n  BIO Tagger Results (50 chunks):")
    print(f"    Mean:   {bio_mean:.2f} ms/chunk")
    print(f"    Median: {bio_median:.2f} ms/chunk")
    print(f"    Std:    {bio_std:.2f} ms")
    
    # Benchmark OpenIE (your actual approach)
    print("\n[3/5] Benchmarking OpenIE...")
    openie_times = benchmark_openie(chunks, num_runs=50)
    
    if openie_times:
        openie_mean = np.mean(openie_times)
        openie_std = np.std(openie_times)
        openie_median = np.median(openie_times)
        
        print(f"\n  OpenIE Results (50 chunks):")
        print(f"    Mean:   {openie_mean:.2f} ms/chunk")
        print(f"    Median: {openie_median:.2f} ms/chunk")
        print(f"    Std:    {openie_std:.2f} ms")
    
    # Benchmark Stanza
    print("\n[4/5] Benchmarking Stanza...")
    stanza_times = benchmark_stanza(chunks, num_runs=50)
    
    if stanza_times:
        stanza_mean = np.mean(stanza_times)
        stanza_std = np.std(stanza_times)
        stanza_median = np.median(stanza_times)
        
        print(f"\n  Stanza Results (50 chunks):")
        print(f"    Mean:   {stanza_mean:.2f} ms/chunk")
        print(f"    Median: {stanza_median:.2f} ms/chunk")
        print(f"    Std:    {stanza_std:.2f} ms")
        
        # Comparison
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(f"  BIO Tagger:          {bio_mean:.2f} ms/chunk")
        print(f"  OpenIE:              {openie_mean:.2f} ms/chunk")
        print(f"  Stanza:              {stanza_mean:.2f} ms/chunk")
        print(f"\n  Speedup vs OpenIE:   {openie_mean / bio_mean:.1f}x faster")
        print(f"  Speedup vs Stanza:   {stanza_mean / bio_mean:.1f}x faster")
        
        # Extrapolate to full dataset
        num_total_chunks = 3000
        bio_total = (bio_mean * num_total_chunks) / 1000 / 60  # minutes
        openie_total = (openie_mean * num_total_chunks) / 1000 / 60  # minutes
        stanza_total = (stanza_mean * num_total_chunks) / 1000 / 60  # minutes
        
        print(f"\n  Extrapolation to {num_total_chunks:,} chunks:")
        print(f"    BIO Tagger:          {bio_total:.1f} minutes")
        print(f"    OpenIE:              {openie_total:.1f} minutes")
        print(f"    Stanza:              {stanza_total:.1f} minutes")
        print(f"    Time saved (vs OpenIE): {openie_total - bio_total:.1f} minutes")
        print(f"    Time saved (vs Stanza): {stanza_total - bio_total:.1f} minutes")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
