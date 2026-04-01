# compare_models.py
"""
Compare Qwen3-Embedding-0.6B (original) vs Model2Vec (distilled)
"""

from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from model2vec import StaticModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_original(texts, tokenizer, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Encode with original Qwen3-Embedding-0.6B"""
    if isinstance(texts, str):
        texts = [texts]
    
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=8192, return_tensors='pt')
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    with torch.no_grad():
        model_output = model(**encoded)
    
    embeddings = mean_pooling(model_output, encoded['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings.cpu().numpy()

def print_similarity_matrix(sim_matrix, texts, title):
    """Pretty print similarity matrix"""
    print(f"\n{title}")
    print("=" * 80)
    print(f"{'':30} Text 1   Text 2   Text 3")
    for i, text in enumerate(texts):
        text_short = text[:28] + "..." if len(text) > 28 else text
        print(f"{text_short:30} {sim_matrix[i, 0]:.3f}    {sim_matrix[i, 1]:.3f}    {sim_matrix[i, 2]:.3f}")

def main():
    print("Loading models...\n")
    
    # Load original Qwen3-Embedding-0.6B
    print("1. Loading original Qwen3-Embedding-0.6B...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True)
    model_original = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True).to(device)
    print(f"   ✓ Loaded on {device}")
    
    # Load Model2Vec distilled version
    print("2. Loading Model2Vec distilled version...")
    model_distilled = StaticModel.from_pretrained("./qwen3_static_embeddings")
    print(f"   ✓ Loaded (256 dimensions)")
    
    # Test texts
    test_texts = [
        "Self-attention has O(n²) complexity",
        "Linear attention reduces computational cost",
        "The weather is nice today"
    ]
    
    print(f"\nTest texts:")
    for i, text in enumerate(test_texts, 1):
        print(f"  {i}. {text}")
    
    # Encode with original model
    print("\nEncoding with original model...")
    embeddings_original = encode_original(test_texts, tokenizer, model_original, device)
    print(f"   ✓ Shape: {embeddings_original.shape}")
    
    # Encode with distilled model
    print("Encoding with distilled model...")
    embeddings_distilled = model_distilled.encode(test_texts)
    print(f"   ✓ Shape: {embeddings_distilled.shape}")
    
    # Compute similarity matrices
    sim_original = cosine_similarity(embeddings_original)
    sim_distilled = cosine_similarity(embeddings_distilled)
    
    # Print comparison
    print_similarity_matrix(sim_original, test_texts, "ORIGINAL Qwen3-Embedding-0.6B (768d)")
    print_similarity_matrix(sim_distilled, test_texts, "DISTILLED Model2Vec (256d)")
    
    # Compute differences
    print("\nDIFFERENCES (Original - Distilled)")
    print("=" * 80)
    print(f"{'':30} Text 1   Text 2   Text 3")
    for i, text in enumerate(test_texts):
        text_short = text[:28] + "..." if len(text) > 28 else text
        diff = sim_original[i] - sim_distilled[i]
        print(f"{text_short:30} {diff[0]:+.3f}    {diff[1]:+.3f}    {diff[2]:+.3f}")
    
    # Key metrics
    print("\nKEY METRICS")
    print("=" * 80)
    
    # Related pairs (Text 1 & 2 - both about attention)
    related_orig = sim_original[0, 1]
    related_dist = sim_distilled[0, 1]
    print(f"Related texts (1 & 2 - both about attention):")
    print(f"  Original:  {related_orig:.3f}")
    print(f"  Distilled: {related_dist:.3f}")
    print(f"  Difference: {related_orig - related_dist:+.3f}")
    
    # Unrelated pairs (Text 1 & 3 - attention vs weather)
    unrelated_orig = sim_original[0, 2]
    unrelated_dist = sim_distilled[0, 2]
    print(f"\nUnrelated texts (1 & 3 - attention vs weather):")
    print(f"  Original:  {unrelated_orig:.3f}")
    print(f"  Distilled: {unrelated_dist:.3f}")
    print(f"  Difference: {unrelated_orig - unrelated_dist:+.3f}")
    
    # Separation quality
    separation_orig = related_orig - unrelated_orig
    separation_dist = related_dist - unrelated_dist
    print(f"\nSeparation quality (related - unrelated):")
    print(f"  Original:  {separation_orig:.3f}")
    print(f"  Distilled: {separation_dist:.3f}")
    print(f"  Loss: {separation_orig - separation_dist:+.3f}")
    
    # Overall assessment
    print("\nASSESSMENT")
    print("=" * 80)
    if separation_dist >= 0.20:
        print("✓ GOOD: Distilled model maintains separation between related/unrelated")
    elif separation_dist >= 0.10:
        print("⚠ ACCEPTABLE: Some separation loss, but usable for retrieval")
    else:
        print("✗ POOR: Significant separation loss - consider re-distilling with 768d")

if __name__ == "__main__":
    main()