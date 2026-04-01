# distill_qwen3.py
"""
Distill Qwen3-Embedding-0.6B to static embeddings using Model2Vec.
"""

from model2vec.distill import distill
import torch

def main():
    print("Starting Model2Vec distillation of Qwen3-Embedding-0.6B...")
    print("This will take ~30-60 seconds on CPU.\n")
    
    # Distill Qwen3-Embedding to static embeddings
    m2v_model = distill(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        # Optional parameters (uncomment to customize):
        pca_dims=64,              # Reduce dimensions (default: keep original)
        # apply_zipf=True,           # Apply Zipf weighting (default: True)
        # use_subword=True,          # Use subword tokenization (default: True)
        # show_progress_bar=True,    # Show progress (default: True)
    )
    
    # Save the distilled model
    output_path = "./qwen3_static_embeddings"
    m2v_model.save_pretrained(output_path)
    print(f"\n✓ Distilled model saved to: {output_path}")
    
    # Test the model
    print("\nTesting the distilled model...")
    test_texts = [
        "Self-attention has O(n²) complexity",
        "Linear attention reduces computational cost",
        "The weather is nice today"
    ]
    
    embeddings = m2v_model.encode(test_texts)
    print(f"✓ Generated embeddings shape: {embeddings.shape}")
    print(f"  - {len(test_texts)} texts")
    print(f"  - {embeddings.shape[1]} dimensions")
    
    # Test similarity
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    sim_matrix = cosine_similarity(embeddings)
    print("\nCosine similarity matrix:")
    print("                          Text 1   Text 2   Text 3")
    for i, text in enumerate(test_texts):
        text_short = text[:25] + "..." if len(text) > 25 else text
        print(f"{text_short:25} {sim_matrix[i, 0]:.3f}    {sim_matrix[i, 1]:.3f}    {sim_matrix[i, 2]:.3f}")
    
    print("\n✓ Distillation complete!")
    print(f"  Model saved to: {output_path}")
    print(f"  Load with: StaticModel.from_pretrained('{output_path}')")

if __name__ == "__main__":
    main()