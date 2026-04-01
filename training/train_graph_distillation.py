"""
Train Graph Distillation Model

Two-phase approach:
1. Node2Vec: Extract graph embeddings from AOKG built on 250 training chunks
2. Distillation: Train DistilBERT to predict Node2Vec embeddings from raw text

Output: graph_distillation_model.pt (fast inference model)
"""

import argparse
import msgpack
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# Import transformers components separately to avoid lazy loading issues
import transformers
transformers.utils.logging.set_verbosity_error()
from transformers.models.distilbert.modeling_distilbert import DistilBertModel
from transformers.models.distilbert.tokenization_distilbert import DistilBertTokenizer

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path
import time
from tqdm import tqdm
import networkx as nx
from karateclub import Node2Vec
from collections import defaultdict

# Import BIO tagger for triplet extraction
from inference_bio_tagger import BIOTripletExtractor


class AOKGBuilder:
    """Build 4-layer AOKG from SPO triplets"""
    
    def __init__(self):
        from nltk.corpus import wordnet as wn
        self.wn = wn
        
    def build_graph(self, triplets):
        """
        Build AOKG from triplets
        
        Returns:
            nx.DiGraph with nodes at 4 layers (L0-L3)
        """
        G = nx.DiGraph()
        
        for s, p, o in triplets:
            # Clean tokens (remove special chars, lowercase)
            s_clean = [t.lower().strip() for t in s if t.strip()]
            p_clean = [t.lower().strip() for t in p if t.strip()]
            o_clean = [t.lower().strip() for t in o if t.strip()]
            
            all_tokens = s_clean + p_clean + o_clean
            
            for token in all_tokens:
                if not token:
                    continue
                    
                # L0: Surface form
                l0_node = f"L0:{token}"
                G.add_node(l0_node, layer=0, token=token)
                
                # L1: Lemma (use wordnet lemmatization)
                lemma = self._get_lemma(token)
                l1_node = f"L1:{lemma}"
                G.add_node(l1_node, layer=1, token=lemma)
                G.add_edge(l0_node, l1_node, relation="lemmatizes_to")
                
                # L2: Synsets (first sense only for simplicity)
                synsets = self.wn.synsets(lemma)
                if synsets:
                    synset = synsets[0]
                    l2_node = f"L2:{synset.name()}"
                    G.add_node(l2_node, layer=2, synset=synset.name())
                    G.add_edge(l1_node, l2_node, relation="belongs_to_synset")
                    
                    # L3: Hypernyms (first hypernym only)
                    hypernyms = synset.hypernyms()
                    if hypernyms:
                        hypernym = hypernyms[0]
                        l3_node = f"L3:{hypernym.name()}"
                        G.add_node(l3_node, layer=3, synset=hypernym.name())
                        G.add_edge(l2_node, l3_node, relation="is_a")
        
        return G
    
    def _get_lemma(self, token):
        """Get lemma using WordNet morphy"""
        lemma = self.wn.morphy(token)
        return lemma if lemma else token


class GraphEmbeddingDataset(Dataset):
    """Dataset of (text, graph_embedding) pairs"""
    
    def __init__(self, texts, embeddings, tokenizer, max_length=512):
        self.texts = texts
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        embedding = self.embeddings[idx]
        
        # Tokenize text
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'graph_embedding': torch.tensor(embedding, dtype=torch.float32)
        }


class GraphDistillationModel(nn.Module):
    """DistilBERT + projection layer to predict graph embeddings"""
    
    def __init__(self, embed_dim=128):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.projection = nn.Linear(768, embed_dim)
        
    def forward(self, input_ids, attention_mask):
        # Get DistilBERT output
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        
        # Project to graph embedding space
        graph_embedding = self.projection(cls_embedding)  # [batch, 128]
        
        return graph_embedding


def extract_triplets_with_bio_tagger(chunks, model_path='bio_tagger_best.pt'):
    """Extract SPO triplets from chunks using BIO tagger"""
    print("\n" + "="*60)
    print("PHASE 1A: Extracting Triplets with BIO Tagger")
    print("="*60)
    
    extractor = BIOTripletExtractor(model_path=model_path)
    
    all_triplets = []
    for i, chunk in enumerate(tqdm(chunks, desc="Extracting triplets")):
        triplets = extractor.extract_triplets(chunk['sentence'])  # Use 'sentence' key from training data
        all_triplets.append(triplets)
        
        if i == 0:
            print(f"\nExample chunk 0 triplets ({len(triplets)} total):")
            for j, (s, p, o) in enumerate(triplets[:3]):
                print(f"  {j+1}. S={s[:3]}... P={p[:2]}... O={o[:3]}...")
    
    print(f"\nExtracted {sum(len(t) for t in all_triplets)} total triplets from {len(chunks)} chunks")
    return all_triplets


def build_aokgs_from_triplets(all_triplets):
    """Build AOKG for each chunk"""
    print("\n" + "="*60)
    print("PHASE 1B: Building AOKGs")
    print("="*60)
    
    builder = AOKGBuilder()
    graphs = []
    
    for i, triplets in enumerate(tqdm(all_triplets, desc="Building graphs")):
        if not triplets:
            # Empty graph
            graphs.append(nx.DiGraph())
            continue
            
        graph = builder.build_graph(triplets)
        graphs.append(graph)
        
        if i == 0:
            print(f"\nExample AOKG 0:")
            print(f"  Nodes: {graph.number_of_nodes()}")
            print(f"  Edges: {graph.number_of_edges()}")
            layers = defaultdict(int)
            for node, data in graph.nodes(data=True):
                layers[data.get('layer', -1)] += 1
            print(f"  Layer distribution: {dict(layers)}")
    
    return graphs


def derive_node2vec_embeddings(graphs, dimensions=128, walk_length=80, walk_number=10):
    """Derive Node2Vec embeddings and aggregate per chunk"""
    print("\n" + "="*60)
    print("PHASE 1C: Deriving Node2Vec Embeddings")
    print("="*60)
    print(f"Parameters: dim={dimensions}, walk_len={walk_length}, walks={walk_number}")
    
    chunk_embeddings = []
    
    for i, G in enumerate(tqdm(graphs, desc="Node2Vec")):
        if G.number_of_nodes() == 0:
            # Empty graph -> zero embedding
            chunk_embeddings.append(np.zeros(dimensions))
            continue
        
        # Convert to undirected for Node2Vec
        G_undirected = G.to_undirected()
        
        # Relabel nodes to integers (karateclub requirement)
        node_mapping = {node: idx for idx, node in enumerate(G_undirected.nodes())}
        G_relabeled = nx.relabel_nodes(G_undirected, node_mapping)
        
        # Run Node2Vec
        model = Node2Vec(dimensions=dimensions, walk_length=walk_length, walk_number=walk_number)
        model.fit(G_relabeled)
        
        # Get embeddings for all nodes
        node_embeddings = model.get_embedding()  # [num_nodes, dimensions]
        
        # Aggregate via mean pooling
        chunk_embedding = np.mean(node_embeddings, axis=0)  # [dimensions]
        chunk_embeddings.append(chunk_embedding)
        
        if i == 0:
            print(f"\nExample chunk 0:")
            print(f"  Graph nodes: {G.number_of_nodes()}")
            print(f"  Node embeddings shape: {node_embeddings.shape}")
            print(f"  Chunk embedding: {chunk_embedding[:5]}... (mean of {len(node_embeddings)} nodes)")
    
    chunk_embeddings = np.array(chunk_embeddings)  # [num_chunks, dimensions]
    print(f"\nFinal embeddings shape: {chunk_embeddings.shape}")
    print(f"Mean norm: {np.mean(np.linalg.norm(chunk_embeddings, axis=1)):.4f}")
    
    return chunk_embeddings


def train_distillation_model(texts, graph_embeddings, epochs=20, batch_size=16, lr=3e-5, val_split=0.2):
    """Train DistilBERT to predict graph embeddings from text"""
    print("\n" + "="*60)
    print("PHASE 2: Training Distillation Model")
    print("="*60)
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {lr}, Val split: {val_split}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Split train/val
    num_val = int(len(texts) * val_split)
    num_train = len(texts) - num_val
    
    indices = np.random.permutation(len(texts))
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]
    
    print(f"Train: {num_train}, Val: {num_val}")
    
    # Create datasets
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_dataset = GraphEmbeddingDataset(
        [texts[i] for i in train_idx],
        [graph_embeddings[i] for i in train_idx],
        tokenizer
    )
    
    val_dataset = GraphEmbeddingDataset(
        [texts[i] for i in val_idx],
        [graph_embeddings[i] for i in val_idx],
        tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = GraphDistillationModel(embed_dim=graph_embeddings.shape[1])
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['graph_embedding'].to(device)
            
            optimizer.zero_grad()
            predicted = model(input_ids, attention_mask)
            loss = criterion(predicted, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        all_predicted = []
        all_target = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target = batch['graph_embedding'].to(device)
                
                predicted = model(input_ids, attention_mask)
                loss = criterion(predicted, target)
                
                val_loss += loss.item()
                all_predicted.append(predicted.cpu().numpy())
                all_target.append(target.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Calculate correlation
        all_predicted = np.concatenate(all_predicted, axis=0)
        all_target = np.concatenate(all_target, axis=0)
        
        # Flatten and compute correlation
        pred_flat = all_predicted.flatten()
        target_flat = all_target.flatten()
        correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, Corr={correlation:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'graph_distillation_model.pt')
            print(f"  → Best model saved (corr={correlation:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to: graph_distillation_model.pt")
    
    return model, correlation


def visualize_embeddings(graph_embeddings, texts, output_path='node2vec_tsne.png'):
    """Visualize embeddings with t-SNE"""
    print("\n" + "="*60)
    print("Visualizing Embeddings with t-SNE")
    print("="*60)
    
    if len(graph_embeddings) < 5:
        print("Too few samples for t-SNE visualization")
        return
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(graph_embeddings)-1))
    embeddings_2d = tsne.fit_transform(graph_embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=50)
    
    # Annotate a few points
    for i in range(min(10, len(texts))):
        plt.annotate(
            f"{i}: {texts[i][:30]}...",
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            fontsize=8,
            alpha=0.7
        )
    
    plt.title("Node2Vec Graph Embeddings (t-SNE)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train graph distillation model")
    parser.add_argument('--data', type=str, default='data/bio_training_250chunks_complete_FIXED.msgpack',
                       help='Training data msgpack file')
    parser.add_argument('--bio-model', type=str, default='bio_tagger_best.pt',
                       help='BIO tagger model path')
    parser.add_argument('--dimensions', type=int, default=128,
                       help='Node2Vec embedding dimensions')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-5,
                       help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--visualize', action='store_true',
                       help='Create t-SNE visualization')
    
    args = parser.parse_args()
    
    print("="*60)
    print("GRAPH DISTILLATION TRAINING")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"BIO Model: {args.bio_model}")
    print(f"Dimensions: {args.dimensions}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*60)
    
    # Load training data
    print("\nLoading training data...")
    with open(args.data, 'rb') as f:
        data = msgpack.unpack(f, raw=False)
    
    chunks = data['training_data'][:250]  # Use all 250 chunks
    texts = [chunk['sentence'] for chunk in chunks]  # Use 'sentence' key from training data
    print(f"Loaded {len(chunks)} chunks")
    
    # Phase 1A: Extract triplets
    all_triplets = extract_triplets_with_bio_tagger(chunks, args.bio_model)
    
    # Phase 1B: Build AOKGs
    graphs = build_aokgs_from_triplets(all_triplets)
    
    # Phase 1C: Derive Node2Vec embeddings
    graph_embeddings = derive_node2vec_embeddings(
        graphs,
        dimensions=args.dimensions,
        walk_length=80,
        walk_number=10
    )
    
    # Optional visualization
    if args.visualize:
        visualize_embeddings(graph_embeddings, texts)
    
    # Phase 2: Train distillation model
    model, final_corr = train_distillation_model(
        texts,
        graph_embeddings,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"✅ Model saved: graph_distillation_model.pt")
    print(f"✅ Final correlation: {final_corr:.4f}")
    print(f"✅ Target: r > 0.7 {'✓ ACHIEVED' if final_corr > 0.7 else '✗ NOT MET'}")
    print("\nNext steps:")
    print("1. Run: python apply_graph_embeddings.py (derive embeddings for full corpus)")
    print("2. Run: python build_graph_hnsw.py (build HNSW index)")
    print("3. Run: python graph_reranker.py (implement reranking)")
    print("="*60)


if __name__ == '__main__':
    main()
