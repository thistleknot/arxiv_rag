"""Quick debug of the new GraphRetriever extraction pipeline."""
import json
import ollama
import time

paper = (
    "Transformers are a neural network architecture introduced by Vaswani et al. in 2017. "
    "The key innovation is the self-attention mechanism which allows the model to attend "
    "to all positions in the input sequence simultaneously. BERT uses masked language modeling "
    "as its pre-training objective. GPT uses autoregressive language modeling instead. "
    "Both models build on the transformer architecture but differ in their training approach."
)

schema = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                },
                "required": ["name", "type"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation_type": {"type": "string"},
                },
                "required": ["source", "target", "relation_type"],
            },
        },
    },
    "required": ["entities", "relations"],
}

c = ollama.Client(host="http://127.0.0.1:11434")
prompt = (
    "/no_think\n"
    "Extract all important entities and their relations from the following academic text.\n"
    "Focus on: methods, models, algorithms, datasets, metrics, concepts, and their relationships.\n"
    "Text:\n" + paper
)

print("1) Raw Ollama call...")
t0 = time.time()
try:
    resp = c.chat(
        model="qwen3.5:2b",
        messages=[{"role": "user", "content": prompt}],
        format=schema,
        options={"num_predict": 2048, "temperature": 0.1},
    )
    raw = resp.message.content
    print(f"   Response length: {len(raw)} chars  ({time.time()-t0:.1f}s)")
    print(f"   First 400 chars: {raw[:400]}")
    data = json.loads(raw)
    ents = data.get("entities", [])
    rels = data.get("relations", [])
    print(f"   Entities: {len(ents)}")
    for e in ents:
        print(f"     {e}")
    print(f"   Relations: {len(rels)}")
    for r in rels:
        print(f"     {r}")
except Exception as ex:
    print(f"   ERROR: {type(ex).__name__}: {ex}")

print("\n2) GraphRetriever integration test...")
from graph.graph_retriever import GraphRetriever

gr = GraphRetriever()
print(f"   client OK: {gr._client is not None}")
print(f"   embedder OK: {gr._embedder is not None}")

# Test chunk extraction
print("   Extracting single chunk...")
t0 = time.time()
result = gr._extract_chunk(paper)
print(f"   Done in {time.time()-t0:.1f}s")
if result is None:
    print("   RESULT: None (extraction failed)")
else:
    ents = result.get("entities", [])
    rels = result.get("relations", [])
    print(f"   Entities: {len(ents)}, Relations: {len(rels)}")

# Test full retrieve_context
print("\n   Full retrieve_context...")
t0 = time.time()
ctx = gr.retrieve_context(
    query="attention mechanism in transformers",
    paper_texts={"test_001": paper},
    top_k=10,
)
print(f"   Done in {time.time()-t0:.1f}s")
print(f"   Context length: {len(ctx)} chars")
if ctx:
    for line in ctx.split("\n"):
        print(f"   > {line}")
else:
    print("   EMPTY — debugging _extract_paper_graph:")
    pg = gr._extract_paper_graph("test_002", paper)
    print(f"   entities: {len(pg.entities)}")
    print(f"   relations: {len(pg.relations)}")
    print(f"   triplets: {len(pg.triplets)}")
    for t in pg.triplets[:5]:
        print(f"     {t}")
