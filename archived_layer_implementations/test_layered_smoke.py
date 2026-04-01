"""Smoke test for the layered knowledge graph."""
import sys
sys.path.insert(0, r'c:\Users\user\arxiv_id_lists')

from streamlit_bio_demo import _resolve_word, render_layered_graph

# Test resolution for each word
print("=== Word Resolution ===")
words_roles = [
    ('humans', 'SUBJ'), ('possess', 'PRED'),
    ('extraordinary', 'OBJ'), ('ability', 'OBJ'),
    ('create', 'OBJ'), ('utilize', 'OBJ'), ('tools', 'OBJ'),
]
for w, role in words_roles:
    info = _resolve_word(w, role)
    print(f"  {w:18s} lemma={info['lemma']:15s} synset={info['synset']:22s} hypernym={info['hypernym']}")

# Check deduplication: if two words share same lemma, they should converge
print("\n=== Deduplication check ===")
# tools -> tool (lemma), tool.n.01 (synset), implement.n.01 (hypernym)
info_tools = _resolve_word('tools', 'OBJ')
info_tool = _resolve_word('tool', 'OBJ')
print(f"  tools: lemma={info_tools['lemma']}, synset={info_tools['synset']}")
print(f"  tool:  lemma={info_tool['lemma']}, synset={info_tool['synset']}")
print(f"  Same lemma? {info_tools['lemma'] == info_tool['lemma']}")

# Render the graph
print("\n=== Render layered graph ===")
fig, details = render_layered_graph(
    ['humans'],
    ['possess'],
    ['extraordinary', 'ability', 'create', 'utilize', 'tools']
)
if fig is not None:
    fig.savefig(r'c:\Users\user\arxiv_id_lists\test_layered_graph.png', dpi=120, bbox_inches='tight')
    print(f"  Saved test_layered_graph.png  (size={fig.get_size_inches()})")
    import matplotlib.pyplot as plt
    plt.close(fig)
else:
    print("  ERROR: fig is None!")

print(f"\n  {len(details)} detail rows:")
for d in details:
    print(f"    {d}")

print("\n=== Count unique nodes per layer ===")
lemmas = set(d['Lemma'] for d in details)
synsets = set(d['Synset'] for d in details)
hypernyms = set(d['Hypernym'] for d in details)
print(f"  Surface words: {len(details)}")
print(f"  Unique lemmas: {len(lemmas)}  {sorted(lemmas)}")
print(f"  Unique synsets: {len(synsets)}  {sorted(synsets)}")
print(f"  Unique hypernyms: {len(hypernyms)}  {sorted(hypernyms)}")

print("\nSMOKE TEST PASSED")
