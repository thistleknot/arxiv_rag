"""
Update feature catalog with hierarchical retrieval design.
Adds Feature 12 with complete design documentation.
"""

import sqlite3
from datetime import datetime

# Hierarchical Retrieval Design Documentation
HIERARCHICAL_DESIGN = """
## Architecture Overview

Hierarchical paper→section retrieval for three-layer φ-retriever system.

**Purpose:** Save expensive late-stage encoders (ColBERT/cross-encoder) for filtered paper sections, not raw chunks.

**Pattern:** chunks → sections → papers → select top k² papers → fetch ALL sections → Layer 3 reranking

## Flow Diagram

```
Layer 1: BM25 + Dense + RRF → 13 seed chunks
  ↓
Layer 2: Graph + Qwen3 expansion → 157 total chunks
  ↓
Layer 2.5: HIERARCHICAL AGGREGATION (NEW)
  • Group 157 chunks → ~40 sections (by paper_id + section_idx)
  • Aggregate RRF scores per section
  • Group sections → ~20 papers (by paper_id)
  • Aggregate RRF scores per paper
  • Select top k² papers (13² = 169 papers)
  • Fetch ALL sections from filtered papers (~500 sections)
  ↓
Layer 3: Cross-Encoder on filtered sections → 13 final
```

## Implementation Source

**Reference Implementation:** gist_retriever.py (lines 1594-1720)

**Key Methods:**
1. `_group_and_reconstruct(chunks, limit)` - Groups chunks by (paper_id, section_idx), fetches complete sections, reconstructs full text with de-overlap
2. `_group_sections_into_papers(sections, max_papers)` - Groups sections by paper_id, aggregates RRF scores, selects top papers
3. `_fetch_all_chunks_for_section(paper_id, section_idx)` - SQL query to get ALL chunks in a section
4. `_de_overlap_strings(texts)` - Removes overlapping text between consecutive chunks

## Data Classes

```python
@dataclass
class RetrievedGroup:
    group_id: str  # "paper_id:s{section_idx}"
    group_key: Tuple[str, int]  # (paper_id, section_idx)
    matched_chunks: List[RetrievedDoc]
    full_text: str  # Complete section (de-overlapped)
    rrf_score: float
    metadata: dict

@dataclass
class RetrievedPaper:
    paper_id: str
    sections: List[RetrievedGroup]
    full_text: str  # All sections concatenated
    rrf_score: float
    metadata: dict
```

## Integration Plan for Three-Layer System

**Location:** three_layer_phi_retriever.py, line 424 (after Layer 2, before Layer 3)

**Approach:**
1. Copy helper methods from gist_retriever.py
2. Add `_layer3_hierarchical_rerank()` method
3. Add config flag: `enable_hierarchical: bool = True`
4. Branch in `retrieve()`: if hierarchical → use paper aggregation, else → direct chunk reranking

**Configuration:**
- `paper_pool_size: int = 169` (k² for k=13)
- `section_limit: int = 39` (k × 3 sections to consider)
- `use_cross_encoder: bool = True`

## Performance Hypothesis

**Goal:** Reduce late encoder workload by filtering papers first

**Example (k=13):**
- Layer 2 output: 157 chunks
- Group → ~40 sections
- Select top 169 papers (k²)
- Fetch all sections → ~500 sections from filtered papers
- Layer 3: Cross-encode 500 sections → 13 final

**Trade-off:** More sections than raw chunks, BUT higher quality context (complete sections from top papers)

## Design Rationale

**Why hierarchical?**
1. Complete context: Full sections preserve narrative flow
2. Quality filtering: Focus on top-ranked papers by aggregated evidence
3. Flexible reranking: Can rerank at paper-level OR section-level
4. Alignment with arXiv structure: Papers naturally organize into sections

**Architectural Decision:** Rerank at section-level (not paper-level) to maintain fine-grained retrieval while benefiting from paper-level filtering.

## Testing Strategy

1. Unit test grouping methods with mock data
2. Integration test: full pipeline with enable_hierarchical=True
3. Benchmark: compare time/quality vs. non-hierarchical
4. Visual inspection: check paper diversity in results

## References

- gist_retriever.py: Complete reference implementation (Steps 5-9)
- pgvector_retriever.py: Base methods (_get_group_key, _fetch_all_chunks_for_group)
- HIERARCHICAL_RETRIEVAL_DESIGN.md: Full design document
- HIERARCHICAL_IMPLEMENTATION_PLAN.md: Step-by-step implementation guide
- HIERARCHICAL_VISUAL_ROADMAP.md: Visual diagrams and insertion points
"""

def add_hierarchical_feature():
    """Add hierarchical retrieval as Feature 12."""
    conn = sqlite3.connect('feature_catalog.sqlite3')
    c = conn.cursor()
    
    # Check if feature 12 already exists
    c.execute('SELECT id FROM features WHERE id = 12')
    if c.fetchone():
        print("Feature 12 already exists, updating...")
        c.execute("""
            UPDATE features 
            SET name = ?,
                description = ?,
                status = ?,
                updated_at = ?,
                validation_notes = ?
            WHERE id = 12
        """, (
            "Hierarchical Paper→Section Retrieval",
            HIERARCHICAL_DESIGN,
            "TODO",
            datetime.now().isoformat(),
            "Design complete. Implementation pending. Reference: gist_retriever.py lines 1594-1720."
        ))
    else:
        print("Adding Feature 12...")
        c.execute("""
            INSERT INTO features (name, description, status, validation_notes)
            VALUES (?, ?, ?, ?)
        """, (
            "Hierarchical Paper→Section Retrieval",
            HIERARCHICAL_DESIGN,
            "TODO",
            "Design complete. Implementation pending. Reference: gist_retriever.py lines 1594-1720."
        ))
    
    conn.commit()
    feature_id = c.lastrowid if c.lastrowid else 12
    
    # Add architectural decision
    c.execute('SELECT COUNT(*) FROM architectural_decisions')
    ad_count = c.fetchone()[0]
    
    print(f"\nAdding Architectural Decision {ad_count + 1}...")
    c.execute("""
        INSERT INTO architectural_decisions (
            decision,
            rationale,
            before_state,
            after_state
        ) VALUES (?, ?, ?, ?)
    """, (
        "Hierarchical aggregation: chunks → sections → papers → Layer 3",
        """
        Save expensive late-stage encoders for aggregates of filtered papers.
        
        Key insight: Instead of reranking 157 raw chunks, group into sections,
        aggregate to top k² papers (169), fetch ALL sections from those papers,
        then rerank at section-level with complete context.
        
        Trade-off: More sections (~500) than raw chunks (157), BUT higher quality
        because sections provide complete narrative context from top-ranked papers.
        
        Design choice: Rerank at section-level (not paper-level) to maintain
        fine-grained retrieval while benefiting from paper-level filtering.
        """,
        "Layer 3 directly reranks 157 chunks from Layer 2 expansion",
        "Layer 2 → hierarchical grouping → top k² papers → Layer 3 reranks sections from filtered papers"
    ))
    
    conn.commit()
    conn.close()
    
    print(f"✓ Feature {feature_id} added/updated")
    print(f"✓ Architectural decision logged")
    print("\nFeature catalog updated successfully!")


if __name__ == "__main__":
    print("="*60)
    print("UPDATING FEATURE CATALOG")
    print("="*60)
    print("\nAdding hierarchical retrieval design documentation...")
    
    add_hierarchical_feature()
    
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    # Verify
    conn = sqlite3.connect('feature_catalog.sqlite3')
    c = conn.cursor()
    
    c.execute('SELECT COUNT(*) FROM features')
    total = c.fetchone()[0]
    print(f"\nTotal features: {total}")
    
    c.execute('SELECT id, name, status FROM features WHERE id = 12')
    row = c.fetchone()
    if row:
        print(f"\nFeature 12: {row[1]}")
        print(f"Status: {row[2]}")
    
    c.execute('SELECT COUNT(*) FROM architectural_decisions')
    ad_count = c.fetchone()[0]
    print(f"\nTotal architectural decisions: {ad_count}")
    
    conn.close()
