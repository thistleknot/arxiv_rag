"""
diagnose_through_line.py - Systematic diagnostic for missing Through-Line synthesis.

Traces the complete 8-stage pipeline to identify why "Through-Line" sections
are missing from retrieval reports despite markdown files existing and synthesis
code being implemented.

Execution Flow:
  1. Load sample query + verify prerequisites
  2. Test Stage 4: Thesis generation via SyllogismFormer with verbose logging
  3. Test Stage 6: Paper loading to verify papers_content availability
  4. Test Stage 8: Through-line derivation with full logging
  5. Generate sample report and compare with expected output

Find-Out and Fix:
  - Thesis being empty or falsy (blocking gate)
  - papers_content being empty dict (Stage 6 failure)
  - LLM failures silently returning empty strings
  - Verbose mode toggling to see what's actually happening
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

# Add imports for testing
sys.path.insert(0, str(Path(__file__).parent))

from reasoning.syllogism_former import SyllogismFormer, SyllogismResult, ChainLink
from reasoning.paper_loader import load_papers
from arxiv_pipeline.syllogism_retriever import SyllogismRetriever


def test_stage_4_thesis_generation():
    """Test thesis generation (Stage 4) with verbose logging."""
    print("\n" + "="*80)
    print("TEST 1: Stage 4 — Thesis Generation (SyllogismFormer)")
    print("="*80)
    
    # Sample test data
    query = "transformer attention mechanisms for long context"
    premises_by_paper = {
        "2005.09007": [
            "Facilitates capturing more contextual information from different scales.",
            "Allows training deep network from scratch without backbones.",
        ],
        "1603.09320": [
            "Efficient nearest neighbor search using HNSW graphs.",
        ],
        "2101.09671": [
            "Reduction in computational requirements through pruning and quantization.",
        ],
    }
    nli_scores = {
        "2005.09007": 0.95,
        "1603.09320": 0.85,
        "2101.09671": 0.72,
    }
    objective_text = "Understanding transformer attention mechanisms for handling long context in sequences"
    
    # Test with verbose=True
    print("\n[INFO] Creating SyllogismFormer with verbose=True...")
    former = SyllogismFormer(verbose=True)
    
    print(f"\n[INFO] Calling form() with:")
    print(f"  - query: {query}")
    print(f"  - {len(premises_by_paper)} papers with premises")
    print(f"  - nli_scores present: {list(nli_scores.keys())}")
    print(f"  - objective_text: {objective_text}")
    
    result: SyllogismResult = former.form(
        query=query,
        premises_by_paper=premises_by_paper,
        nli_scores=nli_scores,
        objective_text=objective_text,
    )
    
    print(f"\n[RESULT] SyllogismFormer.form() returned:")
    print(f"  - thesis: {repr(result.thesis)}")
    print(f"  - thesis truthy: {bool(result.thesis)}")
    print(f"  - chain length: {len(result.chain)}")
    print(f"  - paper_scores: {result.paper_scores}")
    
    # Verify thesis is not empty
    if result.thesis and result.thesis not in ["No thesis generated.", "Thesis formation failed."]:
        print(f"\n✅ PASS: Thesis generated successfully")
        return result
    else:
        print(f"\n❌ FAIL: Thesis is empty or error message")
        return None


def test_stage_6_paper_loading():
    """Test paper loading (Stage 6) to verify papers_content availability."""
    print("\n" + "="*80)
    print("TEST 2: Stage 6 — Paper Loading (load_papers)")
    print("="*80)
    
    # Test with real arXiv IDs from the papers directory
    test_ids = ["1603.09320", "2005.09007", "2101.09671"]
    
    print(f"\n[INFO] Loading papers for IDs: {test_ids}")
    print(f"[INFO] Expected location: papers/post_processed/")
    
    papers_content = load_papers(test_ids)
    
    print(f"\n[RESULT] load_papers() returned:")
    print(f"  - dict size: {len(papers_content)}")
    print(f"  - keys: {list(papers_content.keys())}")
    
    for arxiv_id, content in papers_content.items():
        print(f"  - [{arxiv_id}] content length: {len(content)} chars")
        if len(content) > 100:
            print(f"    Preview: {content[:100]}...")
    
    if papers_content:
        print(f"\n✅ PASS: Paper loading successful")
        return papers_content
    else:
        print(f"\n❌ FAIL: No papers loaded")
        return {}


def test_stage_8_through_line(result: Optional[SyllogismResult], papers_content: Dict[str, str]):
    """Test through-line derivation (Stage 8) with verbose logging."""
    print("\n" + "="*80)
    print("TEST 3: Stage 8 — Through-Line Synthesis (_derive_through_line)")
    print("="*80)
    
    if not result or not result.thesis:
        print(f"\n⚠️  SKIP: Stage 4 failed, cannot test Stage 8")
        return None
    
    query = "transformer attention mechanisms for long context"
    
    # Create retriever with verbose=True to capture through-line generation
    print(f"\n[INFO] Creating SyllogismRetriever with verbose=True...")
    
    try:
        retriever = SyllogismRetriever(verbose=True, use_cache=False)
    except Exception as e:
        print(f"\n❌ ERROR: Could not create SyllogismRetriever: {e}")
        return None
    
    # We can't directly test _derive_through_line() without the full pipeline,
    # but we can at least verify the gate condition
    print(f"\n[INFO] Verifying gate condition...")
    print(f"  - result.thesis truthy: {bool(result.thesis)}")
    print(f"  - papers_content available: {bool(papers_content)}")
    print(f"  - Gate condition (if result.thesis): {bool(result.thesis)}")
    
    if result.thesis:
        print(f"\n✅ PASS: Gate condition allows through-line derivation")
        print(f"    (Full LLM derivation would proceed in production)")
    else:
        print(f"\n❌ FAIL: Gate condition blocks through-line derivation")
    
    retriever.close()
    return True


def test_full_pipeline():
    """Run a sample query through the full 8-stage pipeline."""
    print("\n" + "="*80)
    print("TEST 4: Full 8-Stage Pipeline")
    print("="*80)
    
    query = "transformer attention mechanisms"
    
    print(f"\n[INFO] Running full pipeline for query: {query}")
    
    try:
        retriever = SyllogismRetriever(verbose=True, use_cache=False)
        result = retriever.retrieve_with_syllogism(
            query=query,
            top_k=5,
        )
        
        print(f"\n[RESULT] Pipeline completed:")
        print(f"  - query: {result.query}")
        print(f"  - thesis: {repr(result.thesis[:80])}...")
        print(f"  - through_line: {repr(result.through_line[:80] if result.through_line else '(empty)')}")
        print(f"  - papers count: {len(result.papers)}")
        print(f"  - chain length: {len(result.chain)}")
        print(f"  - graph_context lines: {len(result.graph_context.split(chr(10))) if result.graph_context else 0}")
        
        # Generate markdown
        markdown = result.to_markdown(k=3)
        
        # Check for key sections
        print(f"\n[INFO] Markdown sections present:")
        has_intent = "## Intent" in markdown
        has_thesis = "## Thesis" in markdown
        has_through_line = "## Synthesis" in markdown or "## Through-Line" in markdown
        has_chain = "## Entailment Chain" in markdown
        has_graph = "## Knowledge Graph" in markdown
        
        print(f"  - Intent: {has_intent}")
        print(f"  - Thesis: {has_thesis}")
        print(f"  - Through-Line/Synthesis: {has_through_line}")
        print(f"  - Entailment Chain: {has_chain}")
        print(f"  - Knowledge Graph: {has_graph}")
        
        if has_through_line:
            print(f"\n✅ PASS: Through-Line section present in markdown")
        else:
            print(f"\n❌ FAIL: Through-Line section missing from markdown")
            print(f"    (Check if result.through_line is empty)")
        
        # Save for inspection
        report_path = Path(__file__).parent / "_diagnostic_report.md"
        report_path.write_text(markdown, encoding="utf-8")
        print(f"\n[INFO] Full report saved to: {report_path}")
        
        retriever.close()
        return result
        
    except Exception as e:
        print(f"\n❌ ERROR: Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all diagnostic tests."""
    print("\n" + "🔍 "*40)
    print("THROUGH-LINE SYNTHESIS DIAGNOSTIC")
    print("🔍 "*40)
    
    # Test 1: Thesis generation
    thesis_result = test_stage_4_thesis_generation()
    
    # Test 2: Paper loading
    papers_content = test_stage_6_paper_loading()
    
    # Test 3: Through-line gate condition
    gate_result = test_stage_8_through_line(thesis_result, papers_content)
    
    # Test 4: Full pipeline
    full_result = test_full_pipeline()
    
    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    status = {
        "Stage 4 (Thesis generation)": "✅ PASS" if thesis_result and thesis_result.thesis else "❌ FAIL",
        "Stage 6 (Paper loading)": "✅ PASS" if papers_content else "❌ FAIL",
        "Stage 8 (Gate condition)": "✅ PASS" if gate_result else "❌ FAIL",
        "Full pipeline": "✅ PASS" if full_result and full_result.through_line else "❌ FAIL",
    }
    
    for test_name, result_status in status.items():
        print(f"  {test_name}: {result_status}")
    
    print("\n[NEXT STEPS]")
    if all("PASS" in v for v in status.values()):
        print("  ✅ All tests passed — through-line synthesis is working.")
        print("  Check if retriever was called with verbose=True in production.")
    else:
        failing = [k for k, v in status.items() if "FAIL" in v]
        print(f"  ❌ Tests failing: {', '.join(failing)}")
        print("  Run with verbose output above to identify root cause.")
    
    print("\n")


if __name__ == "__main__":
    main()
