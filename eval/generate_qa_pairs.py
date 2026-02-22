"""
QA Pair Generator for RAGAS Evaluation

Core Thesis:
    Sample random chunks from arxiv_chunks, fetch their full sections,
    use an LLM to derive question+answer pairs that are answerable from
    the section text. Outputs train/test splits for the RAGAS evaluation
    pipeline.

Workflow:
    1. Sample N_SAMPLE random (paper_id, section_idx) pairs from DB
    2. Fetch all chunk content per section → join into section text
    3. LLM generates question + answer from section text
    4. Filter: skip trivially short answers or yes/no questions
    5. Save eval/data/qa_pairs.json  { train: [...], test: [...] }

Output schema per pair:
    {
        "question":      str,      # answerable from section
        "answer":        str,      # gold answer derived from section
        "section_text":  str,      # concatenated section content (ground context)
        "paper_id":      str,
        "section_idx":   int,
        "chunk_ids":     [str]     # chunk_ids in this section
    }
"""

import json
import os
import random
import re
import sys

import psycopg2
import openai

# ── Config ────────────────────────────────────────────────────────────────────
DB = dict(host="localhost", port=5432, dbname="langchain",
          user="langchain", password="langchain")

COPILOT_PROXY = "http://127.0.0.1:8069/v1"
GEN_MODEL     = "gpt-4.1"   # via GitHub Copilot proxy
N_SAMPLE     = 200  # candidate sections to query
N_TARGET     = 150  # QA pairs to actually generate (after LLM filter)
TRAIN_FRAC   = 0.70
SEED         = 42

OUT_DIR  = os.path.join(os.path.dirname(__file__), "data")
OUT_FILE = os.path.join(OUT_DIR, "qa_pairs.json")

QA_PROMPT = """\
Below is a passage from an academic paper. Read it carefully.

PASSAGE:
{section_text}

Your task:
1. Write ONE specific question that is clearly and fully answerable from the passage above.
2. Write the answer to that question, based only on the passage.

Rules:
- The question must NOT be a yes/no question.
- The question must NOT ask for the paper title or authors.
- The answer must be at least 2 sentences drawn from the passage.
- Do not invent information outside the passage.

Respond in this exact format (nothing else):
QUESTION: <your question here>
ANSWER: <your answer here (2+ sentences)>
"""


# ── DB helpers ─────────────────────────────────────────────────────────────────
def fetch_sections(conn, n_sample: int) -> list[dict]:
    """
    Sample random (paper_id, section_idx) pairs, fetch all chunks per section.
    Returns list of {paper_id, section_idx, section_text, chunk_ids}.
    """
    with conn.cursor() as cur:
        # Sample distinct sections (not individual chunks) for diversity
        # Wrap in a subquery so ORDER BY random() is applied to the distinct set
        cur.execute("""
            SELECT paper_id, section_idx
            FROM (
                SELECT DISTINCT paper_id, section_idx
                FROM arxiv_chunks
            ) AS distinct_sections
            ORDER BY random()
            LIMIT %s
        """, (n_sample,))
        sections = cur.fetchall()

    results = []
    with conn.cursor() as cur:
        for paper_id, section_idx in sections:
            cur.execute("""
                SELECT chunk_id, chunk_idx, content
                FROM arxiv_chunks
                WHERE paper_id = %s AND section_idx = %s
                ORDER BY chunk_idx
            """, (paper_id, section_idx))
            rows = cur.fetchall()
            if not rows:
                continue
            chunk_ids = [r[0] for r in rows]
            section_text = "\n\n".join(r[2] for r in rows if r[2])
            if len(section_text.split()) < 80:
                continue  # too short for a meaningful question
            results.append({
                "paper_id": paper_id,
                "section_idx": section_idx,
                "section_text": section_text,
                "chunk_ids": chunk_ids,
            })
    return results


# ── LLM call ──────────────────────────────────────────────────────────────────
def generate_qa(section_text: str) -> tuple[str, str] | None:
    """
    Call gpt-4.1 via Copilot proxy to generate (question, answer) from section_text.
    Returns (question, answer) or None on failure / invalid output.
    """
    client = openai.OpenAI(api_key="dummy-key", base_url=COPILOT_PROXY)
    prompt = QA_PROMPT.format(section_text=section_text[:3000])
    try:
        response = client.chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        )
        text = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [LLM error] {e}")
        return None

    # Parse QUESTION: / ANSWER:
    q_match = re.search(r"QUESTION:\s*(.+?)(?=\nANSWER:|$)", text, re.DOTALL)
    a_match = re.search(r"ANSWER:\s*(.+)", text, re.DOTALL)
    if not q_match or not a_match:
        return None

    question = q_match.group(1).strip()
    answer   = a_match.group(1).strip()

    # Quality filters
    if len(answer.split()) < 15:
        return None
    if question.lower().startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "has ", "have ")):
        return None
    if len(question) < 20:
        return None

    return question, answer


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(SEED)

    print("=" * 60)
    print("QA Pair Generator")
    print("=" * 60)

    print(f"\n[1/3] Fetching {N_SAMPLE} candidate sections from DB...")
    conn = psycopg2.connect(**DB)
    sections = fetch_sections(conn, N_SAMPLE)
    conn.close()
    print(f"  ✓ {len(sections)} sections after length filter")

    random.shuffle(sections)
    sections = sections[:N_TARGET + 30]  # overshoot to account for LLM failures

    print(f"\n[2/3] Generating QA pairs (target={N_TARGET}) with {GEN_MODEL}...")
    pairs = []
    for i, sec in enumerate(sections):
        if len(pairs) >= N_TARGET:
            break
        result = generate_qa(sec["section_text"])
        status = "✓" if result else "✗"
        if result:
            q, a = result
            pairs.append({
                "question":     q,
                "answer":       a,
                "section_text": sec["section_text"],
                "paper_id":     sec["paper_id"],
                "section_idx":  sec["section_idx"],
                "chunk_ids":    sec["chunk_ids"],
            })
        print(f"  [{i+1:3d}/{min(len(sections), N_TARGET+30)}] {status}  "
              f"({len(pairs)} collected)  paper={sec['paper_id'][:20]}", flush=True)

    print(f"\n  Generated {len(pairs)} valid QA pairs")

    print(f"\n[3/3] Splitting and saving...")
    random.shuffle(pairs)
    n_train = int(len(pairs) * TRAIN_FRAC)
    output = {
        "meta": {
            "n_total": len(pairs),
            "n_train": n_train,
            "n_test": len(pairs) - n_train,
            "gen_model": GEN_MODEL,
            "seed": SEED,
        },
        "train": pairs[:n_train],
        "test":  pairs[n_train:],
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Saved {OUT_FILE}")
    print(f"    train: {n_train}  test: {len(pairs)-n_train}")


if __name__ == "__main__":
    main()
