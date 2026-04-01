#!/usr/bin/env python
"""
autoharness_arxiv_extractor.py
═══════════════════════════════════════════════════════════════════════════════
ReAct autonomous agent harness that drives arxiv_gap_extractor.py to completion,
diagnosing and self-correcting failures using:

    gpt-4o  → REASON, OBSERVE, REFLECT  (diagnosis, planning)
    gpt-4.1 → ACT                        (code edits, patches)

LLM backend: copilot-proxy at http://127.0.0.1:8069/v1

ReAct state machine per iteration:
    REASON   → decide what to run / how to fix
    VALIDATE → confirm extractor exists and state is sane
    ACT      → run extractor or apply fix
    OBSERVE  → read stdout/stderr/CSV/state
    REFLECT  → success → next batch; fail → diagnose, fix, retry

Memory files (append-only):
    .react_agent/changes.jsonl   — action log (what was run / changed)
    .react_agent/memory.jsonl    — outcome log (success/fail + diagnosis)
    .react_agent/progress.md     — human-readable current status
    .react_agent/corrections.md  — behavioral mistake log (read at Phase 0 startup)
    .react_agent/evidence.md     — evidence gate artifact (written before completion)
    .react_agent/fix_{i}.py      — backup of extractor before each patch
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# ─── Pipeline constants ────────────────────────────────────────────────────────
PYTHON      = r"c:\users\user\py310\scripts\python.exe"
EXTRACTOR   = Path(r"C:\Users\user\arxiv_id_lists\arxiv_gap_extractor.py")
STATE_FILE  = Path(r"C:\Users\user\arxiv_id_lists\_extractor_state.json")
CSV_PATH    = Path(r"C:\Users\user\arxiv_id_lists\papers\post_processed\arxiv_data_with_analysis.csv")
WORK_DIR    = Path(r"C:\Users\user\arxiv_id_lists\.react_agent")
WORK_DIR.mkdir(exist_ok=True)

CHANGES_LOG    = WORK_DIR / "changes.jsonl"
MEMORY_LOG     = WORK_DIR / "memory.jsonl"
PROGRESS_MD    = WORK_DIR / "progress.md"
CORRECTIONS_MD = WORK_DIR / "corrections.md"
EVIDENCE_MD    = WORK_DIR / "evidence.md"

PROXY_BASE  = "http://127.0.0.1:8069/v1"
PROXY_KEY   = "dummy-key"

BATCH_SIZE    = 20   # papers per run_extractor call
MAX_RETRIES   = 5    # consecutive failures before BLOCKED
TOTAL_GAP     = 283  # gap papers remaining (calculated from prior recon)

# ─── Troubleshooting levels (SKILL.md) ─────────────────────────────────────────
TROUBLESHOOTING = {
    1: ("REREAD",   "Re-read error output and memory patterns; retry with same approach"),
    2: ("REREAD",   "Re-read more broadly — check recent changes.jsonl for side effects"),
    3: ("ISOLATE",  "Isolate the failure — run minimal reproduction, check a single paper"),
    4: ("PIVOT",    "Entirely different strategy — the current approach is not working"),
    5: ("PIVOT",    "Final pivot attempt — last chance before escalation"),
}

# ─── LLM clients ──────────────────────────────────────────────────────────────
_reasoning = OpenAI(api_key=PROXY_KEY, base_url=PROXY_BASE)
_coding    = OpenAI(api_key=PROXY_KEY, base_url=PROXY_BASE)


def think(user_prompt: str, system: str = "", model: str = "gpt-4o") -> str:
    """REASON / OBSERVE / REFLECT step — gpt-4o."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_prompt})
    resp = _reasoning.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def code_edit(bug_description: str, file_content: str) -> str:
    """ACT step — gpt-4.1 returns COMPLETE fixed Python file (no stubs, no fences)."""
    system = (
        "You are a precise Python code editor. "
        "The user describes a bug. You return the COMPLETE fixed Python file — "
        "no markdown fences, no commentary, just raw Python source from line 1 to EOF. "
        "Do not truncate. Do not add stubs. Return the entire working file."
    )
    user = (
        f"Bug to fix:\n{bug_description}\n\n"
        f"Current file:\n```python\n{file_content}\n```"
    )
    resp = _coding.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.1,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip accidental markdown fences
    raw = re.sub(r'^```\w*\n?', '', raw)
    raw = re.sub(r'\n?```\s*$', '', raw.rstrip())
    return raw


# ─── Memory / logging helpers ──────────────────────────────────────────────────
def _append_jsonl(path: Path, entry: dict) -> None:
    entry["ts"] = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def log_memory(entry: dict) -> None:
    _append_jsonl(MEMORY_LOG, entry)


def log_change(entry: dict) -> None:
    _append_jsonl(CHANGES_LOG, entry)


def recent_failures(n: int = 3) -> list[dict]:
    """Return last n failure entries from memory log."""
    if not MEMORY_LOG.exists():
        return []
    lines = [l.strip() for l in MEMORY_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    entries = [json.loads(l) for l in lines]
    return [e for e in entries if e.get("outcome") == "fail"][-n:]


# ─── Progress helpers ──────────────────────────────────────────────────────────
def processed_count() -> int:
    """Number of papers in the state JSON's 'processed' list."""
    if not STATE_FILE.exists():
        return 0
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8-sig"))
        return len(data.get("processed", []))
    except Exception:
        return 0


def csv_row_count() -> int:
    """Current data rows in the output CSV (excludes header)."""
    if not CSV_PATH.exists():
        return 0
    try:
        with CSV_PATH.open(encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in csv.reader(f)) - 1
    except Exception:
        return -1


def write_progress(done: int, total: int, status: str, notes: str = "") -> None:
    pct = (done / total * 100) if total > 0 else 0
    bar_filled = int(pct / 5)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    PROGRESS_MD.write_text(
        f"# ReAct Harness — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"**Task**: Extract utility/barriers/thesis for {total} gap papers into CSV\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Status | `{status}` |\n"
        f"| Progress | {bar} {pct:.0f}% |\n"
        f"| State-file | {done} / {total} |\n"
        f"| CSV rows | {csv_row_count()} |\n\n"
        f"## Notes\n{notes}\n",
        encoding="utf-8",
    )


# ─── Worker: run extractor subprocess ─────────────────────────────────────────
def run_extractor(limit: int | None = None) -> tuple[int, str, str]:
    """
    Invoke arxiv_gap_extractor.py as subprocess.
    Returns (returncode, stdout, stderr).
    """
    cmd = [PYTHON, str(EXTRACTOR)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(EXTRACTOR.parent),
        timeout=900,   # 15 min — a batch of 20 papers can take 5+ min with backoffs
    )
    return result.returncode, result.stdout, result.stderr


# ─── Enhanced reasoning helpers (SKILL.md / HARNESS.md / QUALITY.md) ───────────

def scan_memory_patterns() -> str:
    """Scan memory.jsonl for recurring failure patterns (anti-tunnel-vision)."""
    if not MEMORY_LOG.exists():
        return "No prior memory entries."
    lines = [l.strip() for l in MEMORY_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    entries = [json.loads(l) for l in lines]

    failures = [e for e in entries if e.get("outcome") == "fail"]
    successes = [e for e in entries if e.get("outcome") == "success"]

    # Detect repeated error patterns
    error_previews = [e.get("error_preview", "")[:80] for e in failures if e.get("error_preview")]
    error_freq = Counter(error_previews).most_common(3)

    summary_parts = [
        f"Total entries: {len(entries)} ({len(successes)} success, {len(failures)} fail)",
    ]
    if error_freq:
        summary_parts.append("Recurring errors:")
        for preview, count in error_freq:
            summary_parts.append(f"  [{count}x] {preview}")

    # Check for tunnel vision — same fix_key applied multiple times
    fix_keys = [e.get("fix_key", "") for e in failures if e.get("fix_key")]
    fix_freq = Counter(fix_keys).most_common(3)
    repeated = [(k, c) for k, c in fix_freq if c >= 2]
    if repeated:
        summary_parts.append("⚠ TUNNEL VISION — repeated diagnoses:")
        for key, count in repeated:
            summary_parts.append(f"  [{count}x] {key[:60]}")

    return "\n".join(summary_parts)


def society_of_thought(error_context: str, extractor_head: str, recent_ctx: str) -> str:
    """
    Society of Thought diagnosis — three perspectives before deciding on a fix.
    Ref: SKILL.md § Society of Thought.
    """
    sot_prompt = (
        f"=== ERROR CONTEXT ===\n{error_context}\n\n"
        f"=== RECENT FAILURES ===\n{recent_ctx}\n\n"
        f"=== EXTRACTOR (first 3000 chars) ===\n{extractor_head}\n\n"
        "Analyze this failure from THREE perspectives:\n\n"
        "**PROPOSER**: What is the most likely root cause? Name the broken function "
        "and propose a specific fix.\n\n"
        "**SKEPTIC**: Challenge the Proposer's diagnosis. What alternative causes "
        "could produce this error? What could go wrong with the proposed fix? "
        "Is this a symptom of a deeper issue?\n\n"
        "**VERIFIER**: Given both views, what is the safest, most targeted fix? "
        "How do we verify it worked? What regression risk exists?\n\n"
        "Conclude with:\n"
        "1. Root cause (one sentence)\n"
        "2. Broken function name\n"
        "3. Exact fix — what to change and how\n"
        "4. Verification: how to confirm the fix worked"
    )
    return think(
        user_prompt=sot_prompt,
        system="You are a senior Python debugging expert using Society of Thought analysis. "
               "Be terse and precise.",
    )


def adversarial_preact_check(action: str, batch: int, done: int) -> tuple[bool, str]:
    """
    Adversarial Pre-ACT Check (HARNESS.md) — three questions before executing.
    Returns (proceed: bool, reason: str).
    """
    # Skeptic: Could this action cause harm?
    if not EXTRACTOR.exists():
        return False, "SKEPTIC: Extractor file missing — cannot proceed"

    # Scope Guard: Is this action within our plan?
    if action not in ("run_extractor", "apply_fix"):
        return False, f"SCOPE GUARD: Unexpected action '{action}' — not in plan"

    # Verifier: Are preconditions met?
    if action == "run_extractor":
        if done > 0 and not STATE_FILE.exists():
            return False, "VERIFIER: State file missing after prior batches — data loss risk"
        if batch <= 0:
            return False, f"VERIFIER: Invalid batch size {batch}"

    # Environmental check: working directory exists
    if not EXTRACTOR.parent.exists():
        return False, "ENVIRONMENTAL: Working directory missing"

    return True, "All pre-ACT checks passed"


def _write_corrections_entry(mistake: str, correction: str) -> None:
    """Append a behavioral mistake entry to corrections.md (MEMORY.md spec)."""
    entry = (
        f"\n### {datetime.now().strftime('%Y-%m-%d %H:%M')}: {mistake}\n"
        f"- **Mistake:** {mistake}\n"
        f"- **Correction:** {correction}\n"
    )
    with CORRECTIONS_MD.open("a", encoding="utf-8") as fh:
        fh.write(entry)


def write_evidence(done: int, total: int, resolution: str) -> None:
    """Write evidence gate artifact before declaring completion (SKILL.md evidence gate)."""
    csv_rows = csv_row_count()
    EVIDENCE_MD.write_text(
        f"# Evidence Gate\n\n"
        f"**Problem statement:** Drive arxiv_gap_extractor.py to process all {total} gap papers.\n\n"
        f"**Root cause (if failures occurred):** See memory.jsonl for per-failure diagnoses.\n\n"
        f"**Resolution:** {resolution}\n\n"
        f"**Validation:**\n"
        f"- Papers processed: {done}/{total}\n"
        f"- CSV rows: {csv_rows}\n"
        f"- Extractor intact: {EXTRACTOR.exists()}\n"
        f"- State file intact: {STATE_FILE.exists()}\n\n"
        f"**Completion conditions met:**\n"
        f"- [{'x' if done >= total else ' '}] All {total} papers processed\n"
        f"- [{'x' if csv_rows >= done else ' '}] CSV has >= {done} rows\n"
        f"- [{'x' if EXTRACTOR.exists() else ' '}] Extractor executable present\n\n"
        f"**Residual uncertainty:** None if all three conditions above are checked.\n",
        encoding="utf-8",
    )


def completion_quality_check(done: int, total: int) -> str:
    """
    Six-dimension quality check at task completion (QUALITY.md).
    Returns assessment string.
    """
    checks = []

    # 1. Safety — did we corrupt any data?
    csv_rows = csv_row_count()
    safety = "Good" if csv_rows >= done else "Poor"
    checks.append(f"Safety:          {safety} — CSV has {csv_rows} rows for {done} processed papers")

    # 2. Completeness — did we finish?
    completeness = "Good" if done >= total else ("Adequate" if done >= total * 0.9 else "Poor")
    checks.append(f"Completeness:    {completeness} — {done}/{total} papers processed")

    # 3. Executability — can we re-run cleanly?
    extractor_ok = EXTRACTOR.exists()
    state_ok = STATE_FILE.exists()
    executability = "Good" if extractor_ok and state_ok else "Poor"
    checks.append(f"Executability:   {executability} — extractor={'OK' if extractor_ok else 'MISSING'}, "
                  f"state={'OK' if state_ok else 'MISSING'}")

    # 4. Maintainability — are backups and logs intact?
    backups = list(WORK_DIR.glob("fix_*.py"))
    maintainability = "Good" if MEMORY_LOG.exists() and CHANGES_LOG.exists() else "Adequate"
    checks.append(f"Maintainability: {maintainability} — {len(backups)} backups, "
                  f"memory.jsonl={'OK' if MEMORY_LOG.exists() else 'MISSING'}")

    # 5. Cost-awareness — how many iterations did it take?
    changes_count = 0
    if CHANGES_LOG.exists():
        changes_count = sum(1 for l in CHANGES_LOG.read_text(encoding="utf-8").splitlines() if l.strip())
    cost = "Good" if changes_count <= total / BATCH_SIZE * 1.5 else "Adequate"
    checks.append(f"Cost-Awareness:  {cost} — {changes_count} actions logged")

    # 6. Acceptance Alignment — local checks map onto all completion_conditions (QUALITY.md)
    all_conditions_met = (done >= total) and (csv_rows >= done) and extractor_ok and state_ok
    acceptance = "Good" if all_conditions_met else ("Adequate" if done >= total * 0.9 else "Poor")
    checks.append(
        f"Acceptance:      {acceptance} — "
        f"completion_conditions {'all satisfied' if all_conditions_met else 'NOT all satisfied'}"
    )

    # Hard blocks
    if safety == "Poor" or executability == "Poor" or acceptance == "Poor":
        checks.append("\n⚠ HARD BLOCK: Safety, Executability, or Acceptance failed — review before declaring complete")

    return "\n".join(checks)


def get_troubleshooting_level(consecutive_errors: int) -> tuple[int, str, str]:
    """Map consecutive error count to troubleshooting level (SKILL.md)."""
    level, name, desc = 1, "REREAD", "Re-read error and retry"
    for threshold, (n, d) in TROUBLESHOOTING.items():
        if consecutive_errors >= threshold:
            level, name, desc = threshold, n, d
    return level, name, desc


# ─── Main ReAct loop ───────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 65)
    print(" ReAct Autoharness — arxiv gap extractor")
    print("=" * 65)

    # Guard: extractor must exist
    if not EXTRACTOR.exists():
        print(f"[FATAL] Extractor not found: {EXTRACTOR}")
        sys.exit(1)

    # Phase 0: read behavioral corrections before acting (MEMORY.md — always read at Phase 0)
    prior_corrections = ""
    if CORRECTIONS_MD.exists():
        prior_corrections = CORRECTIONS_MD.read_text(encoding="utf-8").strip()
        if prior_corrections:
            print(f"[STARTUP] Prior corrections found in {CORRECTIONS_MD.name} — injecting into diagnosis context")

    consecutive_errors = 0
    applied_fix_keys: list[str] = []

    write_progress(processed_count(), TOTAL_GAP, "STARTING")

    for iteration in range(200):   # safety cap — 200 batches of 20 = up to 4,000 papers

        # ── CHECK COMPLETION ───────────────────────────────────────────────────
        done = processed_count()
        if done >= TOTAL_GAP:
            # Evidence gate — write before declaring complete (SKILL.md)
            write_evidence(done, TOTAL_GAP, f"All {done} papers processed across {iteration} iterations.")
            quality = completion_quality_check(done, TOTAL_GAP)
            print(f"\n[QUALITY CHECK]\n{quality}")
            write_progress(done, TOTAL_GAP, "COMPLETE")
            log_memory({"outcome": "success", "summary": f"All {done} papers complete",
                        "evidence": str(EVIDENCE_MD)})
            print(f"\n✅  All {TOTAL_GAP} papers processed. Evidence → {EVIDENCE_MD}")
            break

        # ── REASON ────────────────────────────────────────────────────────────
        remaining = TOTAL_GAP - done
        batch     = min(BATCH_SIZE, remaining)
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] iter={iteration + 1}  "
              f"processed={done}/{TOTAL_GAP}  batch={batch}")

        # ── VALIDATE ──────────────────────────────────────────────────────────
        assert EXTRACTOR.exists(), "Extractor deleted mid-run!"
        assert STATE_FILE.exists() or done == 0, "State file missing after first batch!"

        # ── ACT: run the extractor for one batch ──────────────────────────────
        log_change({"action": "run_extractor", "limit": batch, "iter": iteration})
        try:
            rc, stdout, stderr = run_extractor(limit=batch)
        except subprocess.TimeoutExpired:
            stderr = "TimeoutExpired after 900s"
            stdout = ""
            rc = 1
        except Exception as exc:
            stderr = str(exc)
            stdout = ""
            rc = 1

        # ── OBSERVE ───────────────────────────────────────────────────────────
        combined = (stdout + "\n" + stderr).strip()
        tail = combined[-2000:] if len(combined) > 2000 else combined
        print(tail)

        # ── REFLECT ───────────────────────────────────────────────────────────
        success = (rc == 0) and ("Appended" in stdout or "papers" in stdout.lower())
        if success:
            newly_done = processed_count()
            log_memory({"outcome": "success", "batch": batch, "total_done": newly_done})
            write_progress(newly_done, TOTAL_GAP, "IN_PROGRESS",
                           f"Last batch: +{newly_done - done} papers at {datetime.now().strftime('%H:%M')}")
            consecutive_errors = 0
            time.sleep(1)   # brief breath between batches
            continue

        # — Failure path ────────────────────────────────────────────────────────
        consecutive_errors += 1
        print(f"\n[REFLECT] Failure #{consecutive_errors} (rc={rc}). Diagnosing...")

        if consecutive_errors > MAX_RETRIES:
            msg = f"INCOMPLETE — halted after {MAX_RETRIES} consecutive failures. Human review needed."
            write_progress(done, TOTAL_GAP, "INCOMPLETE",
                           f"{msg}\n\nLast error tail:\n```\n{combined[-600:]}\n```")
            log_memory({"outcome": "incomplete", "reason": msg, "last_error": combined[-600:]})
            print(f"\n❌  {msg}")
            break

        # ── DIAGNOSE via gpt-4o ───────────────────────────────────────────────
        recent_ctx = json.dumps(recent_failures(), indent=2)
        extractor_head = EXTRACTOR.read_text(encoding="utf-8")[:3000]

        corrections_section = (
            f"=== prior behavioral corrections ===\n{prior_corrections}\n\n"
            if prior_corrections else ""
        )
        diagnosis = think(
            user_prompt=(
                f"The arxiv_gap_extractor.py script failed (exit {rc}).\n\n"
                f"=== stdout + stderr (last 3000 chars) ===\n{combined[-3000:]}\n\n"
                f"=== recent failures ===\n{recent_ctx}\n\n"
                f"=== extractor (first 3000 chars) ===\n{extractor_head}\n\n"
                f"{corrections_section}"
                "Respond with:\n"
                "1. Root cause (one sentence)\n"
                "2. Broken function name\n"
                "3. Exact fix — what to change and how\n"
                "Be specific. Do not suggest rewriting the whole file unless absolutely necessary."
            ),
            system="You are a senior Python debugging expert. Be terse and precise.",
        )
        print(f"\n[DIAGNOSIS]\n{diagnosis}")

        # Detect repeated diagnosis
        fix_key = diagnosis[:100].strip()
        if fix_key in applied_fix_keys:
            print("[WARN] Repeated diagnosis — forcing alternative approach")
            _write_corrections_entry(
                mistake=f"Repeated diagnosis at iter {iteration}: '{fix_key[:80]}'",
                correction="Change strategy — do not retry the same approach; pivot to an alternative fix.",
            )
            diagnosis = (
                "IMPORTANT: The previous fix attempt failed. Try an entirely different strategy.\n\n"
                + diagnosis
            )

        # ── FIX via gpt-4.1 ──────────────────────────────────────────────────
        current_code = EXTRACTOR.read_text(encoding="utf-8")
        fixed_code = code_edit(
            bug_description=f"{diagnosis}\n\nError output:\n{combined[-2000:]}",
            file_content=current_code,
        )

        if len(fixed_code) < 300:
            # Code edit returned garbage — skip this cycle
            log_memory({"outcome": "fail", "reason": "code_edit < 300 chars", "diagnosis": diagnosis})
            print("[WARN] code_edit returned too short — skipping fix")
            continue

        # ── APPLY FIX ─────────────────────────────────────────────────────────
        # Backup first
        backup = WORK_DIR / f"fix_{iteration:03d}.py"
        backup.write_text(current_code, encoding="utf-8")
        EXTRACTOR.write_text(fixed_code, encoding="utf-8")
        applied_fix_keys.append(fix_key)

        log_change({
            "action": "apply_fix",
            "iter": iteration,
            "backup": str(backup),
            "diagnosis_preview": fix_key,
        })
        log_memory({
            "outcome": "fail",
            "error_preview": combined[-400:],
            "diagnosis": diagnosis,
            "fix_key": fix_key,
        })
        print(f"[FIX] Applied. Backup → {backup.name}. Retrying...")

    else:
        # Exhausted 200 iterations
        write_progress(processed_count(), TOTAL_GAP, "ITER_CAP_REACHED",
                       "Reached 200 iteration cap. Resume by re-running the harness.")
        print("\n[HARNESS] 200-iteration cap reached. Check progress.md")


if __name__ == "__main__":
    main()
