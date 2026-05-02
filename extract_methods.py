#!/usr/bin/env python3
"""
Phase 5: Extract core methods as pseudocode from a processed paper.

Strips everything from the first References/Bibliography/Appendix heading onward,
then sends the trimmed content to gpt-4.1 via copilot-proxy with a methods-focused
system prompt (methods_sspv2.md).

Preconditions:
    - Markdown file exists (output of Phase 4)
    - methods_sspv2.md exists in the same directory as this script
    - copilot-proxy running at PROXY_HOST:PROXY_PORT
Postconditions:
    - {stem}_methods.md written with pseudocode extraction
Failure modes:
    - Proxy unreachable: exits with error
    - No cutoff heading found: processes full document with warning
    - Empty response: exits with error
"""

import sys
import re
import json
import argparse
import requests
from pathlib import Path

# Windows console may default to cp1252; force UTF-8 for streaming output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROXY_HOST = "localhost"
PROXY_PORT = 8069
MODEL = "gpt-4.1"
SYSTEM_PROMPT_FILE = Path(__file__).parent / "methods_sspv2.md"

# Headings that mark end-of-content; first match wins, case-insensitive
_CUTOFF_RE = re.compile(
    r"^#{1,3}\s*(references|bibliography|works cited|appendix|supplementary material)",
    re.IGNORECASE | re.MULTILINE,
)


def strip_post_references(content: str) -> tuple[str, bool]:
    """
    Return (body_text, found_cutoff).

    Require: content is a non-empty markdown string.
    Guarantee: body_text contains no bibliography or appendix entries.
               If no cutoff heading is found, returns (content, False).
    """
    m = _CUTOFF_RE.search(content)
    if m:
        return content[: m.start()].rstrip(), True
    return content, False


def load_system_prompt() -> str:
    """
    Require: SYSTEM_PROMPT_FILE exists on disk.
    Failure mode: FileNotFoundError propagates to caller.
    """
    if not SYSTEM_PROMPT_FILE.is_file():
        raise FileNotFoundError(f"System prompt not found: {SYSTEM_PROMPT_FILE}")
    return SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")


def call_proxy(paper_text: str, system_prompt: str, model: str, host: str, port: int) -> str:
    """
    Stream a chat completion from copilot-proxy and return the assembled text.

    Require: proxy reachable at host:port, paper_text and system_prompt non-empty.
    Guarantee: returns full response text; empty string signals total failure.
    Failure mode: HTTP non-2xx raises requests.HTTPError; SSE parse errors skipped.
    """
    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Authorization": "Bearer dummy", "Content-Type": "application/json"}
    user_content = (
        "Extract the paper's core methods as pseudocode.\n\n"
        "Rules:\n"
        "- Use only evidence from the paper.\n"
        "- Mark inferred steps inline with [inferred].\n"
        "- Omit references, results, and discussion unless needed to reconstruct the method.\n"
        "- Do not summarize the paper. Extract only what is needed to reimplement.\n\n"
        "<<<BEGIN PAPER>>>\n"
        f"{paper_text}\n"
        "<<<END PAPER>>>"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": True,
    }

    parts: list[str] = []
    with requests.post(url, json=payload, headers=headers, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        buf = ""
        done = False
        for raw_bytes in resp.iter_content(chunk_size=None):
            if done:
                break
            buf += raw_bytes.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    done = True
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content")
                    if text:
                        parts.append(text)
                        print(text, end="", flush=True)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    print()  # newline after streaming output
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract core methods as pseudocode from a processed markdown paper"
    )
    parser.add_argument("md_file", help="Path to the processed markdown file")
    parser.add_argument("--model", default=MODEL, help="Model name")
    parser.add_argument("--host", default=PROXY_HOST, help="Proxy host")
    parser.add_argument("--port", type=int, default=PROXY_PORT, help="Proxy port")
    args = parser.parse_args()

    md_path = Path(args.md_file)
    if not md_path.is_file():
        print(f"Error: markdown file not found: {md_path}")
        return 1

    try:
        system_prompt = load_system_prompt()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    content = md_path.read_text(encoding="utf-8")
    paper_text, found_cutoff = strip_post_references(content)
    if not found_cutoff:
        print("Warning: no References/Bibliography/Appendix heading found — processing full document.")
    else:
        orig_lines = content.count("\n")
        kept_lines = paper_text.count("\n")
        print(f"  Stripped to pre-references body: {kept_lines}/{orig_lines} lines kept.")

    out_path = md_path.with_name(md_path.stem + "_methods.md")
    print(f"Extracting methods from: {md_path.name}")
    print(f"Output: {out_path.name}")
    print(f"Model: {args.model} via {args.host}:{args.port}")
    print()

    try:
        result = call_proxy(paper_text, system_prompt, args.model, args.host, args.port)
    except Exception as exc:
        print(f"\nError calling proxy: {exc}")
        return 1

    if not result.strip():
        print("Error: proxy returned empty response.")
        return 1

    out_path.write_text(result, encoding="utf-8")
    print(f"\nWritten: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
