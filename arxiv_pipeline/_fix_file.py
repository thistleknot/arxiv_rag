#!/usr/bin/env python3
"""Reconstruct the corrupted syllogism_retriever.py file."""

import re
from pathlib import Path

file_path = Path(__file__).parent / 'syllogism_retriever.py'

# Read the file
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Step 1: Find where to_markdown method properly ends
# It ends at the first 'return "\n".join(md)' at indent level 8 spaces (method body)
to_md_end_idx = None
method_started = False

for i, line in enumerate(lines):
    if 'def to_markdown(self, k: int = 5)' in line:
        method_started = True
        continue
    
    if method_started:
        # Look for the proper return statement
        if 'return "\n".join(md)' in line and line.lstrip().startswith('return'):
            indent = len(line) - len(line.lstrip())
            # Should be at 8 spaces (method body indentation)
            if indent in (8, 12):  # account for possible variations
                to_md_end_idx = i + 1
                break

print(f"Found to_markdown end at line {to_md_end_idx}")

# Step 2: Find where _coerce_utility starts (properly at module level, not as class method)
coerce_start_idx = None
for i in range(to_md_end_idx or 0, len(lines)):
    line = lines[i]
    # Look for def _coerce_utility at column 0 or after blank lines
    if line.strip().startswith('def _coerce_utility') and not line.lstrip().startswith('def'):
        coerce_start_idx = i
        break

print(f"Found _coerce_utility start at line {coerce_start_idx}")

# Step 3: Find where SyllogismRetriever class starts
class_start_idx = None
for i in range(coerce_start_idx or len(lines) - 50, len(lines)):
    if lines[i].strip().startswith('class SyllogismRetriever'):
        class_start_idx = i
        break

print(f"Found SyllogismRetriever class start at line {class_start_idx}")

# Step 4: Reconstruct the file
cleaned_lines = []

# Keep everything up to and including to_markdown's return statement
if to_md_end_idx:
    cleaned_lines.extend(lines[:to_md_end_idx])
    cleaned_lines.append('\n')
    cleaned_lines.append('\n')

# Add a clean _coerce_utility function
if coerce_start_idx:
    utl_func = '''def _coerce_utility(v: str) -> str:
    """Convert CSV utility value to a plain text string.

    The CSV stores utilities as stringified JSON lists, e.g.:
        "[\\\"Point one.\\\", \\\"Point two.\\\"]"

    This function parses the list and joins items with ". ".
    Falls back to the raw string if parsing fails.
    """
    if not v:
        return ""
    v = v.strip()
    if v.startswith("["):
        try:
            items = ast.literal_eval(v)
            if isinstance(items, list):
                return ". ".join(str(x).strip().rstrip(".") for x in items if x) + "."
        except (ValueError, SyntaxError):
            pass
    return v


'''
    cleaned_lines.append(utl_func)

# Keep everything from class onwards
if class_start_idx:
    cleaned_lines.extend(lines[class_start_idx:])

# Write the clean file
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(cleaned_lines)

print(f"✅ File rebuilt. Total lines: {len(cleaned_lines)}")
