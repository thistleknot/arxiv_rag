#!/usr/bin/env python3
"""Clean up corrupted syllogism_retriever.py file."""

import re
from pathlib import Path

file_path = Path(__file__).parent / 'syllogism_retriever.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Strategy: Find the proper end of to_markdown (first complete return statement with proper balance)
# Remove everything between first complete return and '_coerce_utility'

# Pattern: Find 'def to_markdown' then skip to first proper 'return "\n".join(md)' 
# at the correct indentation level

lines = content.split('\n')
output_lines = []
skip_until_function = False
brace_depth = 0
in_method = False
method_indent = 0

for i, line in enumerate(lines):
    # Detect start of to_markdown
    if 'def to_markdown(self, k: int = 5)' in line:
        in_method = True
        method_indent = len(line) - len(line.lstrip())
        output_lines.append(line)
        continue
    
    # Track when method ends (dedentation)
    if in_method:
        if line.strip() == '':
            output_lines.append(line)
            continue
        
        current_indent = len(line) - len(line.lstrip())
        
        # Check if this is the proper return statement ending the method
        if 'return "\n".join(md)' in line and current_indent == method_indent + 4:
            output_lines.append(line)
            output_lines.append('')  # blank line after method
            in_method = False
            skip_until_function = True
            continue
        
        output_lines.append(line)
        continue
    
    # Skip corrupted content between to_markdown and _coerce_utility
    if skip_until_function:
        if line.strip().startswith('def _coerce_utility'):
            skip_until_function = False
            output_lines.append('')
            output_lines.append('')
            output_lines.append(line)
        continue
    
    output_lines.append(line)

# Write cleaned content
cleaned_content = '\n'.join(output_lines)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(cleaned_content)

print(f"✅ Cleaned {file_path}")
print(f"   Total lines: {len(output_lines)}")
