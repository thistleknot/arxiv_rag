#!/usr/bin/env python3
"""
Post-process Markdown files by extracting base64 image data and creating CSV reference files.

This script:
1. Reads a Markdown file
2. Finds all base64 image data
3. Replaces each image with a reference tag: <image_NNN> (sequential, zero-padded)
4. Creates a CSV file with image_index, line_number, and base64_data columns
"""

import os
import re
import csv
import argparse
from pathlib import Path


def extract_base64_images(markdown_file_path):
    """
    Extract base64 image data from a Markdown file and replace with sequential reference tags.

    Each image gets a 1-based sequential index, zero-padded to 3 digits: <image_001>.
    The CSV maps image_index -> (line_number, base64_data).

    Require: markdown_file_path is a readable .md file.
    Guarantee: returns (modified_lines, image_data_dict) where image_data_dict maps
               int image_index to (int line_number, str base64_data).
    """
    with open(markdown_file_path, 'r', encoding='utf-8') as file:
        content = file.readlines()

    base64_pattern = re.compile(r'!\[.*?\]\(data:image\/[^;]+;base64,([a-zA-Z0-9+/=]+)\)')
    image_data_dict: dict[int, tuple[int, str]] = {}
    modified_content: list[str] = []
    idx = 0

    for line_num, line in enumerate(content, 1):
        matches = list(base64_pattern.finditer(line))
        if not matches:
            modified_content.append(line)
            continue

        new_line = line
        for match in matches:
            idx += 1
            base64_data = match.group(1)
            image_data_dict[idx] = (line_num, base64_data)
            new_line = new_line.replace(match.group(0), f"<image_{idx:03d}>")

        modified_content.append(new_line)

    return modified_content, image_data_dict


def write_modified_markdown(markdown_file_path, modified_content):
    """
    Write the modified content back to the Markdown file.

    Require: markdown_file_path is writable.
    Guarantee: file contains modified_content exactly.
    """
    with open(markdown_file_path, 'w', encoding='utf-8') as file:
        file.writelines(modified_content)


def create_csv_reference(csv_file_path, image_data_dict):
    """
    Create a CSV file with image_index, line_number, and base64_data columns.

    Require: image_data_dict maps int image_index to (int line_number, str base64_data).
    Guarantee: CSV rows are sorted by image_index ascending.
    """
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['image_index', 'line_number', 'base64_data'])
        for idx, (line_num, base64_data) in sorted(image_data_dict.items()):
            csv_writer.writerow([idx, line_num, base64_data])


def main():
    parser = argparse.ArgumentParser(description='Extract base64 images from Markdown files')
    parser.add_argument('markdown_file', help='Path to the Markdown file')
    args = parser.parse_args()
    
    markdown_file_path = args.markdown_file
    
    # Validate file existence
    if not os.path.isfile(markdown_file_path):
        print(f"Error: File {markdown_file_path} not found.")
        return 1
    
    # Validate file is a Markdown file
    if not markdown_file_path.lower().endswith('.md'):
        print(f"Error: File {markdown_file_path} is not a Markdown file.")
        return 1
    
    # Extract base file name without extension for CSV file
    base_name = Path(markdown_file_path).stem
    csv_file_path = f"{base_name}.csv"
    
    # Process the file
    modified_content, image_data_dict = extract_base64_images(markdown_file_path)
    
    # Write modified content back to the Markdown file
    write_modified_markdown(markdown_file_path, modified_content)
    
    # Create CSV reference file
    create_csv_reference(csv_file_path, image_data_dict)
    
    # Print summary
    num_images = len(image_data_dict)
    print(f"Processing complete. {num_images} image(s) extracted.")
    print(f"Modified Markdown saved to: {markdown_file_path}")
    print(f"Full base64 data saved to: {csv_file_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())