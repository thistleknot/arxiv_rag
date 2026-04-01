"""Regenerate training data with fixed adjacent grouping"""
import sys
import subprocess

print("Regenerating training data with fixed create_bio_labels()...")
print("This will overwrite bio_training_atomic_clean.msgpack")
print()

result = subprocess.run([
    r'c:\users\user\py310\scripts\python.exe',
    'extract_bio_atomic_clean.py',
    '--chunks', '250',
    '--output', 'bio_training_250chunks_grouped.msgpack'
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("ERRORS:")
    print(result.stderr)

print("\n" + "="*80)
print("NEXT STEP: Test training with fixed labels")
print("="*80)
print("Run: python quick_train_test.py")
print("Expected: F1 should exceed 0.1218 (previous best with incomplete labels)")
