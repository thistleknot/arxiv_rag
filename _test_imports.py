"""Test if training dependencies are available"""
import sys
print("Python:", sys.version)

try:
    import torch
    print(f"✅ torch {torch.__version__}")
except Exception as e:
    print(f"❌ torch: {e}")

try:
    import transformers
    print(f"✅ transformers {transformers.__version__}")
except Exception as e:
    print(f"❌ transformers: {e}")

try:
    import optuna
    print(f"✅ optuna {optuna.__version__}")
except Exception as e:
    print(f"❌ optuna: {e}")

try:
    import msgpack
    print(f"✅ msgpack")
except Exception as e:
    print(f"❌ msgpack: {e}")

print("\nAll imports successful!")
