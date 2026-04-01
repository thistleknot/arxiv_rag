"""Test training script execution"""
import sys
import traceback

print("Attempting to run training script...")

try:
    sys.argv = ['train_bert_bio_optuna.py', 
                '--train', 'checkpoints/quotes_bio_training.msgpack',
                '--output', 'quotes_bio_tagger.pt',
                '--trials', '2',
                '--max-epochs', '5',
                '--patience', '3']
    
    # Try to execute the training script
    exec(open('training/train_bert_bio_optuna.py').read())
    
except SystemExit as e:
    print(f"SystemExit: {e.code}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
