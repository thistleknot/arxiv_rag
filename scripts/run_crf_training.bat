@echo off
cd c:\Users\user\arxiv_id_lists
set HF_HUB_DISABLE_TELEMETRY=1
set TRANSFORMERS_NO_TELEMETRY=1
c:\users\user\py310\scripts\python.exe train_bert_bio_crf.py > crf_training_output.txt 2>&1
