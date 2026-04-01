import msgpack

with open('data/bio_training_test10.msgpack', 'rb') as f:
    data = msgpack.unpackb(f.read())

training_data = data['training_data']
total = len(training_data)

subj_count = sum(1 for ex in training_data if any(ex['labels']['B-SUBJ']))
pred_count = sum(1 for ex in training_data if any(ex['labels']['B-PRED']))
obj_count = sum(1 for ex in training_data if any(ex['labels']['B-OBJ']))
complete_count = sum(1 for ex in training_data 
                     if any(ex['labels']['B-SUBJ']) 
                     and any(ex['labels']['B-PRED']) 
                     and any(ex['labels']['B-OBJ']))

print(f"\n{'='*50}")
print(f"BEFORE FIX (28 examples from previous run):")
print(f"  Examples with subject: 12 (42.9%)")
print(f"  Examples with predicate: 27 (96.4%)")
print(f"  Examples with object: 8 (28.6%)")
print(f"  Complete S+P+O: 4 (14.3%)")
print(f"{'='*50}")
print(f"AFTER FIX ({total} examples from current run):")
print(f"  Examples with subject: {subj_count} ({100*subj_count/total:.1f}%)")
print(f"  Examples with predicate: {pred_count} ({100*pred_count/total:.1f}%)")
print(f"  Examples with object: {obj_count} ({100*obj_count/total:.1f}%)")
print(f"  Complete S+P+O: {complete_count} ({100*complete_count/total:.1f}%)")
print(f"{'='*50}\n")

print("Sample improvements:")
print("\n1. 'KE and CL' - now includes 'and' in subject span")
print("2. 'updating enormous knowledge in LLMs' - now includes 'in' in subject span")
print("3. 'Existing KE methods' - multi-word subject preserved")
