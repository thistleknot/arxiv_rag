import re

def normalize_for_matching(text):
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

sentence = "However, KE provides fine-grained controllability when specific knowledge needs to be altered, which is unachievable by CL; 2 Forgetting ."
obj = "fine - grained controllability , which is unachievable by CL"

sent_norm = normalize_for_matching(sentence)
obj_norm  = normalize_for_matching(obj)

print("Normalized sentence:")
print(" ", sent_norm)
print()
print("Normalized object:")
print(" ", obj_norm)
print()

# Show how tokens align
sent_words = sent_norm.split()
obj_words  = obj_norm.split()
print("Object words:", obj_words)
print()

# Find positions of each object word in sentence
for w in obj_words:
    positions = [i for i, sw in enumerate(sent_words) if sw == w]
    print(f"  '{w}' found at positions: {positions}")
