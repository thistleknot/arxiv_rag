import stanza

nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', verbose=False)

sentence = "Existing KE methods focus on updating small-scale and localized knowledge, typically on synthetic fact pairs (Mitchell et al., 2022a; Meng et al., 2022a)."

doc = nlp(sentence)
sent = doc.sentences[0]

print("Full dependency parse:")
for word in sent.words:
    head_text = sent.words[word.head - 1].text if word.head > 0 else 'ROOT'
    print(f"  [{word.id:2d}] {word.text:30s} upos={word.upos:8s} deprel={word.deprel:15s} head={word.head}({head_text})")
