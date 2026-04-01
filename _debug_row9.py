import stanza, re, string

nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', verbose=False)

sent = "However, perhaps due to the inductive bias of gradient-based training [84], deep learning models tend to diffuse information across the entire representation vector."
doc = nlp(sent)

for s in doc.sentences:
    print("=== DEP PARSE ===")
    for w in s.words:
        head_text = s.words[w.head-1].text if w.head > 0 else "ROOT"
        print("  [%d] %s dep=%s head=%d(%s)" % (w.id, w.text, w.deprel, w.head, head_text))
    
    # Find root
    root = next((w for w in s.words if w.head == 0), None)
    print("\nRoot:", root.text if root else None)
    
    if root:
        # Find subj, obj, obl
        subj_head = obj_head = obl_head = None
        for w in s.words:
            if w.head == root.id:
                if w.deprel in ['nsubj','nsubj:pass']:
                    subj_head = w
                elif w.deprel in ['obj','dobj','iobj']:
                    obj_head = w
                elif w.deprel == 'obl':
                    if obj_head is None or obj_head.deprel == 'obl':
                        obl_head = w
        print("\nSubj head:", subj_head.text if subj_head else None)
        print("Obj/obl head:", (obj_head or obl_head).text if (obj_head or obl_head) else None)
