"""Patch eval/textgrad_optimize.py — remove answer_relevancy references."""
import re

p = "eval/textgrad_optimize.py"
text = open(p, encoding="utf-8").read()

before = text.count("answer_relevancy")
print(f"References before: {before}")

# 1. score_to_loss_text: 4-metric mean → 3-metric mean
text = re.sub(
    r'scores\[k\] for k in \("context_precision", "context_recall",\s*"faithfulness", "answer_relevancy"\)\s*\) / 4\.0',
    'scores[k] for k in ("context_precision", "context_recall", "faithfulness")\n    ) / 3.0',
    text,
)

# 2. Remove the answer_relevancy line from the feedback f-string
text = re.sub(
    r"  answer_relevancy\s*: \{scores\['answer_relevancy'\]:.+?\}\s*\n",
    "",
    text,
)

# 3. Remove the answer_relevancy analysis line
text = re.sub(
    r'\{"answer_relevancy is LOW[^}]+\}\s*\n',
    "",
    text,
)

# 4. log_iteration: 4-metric mean → 3-metric mean
text = re.sub(
    r'scores\[k\] for k in \("context_precision", "context_recall",\s*"faithfulness", "answer_relevancy"\)\s*\) / 4\.0\s*\n(\s*mlflow\.log_metric)',
    'scores[k] for k in ("context_precision", "context_recall", "faithfulness")\n        ) / 3.0\n\\1',
    text,
)

# 5. Main loop: 4-metric mean → 3-metric mean (iteration scoring)
text = re.sub(
    r'scores\[k\] for k in \("context_precision", "context_recall",\s*"faithfulness", "answer_relevancy"\)\s*\) / 4\.0',
    'scores[k] for k in ("context_precision", "context_recall", "faithfulness")\n            ) / 3.0',
    text,
)

# 6. Main loop print: remove AR= part
text = re.sub(
    r'f"AR=\{scores\[.answer_relevancy.\]:.+?\}  "\s*\n',
    "",
    text,
)

# 7. Final eval: 4-metric mean → 3-metric mean
text = re.sub(
    r'final_scores\[k\] for k in \("context_precision", "context_recall",\s*"faithfulness", "answer_relevancy"\)\s*\) / 4\.0',
    'final_scores[k] for k in ("context_precision", "context_recall", "faithfulness")\n            ) / 3.0',
    text,
)

after = text.count("answer_relevancy")
print(f"References after:  {after}")
open(p, "w", encoding="utf-8").write(text)
print("Done.")
