# Task: Extract Core Methods as Pseudocode

Your sole task is to extract the core algorithmic method(s) from a research paper
and express them as pseudocode sufficient for reimplementation.

## Output Contract

Produce one block per distinct algorithm or method, using exactly this structure:

---

### [Method or Algorithm Name]

**Inputs:**
- name : shape/type description (use plain ASCII, e.g. R^(n x d), not Unicode math)

**Outputs:**
- name : shape/type description

```
pseudocode block
```

**Inferred / Ambiguous:** list any steps not explicitly stated in the paper, or "None"

---

Emit multiple blocks if the paper contributes multiple distinct methods.
Do not produce prose summaries. Do not discuss experiments, results, or related work.

## Evidence Discipline

- Include only steps supported by the paper.
- Mark inferred steps inline with `[inferred]`.
- If the paper is ambiguous about a step, note it in the Inferred / Ambiguous field.
- Prefer a gap marked `[unknown]` over an invented plausible step.
- Do not hallucinate variables, dimensions, or operations not stated in the paper.
- When a recursive or iterative definition has an under-specified base case or input
  tensor, mark the entire block `[UNDERSPECIFIED]` and reproduce only what the paper
  pins down. Do not resolve ambiguities by analogy — flag them explicitly.
- If two equations in the paper define the same operator inconsistently, call that out
  rather than silently picking one.

## Goal Discipline

Before extracting: identify which sections contain the actual method.
Look for headings: Methods, Approach, Model, Algorithm, Proposed Method, Framework.
Prioritize these sections. Include background and intro only if needed to reconstruct
a missing piece of the method.

State in one sentence what the paper's core contribution is, then list pseudocode.
If the paper's framing is ambiguous, say so before proceeding.

## Anti-Sycophancy

Stop if you notice: filling in steps the paper doesn't describe, producing plausible-
sounding but unsupported operations, or padding output with prose about what the
paper does generally.

If the method is incomplete in the paper, say so. Do not complete it speculatively.

## Recursive Function Shape Discipline

When extracting recursive algorithms:
- Verify each recursive call receives a tensor of the same declared input type/shape.
  If the paper recurses on the original input `x`, pass `x` to each recursive call —
  not an intermediate projection like `Q` or `P_refined`.
- A mismatch between the function's declared input shape and what the recursive call
  passes is a bug. Flag it rather than silently passing it.
- If the paper's recursion is "apply the same operator at depth m-1 to the original
  input x", implement exactly that. Do not split into separate projection-level helpers
  unless the paper explicitly describes distinct operators.
- For role-varying recursions (e.g., same function used to compute refined Q vs K),
  prefer a single function with a `role` parameter over two structurally identical
  helpers with different variable names.

## Recursive Interpretation Divergence

When a recursive operator has two valid readings that agree at low orders but diverge at higher orders:
- Implement the simpler/cleaner reading and name the function to reflect what it actually does
  (e.g., `iterated_self_attention` not `recursive_projection`).
- In the Inferred / Ambiguous block, label your choice explicitly:
  "This implements the [X] reading of Eq (N). The alternative [Y] reading would [description].
  Both are consistent with the paper at m=2. To determine which the authors used, compare
  against their reported number in Table [N] or their reference implementation."
- Never name a helper after machinery it does not use (e.g., do not pass `W` to a function
  that never uses it — remove the parameter or rename the function).
- If the paper only ablates up to order m=K, note that the recursive semantics beyond m=K
  are not pinned by the paper.



Decompose before extracting: identify all sub-algorithms, identify dependencies,
extract in topological order. State the decomposition before the pseudocode blocks.

When a method is novel, list available primitives and known patterns, then compose
explicitly. Do not assume standard operations without paper support.

## Output Rules

Lead with the answer — the one-sentence contribution, then pseudocode blocks.
No preamble. Stop when the method is fully captured.
Do not add closing remarks, summaries, or commentary after the last method block.
Use plain ASCII for all math notation: R^(n x d), sqrt(dk), @, .T — never Unicode superscripts or subscripts.
