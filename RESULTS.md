# Results

Every number here was measured by this repo. Where a number is not trustworthy, it says so
and says why. Runs are in `runs/ledger.jsonl` with provenance attached.

Reader: `unsloth/Llama-3.2-{1B,3B}-Instruct` and `unsloth/Meta-Llama-3.1-8B-Instruct`. All
three share tokenizer fingerprint `b47fcb7017102fb3` — verified, not assumed, which is what
makes the sizes comparable at all.

---

## 1. Retrieval already wins the token argument

LongMemEval-S dev split, n=100 (94 answerable). **Evidence recall** asks whether the arm
retrieved the turn containing the answer at all. It is a ceiling, not an accuracy: the
reader still has to use what it is handed.

| Arm | Context tokens (median) | Evidence recall |
|---|---|---|
| A. Full context | 105,708 | 1.000 (94/94) |
| B. Grep | 4,075 | 0.809 (76/94) |
| C. RAG (BM25 + dense + RRF) | 4,061 | **0.968** (91/94) |
| D. Weight memory | **0** | §2 |

**This is not the result the project was designed around.** The premise was that retrieval
is expensive and weight memory is cheap. Retrieval is not expensive: arm C reaches 96.8% of
the ceiling's evidence recall at **1/26th the tokens**. The 105K-token arm buys almost
nothing a competent retriever does not already get.

So weight memory is not competing against a costly strawman. It is competing against a
baseline that is already cheap and already near the ceiling. That makes the bar much higher
and the question much more interesting.

Arm B exists to keep the others honest: BM25+dense+RRF beats plain string matching by 16
points at an identical token budget, so the complexity is earned.

---

## 2. Weight memory does not work at these scales

Protocol: 16 short episodes, facts stated plainly, answers are invented proper nouns so
nothing can leak from pretraining. Ingest by plain causal-LM loss over the raw transcript,
then answer with an **empty context window**. Rank 16, lr 2e-3, 10 epochs, 5 seeds.

Probes are only counted when they pass two controls: the model must NOT answer them before
ingestion (no pretraining leak) and MUST answer them with the full text in context (the
question is within the reader's ability). Sizes see different valid sets, so the comparable
figure is the 5 probes valid at every size.

| Size | As reported | Common probe set |
|---|---|---|
| **1B** | 0.500 | **0.400** |
| 3B | 0.114 | 0.160 |
| 8B | 0.200 | 0.160 |

Per probe, hits out of 5 seeds:

```
                          1B  3B  8B
Where do I work?           4   2   1
What is the ferret named?  3   1   1
What car do I drive?       2   1   2
What is my manager named?  1   0   0
What is my sister named?   0   0   0     never recalled, at any size
```

### What the failure looks like up close

**Writing in and reading back are different problems.** Fact NLL falls sharply after
ingestion — the fact demonstrably enters the weights — while recall stays near a third. It
is stored as text without being retrievable as an answer. Exact-match scoring alone cannot
see this distinction; it took the likelihood diagnostic to separate "did not learn it" from
"learned it, cannot surface it".

**More training is actively harmful past a narrow window.** At lr=2e-3, ten epochs helps.
Fifty epochs drives fact NLL *above the untrained baseline* — the model ends up worse at
the exact fact it just trained on. That is catastrophic forgetting appearing on the target
itself, at the smallest scale it could possibly appear.

**More capacity does not help.** Rank 16 → 64 → 256 degrades monotonically.

**Seed variance swamps the effect.** Standard deviation across seeds is 0.12–0.23, which is
larger than most differences worth measuring. A single unseeded run is a draw from a wide
distribution, not a result.

### ⚠️ The scaling shape is not yet a claim

Learning rate was held at 2e-3 across all three sizes. A rank-16 adapter is a
proportionally much larger perturbation on a 1B model than on an 8B one, so identical
hyperparameters are a far larger *effective* step at the small end. This measures recall at
a fixed step size, not recall at scale.

The honest statement today is: **at fixed hyperparameters, recall falls with model size.**
Turning that into a statement about scale requires tuning the learning rate per rung and
comparing each size's best. Until then the ordering above should not be read as a scaling
law.

---

## 3. What the harness caught before any money was spent

Every one of these produced output that looked completely normal.

| Bug | Consequence if unnoticed |
|---|---|
| `words × 1.3` token estimate | Off by 0.79x–3.46x, in *both* directions by text shape. Corrupts the headline cost ratio and does not cancel in it. |
| PEFT stacking adapters | Each sweep config inherited earlier training. Produced a false "full recall" that was pure contamination. |
| Unseeded LoRA init | Same config returned 1/2, 1/2, 2/2. Would have published an unreproducible number. |
| `fact_nll` scoring the wrong tokens | `'pemberton'` is `[79, 9034, 37733]`; the model emits `' Pemberton'` = `[69383, 37733]`. Measured the likelihood of a string the model never produces. |
| Abstention and correctness both firing | "I don't know, but maybe Pemberton" scored as abstained *and* correct, so the three numbers stopped partitioning the probe set. |
| Dense/lexical lane asymmetry | BM25 returned only genuine matches; dense returned its top 100 regardless of quality. Real flaw, but re-measurement showed **no effect on the result** — the 0.968 stood. |
| Missing freshness guard in the repeatability script | The same contamination hole as the PEFT bug, unguarded, in the script whose numbers were being reported. |

The fifth, sixth, and seventh were found by reviewing code that had already produced
published-looking numbers. The sixth is worth singling out for the opposite reason: it was
a genuine defect that turned out **not** to matter, and saying so is part of the record.

---

## 4. Cost

| | |
|---|---|
| GPU time | ~6 minutes (1B 77s, 3B 76s, 8B 177s) on 2×L4 |
| Spend | ~$0.17 |
| Everything else | local, free |

The scaling run cost less than a coffee because the failure modes were found on a laptop
first. That is the argument for the harness, stated as a number.

---

## 5. Open

- **Per-rung learning-rate sweep** (~$0.70). The only way the scaling shape becomes a claim.
- **The forgetting curve.** Held-out capability against update count. This measures what
  weight memory *destroys*, which is the half nobody publishes.
- **BEAM.** The regime where memory is not substitutable by a bigger window. LongMemEval
  fits in 131K, so on that corpus a larger window substitutes for memory outright.
- **The validity gate is noisy.** Single greedy generation, substring match. 3B was judged
  unable to answer a question that 1B answered 5/5 — almost certainly phrasing, not
  capability.
