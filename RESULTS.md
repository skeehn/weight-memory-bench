# Results

Every number here was measured by this repo. Where a number is not trustworthy, it says so
and says why. Runs are in `runs/ledger.jsonl` with provenance attached.

Reader: `unsloth/Llama-3.2-{1B,3B}-Instruct` and `unsloth/Meta-Llama-3.1-8B-Instruct`. All
three share tokenizer fingerprint `b47fcb7017102fb3` — verified, not assumed, which is what
makes the sizes comparable at all.

---

## 0. The reader is the bottleneck, not retrieval

LongMemEval dev, n=100, Llama-3.2-1B, greedy decoding. **Accuracy, not evidence recall.**

| Arm | Tokens | Evidence recall | answered | acc \| answered | **acc over all** | fabrications |
|---|---|---|---|---|---|---|
| Full context | 105,708 | 1.000 | 0.740 | 0.216 | **0.160** | 3 |
| Grep | 4,075 | 0.809 | 0.200 | 0.200 | **0.040** | 3 |
| RAG | 4,061 | 0.968 | 0.230 | 0.174 | **0.040** | 1 |

**Full context has the evidence 100% of the time and answers 16% of questions correctly.**
No retrieval improvement can move a number capped that far below its own ceiling. On this
benchmark, at this model size, retrieval quality is not what is being measured — the reader
is. Evidence recall (§1) is the more informative measurement, and presenting accuracy as an
upgrade to it would be backwards.

Arm A also costs **20 seconds per query** (2,033s for 100) against grep's 0.7s. The
expensive arm loses on tokens, on latency, and wins on accuracy only against baselines that
are themselves near the floor.

### What the three-number rule caught

Grep and RAG return *identical* scores despite 53 of 100 answers differing and evidence
recall of 0.809 vs 0.968. Not a caching bug — verified by diffing the cached generations.
The reader is weak enough that a 16-point difference in evidence recall produces no
difference in accuracy at all.

And on the gameable metric, grep (0.200) looks competitive with full context (0.216). On
the honest one it is **four times worse** (0.040 vs 0.160). The entire gap is that grep
abstains 80% of the time and full context 26%. A benchmark reporting only
`accuracy | answered` would conclude that keyword matching rivals a 106K-token window.

### Sensitivity to the abstention definition

The reader refuses in phrasings it was never told to use — "I can't answer that", "I don't
have access to...". Detection is string matching and irreducibly fuzzy: "I don't have
personal opinions, but I can provide..." is a partial refusal, "I'm happy to help! How..."
is neither refusal nor answer. Across three versions of the marker list:

```
answered_rate            0.970 -> 0.740    moved 0.23
accuracy_given_answered  0.175 -> 0.216    moved 0.04
accuracy_over_all        0.170 -> 0.160    moved 0.01
```

`accuracy_over_all` is ~23x more stable, **not immune** — it moves because abstention takes
precedence over a keyword match, so widening the refusal set can reclassify a response that
carried the right answer beside a refusal phrase.

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

### The per-rung learning-rate sweep

The first scaling run held lr at 2e-3 across all sizes, which confounds scale with
effective step size: a rank-16 adapter perturbs a 1B model far more than an 8B one. So
every size was re-run across four learning rates, 5 seeds each.

```
        5e-4    1e-3    2e-3    5e-3
1B     0.000   0.000   0.500   0.000
3B     0.000   0.000   0.057   0.000
8B     0.100   0.200   0.300   0.000
```

**All three sizes peak at the same learning rate**, so the confound was real in principle
and absent in fact — the original comparison was valid as run. That is worth stating
plainly rather than quietly dropping, because the reasoning that motivated this run was
sound and the conclusion it predicted was wrong.

On the 5 probes valid at every size: **1B 0.400, 8B 0.280, 3B 0.080.** Non-monotonic after
per-rung tuning, so: report the grid, not a trend. (That threshold was fixed in
`summarize_lr_sweep.py` before the data arrived.)

**The usable window is narrower than a factor of two.** 1B is 0.000 at 1e-3, 0.500 at
2e-3, 0.000 at 5e-3. A mechanism that only works inside a sub-2x learning-rate band is
impractical on its own terms, independent of its peak accuracy: there is no gradient to
tune along, because everything outside the band reads as an identical zero.

**Scale improves writing, not reading.** 8B reaches the lowest fact NLL of any
configuration measured (4.75 at lr=1e-3, against 1B's best of 7.00) while recalling less
(0.300 vs 0.500). The bigger model stores the fact *better* and retrieves it *worse*. The
write path works and scales; the read path is what is missing.

### ⚠️ Error bars here understate the true variance

Reported standard deviations measure variance across seeds. There is additional variance
*within* a seed that this protocol does not capture.

Evidence: 3B at lr=2e-3 returned 0.114 on one run and 0.057 on another, same rank, same
learning rate, same epochs, same five seeds. The valid probe set was **identical**, and the
entire difference was one probe ("Where do I work?") going from 2 hits to 0. Meanwhile 1B
reproduced to three decimals across the same pair of runs.

`torch.manual_seed` fixes LoRA initialization but not GPU kernel non-determinism — larger
models dispatch to different kernels, and cuBLAS reductions are not deterministic by
default. Runs in this document predate the fix (`use_deterministic_algorithms`,
`CUBLAS_WORKSPACE_CONFIG`), now in `arms/weight_memory.py`.

Consequence: **3B's 0.057 is not distinguishable from zero, or from 0.114.** The 1B-vs-3B
and 8B-vs-3B gaps (5x-9x) are large enough to survive this, but no difference of ~0.06 in
these tables should be read as real.

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
