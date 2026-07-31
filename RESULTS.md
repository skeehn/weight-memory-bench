# Results

> **Superseded headline.** Sections 2 and 2.5-2.7 below conclude that online weight memory
> does not work at 1B. **That conclusion was wrong**, and wrong because of three defects in
> my own setup, not because of the mechanism. See §-1. The old sections are kept because the
> corrected result is only interpretable next to what it replaces.

---

## -2. AlphaEdit: attempted, NOT evaluated

Three GPU runs, three bugs of mine, and **zero information about whether the method works**.
Recorded so the numbers are not mistaken for a result about AlphaEdit.

| run | retention | ppl | cause |
|---|---|---|---|
| 1 | 0.150 | 123,489x | bf16/float32 dtype error, then a null space of 8010/8192 |
| 2 | — | — | same, unfixed |
| 3 | 0.100 | 14,845x | preserve corpus never loaded; diagnostic was a constant |

**The diagnostic was hardcoded.** `EditReport(projected_norm=raw_norm)` sets it to the same
variable as `delta_norm`, so `null_space_retention` computes `x/x = 1.0` unconditionally. I
cited "survived projection 1.000" as evidence the projection did nothing. It was never a
measurement, and the interpretation of run 1 that rested on it is void.

**The preserve corpus never loaded.** Run 3 reports `n_preserve_texts: 8` and
`token_positions: 346` despite a fix intended to supply thousands of LongMemEval turns. The
loader sits inside a bare `try/except: pass`, which turned a loud failure into a silent one.
346 token positions cannot estimate an 8192x8192 covariance, so the null space was again
sampling noise and the "constrained" edit ran unconstrained.

That `except: pass` is the root cause of run 3 and is the same failure shape as most of this
document: instrumentation that concealed a problem instead of surfacing it.

**What did work:** `restored_ppl_ratio` came back 1.000x on every run, so the edit was fully
contained in the tracked layers. The plumbing is sound. Nothing else here is evidence about
anything.

---

## -1. Weight memory works: 1.00 retention

Llama-3.2-1B, LoRA rank 16, 20 invented facts, **answered with an empty context window**.

| lr | steps | seed | retention | held-out ppl |
|---|---|---|---|---|
| 2e-3 | 45 | 0 | **0.70** | **1.65x** |
| 2e-3 | 45 | 1 | 0.75 | 2.14x |
| 2e-3 | 120 | 0 | 0.95 | 6.15x |
| 2e-3 | 120 | 1 | **1.00** | 14.97x |

The model learns **every** injected fact at the longer budget, and 70-75% of them for a
1.65-2.14x perplexity cost. Previous best across every method in this document: **0.083**.

### The Pareto frontier

```
retention  damage    config
0.200       1.28x    5e-4 / 45
0.375       2.02x    1e-3 / 45
0.475       4.14x    5e-4 / 120
0.700       1.65x    2e-3 / 45    <- dominates the three above it on BOTH axes
1.000      14.97x    2e-3 / 120
```

High learning rate, **short** budget, gradient clipping, augmented data. Nearly the
opposite corner of the search space from where the failed methods were looking.

### The three defects that produced the old negative result

**1. One phrasing per fact.** Allen-Zhu & Li (arXiv 2309.14316): knowledge seen in a single
surface form is *"memorized but not extractable, 0% accuracy, regardless of subsequent
instruction fine-tuning."* The old runs trained on one sentence per fact and measured
exactly that symptom — fact NLL collapsing while recall stayed at zero — and read it as
proof the mechanism could not work. With ~10 generated paraphrases per fact: 0.083 -> 0.750.

**2. The abstention system prompt.** Measured on 50 facts, each asked with its own statement
supplied as context:

```
STRICT  ("reply exactly: I don't know")   19/50 answerable   38%
SOFT                                       38/50             76%
NONE                                       49/50             98%
```

A 1B instruct model given a firm refusal instruction declines questions whose answers are
directly in front of it. STRICT was used in **every** measurement in this document, so
reader accuracy is understated throughout — including the "the reader is the bottleneck"
conclusion in §0. On identical weights it also hid 17.5 points of retention (0.575 vs 0.750).

**3. Learning rate 10x too high with no stabilisers.** 2e-3 against a standard LoRA range of
1e-4 to 5e-4, no gradient clipping, no warmup, and 25-100 epochs over ~200 tokens. The
divergence attributed to the mechanism was self-inflicted.

### A claim from §2.6 that is now falsified

That section states that "the update that encodes the fact and the update that damages the
model appear to be **the same update**." They are not. Same config, same seeds, different
initialization:

```
2e-3 /  45 steps:  0.75 @ 286.2x   and  0.75 @   2.2x
1e-3 / 120 steps:  0.65 @   6.1x   and  0.80 @ 106.8x
```

Retention is stable across seeds while damage varies by two orders of magnitude. The two
axes are separable; the earlier claim was only sustainable while retention was pinned at
zero everywhere, which made any correlation unmeasurable.

### Against the pre-registered bar

The bar was retention >= 0.50 **and** ppl_ratio <= 1.5x, fixed before any of this was known
and **not moved**. The best point clears the retention floor comfortably (0.70) and misses
the perplexity ceiling by 10% (1.65x). No configuration clears both on *every* seed.

Stated plainly: **fails the strict bar; answers the question as posed.** 1.65x perplexity on
held-out prose is a real cost and is not catastrophic forgetting. Catastrophic forgetting is
the 20,371x baseline this began from.

### Still open

- **Restart on divergence** (built, unit-tested, not yet run on real weights). Seed 0 gives
  1.65x and seed 1 gives 2.14x on the same config; taking the first attempt under threshold
  should turn that spread into a reliable operating point.
- **AlphaEdit** — deployed, blocked on a bf16/float32 dtype bug, now fixed and awaiting a
  rerun. Its `null_space_retention` diagnostic is the sharpest available test of whether
  fact-storing and knowledge-preserving directions genuinely overlap in a 1B model.

---

Every number here was measured by this repo. Where a number is not trustworthy, it says so
and says why. Runs are in `runs/ledger.jsonl` with provenance attached.

Reader: `unsloth/Llama-3.2-{1B,3B}-Instruct` and `unsloth/Meta-Llama-3.1-8B-Instruct`. All
three share tokenizer fingerprint `b47fcb7017102fb3` — verified, not assumed, which is what
makes the sizes comparable at all.

---

## 0. Accuracy: the reader looks like the bottleneck

LongMemEval **dev** split, n=100, Llama-3.2-1B, greedy decoding.

⚠️ **`dev` is the tuning split.** This repo implements stratified dev/test discipline
precisely so results would be reported on `test`, and then reports dev. The 400-instance
test split has not been run. Everything below is provisional.

| Arm | Tokens | Evidence recall | answered | acc \| answered | **acc over all** | 95% CI | fabr |
|---|---|---|---|---|---|---|---|
| Full context | 105,708 | 1.000 | 0.740 | 0.216 | **0.160** | [0.101, 0.244] | 3 |
| Grep | 4,075 | 0.809 | 0.200 | 0.200 | **0.040** | [0.016, 0.098] | 3 |
| RAG | 4,061 | 0.968 | 0.230 | 0.174 | **0.040** | [0.016, 0.098] | 1 |

Full context's interval does not overlap the retrieval arms', so that gap is real. Grep and
RAG share a point estimate *and* an interval: they are **indistinguishable at n=100**, which
is a weaker and different claim than being equal.

**Full context has the evidence 100% of the time and answers 16% of questions correctly.**
No retrieval improvement can move a number capped that far below its own ceiling.

⚠️ **But "the reader is the bottleneck" is an inference, not a measurement.** Full context's
16% is confounded with long-context degradation across 105,708 tokens of distractors. The
`oracle` arm — built, unrun — hands the reader only the tagged evidence turns, a few hundred
tokens with no distractors, and would separate the two. Until then the live alternative is
that the reader is competent on clean evidence and failing on long noisy context: a
different problem with different fixes.

Evidence recall (§1) remains the better-supported measurement of the two.

Arm A also costs **20 seconds per query** (2,033s for 100) against grep's 0.7s. The
expensive arm loses on tokens, on latency, and wins on accuracy only against baselines that
are themselves near the floor.

### What the three-number rule caught

Grep and RAG return the same point estimate despite 53 of 100 answers differing and
evidence recall of 0.809 vs 0.968. Not a caching bug — verified by diffing the cached
generations. But with overlapping intervals of [0.016, 0.098], the correct reading is that
a 16-point evidence-recall difference **does not resolve at n=100**, not that it produces no
effect. Distinguishing those needs the test split.

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
| D. Weight memory | **0** | §2 — toy corpus only, never run on LongMemEval |

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

## 2.5 The forgetting curve: what online updating destroys

100 online updates applied as **one continuous stream**, not as independent fine-tunes.
That distinction is the point: *LoRA Learns Less and Forgets Less* measures forgetting after
a single fine-tune, and a memory system updates forever. Llama-3.2-1B, rank 16, lr 2e-3,
seed 0, measured at six checkpoints.

| Updates | Retention | Held-out perplexity | vs baseline |
|---|---|---|---|
| 0 | 0.000 | 30.58 | 1.00x |
| 5 | 0.000 | 37.79 | 1.24x |
| 10 | 0.000 | 36.22 | 1.18x |
| 25 | 0.000 | 73.75 | 2.41x |
| 50 | **0.375** | 121.29 | 3.97x |
| 100 | 0.000 | **1126.25** | **36.83x** |

**The model is destroyed and remembers nothing.** Held-out perplexity — measured on prose
sharing no vocabulary with the injected episodes — rises **36.8x**, and retention at that
point is zero. The damage is not a side effect of successful memorization; there is no
successful memorization to trade against.

**There is no usable operating window.** Peak retention is 0.375 at 50 updates, and it
already costs 4x perplexity. Retention never exceeds a third even at its best, while damage
compounds monotonically from update 25 onward and then goes vertical.

This is the clearest statement the project produces. Weight memory at this scale does not
have a bad accuracy/cost trade-off — it has no trade-off, because the accuracy side never
materializes while the cost side runs away.

### ⚠️ The capability measure failed, and the perplexity measure is why both existed

The design tracked capability two ways: continuous perplexity, and discrete probes on
general knowledge the base model already had.

**Only 1 of 6 discrete probes survived the validity filter.** The 1B reader could not
reliably answer "What is 2 plus 2?", "What is the capital of France?", "How many days are
in a week?", or "What is the largest ocean?" in this prompt format, so they were correctly
excluded as unanswerable — a probe never known cannot be forgotten. That left a single
binary probe, which is why the column reads 1.000 / 0.000 / 1.000 / 1.000 / 1.000 / 0.000.
It is one coin, not a rate, and should be ignored.

Perplexity carried the result alone. Having a continuous measure alongside a discrete one
is what kept the run from producing nothing: the discrete measure was still reporting
perfect capability at 50 updates while perplexity had already quadrupled.

Fixing this means capability probes calibrated to a 1B model rather than to what an adult
knows. Not re-run here, and the perplexity finding does not depend on it.

**n=1 seed.** Given measured seed variance elsewhere in this document, the retention column
in particular should not be read as precise.

---

## 2.6 Four anti-forgetting methods, and why none of them worked

The forgetting result raises the obvious question: can the damage be prevented? Four
composable mitigations, each motivated by one of the two measured failure modes rather than
picked off a list. Llama-3.2-1B, rank 16, lr 2e-3, 25 update passes, 2 seeds, 6 valid probes.

**The bar was fixed in source before any of these ran:** retention >= 0.50 **and**
ppl_ratio <= 1.5x. Deliberately stricter than "beat the baseline", because beating a method
that destroys the model is meaningless.

| Method | Retention | Held-out ppl | Steps | Verdict |
|---|---|---|---|---|
| naive | 0.083 | 20,371x | 25 | damages, learns nothing |
| replay | 0.000 | 46,717x | 50 | damages, learns nothing |
| chat-format | 0.083 | 178x | 25 | damages, learns nothing |
| replay+chatfmt | 0.083 | 5,019x | 50 | damages, learns nothing |
| KL anchor | 0.000 | **1.51x** | 25 | safe, learns nothing |
| ppl gate 1.5x | 0.000 | 1.70x | 6 | safe, learns nothing |
| replay+chatfmt+gate | 0.000 | 116x | 9 | damages, learns nothing |
| **everything** | 0.000 | **1.80x** | **23** | safe, learns nothing |

### The finding

**Damage is controllable across four and a half orders of magnitude. Retention is not
controllable at all.** Perplexity spans 46,717x to 1.51x across these configs; retention
never exceeds 0.083 and is exactly 0.000 in every configuration that controls damage.

**`everything` rules out the "stopped too early" objection.** The gated configs learned
nothing, but they also only ran 6-9 steps. `everything` ran **23 of 25 steps**, held the
model at 1.80x, and still retained zero. Full training budget, model intact, nothing stuck.

> At 1B with rank-16 LoRA, the update that encodes the fact and the update that damages the
> model appear to be **the same update**. Every method that suppressed the damage suppressed
> the learning by the same amount.

### What each method actually did

- **Replay made it worse** (46,717x vs naive's 20,371x). It doubles the step count, and
  damage tracks steps. Rehearsal is the right tool for distributional drift and the wrong
  tool for a diverging optimizer.
- **Chat-format cut per-step damage ~114x** (178x vs 20,371x) with no retention benefit.
  Wrapping text in the template the model expects lowers the loss, which shrinks the
  gradient. It is a gradient-magnitude technique, not a memory technique.
- **KL anchor produced the least damage of anything** (1.51x) and zero retention. This is
  the stability-plasticity tradeoff at its stability extreme.
- **The perplexity gate works only if it samples faster than the damage.** Plain gate:
  checked every 1-2 steps, caught it at 1.70x. With replay: checked between epochs of 4-5
  steps, and by the second check the model was at **116x**. Same controller, same threshold,
  68x different outcome purely from checking granularity.

### ⚠️ Caveats

- **`kl_weight` was sampled at exactly one point (0.5), and it froze the model.** The entire
  question is whether some weight buys retention before perplexity moves. That is a cheap
  one-dimensional sweep and it has not been run. This result does not rule out a working KL
  setting; it rules out *this* one.
- **This "naive" is harsher than §2.5's.** Here 25 passes run under one optimizer, so Adam's
  moments compound; the forgetting curve used 25 separate single-epoch calls with a fresh
  optimizer each time. That earlier choice, made for reproducibility, was accidentally
  protective by 2-3 orders of magnitude. 20,371x and 36.83x are not the same measurement.
- **2 seeds, 6 probes.** Retention differences below ~0.17 are one probe on one seed.
- **1B only.** 8B showed the lowest fact-NLL of any configuration measured (§2), so the
  separability this run failed to find may exist at scale.
- **MEMIT-style targeted editing was not implemented.** It is the one approach that does not
  run gradient descent over the whole adapter, and therefore the one most likely to separate
  what these four could not.

---

## 2.7 Can it be fixed? A KL sweep, and the answer

`kl_weight` was swept 0 to 0.5, then finer between 0 and 0.005 where the entire regime
change happens, then confirmed at 6 seeds on the three most promising weights. Chat-format
held on throughout, since it cut per-step damage ~114x for free.

**Confirmation run, 6 seeds, 36 probe-trials per weight:**

| kl_weight | retention | hits/36 | 95% CI | diverged |
|---|---|---|---|---|
| 0.0005 | 0.139 | 5/36 | [0.061, 0.287] | 1/6 |
| **0.001** | **0.222** | **8/36** | [0.117, 0.381] | **0/6** |
| 0.002 | 0.083 | 3/36 | [0.029, 0.218] | 1/6 |

kl=0.001 is best on every axis — highest retention, lowest perplexity (1.63x), the only
weight with zero divergences. **And none of the differences are significant**: all three
intervals overlap. That is the honest limit of 36 trials.

### The distance travelled, and the distance remaining

| | Retention | Held-out ppl |
|---|---|---|
| Naive | 0.083 | 20,371x |
| Best found (kl=0.001 + chat-format) | 0.222 | 1.63x |

**~2.7x the retention at ~12,000x less damage.** The model goes from destroyed and
remembering nothing to intact and remembering about a fifth. That is real progress on the
problem.

It still fails the pre-registered bar, and badly on the axis that matters: 0.222 against a
0.50 floor. A memory system that recalls one fact in five is not a memory system.

### ⚠️ The process is bimodal, so every mean in this document understates the problem

At kl=0.003, one seed diverged to **259.96x** and the other sat at **1.34x** — a 194-fold
gap inside a single configuration. The process does not degrade smoothly; it either blows
up or it does not, and the seed decides which.

Every mean reported across the method comparison and the coarse sweep is therefore an
average of divergences and non-divergences, describing neither. `kl_sweep.py` now prints
per-seed values and a divergence count so this cannot hide behind an average again. The
method-comparison table (§2.6) predates that fix and should be read with it in mind.

**This was caught three times only by the next data point, never by applying the intervals
already in the repo.** 2/12 was called a "strict improvement"; 2/12 and 3/12 together were
called "a coherent pattern"; the damage axis was called "real and consistent" one message
before a 100x spike appeared in it. The discipline existed in code and kept failing at the
moment of interpretation.

### Still open

- **8B.** It showed the lowest fact-NLL of any configuration measured (§2). The separability
  absent at 1B may exist at scale.
- **MEMIT-style targeted editing.** The only approach that does not gradient-descend the
  whole adapter, and the one designed specifically for this failure. Not implemented.

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
