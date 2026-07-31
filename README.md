# weight-memory-bench

Should AI memory live in **weights** or in a **vector store**? This measures both, on the
two axes the answer turns on: what it costs, and what it destroys.

Every number below was produced by this repo and is in `runs/ledger.jsonl` with provenance
attached. Total GPU spend: **under $2**.

---

## The four findings

### 1. Retrieval already wins the token argument

LongMemEval-S dev, n=100. *Evidence recall* asks whether the arm retrieved the turn
containing the answer at all.

| Arm | Context tokens | Evidence recall |
|---|---|---|
| Full context | 105,708 | 1.000 |
| Grep | 4,075 | 0.809 |
| RAG (BM25+dense+RRF) | 4,061 | **0.968** |
| Weight memory | **0** | see 3 & 4 |

A competent retriever reaches 96.8% of the ceiling at **1/26th the tokens**. The premise
this project was built on — that retrieval is expensive and weight memory is cheap — is
wrong. Weight memory is not competing against an expensive strawman. It is competing
against something already cheap and already near the ceiling.

### 2. The reader is the bottleneck, not retrieval

| Arm | answered | acc \| answered | **acc over all** |
|---|---|---|---|
| Full context | 0.740 | 0.216 | **0.160** |
| Grep | 0.200 | 0.200 | **0.040** |
| RAG | 0.230 | 0.174 | **0.040** |

**Full context has the evidence 100% of the time and answers 16% correctly.** No retrieval
improvement can move a number capped that far below its own ceiling. Grep and RAG score
*identically* despite 53 of 100 answers differing and a 16-point evidence-recall gap — the
reader is too weak for retrieval quality to propagate into accuracy at all.

Full context also costs **20 seconds per query** against grep's 0.7s. It loses on tokens, on
latency, and beats baselines that are themselves near the floor.

### 3. Scale improves *writing* memory, not *reading* it

Per-rung learning-rate sweep, mean fact recall over 5 seeds:

```
        5e-4    1e-3    2e-3    5e-3
1B     0.000   0.000   0.500   0.000
3B     0.000   0.000   0.057   0.000
8B     0.100   0.200   0.300   0.000
```

8B reaches the **lowest fact NLL of any configuration measured** (4.75, against 1B's best of
7.00) while recalling *less* (0.300 vs 0.500). The bigger model stores the fact better and
retrieves it worse. The write path works and scales; the read path is what is missing.

The usable learning-rate window is **narrower than a factor of two** — 1B is 0.000 at 1e-3,
0.500 at 2e-3, 0.000 at 5e-3. A mechanism that only works inside a sub-2x band is
impractical on its own terms: outside it, everything reads as an identical zero, so there is
no gradient to tune along.

### 4. Online updating destroys the model without establishing the memory

100 updates applied as **one continuous stream**, not independent fine-tunes:

| Updates | Retention | Held-out perplexity |
|---|---|---|
| 0 | 0.000 | 30.58 (1.00x) |
| 25 | 0.000 | 73.75 (2.41x) |
| 50 | **0.375** | 121.29 (3.97x) |
| 100 | 0.000 | **1126.25 (36.83x)** |

**There is no usable operating window.** Peak retention is 0.375 and already costs 4x
perplexity. By 100 updates the model is 36.8x worse on held-out text and has retained
nothing. This is not a bad accuracy/cost trade-off — there is no trade-off, because the
accuracy side never materializes while the cost side runs away.

*LoRA Learns Less and Forgets Less* measures forgetting after a **single** fine-tune. A
memory system updates forever. This is that regime.

---

## What is wrong with these numbers

Stated here rather than buried, because a results document without this section is
advertising.

- **The capability probe measure failed.** Only 1 of 6 general-knowledge probes survived
  validity filtering — a 1B reader cannot reliably answer "What is 2 plus 2?" in this prompt
  format. That column is a single coin flip. Perplexity carried the forgetting result alone.
- **The forgetting curve is n=1 seed.**
- **Error bars understate true variance.** 3B returned 0.114 and 0.057 on an identical
  config with an identical probe set — GPU kernel non-determinism, which `manual_seed` does
  not fix. 3B's 0.057 is not distinguishable from zero.
- **LongMemEval fits in a 131K window**, so on this corpus a bigger context substitutes for
  memory. BEAM is loaded but unrun.
- **Abstention detection is string matching and irreducibly fuzzy.** Moving the line moves
  `answered_rate` by 0.23 and `accuracy_over_all` by 0.01 — the honest metric is ~23x more
  stable, but not immune.

Full detail in [`RESULTS.md`](RESULTS.md).

---

## Method

The measurement rules were written before there was anything to measure, and several caught
real errors later.

**One tokenizer, no fallback.** Every arm counts with `harness/tokens.py`, which loads the
real reader tokenizer and *raises* if it cannot. The usual shortcut, `words * 1.3`, lands
between **0.79x and 3.46x** of the real count on this repo's own samples, erring in both
directions by text shape. It cannot be calibrated away and does not cancel in a ratio.

**Three numbers on every accuracy report.** `answered_rate`, `accuracy | answered`, and
`accuracy over all` with abstentions counted wrong. This caught a **4x error**: on the
gameable metric grep (0.200) looks competitive with a 106K-token window (0.216); on the
honest one it is four times worse.

**Provenance or no row.** Eight fields, or `ledger.append` raises.

**The ledger is read line by line.** A whole-file `json.loads` on a JSON-lines file dies on
line 2 and reports one unreadable file while inspecting zero rows — silently exempting the
artifact the audit exists to police.

**Degenerate runs are caught.** An all-abstain run passes every other gate. It is a
well-formed measurement of nothing.

---

## Seventeen bugs, in three classes

Every one produced output that looked entirely normal. The taxonomy matters more than the
count, because only the first class is what "benchmark bugs" usually means:

**Corrupted the numbers** — `words * 1.3` estimates; PEFT silently stacking adapters so each
sweep config inherited the last; unseeded LoRA init (same config gave 1/2, 1/2, 2/2);
`fact_nll` scoring `'pemberton'` `[79, 9034, 37733]` when the model emits `' Pemberton'`
`[69383, 37733]`; abstention and correctness both firing on one response; abstention
detection matching only the sanctioned phrase; GPU non-determinism at 3B but not 1B.

**Destroyed work** — a fatal transport error with no retry; no resumability, discarding paid
GPU time; `/environments/production/` routing to the *previous* build; a 24GB image that
never finished deploying.

**Cost money directly** — teardown on the last line of a script instead of in a trap, killed
before it ran, leaving a replica billing; teardown aborting on the first already-inactive
deployment while printing a false "shut down manually".

Two were real defects that turned out to change **nothing**: the RAG lane asymmetry, and the
learning-rate confound that motivated an entire re-run (all three sizes peak at the same
lr, so the original comparison was valid). Both are recorded. A correction list containing
only corrections that mattered is a filtered list.

---

## Layout

```
harness/tokens.py    one tokenizer, shared by every arm, no fallback
harness/ledger.py    append-only JSONL, read line by line
harness/gates.py     validity gates and the three-number rule
harness/reader.py    the single reader model and abstention contract
arms/                full_context, grep, rag, weight_memory
data/                LongMemEval and BEAM loaders
scripts/             sweeps, gates, benchmark and forgetting-curve runners
baseten/             deployment configs and cost-safe run scripts
runs/ledger.jsonl    every run, with provenance
```

## Running

```
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest
```

128 tests. First run downloads the reader tokenizer.
