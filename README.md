# weight-memory-bench

Measuring memory that lives in **weights** against memory that lives in a **vector store**, on
the two axes that decide whether the trade is worth making: token cost and forgetting.

Status: **arms A-C measured, arm D characterized at 1B, scaling curve in progress.**

## What is measured so far

Selection cost and evidence recall on the LongMemEval dev split (n=100, 94 answerable).
Evidence recall asks whether the arm retrieved the turn containing the answer at all -- it
is a *ceiling*, not an accuracy, since the reader still has to use what it is handed.

| Arm | Context tokens (median) | Evidence recall |
|---|---|---|
| A. Full context | 105,708 | 1.000 (94/94) |
| B. Grep | 4,075 | 0.809 (76/94) |
| C. RAG (BM25+dense+RRF) | 4,061 | 0.968 (91/94) |
| D. Weight memory | **0** | see below |

**The headline is not the one the plan predicted.** Good retrieval already reaches the
ceiling's evidence recall at 1/26th the tokens. The expensive arm buys nothing a
well-built retriever does not already get, so weight memory is not competing against a
costly strawman -- it is competing against a baseline that is already cheap and already
at the ceiling.

**Arm D at 1B does not work.** On the easiest case constructible -- 16 short episodes,
facts stated plainly, invented proper nouns so nothing can leak from pretraining -- online
LoRA over the raw transcript recalls **33% of facts, with the random seed alone moving
that between 17% and 50%** (5 seeds, 6 valid probes).

Three observations behind that number:

- **Writing in and reading back are different problems.** Fact NLL drops sharply after
  ingestion, so the fact demonstrably enters the weights. Recall stays at a third. It is
  stored as text without being retrievable as an answer.
- **More training is actively harmful past a narrow window.** At lr=2e-3, ten epochs helps;
  fifty drives the model *worse than untrained* on the very fact it just trained on.
- **More capacity does not help.** Rank 16 to 64 to 256 degrades monotonically.

Whether this is a property of the mechanism or of the model size is what the scaling curve
(1B / 3B / 8B, identical protocol and seeds, same hardware) is being run to answer.

## The question

Retrieval pays context tokens on every turn, forever. Weight memory pays a one-time update and
then answers with a near-empty context window. If that trade works, the token cost of memory
collapses. If it does not, it fails in a specific and measurable way: the model either does not
retain what you wrote into it, or it retains it and quietly loses something else.

Almost nobody publishes both halves. This measures both.

## Four arms, one reader

The reader model is **identical across all four arms**. Only the memory mechanism differs. That
is what makes it a controlled comparison rather than a demo.

Reader: **`unsloth/Llama-3.2-1B-Instruct`**, 131,072-token context.

The choice is forced by two constraints pulling opposite ways. Arm A must hold the entire
LongMemEval haystack, measured at **105,636 tokens median / 107,182 max**, so the reader needs
>110K of context. Arm D must LoRA that same model on a 24GB L4. That rules out the Qwen3 family
(40,960) and SmolLM3 (65,536). Of what is left, Llama-3.2-1B needs 3.3GB of KV cache at full
context where Phi-4-mini needs 13.7GB, and costs roughly a third of the prefill FLOPs that arm A
pays on every query.

| Arm | Mechanism | Context tokens/query |
|---|---|---|
| A. Full context | the entire haystack, in the window | ~106K |
| B. Grep | string search over the transcript | varies |
| C. Classical RAG | BM25 + dense + RRF | ~2-8K |
| D. Weight memory | frozen LM + LoRA updated online | ~0 |

Arm B exists because if weight memory cannot beat grep, the complexity is not earned.

## What gets measured

1. **Retention** - can it answer about facts ingested N episodes ago?
2. **Token cost per correct answer** - the whole point, made arithmetic.
3. **Forgetting** - held-out general capability as a function of online update count.
4. **The frontier** - LoRA rank against retention against degradation. A curve, not a number.

## Rules, from commit one

These are not a later cleanup. They are why the numbers will be worth reading.

**One tokenizer, no fallback.** Every arm counts with `harness/tokens.py`, which loads the real
reader tokenizer and *raises* if it cannot. There is no estimate path. The usual shortcut,
`words * 1.3`, is not a noisy approximation of the truth - measured on this repo's own samples it
lands between **0.79x and 3.46x** of the real count, erring in both directions depending on text
shape. It overshoots on prose and undershoots a JSON chunk by 3.5x. It cannot be calibrated away
and it does not cancel when you divide one arm's tokens by another's. The verdict survived a
reader swap: on Qwen3's vocabulary the same samples spanned 0.79x to 4.23x. Pinned in
`tests/test_tokens.py`, with the golden fingerprint that caught the swap.

**Three numbers on every accuracy report.** `answered_rate`, `accuracy | answered`, and
`accuracy over all` with abstentions counted wrong. Answered-only accuracy is trivially gamed:
an arm that answers one probe correctly and abstains on the other 299 reports 1.00.

**Provenance or no row.** Eight fields - reader model, revision, tokenizer fingerprint, arm,
split, seed, timestamp, corpus hash. `harness/ledger.append` raises rather than writing a row
nobody could reproduce.

**The ledger is read line by line.** A whole-file `json.loads` on a JSON-lines file dies on line
2, gets reported as one unreadable file, and inspects zero rows - silently exempting the exact
artifact the audit exists to police. A corrupt line here costs one row.

**Degenerate runs are caught.** An all-abstain run passes every other gate: provenance complete,
three numbers present, `accuracy_given_answered` correctly `None`. It is a well-formed
measurement of nothing. So is a run that never abstains.

**n >= 300 before any tail statistic.**

## Benchmarks

- **LongMemEval-S** for the token-cost axis. ~115K tokens/question, 500 instances. It fits
  inside a modern context window, which is what makes full-context a legitimate ceiling and
  makes 115K-versus-zero legible. Stated caveat: on this corpus a bigger window substitutes for
  memory, so it measures context-management efficiency as much as memory.
- **BEAM (128K split)** for the retention axis - the regime where memory is provably *not*
  substitutable by a bigger window.

## Layout

```
harness/tokens.py    one tokenizer, shared by every arm. No fallback.
harness/ledger.py    append-only JSONL, read line by line
harness/gates.py     the validity gates and the three-number rule
arms/                the four arms
data/                benchmark loaders
runs/ledger.jsonl    results
```

## Running

```
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest
```

First run downloads the reader tokenizer.
