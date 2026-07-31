"""Record the scaling run into the append-only ledger.

The ledger is the repo's actual artifact: every run ever done, with enough provenance that
someone else can tell whether two rows are comparable. Results sitting in a loose JSON file
are not that -- they carry no tokenizer fingerprint, no corpus hash, and no record of which
seed produced which number.

**One row per (size, seed).** Aggregates cannot be decomposed, so a row per rung would make
`seed` a lie -- and seed is one of the eight required provenance fields precisely because a
result that cannot be traced to its initialization is not reproducible. Runs predating the
`per_seed_rates` fix have only aggregates, and are recorded as such with an explicit
`aggregate` marker rather than being given a fabricated seed.

    uv run python -m scripts.record_scaling runs/scaling_curve_remote.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from harness import ledger
from harness.tokens import Tokenizer


def tokenizer_fingerprint(model: str) -> str:
    """Fingerprint the rung's own tokenizer.

    Fetched per model rather than assumed shared. Llama 3.1 and 3.2 look like they use the
    same vocabulary, and they may well -- but 'looks the same' is what the fingerprint
    exists to replace.
    """
    return Tokenizer(model=model).fingerprint


def corpus_hash() -> str:
    from scripts.memory_gate import EPISODES, PROBES

    h = hashlib.sha256()
    for line in EPISODES:
        h.update(line.encode())
        h.update(b"\x00")
    for question, expected in PROBES:
        h.update(question.encode())
        h.update(b"\x00")
        h.update(expected.encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def rows_for(result: dict, corpus: str):
    """Yield ledger rows for one rung."""
    base = {
        "reader_model": result["model"],
        "reader_revision": "main",
        "tokenizer_fingerprint": tokenizer_fingerprint(result["model"]),
        "arm": "weight_memory",
        "split": "toy_gate_v2",
        "corpus_hash": corpus,
        "size": result["size"],
        "lora_rank": result.get("rank"),
        "learning_rate": result.get("lr"),
        "epochs": result.get("epochs"),
        "valid_probes": result["valid_probes"],
        "excluded_probes": [e["question"] for e in result.get("excluded", [])],
    }

    per_seed = result.get("per_seed_rates")
    if per_seed:
        for seed, rate in enumerate(per_seed):
            yield {**base, "seed": seed, "recall": rate, "aggregate": False}
        return

    # Pre-fix data: only aggregates survive. Recorded honestly rather than back-filled with
    # an invented seed, and flagged so nobody averages it against per-seed rows later.
    yield {
        **base,
        "seed": f"0-{result['seeds'] - 1}",
        "aggregate": True,
        "seeds": result["seeds"],
        "mean_recall": result["mean_recall"],
        "min_recall": result["min_recall"],
        "max_recall": result["max_recall"],
        "sd_recall": result["sd_recall"],
        "mean_nll_after": result.get("mean_nll_after"),
        "baseline_nll": result.get("baseline_nll"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = json.loads(Path(args.path).read_text())
    results = payload.get("results", payload)
    if isinstance(results, list):
        results = {r["size"]: r for r in results if "size" in r}

    corpus = corpus_hash()
    written = 0
    for size in sorted(results):
        result = results[size]
        if "error" in result or not result.get("valid_probes"):
            print(f"{size}: skipped ({result.get('error', 'no valid probes')})")
            continue
        for row in rows_for(result, corpus):
            if args.dry_run:
                missing = ledger.missing_provenance({**row, "timestamp_utc": "x"})
                print(f"  {size} seed={row['seed']} missing={missing or 'none'}")
            else:
                ledger.append(row)
                written += 1

    if args.dry_run:
        print("\ndry run; nothing written")
        return

    print(f"\nwrote {written} rows")
    audit = ledger.audit()
    print(f"audit: {audit['rows_read']} rows, clean={audit['clean']}")
    if audit["bad_lines"]:
        print(f"  bad lines: {audit['bad_lines']}")
    if audit["rows_missing_provenance"]:
        print(f"  incomplete: {audit['rows_missing_provenance']}")


if __name__ == "__main__":
    main()
