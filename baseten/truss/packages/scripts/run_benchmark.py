"""Real accuracy on LongMemEval, with the three-number rule.

Everything reported so far for arms A-C is *evidence recall* -- did the arm retrieve the
turn containing the answer. That is a ceiling, not an accuracy: the reader still has to use
what it was handed, and a 1B model handed the right passage still gets things wrong. This
runs the actual questions through the actual reader and scores the answers.

**Selection is local, inference is remote.** Choosing what goes in the context window is
pure text manipulation and costs nothing, so it happens here. Only the selected context
travels to the GPU. That is why the 278MB corpus never has to be uploaded, and why arm A's
106K-token contexts are assembled on a laptop rather than on rented hardware.

**Abstention probes are scored inversely and never filtered out.** 30 of the 500 questions
have no answer in the haystack; declining is correct and answering is a fabrication. They
are what make `answered_rate` a measurement rather than a formality.

    uv run python -m scripts.run_benchmark --split dev --arms full_context grep rag
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

from arms.full_context import FullContextArm
from arms.grep import GrepArm
from arms.rag import RagArm
from data import longmemeval as lme
from harness import ledger
from harness.gates import check, score_response
from harness.tokens import READER_LADDER, shared

MODEL_ID = os.environ.get("WMB_BASETEN_MODEL_ID", "qzkme4kq")
ENDPOINT = f"https://model-{MODEL_ID}.api.baseten.co/environments/production/predict"

# Arm A ships ~400KB of context per item. Five keeps a request near 2MB; the retrieval arms
# are ~40x smaller and could batch far larger, but one knob is easier to reason about than
# three and the request overhead is not the bottleneck.
DEFAULT_BATCH = 5


def api_key() -> str:
    key = os.environ.get("BASETEN_API_KEY")
    if key:
        return key
    from pathlib import Path

    for line in Path.home().joinpath(".trussrc").read_text().splitlines():
        if line.strip().startswith("api_key"):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("no Baseten API key found")


def remote_generate(items: list[dict], size: str, key: str, timeout: int = 1800) -> list[dict]:
    payload = json.dumps({"action": "generate", "size": size, "items": items}).encode()
    request = urllib.request.Request(
        ENDPOINT,
        data=payload,
        headers={"Authorization": f"Api-Key {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())["results"]


def build_arm(name: str, budget: int, dense: bool):
    if name == "full_context":
        return FullContextArm()
    if name == "grep":
        return GrepArm(budget=budget)
    if name == "rag":
        embedder = None
        if dense:
            from arms.rag import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder()
        return RagArm(budget=budget, embedder=embedder)
    raise ValueError(f"unknown arm {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--arms", nargs="+", default=["full_context", "grep", "rag"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--size", default="1B", choices=list(READER_LADDER))
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()

    key = api_key()
    tk = shared()
    instances = lme.split(lme.load(), args.split)
    if args.limit:
        instances = instances[: args.limit]
    corpus = lme.corpus_hash(instances)

    print(f"split={args.split} n={len(instances)} size={args.size} corpus={corpus}\n")

    for arm_name in args.arms:
        arm = build_arm(arm_name, args.budget, args.dense)
        started = time.time()

        # Selection first, entirely local. Doing this up front means a failure in the
        # remote call cannot waste the selection work, and the token cost is known before
        # a single GPU second is spent.
        prepared = []
        for inst in instances:
            arm.prepare(inst)
            selection = arm.select(inst.question)
            prepared.append((inst, selection))
        select_s = time.time() - started
        median_tokens = sorted(s.tokens for _, s in prepared)[len(prepared) // 2]
        print(f"{arm_name}: selected in {select_s:.0f}s, median {median_tokens:,} tokens")

        answers: dict[str, str] = {}
        for i in range(0, len(prepared), args.batch):
            chunk = prepared[i : i + args.batch]
            items = [
                {"id": inst.question_id, "context": sel.text, "question": inst.question}
                for inst, sel in chunk
            ]
            for row in remote_generate(items, args.size, key):
                answers[row["id"]] = row.get("text", "") if "error" not in row else ""
                if "error" in row:
                    print(f"  {row['id']}: {row['error'][:100]}")
            print(f"  {min(i + args.batch, len(prepared))}/{len(prepared)}", end="\r")

        probes = [
            score_response(
                answers.get(inst.question_id, ""),
                inst.answer,
                is_abstention_probe=inst.is_abstention,
            )
            for inst, _ in prepared
        ]
        report = check(probes, tokenizer_fingerprint=tk.fingerprint)
        nums = report.numbers
        elapsed = time.time() - started

        # `accuracy_given_answered` is legitimately None when nothing was answered. That is
        # a valid state, not a zero, so it is printed as None rather than formatted.
        given = nums["accuracy_given_answered"]
        given_str = "None" if given is None else f"{given:.3f}"
        print(
            f"\n  answered_rate           {nums['answered_rate']:.3f}\n"
            f"  accuracy | answered     {given_str}\n"
            f"  accuracy over all       {nums['accuracy_over_all']:.3f}\n"
            f"  median context tokens   {median_tokens:,}\n"
            f"  reportable              {report.reportable}  ({elapsed:.0f}s)"
        )
        for gate in report.failures:
            print(f"    {gate.severity} {gate.name}: {gate.detail}")

        if not args.no_ledger and report.reportable:
            ledger.append(
                {
                    "reader_model": READER_LADDER[args.size],
                    "reader_revision": "main",
                    "tokenizer_fingerprint": tk.fingerprint,
                    "arm": arm_name,
                    "split": args.split,
                    "seed": 0,  # greedy decoding; no sampling seed to record
                    "corpus_hash": corpus,
                    "size": args.size,
                    "budget": args.budget,
                    "median_context_tokens": median_tokens,
                    **nums,
                    "gate_failures": [g.name for g in report.failures],
                }
            )
        print()


if __name__ == "__main__":
    main()
