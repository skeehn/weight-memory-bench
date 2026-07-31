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
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

from arms.full_context import FullContextArm
from arms.grep import GrepArm
from arms.rag import RagArm
from data import longmemeval as lme
from harness import ledger
from harness.gates import check, score_response
from harness.tokens import READER_LADDER, shared

MODEL_ID = os.environ.get("WMB_BASETEN_MODEL_ID", "qzkme4kq")

# Addressed by deployment ID, never by environment. `truss push` creates a deployment that
# is NOT promoted to production, so /environments/production/predict routes to whatever was
# deployed before -- which has already, once, meant measuring the wrong build and getting
# entirely plausible numbers back.
DEPLOYMENT_ID = os.environ.get("WMB_BASETEN_DEPLOYMENT_ID")
if not DEPLOYMENT_ID:
    raise SystemExit("set WMB_BASETEN_DEPLOYMENT_ID to the deployment you intend to call")
ENDPOINT = f"https://model-{MODEL_ID}.api.baseten.co/deployment/{DEPLOYMENT_ID}/predict"

# Arm A ships ~400KB of context per item. Five keeps a request near 2MB; the retrieval arms
# are ~40x smaller and could batch far larger, but one knob is easier to reason about than
# three and the request overhead is not the bottleneck.
DEFAULT_BATCH = 5

# Answers survive a crash here. Only GPU time costs money; recomputing it does not.
CACHE_DIR = Path(__file__).resolve().parent.parent / "runs" / "answers"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def api_key() -> str:
    key = os.environ.get("BASETEN_API_KEY")
    if key:
        return key
    from pathlib import Path

    for line in Path.home().joinpath(".trussrc").read_text().splitlines():
        if line.strip().startswith("api_key"):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("no Baseten API key found")


def remote_generate(
    items: list[dict], size: str, key: str, timeout: int = 1800, attempts: int = 4
) -> list[dict]:
    """Send one batch, retrying transient network failures.

    A single `SSLV3_ALERT_BAD_RECORD_MAC` on a 2MB POST killed a run ten instances in and
    discarded the GPU time already paid for. Transport failures on multi-megabyte requests
    are ordinary, not exceptional, and a benchmark driver that treats them as fatal will
    keep losing paid work at random points.

    HTTPError is caught explicitly and FIRST, because it subclasses URLError -- so catching
    URLError alone silently retries server errors too. A 4xx means the request is wrong and
    will be wrong every time; retrying it just burns the clock. Only 5xx and 429 are worth
    another attempt.
    """
    payload = json.dumps({"action": "generate", "size": size, "items": items}).encode()
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                ENDPOINT,
                data=payload,
                headers={"Authorization": f"Api-Key {key}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())["results"]
        except urllib.error.HTTPError as exc:
            if exc.code < 500 and exc.code != 429:
                body = exc.read()[:300].decode("utf-8", "replace")
                raise RuntimeError(f"HTTP {exc.code} (not retryable): {body}") from exc
            last = exc
            reason = f"HTTP {exc.code}"
        except (ssl.SSLError, urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            reason = type(exc).__name__

        if attempt < attempts - 1:
            delay = 2**attempt
            print(f"\n  retryable failure ({reason}), retry in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"batch failed after {attempts} attempts: {last}")


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


def activate_and_wait(key: str, timeout: int = 900) -> None:
    """Wake the target deployment and block until it can serve.

    Necessary because `deactivate` is NOT scale-to-zero. A scaled-to-zero deployment wakes
    on request; a deactivated one is off and stays off, so calls against it fail as broken
    pipes rather than as anything diagnosable. The cost-safety teardown therefore makes the
    next run impossible unless it explicitly turns the deployment back on.
    """
    api = f"https://api.baseten.co/v1/models/{MODEL_ID}/deployments"

    def status() -> str:
        request = urllib.request.Request(api, headers={"Authorization": f"Api-Key {key}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            for dep in json.loads(response.read())["deployments"]:
                if dep["id"] == DEPLOYMENT_ID:
                    return dep["status"]
        return "MISSING"

    current = status()
    if current == "ACTIVE":
        return
    print(f"deployment {DEPLOYMENT_ID} is {current}; activating")
    try:
        post = urllib.request.Request(
            f"{api}/{DEPLOYMENT_ID}/activate",
            data=b"",
            headers={"Authorization": f"Api-Key {key}"},
        )
        with urllib.request.urlopen(post, timeout=60):
            pass
    except urllib.error.HTTPError as exc:
        if exc.code != 400:  # 400 == already active, which is fine
            raise

    deadline = time.time() + timeout
    while time.time() < deadline:
        current = status()
        if current == "ACTIVE":
            print("  active")
            return
        if current in {"BUILD_FAILED", "DEPLOY_FAILED", "FAILED", "MISSING"}:
            raise RuntimeError(f"deployment is {current}")
        time.sleep(15)
    raise RuntimeError(f"deployment did not become ACTIVE within {timeout}s")


def deactivate_all(key: str) -> None:
    """Shut every deployment on the model down.

    Registered with atexit, because scale_down_delay is 900s: finishing the run and simply
    returning still bills fifteen minutes of warm GPU. Every deployment, not just the one
    driven here -- a stale warm replica costs exactly as much as a live one.
    """
    api = f"https://api.baseten.co/v1/models/{MODEL_ID}/deployments"
    try:
        request = urllib.request.Request(api, headers={"Authorization": f"Api-Key {key}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            deployments = json.loads(response.read())["deployments"]
    except Exception as exc:
        print(f"  COULD NOT LIST DEPLOYMENTS ({exc}) -- SHUT DOWN MANUALLY at app.baseten.co")
        return

    still_billing = []
    for dep in deployments:
        # Already-inactive deployments return HTTP 400. Skipping them is not cosmetic:
        # previously the first 400 aborted the loop and printed "SHUT DOWN MANUALLY" even
        # though every replica was already off. A teardown that cries wolf stops being
        # believed, which is worse than not having one.
        if dep.get("active_replica_count", 0) == 0 and dep.get("status") == "INACTIVE":
            continue
        try:
            post = urllib.request.Request(
                f"{api}/{dep['id']}/deactivate",
                data=b"",
                headers={"Authorization": f"Api-Key {key}"},
            )
            with urllib.request.urlopen(post, timeout=60) as response:
                print(f"  deactivated {dep['id']} -> {response.status}")
        except Exception as exc:
            # One failure must not stop the others: every deployment left warm bills.
            print(f"  could not deactivate {dep['id']}: {exc}")
            still_billing.append(dep["id"])

    if still_billing:
        print(f"  STILL BILLING: {', '.join(still_billing)} -- shut down at app.baseten.co")


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
    parser.add_argument("--fresh", action="store_true",
                        help="ignore cached answers and re-run every instance")
    # Off by default: the safe thing must be the thing that happens when nobody thinks
    # about it. Only pass this when another run follows immediately.
    parser.add_argument("--keep-warm", action="store_true",
                        help="skip deactivation on exit (leaves the GPU billing)")
    args = parser.parse_args()

    key = api_key()
    activate_and_wait(key)
    if not args.keep_warm:
        import atexit

        atexit.register(deactivate_all, key)
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

        # Answers are cached to disk per batch and reloaded on start. GPU time is the only
        # thing here that costs money, so a crash at instance 90 must not re-buy the first
        # 89 -- which is exactly what happened when a transport error killed a run.
        cache_path = CACHE_DIR / f"{args.split}_{args.size}_{arm_name}_{corpus}.json"
        answers: dict[str, str] = {}
        if cache_path.exists() and not args.fresh:
            answers = json.loads(cache_path.read_text())
            print(f"  resuming with {len(answers)} cached answers")

        todo = [(inst, sel) for inst, sel in prepared if inst.question_id not in answers]
        for i in range(0, len(todo), args.batch):
            chunk = todo[i : i + args.batch]
            items = [
                {"id": inst.question_id, "context": sel.text, "question": inst.question}
                for inst, sel in chunk
            ]
            for row in remote_generate(items, args.size, key):
                answers[row["id"]] = row.get("text", "") if "error" not in row else ""
                if "error" in row:
                    print(f"  {row['id']}: {row['error'][:100]}")
            cache_path.write_text(json.dumps(answers))
            print(f"  {min(i + args.batch, len(todo))}/{len(todo)}", end="\r")

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
