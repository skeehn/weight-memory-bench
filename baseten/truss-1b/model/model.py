"""Baseten deployment that runs the weight-memory scaling measurement.

This is a measurement wearing a deployment's clothes. Baseten Training is not enabled on
this account, but dedicated deployments are, and a deployment is just a container on a GPU
that runs code when poked. So `predict` runs one rung of the scaling ladder and returns
the numbers.

**One rung per call.** The three sizes together take longer than any sensible HTTP request.
Splitting them keeps each call short enough to survive, and means a failure at 8B does not
discard 1B and 3B.

**Every completed rung is written to disk before it is returned.** If the client times out
waiting, the GPU time is not lost -- a follow-up `{"action": "fetch"}` returns everything
that finished. Losing a completed measurement to a dropped connection would be paying twice
for the same number.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

RESULTS_DIR = Path("/tmp/wmb_results")


class Model:
    def __init__(self, **kwargs):
        self._loaded = False

    def load(self):
        # Nothing is loaded up front. Each rung builds its own reader and tears it down, so
        # holding a model here would only waste VRAM the next rung needs -- and at 8B on a
        # 24GB card there is no headroom to waste.
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self._loaded = True

    def _saved(self) -> dict:
        out = {}
        for path in sorted(RESULTS_DIR.glob("*.json")):
            try:
                out[path.stem] = json.loads(path.read_text())
            except json.JSONDecodeError:
                out[path.stem] = {"error": "unreadable result file"}
        return out

    def _reader(self, size: str):
        """Cache one reader per size.

        Reloading a model per request would dominate a 400-instance run completely -- the
        weights take longer to load than the generation takes to run.
        """
        if not hasattr(self, "_readers"):
            self._readers = {}
        if size not in self._readers:
            from harness.reader import Reader
            from harness.tokens import READER_LADDER

            self._readers[size] = Reader(model=READER_LADDER.get(size, size))
            self._readers[size].model  # force load now, not mid-batch
        return self._readers[size]

    def predict(self, request: dict) -> dict:
        action = request.get("action", "run")

        if action == "generate":
            # Batched inference for the accuracy run. Context selection happens on the
            # client -- it is pure text manipulation and costs nothing -- so the 278MB
            # corpus never has to be shipped here. Only the selected context travels.
            size = request.get("size", "1B")
            items = request.get("items") or []
            max_new = int(request.get("max_new_tokens", 32))
            reader = self._reader(size)

            out = []
            for item in items:
                try:
                    gen = reader.generate(
                        item.get("context", ""),
                        item.get("question", ""),
                        max_new_tokens=max_new,
                    )
                    out.append(
                        {
                            "id": item.get("id"),
                            "text": gen.text,
                            "prompt_tokens": gen.prompt_tokens,
                            "generated_tokens": gen.generated_tokens,
                        }
                    )
                except Exception as exc:
                    # One bad instance must not discard the rest of an expensive batch.
                    out.append({"id": item.get("id"), "error": f"{type(exc).__name__}: {exc}"})
            return {"size": size, "count": len(out), "results": out}

        if action == "sweep":
            # lr x step-budget sweep with augmentation, the Stage 1 experiment.
            from scripts.stage1_sweep import main as sweep_main
            import sys

            argv = ["stage1_sweep"]
            for key in ("facts", "seeds", "generated"):
                if key in request:
                    argv += [f"--{key}", str(request[key])]
            if request.get("lrs"):
                argv += ["--lrs"] + [str(x) for x in request["lrs"]]
            if request.get("epochs"):
                argv += ["--epochs"] + [str(x) for x in request["epochs"]]
            argv += ["--out", str(RESULTS_DIR / "sweep.json")]
            old, sys.argv = sys.argv, argv
            try:
                sweep_main()
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-2500:]}
            finally:
                sys.argv = old
            path = RESULTS_DIR / "sweep.json"
            return json.loads(path.read_text()) if path.exists() else {"error": "no output"}

        if action == "alphaedit":
            from scripts.alphaedit_experiment import run as alphaedit_run

            try:
                result = alphaedit_run(
                    n_facts=int(request.get("facts", 20)),
                    threshold=float(request.get("threshold", 1e-4)),
                    edit_steps=int(request.get("edit_steps", 25)),
                    edit_lr=float(request.get("edit_lr", 0.5)),
                )
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-2500:]}
            (RESULTS_DIR / "alphaedit.json").write_text(json.dumps(result, indent=2))
            return result

        if action == "forgetting":
            from scripts.forgetting_curve import run_curve

            try:
                result = run_curve(
                    checkpoints=request.get("checkpoints", [0, 10, 25, 50, 100]),
                    rank=int(request.get("rank", 16)),
                    lr=float(request.get("lr", 2e-3)),
                    seed=int(request.get("seed", 0)),
                    size=request.get("size", "1B"),
                )
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]}
            (RESULTS_DIR / f"forgetting_{result['size']}_seed{result['seed']}.json").write_text(
                json.dumps(result, indent=2)
            )
            return result

        if action == "fetch":
            return {"action": "fetch", "results": self._saved()}

        if action == "gpu":
            import torch

            return {
                "cuda": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "total_memory_gb": (
                    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
                    if torch.cuda.is_available()
                    else None
                ),
            }

        size = request.get("size", "1B")
        seeds = int(request.get("seeds", 5))
        rank = int(request.get("rank", 16))
        lr = float(request.get("lr", 2e-3))
        epochs = int(request.get("epochs", 10))

        from harness.tokens import READER_LADDER
        from scripts.scaling_curve import run_one

        model_name = READER_LADDER.get(size, size)
        try:
            result = run_one(size, model_name, seeds, rank, lr, epochs)
        except Exception as exc:
            # Returned rather than raised, so an OOM at 8B produces a recorded negative
            # result instead of an opaque 500 that costs a second run to diagnose.
            return {
                "size": size,
                "model": model_name,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:],
            }

        result["rank"] = rank
        result["lr"] = lr
        result["epochs"] = epochs

        # Keyed by the full configuration, not just the size. Keying on size alone meant
        # four learning rates at one size silently overwrote each other, leaving only the
        # last -- and the whole point of the sweep is comparing them.
        key = f"{size}_lr{lr:g}_r{rank}_e{epochs}"
        (RESULTS_DIR / f"{key}.json").write_text(json.dumps(result, indent=2))

        # Free the GPU before the next rung is requested. Without this, 1B's allocator
        # blocks still hold memory when 8B tries to load onto the same 24GB card.
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

        return result
