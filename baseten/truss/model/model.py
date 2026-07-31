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

    def predict(self, request: dict) -> dict:
        action = request.get("action", "run")

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
