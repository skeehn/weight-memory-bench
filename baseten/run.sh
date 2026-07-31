#!/usr/bin/env bash
# Remote entrypoint for the scaling run.
#
# COST SAFETY. Baseten bills H100 at $0.10833/min and the whole budget is $4.62, which is
# 42 minutes total. The `timeout` below is the hard stop: if anything hangs -- a stalled
# weight download, a wedged CUDA call, a model that will not load -- the job dies rather
# than quietly draining the account. 1500s is 25 minutes, about $2.71 worst case, leaving
# margin for one retry.
#
# The timeout is deliberately on the python process rather than trusted to the platform.
set -euo pipefail

MAX_SECONDS="${MAX_SECONDS:-1500}"

echo "=== installing deps ==="
pip install --quiet --no-cache-dir "transformers>=4.44" "peft>=0.11" "numpy>=1.26"

echo "=== gpu ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

OUT="${BT_CHECKPOINT_DIR:-.}/scaling_curve.json"
echo "=== scaling run (hard cap ${MAX_SECONDS}s) ==="

# `|| true` so a timeout still lets the partial JSON be reported below. scaling_curve.py
# writes its output file after every rung, so a kill at 8B keeps 1B and 3B.
timeout --signal=INT "${MAX_SECONDS}" \
  python -u -m scripts.scaling_curve \
    --sizes 1B 3B 8B \
    --seeds 5 \
    --out "${OUT}" || echo "=== run ended early (timeout or error) ==="

echo "=== results ==="
cat "${OUT}" 2>/dev/null || echo "no results file written"
