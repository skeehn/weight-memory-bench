#!/usr/bin/env bash
# Per-rung learning-rate sweep: 3 sizes x 4 learning rates x 5 seeds.
#
# WHY THIS RUN EXISTS. The first scaling run held lr at 2e-3 across all three sizes and
# found recall falling with scale. That is not a scaling result: a rank-16 adapter is a
# proportionally much larger perturbation on 1B than on 8B, so identical hyperparameters
# are a much larger *effective* step at the small end. The same confound already forced a
# walk-back of the rank sweep. Letting each size find its own best lr and comparing bests
# is the only version of this comparison that means anything.
#
# COST SAFETY. The trap deactivates on every exit path. scale_down_delay is 900s, so
# finishing without deactivating still burns 15 minutes of warm GPU.
set -uo pipefail

MODEL_ID="${MODEL_ID:-qzkme4kq}"
KEY="$(grep -m1 '^api_key' ~/.trussrc | sed 's/.*= *//')"
API="https://api.baseten.co/v1/models/${MODEL_ID}"

# Target the deployment by ID, not by environment. `truss push` creates a NEW deployment
# that is NOT promoted to production, so /environments/production/predict routes to the
# previous one -- which here still carried the code that overwrote results across learning
# rates. Addressing the deployment explicitly makes it impossible to silently measure the
# wrong build.
DEPLOYMENT_ID="${DEPLOYMENT_ID:?set DEPLOYMENT_ID to the deployment you intend to call}"
BASE="https://model-${MODEL_ID}.api.baseten.co/deployment/${DEPLOYMENT_ID}/predict"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/runs/lr_sweep"
mkdir -p "$OUT_DIR"

SEEDS="${SEEDS:-5}"
SIZES=(1B 3B 8B)
LRS=(5e-4 1e-3 2e-3 5e-3)

deactivate_deployment() {
    echo
    echo "=== deactivating ALL deployments (trap) ==="
    # Every deployment, not just the one being driven. A stale deployment left warm bills
    # exactly the same as the current one, and there are now two on this model.
    curl -s -H "Authorization: Api-Key $KEY" "${API}/deployments" \
        | python3 -c "import json,sys; [print(d['id']) for d in json.load(sys.stdin)['deployments']]" 2>/dev/null \
        | while read -r DEP; do
            [ -z "$DEP" ] && continue
            curl -s -X POST -H "Authorization: Api-Key $KEY" \
                "${API}/deployments/${DEP}/deactivate" -o /dev/null \
                -w "  ${DEP} deactivate -> HTTP %{http_code}\n"
        done
}
trap deactivate_deployment EXIT INT TERM

call() {
    curl -s --max-time 1800 -X POST "$BASE" \
        -H "Authorization: Api-Key $KEY" -H "Content-Type: application/json" -d "$1"
}

echo "=== gpu ==="; call '{"action":"gpu"}'; echo; echo

for SIZE in "${SIZES[@]}"; do
    for LR in "${LRS[@]}"; do
        echo "=== ${SIZE} lr=${LR} seeds=${SEEDS} ==="
        START=$(date +%s)
        call "{\"action\":\"run\",\"size\":\"${SIZE}\",\"lr\":${LR},\"seeds\":${SEEDS}}" \
            | tee "$OUT_DIR/${SIZE}_lr${LR}.json" \
            | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
except Exception:
    print('  unparseable response'); raise SystemExit
if 'error' in d:
    print('  ERROR:', d['error'][:160])
else:
    print(f\"  recall {d['mean_recall']:.3f}  range {d['min_recall']:.2f}-{d['max_recall']:.2f}  \"
          f\"sd {d['sd_recall']:.3f}  probes {d['valid_probes']}  nll {d['mean_nll_after']:.2f}\")
" 2>/dev/null || echo "  (no summary)"
        echo "  took $(( $(date +%s) - START ))s"
        echo
    done
done

echo "=== fetching everything the server kept ==="
call '{"action":"fetch"}' > "$OUT_DIR/all.json"
echo "wrote $OUT_DIR/all.json"
