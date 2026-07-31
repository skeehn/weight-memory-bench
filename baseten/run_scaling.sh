#!/usr/bin/env bash
# Drive the scaling measurement on the deployed Truss, then shut it off.
#
# COST SAFETY, which is the main thing this script is for. The deployment bills while a
# replica is warm, and `scale_down_delay` is 900s -- so doing nothing after the last call
# still costs 15 minutes of GPU. The trap deactivates on ANY exit path: success, error,
# Ctrl-C. A forgotten warm replica is the single most likely way to lose this budget, and
# it would happen silently.
set -uo pipefail

MODEL_ID="${MODEL_ID:-qzkme4kq}"
KEY="$(grep -m1 '^api_key' ~/.trussrc | sed 's/.*= *//')"
BASE="https://model-${MODEL_ID}.api.baseten.co/environments/production/predict"
API="https://api.baseten.co/v1/models/${MODEL_ID}"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/runs"
mkdir -p "$OUT_DIR"

SEEDS="${SEEDS:-5}"

deactivate_deployment() {
    echo
    echo "=== deactivating deployment (trap) ==="
    DEP=$(curl -s -H "Authorization: Api-Key $KEY" "${API}/deployments" \
        | python3 -c "import json,sys; print(json.load(sys.stdin)['deployments'][0]['id'])" 2>/dev/null)
    if [ -n "${DEP:-}" ]; then
        curl -s -X POST -H "Authorization: Api-Key $KEY" \
            "${API}/deployments/${DEP}/deactivate" -o /dev/null -w "  deactivate -> HTTP %{http_code}\n"
    else
        echo "  could not resolve deployment id; DEACTIVATE MANUALLY at app.baseten.co"
    fi
}
trap deactivate_deployment EXIT INT TERM

call() {
    curl -s --max-time 1800 -X POST "$BASE" \
        -H "Authorization: Api-Key $KEY" \
        -H "Content-Type: application/json" \
        -d "$1"
}

echo "=== gpu check ==="
call '{"action":"gpu"}' | tee "$OUT_DIR/gpu.json"
echo

for SIZE in 1B 3B 8B; do
    echo "=== rung ${SIZE} (seeds=${SEEDS}) ==="
    START=$(date +%s)
    call "{\"action\":\"run\",\"size\":\"${SIZE}\",\"seeds\":${SEEDS}}" \
        | tee "$OUT_DIR/rung_${SIZE}.json"
    echo
    echo "  ${SIZE} took $(( $(date +%s) - START ))s"
    echo
done

echo "=== fetching everything the server kept ==="
# The server persists each completed rung, so this recovers any result whose HTTP response
# was lost to a timeout. GPU time already paid for should not have to be paid for twice.
call '{"action":"fetch"}' | tee "$OUT_DIR/scaling_curve_remote.json"
echo
echo "results in $OUT_DIR"
