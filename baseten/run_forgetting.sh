#!/usr/bin/env bash
# Forgetting curve on the deployed 1B model.
#
# COST SAFETY VIA TRAP, NOT VIA THE LAST LINE. An earlier attempt at this measurement put
# the teardown at the end of an inline command; the process was killed mid-run and the
# teardown never executed, leaving a replica ACTIVE and billing until it was noticed. A
# trap fires on EXIT, INT and TERM, so a kill still shuts the GPU down.
#
# This pattern already existed in run_lr_sweep.sh. Not reusing it cost real money.
set -uo pipefail

MODEL_ID="${MODEL_ID:-qvv7nrgq}"
DEPLOYMENT_ID="${DEPLOYMENT_ID:?set DEPLOYMENT_ID}"
KEY="$(grep -m1 '^api_key' ~/.trussrc | sed 's/.*= *//')"
API="https://api.baseten.co/v1/models/${MODEL_ID}"
BASE="https://model-${MODEL_ID}.api.baseten.co/deployment/${DEPLOYMENT_ID}/predict"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/runs"
mkdir -p "$OUT_DIR"

teardown() {
    echo
    echo "=== teardown (trap) ==="
    curl -s -H "Authorization: Api-Key $KEY" "${API}/deployments" \
        | python3 -c "
import json,sys
for d in json.load(sys.stdin)['deployments']:
    if d['active_replica_count'] or d['status'] not in ('INACTIVE',):
        print(d['id'])
" 2>/dev/null \
        | while read -r DEP; do
            [ -z "$DEP" ] && continue
            curl -s -X POST -H "Authorization: Api-Key $KEY" \
                "${API}/deployments/${DEP}/deactivate" -o /dev/null \
                -w "  ${DEP} -> HTTP %{http_code}\n"
        done
}
trap teardown EXIT INT TERM

status() {
    curl -s -H "Authorization: Api-Key $KEY" "${API}/deployments" \
        | python3 -c "
import json,sys
d=[x for x in json.load(sys.stdin)['deployments'] if x['id']=='${DEPLOYMENT_ID}']
print(d[0]['status'] if d else 'MISSING')" 2>/dev/null
}

echo "=== activating ${DEPLOYMENT_ID} ==="
if [ "$(status)" != "ACTIVE" ]; then
    curl -s -X POST -H "Authorization: Api-Key $KEY" \
        "${API}/deployments/${DEPLOYMENT_ID}/activate" -o /dev/null -w "  activate -> HTTP %{http_code}\n"
fi
for _ in $(seq 1 90); do
    S="$(status)"
    [ "$S" = "ACTIVE" ] && break
    echo "  $(date +%H:%M:%S) $S"
    sleep 20
done
[ "$(status)" != "ACTIVE" ] && { echo "never became ACTIVE"; exit 1; }
echo "  ACTIVE"

echo
echo "=== forgetting curve (seed ${SEED:-0}) ==="
curl -s --max-time 3000 -X POST "$BASE" \
    -H "Authorization: Api-Key $KEY" -H "Content-Type: application/json" \
    -d "{\"action\":\"forgetting\",\"checkpoints\":[0,5,10,25,50,100],\"seed\":${SEED:-0}}" \
    | tee "$OUT_DIR/forgetting_curve_remote.json" \
    | python3 -c "
import json,sys
d=json.load(sys.stdin)
if 'error' in d:
    print('ERROR:', d['error'][:200]); raise SystemExit
print(f\"base held-out ppl {d['base_heldout_ppl']:.2f}\")
print(f\"{'updates':>8} {'retention':>10} {'capability':>11} {'ppl_ratio':>10}\")
print('-'*44)
for r in d['rows']:
    print(f\"{r['updates']:>8} {r['retention']:>10.3f} {r['capability']:>11.3f} {r['ppl_ratio']:>9.2f}x\")
" 2>/dev/null || echo "  (could not parse response; raw JSON saved)"
echo
echo "saved to $OUT_DIR/forgetting_curve_remote.json"
