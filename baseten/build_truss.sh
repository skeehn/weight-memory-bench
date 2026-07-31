#!/usr/bin/env bash
# Assemble the deployable Truss by copying the measurement code into it.
#
# Regenerated rather than maintained: a hand-copied `packages/` drifts from the repo
# silently, and a scaling curve measured with stale code is worse than no curve. Run this
# before every push.
set -euo pipefail

# Usage: build_truss.sh [truss|truss-1b]
# truss     = all three models, for the scaling sweep
# truss-1b  = 1B only, ~2.5GB image instead of ~24GB, for everything else

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRUSS="$ROOT/baseten/${1:-truss}"
PKG="$TRUSS/packages"

rm -rf "$PKG"
mkdir -p "$PKG"

for module in harness arms data scripts; do
    cp -R "$ROOT/$module" "$PKG/$module"
done

# Strip caches so they cannot shadow the fresh copies inside the container.
find "$PKG" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$PKG" -name '*.pyc' -delete 2>/dev/null || true

echo "packaged into $PKG:"
du -sh "$PKG"/* | sed 's/^/  /'
echo
echo "total: $(du -sh "$TRUSS" | cut -f1)"
