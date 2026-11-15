#!/usr/bin/env bash
#
# The docs name concrete release assets so a reader can copy and paste. That
# makes them one more copy of the version -- and this project has already been
# bitten by version copies three times (four in the build, a stale symbol count,
# and the download examples after 0.1.0). This asserts they agree with VERSION.
#
#   ci/check-docs.sh        # no GPU, no ROCm, no network
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(head -n1 "$ROOT/VERSION" | tr -d '[:space:]')"
[[ -n "$VERSION" ]] || { echo "check-docs: VERSION is empty" >&2; exit 1; }

FAILURES=0
# Concrete asset names: pasrocm-shim-X.Y.Z-... and pasrocm-X.Y.Z-src.tar.gz
while IFS= read -r hit; do
    file="${hit%%:*}"
    rest="${hit#*:}"
    line="${rest%%:*}"
    text="${rest#*:}"
    found="$(printf '%s' "$text" \
        | grep -oE 'pasrocm(-shim)?-[0-9]+\.[0-9]+\.[0-9]+' \
        | head -1 | sed 's/^pasrocm-shim-//; s/^pasrocm-//')"
    [[ -n "$found" ]] || continue
    if [[ "$found" != "$VERSION" ]]; then
        printf 'FAIL %s:%s names %s, VERSION is %s\n' "$file" "$line" "$found" "$VERSION"
        FAILURES=$((FAILURES + 1))
    fi
done < <(grep -rnE 'pasrocm(-shim)?-[0-9]+\.[0-9]+\.[0-9]+' \
            "$ROOT/README.md" "$ROOT"/docs/*.md "$ROOT/SECURITY.md" 2>/dev/null \
         | sed "s|$ROOT/||")

if [[ "$FAILURES" -gt 0 ]]; then
    echo "check-docs: $FAILURES stale version reference(s); update them or use vX.Y.Z" >&2
    exit 1
fi
echo "== docs name release assets for $VERSION"
