#!/usr/bin/env bash
#
# Compile every example. Nothing else in the project did this: test.sh builds
# tests/TestRunner.pas and CI builds the shim, so the examples rotted unnoticed
# through a unit refactor -- 14 of 64 stopped compiling while the README still
# called them "compile-ready".
#
#   ci/check-examples.sh            compile and link (needs the ROCm libraries)
#   ci/check-examples.sh --no-link  compile only, no ROCm needed. This is what CI
#                                   runs: it catches missing units, types and
#                                   syntax, which is the rot that happened here.
#
# Neither mode runs the programs; that needs a GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXTRA=()
LABEL=""
if [[ "${1:-}" == "--no-link" ]]; then
    EXTRA=(-Cn)
    LABEL=" (compile only)"
fi

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

TOTAL=0
FAILED=0
for src in examples/*/*.pas; do
    [[ -f "$src" ]] || continue
    TOTAL=$((TOTAL + 1))
    if fpc @pasrocm.cfg "${EXTRA[@]}" -FU"$OUT" -FE"$OUT" "$src" >"$OUT/last.log" 2>&1; then
        continue
    fi
    FAILED=$((FAILED + 1))
    printf 'FAIL %s\n' "$src"
    grep -m3 -iE 'error|fatal|cannot find' "$OUT/last.log" | sed 's/^/     /'
done

if [[ "$FAILED" -gt 0 ]]; then
    echo "check-examples: $FAILED of $TOTAL examples failed to compile" >&2
    exit 1
fi
echo "== all $TOTAL examples compile$LABEL"
