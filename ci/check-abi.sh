#!/usr/bin/env bash
#
# ABI gate for the shim.
#
#   ci/check-abi.sh --lib dist/work/<id>-<target>/libpasrocm_shim_<gfx>.so
#   ci/check-abi.sh --update        # refresh ci/abi-expected.txt from the spec
#
# Checks
#   0. src/rocprim/rocprim_shim_api.pas and rocprim_bindings.pas are current
#      with src/rocprim/rocprim_shim_abi.inc (ci/gen-shim-bindings.py --check)
#   1. src/rocprim/rocprim_shim_abi.inc  ==  ci/abi-expected.txt
#      (the declared contract may not drift without the golden list following)
#   2. every declared name is exported by the library under test
#
# The specification is the hand-edited file; the two Pascal units are generated
# from it. Checking the generator before the golden list means a stale unit is
# reported as stale, never as an ABI change.
#
# Extra rocprim_* exports are reported, and fatal only with --strict. They are
# the visible symptom of the missing -fvisibility=hidden described in
# DESIGN.md section 4; until the public header (phase 1) lands, the library
# exports more than it promises, so this is a warning by default rather than a
# gate that can never pass.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINDINGS="$ROOT/src/rocprim/rocprim_shim_abi.inc"
EXPECTED="$ROOT/ci/abi-expected.txt"
LIB=""
UPDATE=""
STRICT=""

usage() {
    cat <<'EOF'
usage: ci/check-abi.sh (--lib <path> | --update) [--strict]

  --lib <path>  shared library to inspect (default: the one in src/rocprim)
  --update      rewrite ci/abi-expected.txt from the Pascal declarations
  --strict      treat unexpected rocprim_* exports as errors
  -h, --help    this text
EOF
}

die() { echo "check-abi: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lib)    LIB="${2:?--lib needs a value}"; shift 2 ;;
        --update) UPDATE=1; shift ;;
        --strict) STRICT=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown argument: $1" ;;
    esac
done

[[ -f "$BINDINGS" ]] || die "missing $BINDINGS"

# Check 0: the generated units must match the specification. The generator is
# the last Python in the tree (DESIGN.md section 14.4); a Pascal machine without
# it can still run the export checks below, but CI always proves staleness.
if [[ -f "$ROOT/ci/gen-shim-bindings.py" ]] && command -v python3 >/dev/null 2>&1; then
    python3 "$ROOT/ci/gen-shim-bindings.py" --check \
        || die "generated units are stale; run 'python3 ci/gen-shim-bindings.py' and commit"
elif [[ -n "${CI:-}" ]]; then
    die "python3 is required in CI to prove the generated units are current"
else
    echo "check-abi: WARNING: python3 not found; skipping the generated-units staleness check" >&2
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

grep -oE '^function[[:space:]]+[A-Za-z_][A-Za-z0-9_]*' "$BINDINGS" \
    | awk '{print $2}' | LC_ALL=C sort -u >"$TMP/declared"
DECLARED_COUNT="$(wc -l <"$TMP/declared")"
EXTERNAL_COUNT="$(grep -cE 'external[[:space:]]+librocprimshim' "$BINDINGS" || true)"
if [[ "$DECLARED_COUNT" != "$EXTERNAL_COUNT" ]]; then
    echo "check-abi: WARNING: $DECLARED_COUNT function declarations but $EXTERNAL_COUNT 'external librocprimshim' clauses" >&2
fi

if [[ -n "$UPDATE" ]]; then
    cp "$TMP/declared" "$EXPECTED"
    echo "== wrote $EXPECTED ($DECLARED_COUNT names)"
    exit 0
fi

[[ -f "$EXPECTED" ]] || die "$EXPECTED is missing; run 'ci/check-abi.sh --update' and commit it"
if ! diff -u "$EXPECTED" "$TMP/declared" >"$TMP/contract.diff"; then
    echo "check-abi: FAIL: ci/abi-expected.txt does not match rocprim_bindings.pas" >&2
    sed 's/^/  /' "$TMP/contract.diff" >&2
    echo "check-abi: if the change is intended, run 'ci/check-abi.sh --update'" >&2
    exit 1
fi

if [[ -z "$LIB" && -z "$UPDATE" ]]; then
    # One library per target, so a checkout with exactly one build needs no
    # argument. More than one means the caller must say which.
    shopt -s nullglob
    CANDIDATES=("$ROOT"/src/rocprim/libpasrocm_shim_*.so)
    shopt -u nullglob
    case ${#CANDIDATES[@]} in
        1) LIB="${CANDIDATES[0]}" ;;
        0) usage >&2; die "no library in src/rocprim -- run 'cd src/rocprim && ./build.sh' first, or pass --lib" ;;
        *) usage >&2; die "${#CANDIDATES[@]} libraries in src/rocprim -- name one with --lib" ;;
    esac
fi
[[ -n "$LIB" ]] || { usage >&2; die "--lib is required (or pass --update)"; }
[[ -f "$LIB" ]] || die "no such library: $LIB"

nm -D --defined-only "$LIB" | awk '$3 ~ /^(rocprim|pasrocm)_/ {print $3}' \
    | LC_ALL=C sort -u >"$TMP/exported"

comm -23 "$TMP/declared" "$TMP/exported" >"$TMP/missing"
comm -13 "$TMP/declared" "$TMP/exported" >"$TMP/extra"

STATUS=0
if [[ -s "$TMP/missing" ]]; then
    echo "check-abi: FAIL: declared but not exported ($(wc -l <"$TMP/missing")):" >&2
    sed 's/^/  - /' "$TMP/missing" >&2
    STATUS=1
fi
if [[ -s "$TMP/extra" ]]; then
    if [[ -n "$STRICT" ]]; then
        echo "check-abi: FAIL (--strict): exported but not declared ($(wc -l <"$TMP/extra")):" >&2
        STATUS=1
    else
        echo "check-abi: WARN: exported but not declared ($(wc -l <"$TMP/extra")):" >&2
    fi
    sed 's/^/  + /' "$TMP/extra" >&2
fi

if [[ "$STATUS" -eq 0 ]]; then
    echo "== ABI ok: $DECLARED_COUNT declared names exported ($(wc -l <"$TMP/exported") shim symbols in the library)"
fi
exit "$STATUS"
