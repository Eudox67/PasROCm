#!/usr/bin/env bash
#
# Regression test for setup_env.sh's ROCm detection.
#
# A minimal ROCm 10.1 install ships amdclang++ and no hipcc. Checking only for
# hipcc made setup_env.sh blind to it -- reported by a user on 2026-11-15, and
# invisible on the development machine because hipcc happened to be installed.
# These synthetic trees reproduce both layouts without needing either SDK.
#
#   ci/check-env.sh          # no GPU, no ROCm, no network
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

die() { echo "check-env: $*" >&2; exit 1; }

make_tree() {
    local name="$1" kind="$2"
    local dir="$WORK/$name"
    mkdir -p "$dir/bin" "$dir/lib"
    case "$kind" in
        hipcc)      printf '#!/bin/sh\n' >"$dir/bin/hipcc"; chmod +x "$dir/bin/hipcc" ;;
        clang-lib)  mkdir -p "$dir/lib/llvm/bin"
                    printf '#!/bin/sh\n' >"$dir/lib/llvm/bin/amdclang++"
                    chmod +x "$dir/lib/llvm/bin/amdclang++" ;;
        clang-llvm) mkdir -p "$dir/llvm/bin"
                    printf '#!/bin/sh\n' >"$dir/llvm/bin/amdclang++"
                    chmod +x "$dir/llvm/bin/amdclang++" ;;
        runtime)    : >"$dir/lib/libamdhip64.so.7" ;;
        headers)    mkdir -p "$dir/include/hip" ;;
    esac
    printf '%s\n' "$dir"
}

# Run setup_env.sh against a tree with no inherited ROCm/HIP variables.
probe() {
    env -u HIP_CC -u HIP_DEVICE_LIB_PATH -u DEVICE_LIB_PATH \
        -u ROCM_HOME -u HIP_PATH -u HIP_CLANG_PATH \
        ROCM_PATH="$1" bash -c ". '$ROOT/setup_env.sh' >/dev/null 2>&1; printf '%s' \"\${HIP_CC:-}\""
}

FAILURES=0
expect() {
    local label="$1" tree="$2" want="$3" got
    got="$(probe "$tree")"
    if [[ "$got" == *"$want" ]]; then
        printf 'ok   %-34s -> %s\n' "$label" "$got"
    else
        printf 'FAIL %-34s -> %s (wanted *%s)\n' "$label" "$got" "$want"
        FAILURES=$((FAILURES + 1))
    fi
}

expect 'hipcc only (6.x/7.x)'      "$(make_tree t-hipcc hipcc)"      'hipcc'
expect 'amdclang++ in lib/llvm/bin' "$(make_tree t-clanglib clang-lib)" 'lib/llvm/bin/amdclang++'
expect 'amdclang++ in llvm/bin'     "$(make_tree t-clangllvm clang-llvm)" 'llvm/bin/amdclang++'
expect 'runtime only, no compiler'  "$(make_tree t-runtime runtime)" 'hipcc'

# A runtime-only install must warn, not fail: the test suite still works.
if env -u HIP_CC -u ROCM_HOME ROCM_PATH="$WORK/t-runtime" \
     bash -c ". '$ROOT/setup_env.sh' 2>&1 >/dev/null" | grep -q 'WARNING'; then
    echo 'ok   runtime-only install warns'
else
    echo 'FAIL runtime-only install did not warn'
    FAILURES=$((FAILURES + 1))
fi

[[ "$FAILURES" -eq 0 ]] || die "$FAILURES check(s) failed"
echo '== setup_env.sh detects hipcc and amdclang++ layouts'
