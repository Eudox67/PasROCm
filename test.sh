#!/bin/bash
# ---------------------------------------------------------------------------
# PasROCm - build and run the FPCUnit test suite (no external build tool).
#
#   ./test.sh                            # full suite
#   ./test.sh --suite=TTest_rocSPARSE    # one suite
#   ./test.sh --all --format=plain       # explicit runner flags
#   SHIM_RESOLVE_ONLY=1 ./test.sh        # print which shim would be used, then stop
#
# The shim is built per GPU target (libpasrocm_shim_<gfx>.so). This script
# detects the GPU, builds that target if needed, and points the runtime loader
# at this directory with PASROCM_SHIM_DIR. Nothing links against the shim any
# more: rocprim_shim_loader.pas opens the matching file on the first rocPRIM
# call and refuses it if its identity does not match the device.
#
# Everything is derived from this script's location, so it works from any
# directory and from any checkout path.
# ---------------------------------------------------------------------------
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
SHIM_DIR="$ROOT/src/rocprim"

detect_gfx() {
    local gfx=""
    if [ -x "$ROCM_PATH/bin/rocm_agent_enumerator" ]; then
        gfx="$("$ROCM_PATH/bin/rocm_agent_enumerator" 2>/dev/null \
            | grep -E '^gfx[0-9]+$' | head -n1 || true)"
    fi
    if [ -z "$gfx" ] && command -v rocminfo >/dev/null 2>&1; then
        gfx="$(rocminfo 2>/dev/null \
            | awk '/Name:[[:space:]]+gfx[0-9]+/ { print $2; exit }' || true)"
    fi
    printf '%s' "$gfx"
}

GFX="$(detect_gfx)"
if [ -z "$GFX" ]; then
    echo "==> no AMD GPU detected; cannot choose a shim library." >&2
    echo "    Build one explicitly: (cd src/rocprim && GFX_ARCH=gfx1100 REQUIRE_ARCH=1 ./build.sh)" >&2
    exit 1
fi

SHIM_FILE="$SHIM_DIR/libpasrocm_shim_$GFX.so"
echo "==> GPU $GFX -> $SHIM_FILE"

if [ ! -f "$SHIM_FILE" ]; then
    echo "==> missing; building it now (needs a ROCm compiler)"
    ( cd "$SHIM_DIR" && GFX_ARCH="$GFX" REQUIRE_ARCH=1 ./build.sh )
fi

# Nothing links against the shim: rocprim_shim_loader.pas resolves
# libpasrocm_shim_<gfx>.so at run time. PASROCM_SHIM_DIR names the exact
# directory that holds it, so the loader never has to guess.
export PASROCM_SHIM_DIR="$SHIM_DIR"

if [ -n "${SHIM_RESOLVE_ONLY:-}" ]; then
    echo "==> loader will open: $SHIM_FILE"
    exit 0
fi

export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$ROOT/bin/units"

echo "==> Compiling tests/TestRunner.pas with FPC $(fpc -iV)"
fpc -Mobjfpc -O1 \
    -Fu"$ROOT/src" -Fu"$ROOT/src/rocprim" -Fu"$ROOT/tests" \
    -I"$ROOT/src" \
    -Fl"$ROCM_PATH/lib" -Fl"$ROCM_PATH/lib64" \
    -k"-rpath=$ROCM_PATH/lib:$ROCM_PATH/lib64" \
    -FU"$ROOT/bin/units" -FE"$ROOT/bin" \
    "$ROOT/tests/TestRunner.pas"

echo "==> Running bin/TestRunner $*"
cd "$ROOT"
exec ./bin/TestRunner "$@"
