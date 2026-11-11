#!/bin/bash
# ---------------------------------------------------------------------------
# PasROCm - build and run the FPCUnit test suite (no external build tool).
#
#   ./test.sh                            # full suite
#   ./test.sh --suite=TTest_rocSPARSE    # one suite
#   ./test.sh --all --format=plain       # explicit runner flags
#
# Everything is derived from this script's location, so it works from any
# directory and from any checkout path.
# ---------------------------------------------------------------------------
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
SHIM_DIR="$ROOT/src/rocprim"

export LD_LIBRARY_PATH="$SHIM_DIR:$ROCM_PATH/lib:$ROCM_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# The rocPRIM layer talks to the GPU through librocprim_shim.so (hipcc build).
if [ ! -f "$SHIM_DIR/librocprim_shim.so" ]; then
    echo "==> librocprim_shim.so missing - building it now (needs hipcc)"
    ( cd "$SHIM_DIR" && ./build.sh )
fi

mkdir -p "$ROOT/bin/units"

echo "==> Compiling tests/TestRunner.pas with FPC $(fpc -iV)"
fpc -Mobjfpc -O1 \
    -Fu"$ROOT/src" -Fu"$ROOT/src/rocprim" -Fu"$ROOT/tests" \
    -I"$ROOT/src" \
    -Fl"$SHIM_DIR" -Fl"$ROCM_PATH/lib" -Fl"$ROCM_PATH/lib64" \
    -k"-rpath=$SHIM_DIR:$ROCM_PATH/lib:$ROCM_PATH/lib64" \
    -FU"$ROOT/bin/units" -FE"$ROOT/bin" \
    "$ROOT/tests/TestRunner.pas"

echo "==> Running bin/TestRunner $*"
cd "$ROOT"
exec ./bin/TestRunner "$@"
