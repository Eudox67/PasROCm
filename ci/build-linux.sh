#!/usr/bin/env bash
#
# Build (and package) exactly one (configuration, GPU target) pair.
#
#   ci/build-linux.sh --config ubuntu-24.04-rocm10.1 --target gfx1100
#   ci/build-linux.sh --config ubuntu-24.04-rocm10.1 --target gfx942 --skip-install
#
# Steps, in order: provision the pinned SDK (unless --skip-install), build one
# library, assert the library's own identity against the manifest, ABI-check it,
# package it. The identity assertion is what upgrades "it compiled" into "it
# compiled with the parameters the matrix asked for".
#
# Layout:
#   dist/work/<config>-<gfx>/libpasrocm_shim_<gfx>.so
#   dist/work/<config>-<gfx>/pasrocm-shim-info
#   dist/work/<config>-<gfx>/build-info.env
#   dist/pasrocm-shim-<version>-linux-x86_64-rocm-<line>-<gfx>.tar.gz
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG=""
TARGET=""
SKIP_INSTALL=""
DO_PACKAGE=1
# Phase 1 closed the surface: with -fvisibility=hidden an unexpected export is a
# defect, so the strict ABI gate is the default. --allow-extra-exports exists
# for local debugging only and must never appear in CI.
STRICT_ABI=1
DRY_RUN=""

usage() {
    cat <<'EOF'
usage: ci/build-linux.sh --config <id> --target <gfxNNNN> [options]

  --config <id>        configuration id from ci/supported-builds.json
  --target <gfx>       one GPU target (one library per target, by design)
  --skip-install       do not provision ROCm (already installed)
  --no-package         build, self-test and ABI-check only
  --allow-extra-exports  do not fail on unexpected rocprim_* exports (debugging)
  --dry-run            resolve everything and print, build nothing
  -h, --help           this text
EOF
}

die() { echo "build-linux: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)     CONFIG="${2:?--config needs a value}"; shift 2 ;;
        --target)     TARGET="${2:?--target needs a value}"; shift 2 ;;
        --skip-install) SKIP_INSTALL=1; shift ;;
        --no-package) DO_PACKAGE=0; shift ;;
        --allow-extra-exports) STRICT_ABI=""; shift ;;
        --dry-run)    DRY_RUN=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *)            usage >&2; die "unknown argument: $1" ;;
    esac
done

[[ -n "$CONFIG" ]] || { usage >&2; die "--config is required"; }
[[ -n "$TARGET" ]] || { usage >&2; die "--target is required (one gfx target per build)"; }

set -a
eval "$("$ROOT/ci/manifest.sh" env --id "$CONFIG")"
set +a

case " $CFG_TARGETS " in
    *" $TARGET "*) ;;
    *) die "configuration $CFG_ID does not support target '$TARGET' (has: $CFG_TARGETS)" ;;
esac

WAVEFRONT="$("$ROOT/ci/manifest.sh" wave --target "$TARGET")"
WIDTHS_MASK="$("$ROOT/ci/manifest.sh" mask --target "$TARGET")"
LIB_FILE="${CFG_LIBRARY//\{gfx\}/$TARGET}"
WORK="$ROOT/dist/work/$CFG_ID-$TARGET"
mkdir -p "$WORK"

GIT_COMMIT="${GIT_COMMIT:-$(git -C "$ROOT" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)}"

echo "== config      : $CFG_ID ($CFG_MODE)"
echo "== SDK         : $CFG_SDK_LINE line, pinned $CFG_SDK_VERSION"
echo "== compiler    : $CFG_COMPILER_KIND"
echo "== target      : $TARGET  (wavefront $WAVEFRONT, widths mask $WIDTHS_MASK)"
echo "== library     : $LIB_FILE"
echo "== covers      : $CFG_COVERS"
echo "== work dir    : $WORK"

INSTALL_FLAGS=()
[[ -n "$DRY_RUN" ]] && INSTALL_FLAGS+=(--dry-run)

if [[ -z "$SKIP_INSTALL" ]]; then
    "$ROOT/ci/install-rocm.sh" --config "$CFG_ID" "${INSTALL_FLAGS[@]}"
else
    echo "== ROCm install skipped (--skip-install)"
fi

if [[ -n "$DRY_RUN" ]]; then
    echo "== DRY_RUN: would build $TARGET into $WORK and self-test the result"
    exit 0
fi

echo "== building ======================================================="
GFX_ARCH="$TARGET" \
REQUIRE_ARCH=1 \
COMPILER_KIND="$CFG_COMPILER_KIND" \
OUT_DIR="$WORK" \
VERSION="$CFG_SHIM_VERSION" \
GIT_COMMIT="$GIT_COMMIT" \
ROCM_VERSION="$CFG_SDK_VERSION" \
    "$ROOT/src/rocprim/build.sh"

LIB="$WORK/$LIB_FILE"
[[ -f "$LIB" ]] || die "build produced no $LIB"

echo "== identity self-test ============================================="
# Host-only: the probe makes no HIP calls, so this runs on a GPU-less runner.
# It proves the library carries the target/wavefront/SDK the matrix asked for,
# and unit-tests the pure compatibility predicate (including the mismatch
# branch that no available hardware can reach).
ROCM_ROOT="${ROCM_PATH:-/opt/rocm}"
LD_LIBRARY_PATH="$WORK:$ROCM_ROOT/lib:$ROCM_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH

cc -std=c99 -O1 -Wall -Wextra \
    -o "$WORK/pasrocm-shim-info" \
    "$ROOT/src/rocprim/pasrocm_shim_info.c" \
    -L"$WORK" -l:"$LIB_FILE" -Wl,-rpath,"$WORK" \
    || die "could not link the identity probe against $LIB_FILE"

INFO="$("$WORK/pasrocm-shim-info")" || {
    printf '%s\n' "$INFO" >&2
    die "identity probe failed (compatibility self-test disagreed)"
}
printf '%s\n' "$INFO" | sed 's/^/   /'

expect() {
    printf '%s\n' "$INFO" | grep -qx "$1" || die "shim identity mismatch: expected '$1'"
}
expect "target=$TARGET"
expect "wavefront=$WAVEFRONT"
expect "widths_mask=$WIDTHS_MASK"
expect "rocm=$CFG_SDK_VERSION"
ABI_EXPECTED="$(sed -n 's/^#define PASROCM_SHIM_ABI_VERSION //p' "$ROOT/src/rocprim/rocprim_shim.h")"
[[ -n "$ABI_EXPECTED" ]] || die "cannot read PASROCM_SHIM_ABI_VERSION from rocprim_shim.h"
expect "abi=$ABI_EXPECTED"
echo "== identity ok: $TARGET / wave$WAVEFRONT / masks $WIDTHS_MASK / rocm $CFG_SDK_VERSION"

echo "== ABI check ======================================================"
ABI_FLAGS=()
[[ -n "$STRICT_ABI" ]] && ABI_FLAGS+=(--strict)
"$ROOT/ci/check-abi.sh" --lib "$LIB" "${ABI_FLAGS[@]}"

if [[ "$DO_PACKAGE" == "1" && "$CFG_ARTIFACT" == "1" ]]; then
    echo "== packaging ======================================================"
    "$ROOT/ci/package.sh" \
        --config "$CFG_ID" \
        --target "$TARGET" \
        --lib "$LIB" \
        --build-info "$WORK/build-info.env" \
        --out "$ROOT/dist"
elif [[ "$CFG_ARTIFACT" != "1" ]]; then
    echo "== compile probe passed: $CFG_ID -> $TARGET (no artifact by design)"
else
    echo "== packaging skipped (--no-package)"
fi

echo "== done: $CFG_ID / $TARGET"

