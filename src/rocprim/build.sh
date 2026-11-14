#!/usr/bin/env bash
#
# Version-generic builder for libpasrocm_shim_<gfx>.so
#
#   ./build.sh                                  # auto-detect ROCm and the current GPU(s)
#   ROCM_PATH=/opt/rocm/core-10.1 ./build.sh    # force a specific install
#   GFX_ARCH=gfx1100 ./build.sh                 # one target
#   GFX_ARCH='rdna3;cdna3' ./build.sh           # fat binary across families
#   GFX_ARCH=rdna4 REQUIRE_ARCH=1 ./build.sh    # CI: never infer, fail if unresolved
#   COMPILER_KIND=hipcc ./build.sh              # hipcc (ROCm 6.x / 7.x)
#   COMPILER_KIND=clang ./build.sh              # amdclang++ -x hip (ROCm 10.x)
#   OUT_DIR=/tmp/shim ./build.sh                # keep src/ clean
#   VERSION=0.1.0 GIT_COMMIT=abc1234 ./build.sh # stamp the binary
#   DRY_RUN=1 ./build.sh                        # print the resolved toolchain only
#   NO_STRIP=1 ./build.sh                       # keep debug info
#
# Environment contract:
#   ROCM_PATH      ROCm install root (default: /opt/rocm, then the newest
#                  /opt/rocm-* that has a compiler; header-only leftovers are
#                  never selected)
#   COMPILER_KIND  hipcc | clang. Default: hipcc when the install ships one,
#                  otherwise clang. ROCm 10.x deprecates hipcc, and modern hipcc
#                  is itself a wrapper around `clang++ -x hip`.
#   COMPILER       explicit compiler path; its name selects the kind when
#                  COMPILER_KIND is not given
#   GFX_ARCH       exactly one GPU target: a raw gfx id (gfx1100) or a family
#                  keyword that expands to exactly one target. Family keywords
#                  that expand to several targets are rejected with guidance.
#   REQUIRE_ARCH   non-empty => exit non-zero when no target resolves. CI sets
#                  this because compiler acceptance of a gfx id does not
#                  establish runtime support, and a silent fallback to
#                  "whatever rocm_agent_enumerator says" is not reproducible.
#   OUT_DIR        output directory (default: this script's directory)
#   VERSION        stamped as -DPASROCM_SHIM_VERSION
#   GIT_COMMIT     stamped as -DPASROCM_GIT_COMMIT
#   ROCM_VERSION   stamped as -DPASROCM_ROCM_VERSION (read back by
#                  pasrocm_shim_rocm_version(); defaults to the HIP version)
#   NO_STRIP       keep symbols
#   DRY_RUN        print the toolchain and command, compile nothing
#
# ONE TARGET PER BUILD. The shim is built for exactly one GPU target and named
# after it: libpasrocm_shim_<gfx>.so. A family keyword such as `rdna3` is
# accepted only as an error with guidance -- it expands to several targets, and
# this project ships one small library per architecture instead of a fat binary.
#
# The wavefront width is NOT guessed from preprocessor macros. It is looked up
# in targets.conf (the single mapping table) and injected as
# -DPASROCM_WAVEFRONT_SIZE, because the launch geometry lives in host code that
# is compiled once and cannot know the device. The shim refuses to build without
# it, and refuses to run when the device disagrees.
#
# Compiles BOTH translation units; the resulting per-target library lands in
# OUT_DIR (default: next to the Pascal units in src/rocprim/, which is where
# pasrocm.cfg, test.sh and setup_env.sh expect it).
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$SRC_DIR}"
SOURCES=(rocprim_shim.cpp rocprim_shim_math_kernels.cpp)
# Default on its own line: a '}' inside ${VAR:-default} would close the
# expansion early, truncating the name to "libpasrocm_shim_{gfx".
LIB_PATTERN="${LIB_PATTERN:-}"
[[ -n "$LIB_PATTERN" ]] || LIB_PATTERN='libpasrocm_shim_{gfx}.so'
VERSION="${VERSION:-0.1.0}"
GIT_COMMIT="${GIT_COMMIT:-$(git -C "$SRC_DIR" rev-parse --short=12 HEAD 2>/dev/null || true)}"

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "== $*"; }


# --- 1. Locate a ROCm root -------------------------------------------
has_compiler() {
    local root="$1"
    [[ -x "$root/bin/hipcc" ]] ||
    [[ -x "$root/lib/llvm/bin/amdclang++" ]] ||
    [[ -x "$root/llvm/bin/amdclang++" ]]
}

find_rocm_root() {
    if [[ -n "${ROCM_PATH:-}" ]]; then
        echo "$ROCM_PATH"; return
    fi
    # /opt/rocm is normally a symlink into the real versioned tree
    # (/opt/rocm/bin -> /etc/alternatives/rocm-bin -> .../core-10.1/bin),
    # so this single check covers the whole install.
    if has_compiler /opt/rocm; then
        echo "/opt/rocm"; return
    fi
    # Fallback 1: ROCm 10.x tree without the alternatives symlinks
    # (/opt/rocm/core-10.1). Fallback 2: 6.x/7.x versioned installs.
    local candidate
    while IFS= read -r candidate; do
        has_compiler "$candidate" && { echo "$candidate"; return; }
    done < <(compgen -G '/opt/rocm/core-[0-9]*' | sort -Vr)
    while IFS= read -r candidate; do
        has_compiler "$candidate" && { echo "$candidate"; return; }
    done < <(compgen -G '/opt/rocm-[0-9]*' | sort -Vr)
}

ROCM_ROOT="$(find_rocm_root)"
[[ -n "$ROCM_ROOT" ]] || die "no ROCm install with a compiler found. Set ROCM_PATH=/path/to/rocm and retry."
has_compiler "$ROCM_ROOT" || die "$ROCM_ROOT has no bin/hipcc and no lib/llvm/bin/amdclang++"
[[ -d "$ROCM_ROOT/include/hip" ]] || die "$ROCM_ROOT/include/hip is missing (install the HIP development package)"

LIB_DIR="$ROCM_ROOT/lib64"
[[ -f "$LIB_DIR/libamdhip64.so" ]] || LIB_DIR="$ROCM_ROOT/lib"
[[ -f "$LIB_DIR/libamdhip64.so" ]] || die "libamdhip64.so not found under $ROCM_ROOT (install the HIP runtime development package)"

# --- 2. Resolve the compiler -----------------------------------------
HIPCC_PATH=""
AMDCLANG_PATH=""
[[ -x "$ROCM_ROOT/bin/hipcc" ]] && HIPCC_PATH="$ROCM_ROOT/bin/hipcc"
for candidate in "$ROCM_ROOT/bin/amdclang++" \
                 "$ROCM_ROOT/lib/llvm/bin/amdclang++" \
                 "$ROCM_ROOT/llvm/bin/amdclang++"; do
    if [[ -z "$AMDCLANG_PATH" && -x "$candidate" ]]; then
        AMDCLANG_PATH="$candidate"
    fi
done

if [[ -n "${COMPILER:-}" ]]; then
    CC_BIN="$COMPILER"
    if [[ -z "${COMPILER_KIND:-}" ]]; then
        case "$(basename "$CC_BIN")" in
            *hipcc*) COMPILER_KIND=hipcc ;;
            *)       COMPILER_KIND=clang ;;
        esac
    fi
else
    if [[ -z "${COMPILER_KIND:-}" ]]; then
        if [[ -n "$HIPCC_PATH" ]]; then COMPILER_KIND=hipcc; else COMPILER_KIND=clang; fi
    fi
    case "$COMPILER_KIND" in
        hipcc)
            [[ -n "$HIPCC_PATH" ]] || die "COMPILER_KIND=hipcc but $ROCM_ROOT/bin/hipcc is missing (try COMPILER_KIND=clang)"
            CC_BIN="$HIPCC_PATH" ;;
        clang)
            [[ -n "$AMDCLANG_PATH" ]] || die "COMPILER_KIND=clang but no amdclang++ under $ROCM_ROOT"
            CC_BIN="$AMDCLANG_PATH" ;;
        *)
            die "COMPILER_KIND must be 'hipcc' or 'clang' (got '$COMPILER_KIND')" ;;
    esac
fi
[[ -x "$CC_BIN" ]] || die "compiler is not executable: $CC_BIN"
CC_REAL="$(readlink -f "$CC_BIN" 2>/dev/null || echo "$CC_BIN")"

COMPILER_FLAGS=()
# hipcc injects -x hip itself; a bare clang++ needs it to treat the .cpp files
# as HIP. --rocm-path is a clang flag that both accept.
[[ "$COMPILER_KIND" == "clang" ]] && COMPILER_FLAGS+=(-x hip)

# --- 2b. Toolchain provenance (best effort; never fatal) --------------
HIP_VER=""
if [[ -x "$ROCM_ROOT/bin/hipconfig" ]]; then
    HIP_VER="$("$ROCM_ROOT/bin/hipconfig" --version 2>/dev/null | head -n1 || true)"
fi
if [[ -z "$HIP_VER" && -f "$ROCM_ROOT/.info/version" ]]; then
    HIP_VER="$(head -n1 "$ROCM_ROOT/.info/version" 2>/dev/null || true)"
fi
CC_VERSION="$("$CC_BIN" --version 2>/dev/null | head -n1 || true)"
# hipcc only reports the HIP version; the interesting number is the clang it
# drives. Record both so a package manifest is self-explanatory.
if [[ "$COMPILER_KIND" == "hipcc" && -n "$AMDCLANG_PATH" ]]; then
    CLANG_VERSION="$("$AMDCLANG_PATH" --version 2>/dev/null | head -n1 || true)"
    [[ -n "$CLANG_VERSION" ]] && CC_VERSION="$CLANG_VERSION (via hipcc)"
fi

# --- 3. Resolve the single GPU target and its wavefront width ---------
# Family keywords stay, but only to produce a useful error: this project ships
# one library per architecture, never a fat binary.
expand_family() {
    local family="${1,,}"
    case "$family" in
        rdna1)                     echo "gfx1010;gfx1011;gfx1012" ;;
        rdna2)                     echo "gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036" ;;
        rdna3)                     echo "gfx1100;gfx1101;gfx1102;gfx1103" ;;
        rdna35 | rdna3_5 | rdna3.5) echo "gfx1150;gfx1151;gfx1152;gfx1153" ;;
        rdna4)                     echo "gfx1200;gfx1201" ;;
        rdna4x)                    echo "gfx1250" ;;
        cdna1)                     echo "gfx908" ;;
        cdna2)                     echo "gfx90a" ;;
        cdna3)                     echo "gfx942" ;;
        cdna4)                     echo "gfx950" ;;
        *)                         echo "$1" ;;   # raw gfx passthrough
    esac
}

# The one mapping table. Read directly: the shim build must not need an
# interpreter or a compiled helper, only FPC and ROCm (see README).
TARGET_TABLE="$(cd "$SRC_DIR/../.." && pwd)/targets.conf"

resolve_targets() {
    local raw="$1" item target
    local -a requested=() expanded=() out=()
    IFS=';,' read -ra requested <<<"$raw"
    for item in "${requested[@]}"; do
        [[ -z "$item" ]] && continue
        IFS=';' read -ra expanded <<<"$(expand_family "$item")"
        for target in "${expanded[@]}"; do
            [[ -n "$target" ]] && out+=("$target")
        done
    done
    if [[ ${#out[@]} -gt 0 ]]; then
        printf '%s\n' "${out[@]}" | awk '!seen[$0]++'
    fi
}

if [[ -n "${GFX_ARCH:-}" ]]; then
    mapfile -t SELECTED < <(resolve_targets "$GFX_ARCH")
elif [[ -z "${REQUIRE_ARCH:-}" && -x "$ROCM_ROOT/bin/rocm_agent_enumerator" ]]; then
    # No explicit request: pick the GPU this machine actually has. Inference is
    # disabled when REQUIRE_ARCH is set -- "a GPU happens to be present" is not
    # a reproducible build input.
    mapfile -t SELECTED < <("$ROCM_ROOT/bin/rocm_agent_enumerator" 2>/dev/null \
        | grep -E '^gfx[0-9]+$' || true)
else
    SELECTED=()
fi

if [[ ${#SELECTED[@]} -eq 0 ]]; then
    [[ -n "${REQUIRE_ARCH:-}" ]] && die "REQUIRE_ARCH is set but no GPU target resolved. Pass GFX_ARCH=gfxNNNN."
    die "no GPU target resolved. Pass GFX_ARCH=gfxNNNN (one target per build)."
fi
if [[ ${#SELECTED[@]} -ne 1 ]]; then
    die "GFX_ARCH resolves to ${#SELECTED[@]} targets (${SELECTED[*]}). This project builds ONE target per library; run the build once per target."
fi
TARGET="${SELECTED[0]}"

[[ -f "$TARGET_TABLE" ]] \
    || die "missing $TARGET_TABLE -- the GPU target table ships with the repository"
WAVEFRONT=""
TARGET_WIDTHS=""
while read -r table_target _family table_wave table_widths _min_rocm; do
    case "$table_target" in '' | \#*) continue ;; esac
    if [[ "$table_target" == "$TARGET" ]]; then
        WAVEFRONT="$table_wave"
        TARGET_WIDTHS="$table_widths"
        break
    fi
done <"$TARGET_TABLE"
[[ -n "$WAVEFRONT" ]] \
    || die "unknown GPU target '$TARGET' (not in targets.conf)"
case "$WAVEFRONT" in
    32 | 64) ;;
    *) die "targets.conf has wavefront '$WAVEFRONT' for $TARGET (want 32 or 64)" ;;
esac
# Widths the hardware can actually run (1=32, 2=64, 3=both). The runtime gate
# compares the device's reported width against this set, never against the
# compiled width: measured on gfx1100, a -mwavefrontsize64 kernel reports 64
# while hipDeviceProp_t::warpSize still says 32.
WIDTHS_MASK=0
IFS=',' read -ra _target_widths <<<"$TARGET_WIDTHS"
for _width in "${_target_widths[@]}"; do
    case "$_width" in
        32) WIDTHS_MASK=$((WIDTHS_MASK | 1)) ;;
        64) WIDTHS_MASK=$((WIDTHS_MASK | 2)) ;;
        *) die "targets.conf has width '$_width' for $TARGET (want 32 or 64)" ;;
    esac
done
case "$WIDTHS_MASK" in
    1 | 2 | 3) ;;
    *) die "targets.conf yields width mask '$WIDTHS_MASK' for $TARGET" ;;
esac

ROCM_VERSION="${ROCM_VERSION:-${HIP_VER:-unknown}}"
LIB_FILE="${LIB_PATTERN//\{gfx\}/$TARGET}"
LIB_PATH="$OUT_DIR/$LIB_FILE"

TARGET_FLAGS=(--offload-arch="$TARGET")
# Only the table may select wave64. clang 24 silently accepts -mwavefrontsize64
# on gfx1100 and folds __builtin_amdgcn_wavefrontsize() to 64 (measured), so
# exposing that flag as a knob would be a trap, not a feature.
if [[ "$WAVEFRONT" == "64" ]]; then
    TARGET_FLAGS+=(-mwavefrontsize64)
fi

STAMP_FLAGS=(
    "-DPASROCM_SHIM_VERSION=\"$VERSION\""
    "-DPASROCM_GPU_TARGET=\"$TARGET\""
    "-DPASROCM_WAVEFRONT_SIZE=$WAVEFRONT"
    "-DPASROCM_WAVEFRONT_WIDTHS_MASK=$WIDTHS_MASK"
    "-DPASROCM_ROCM_VERSION=\"$ROCM_VERSION\""
)
[[ -n "$GIT_COMMIT" ]] && STAMP_FLAGS+=("-DPASROCM_GIT_COMMIT=\"$GIT_COMMIT\"")

STRIP_FLAGS=(-s)
[[ -n "${NO_STRIP:-}" ]] && STRIP_FLAGS=()

# --- 4. Report the resolved toolchain ---------------------------------
log "ROCm root : $ROCM_ROOT"
log "compiler  : $CC_REAL  [$COMPILER_KIND]"
[[ -n "$CC_VERSION" ]] && log "compiler v: $CC_VERSION"
[[ -n "$HIP_VER" ]]    && log "HIP ver   : $HIP_VER"
log "lib dir   : $LIB_DIR"
log "target    : $TARGET  (wavefront $WAVEFRONT)"
log "output    : $LIB_PATH"
if [[ "$WAVEFRONT" == "64" ]]; then
    log "wave flag : -mwavefrontsize64"
fi


mkdir -p "$OUT_DIR"

# NOTE: HIP headers come only from the selected ROCm install ($ROCM_ROOT/include).
# Never put a local/copied hip/ header set on the include path: mixing header
# versions fails to compile, because the installed hip/amd_detail/*.h internals
# must match the runtime (ROCm 7.x reported it as 'unknown type name
# hipDriverEntryPointQueryResult').
COMPILE_CMD=(
    "$CC_BIN"
    -O2
    -std=c++17
    "${COMPILER_FLAGS[@]}"
    --rocm-path="$ROCM_ROOT"
    "${TARGET_FLAGS[@]}"
    "${STAMP_FLAGS[@]}"
    -fvisibility=hidden
    -shared -fPIC -Xlinker --no-undefined
    # SONAME is the on-disk file name, deliberately without a .so.N suffix: the
    # Pascal loader resolves the library by exact name, so a versioned soname
    # would silently diverge from it. ABI breaks are gated by
    # ci/check-abi.sh --strict and the package version instead.
    -Wl,-soname,${LIB_PATH##*/}
    # -fvisibility=hidden does not hide __global__ kernels (HIP gives their host
    # stubs default visibility so the runtime can register them, measured
    # 2026-11-14), so the version script is what actually closes the dynamic
    # symbol table to the 149 names. It is generated from the same spec.
    -Wl,--version-script,"$SRC_DIR/rocprim_shim.map"
    -I"$ROCM_ROOT/include"
    -L"$LIB_DIR" -lamdhip64
    -o "$LIB_PATH"
    "${SOURCES[@]}"
)

if [[ -n "${DRY_RUN:-}" ]]; then
    printf '== DRY_RUN: '
    printf '%q ' "${COMPILE_CMD[@]}"
    printf '\n'
    exit 0
fi

cd "$SRC_DIR"
"${COMPILE_CMD[@]}"

strip "${STRIP_FLAGS[@]}" "$LIB_PATH" 2>/dev/null || true

log "Linked against:"
readelf -d "$LIB_PATH" | grep NEEDED || true

# The loader finds the file by name; the soname must therefore be that name.
SONAME="$(readelf -d "$LIB_PATH" 2>/dev/null | sed -n 's/.*SONAME.*\[\(.*\)\]/\1/p')"
[[ "$SONAME" == "${LIB_PATH##*/}" ]] \
    || die "SONAME '$SONAME' must equal the on-disk name '${LIB_PATH##*/}'"
log "SONAME: $SONAME"

# Show where the NEEDED HIP entry actually resolves, i.e. which libamdhip64 the
# shim will load at runtime under the baked-in rpath / LD_LIBRARY_PATH.
if command -v ldd >/dev/null 2>&1; then
    log "Loads at runtime:"
    ldd "$LIB_PATH" 2>/dev/null | grep -E 'amdhip64|not found' || true
fi

EXPORT_COUNT="$(nm -D --defined-only "$LIB_PATH" 2>/dev/null \
    | awk '$3 ~ /^rocprim_/ {print $3}' | sort -u | wc -l || echo 0)"
log "rocprim_* exports: $EXPORT_COUNT"

# --- 5. Build provenance ---------------------------------------------
# Consumed by ci/package.sh to fill manifest.json, so a shipped artifact can
# answer "which toolchain, which SDK, which target, which wavefront" without a
# CI log.
BUILD_INFO="$OUT_DIR/build-info.env"
{
    printf 'BUILD_COMPILER_KIND=%q\n'  "$COMPILER_KIND"
    printf 'BUILD_COMPILER=%q\n'       "$CC_REAL"
    printf 'BUILD_COMPILER_VERSION=%q\n' "$CC_VERSION"
    printf 'BUILD_ROCM_ROOT=%q\n'      "$ROCM_ROOT"
    printf 'BUILD_HIP_VERSION=%q\n'    "$HIP_VER"
    printf 'BUILD_ROCM_VERSION=%q\n'   "$ROCM_VERSION"
    printf 'BUILD_GPU_TARGET=%q\n'     "$TARGET"
    printf 'BUILD_WAVEFRONT=%q\n'      "$WAVEFRONT"
    printf 'BUILD_WIDTHS_MASK=%q\n'    "$WIDTHS_MASK"
    printf 'BUILD_LIBRARY=%q\n'        "$LIB_FILE"
    printf 'BUILD_VERSION=%q\n'        "$VERSION"
    printf 'BUILD_ABI_VERSION=%q\n'    "$(sed -n 's/^#define PASROCM_SHIM_ABI_VERSION //p' "$SRC_DIR/rocprim_shim.h")"
    printf 'BUILD_GIT_COMMIT=%q\n'     "$GIT_COMMIT"
    printf 'BUILD_TIMESTAMP=%q\n'      "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'BUILD_EXPORT_COUNT=%q\n'   "$EXPORT_COUNT"
} >"$BUILD_INFO"
log "build info: $BUILD_INFO"
log "Done."

