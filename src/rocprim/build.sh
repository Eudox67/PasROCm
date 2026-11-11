#!/usr/bin/env bash
#
# Version-generic builder for librocprim_shim.so
#
#   ./build.sh                             # auto-detect ROCm (/opt/rocm, then the
#                                          #   newest /opt/rocm-* that actually
#                                          #   contains bin/hipcc) and the current
#                                          #   GPU arch(s) -> fat binary
#   ROCM_PATH=/opt/rocm/core-10.1 ./build.sh  # force a specific install
#   GFX_ARCH=gfx1100 ./build.sh            # target a specific GPU arch(s), ';' separated
#   GFX_ARCH=rdna4   ./build.sh            # target a whole family keyword (see below)
#   GFX_ARCH=rdna3;cdna3 ./build.sh        # fat binary across multiple families
#   DRY_RUN=1 ./build.sh                   # print resolved archs/toolchain, then exit
#   NO_STRIP=1 ./build.sh                  # keep debug info
#
# Architecture families (keyword -> gfx targets), case-insensitive.
#   RDNA3   : gfx1100/1101/1102 (RX 79xx/76xx) + gfx115x (RDNA 3.5 APUs)
#   RDNA3.5 : gfx1150/1151
#   RDNA4   : gfx1200/1201 (RX 9000 series)
#   CDNA1   : gfx908  (Instinct MI100)
#   CDNA2   : gfx90a  (Instinct MI200/MI250)
#   CDNA3   : gfx940/941/942 (Instinct MI300 series)
#   CDNA4   : gfx950/951 (Instinct MI350/MI355X, ROCm 6.4+)
# Anything that is NOT a known keyword is passed through AS-IS as one or
# more raw gfx-IDs. That pass-through is the forward-compat escape hatch:
# future RDNA5 / CDNA5 parts (e.g. gfx1300, gfx1000) need ZERO script
# change --- target them directly when real hardware ships. New named
# families are a one-line addition above.
#
# Compiles BOTH translation units in place; the resulting librocprim_shim.so
# lands next to the Pascal units in src/rocprim/, which is where pasrocm.cfg,
# test.sh (-Fl plus rpath) and setup_env.sh (LD_LIBRARY_PATH) expect it.
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SRC_DIR"

# --- 1. Locate a ROCm root -------------------------------------------
find_rocm_root() {
    if [[ -n "${ROCM_PATH:-}" ]]; then
        echo "$ROCM_PATH"; return
    fi
    # /opt/rocm is normally a symlink into the real versioned tree
    # (/opt/rocm/bin -> /etc/alternatives/rocm-bin -> .../core-10.1/bin),
    # so this single check covers the whole install.
    if [[ -x "/opt/rocm/bin/hipcc" ]]; then
        echo "/opt/rocm"; return
    fi
    # Fallback: newest versioned tree that actually HAS a usable bin/hipcc.
    # Header-only /opt/rocm-<ver> leftovers must never be selected -- they
    # carry no compiler and no libraries, and picking one used to produce a
    # misleading "no ROCm install found" error while a good install existed.
    local d
    while IFS= read -r d; do
        [[ -x "$d/bin/hipcc" ]] && { echo "$d"; return; }
    done < <(compgen -G '/opt/rocm-[0-9]*' | sort -Vr)
}

ROCM_ROOT="$(find_rocm_root)"
[[ -n "$ROCM_ROOT" && -x "$ROCM_ROOT/bin/hipcc" ]] || {
    echo "ERROR: no ROCm install with an executable bin/hipcc found." \
         "Set ROCM_PATH=/path/to/rocm and retry." >&2
    exit 1
}

HIPCC="$ROCM_ROOT/bin/hipcc"
LIB_DIR="$ROCM_ROOT/lib64"
[[ -f "$LIB_DIR/libamdhip64.so" ]] || LIB_DIR="$ROCM_ROOT/lib"

# --- 1b. Toolchain provenance (best effort; never fatal) --------------
# A build log should answer "which hipcc / which HIP produced this .so?"
# without anyone having to trace symlinks by hand. Every probe here is
# optional: a missing hipconfig or unreadable version must not fail a build.
HIPCC_REAL="$(readlink -f "$HIPCC" 2>/dev/null || echo "$HIPCC")"
HIP_VER=""
if [[ -x "$ROCM_ROOT/bin/hipconfig" ]]; then
    HIP_VER="$("$ROCM_ROOT/bin/hipconfig" --version 2>/dev/null | head -n1 || true)"
fi
if [[ -z "$HIP_VER" ]]; then
    HIP_VER="$("$HIPCC" --version 2>/dev/null | head -n1 \
              | sed -e 's/^HIP version: *//' || true)"
fi

# --- 2. Resolve GPU architecture(s) -----------------------------------
expand_family() {
    local f="${1,,}"
    case "$f" in
        rdna3 | rdna35 | rdna3_5) echo "gfx1100;gfx1101;gfx1102;gfx1150;gfx1151" ;;
        rdna35*)                  echo "gfx1150;gfx1151" ;;
        rdna4)                    echo "gfx1200;gfx1201" ;;
        rdna5)                    echo "gfx1300;gfx1301" ;;
        cdna1)                    echo "gfx908" ;;
        cdna2)                    echo "gfx90a" ;;
        cdna3)                    echo "gfx940;gfx941;gfx942" ;;
        cdna4)                    echo "gfx950;gfx951" ;;
        cdna5)                    echo "gfx1010;gfx1011" ;;
        *)                        echo "$1" ;;   # raw gfx passthrough
    esac
}

ARCH_TOKENS=()
if [[ -n "${GFX_ARCH:-}" ]]; then
    IFS=';,' read -ra parts <<< "$GFX_ARCH"
    for p in "${parts[@]}"; do
        [[ -z "$p" ]] && continue
        IFS=';' read -ra got <<< "$(expand_family "$p")"
        for g in "${got[@]}"; do
            [[ -z "$g" ]] && continue
            ARCH_TOKENS+=("$g")
        done
    done
else
    # No explicit request: build a fat binary covering every GPU HIP can
    # enumerate (portable across RDNA/CDNA boxes sharing this tree).
    if [[ -x "$ROCM_ROOT/bin/rocm_agent_enumerator" ]]; then
        while IFS= read -r line; do
            case "$line" in
                gfx[0-9]*) ARCH_TOKENS+=("$line") ;;
            esac
        done < <("$ROCM_ROOT/bin/rocm_agent_enumerator" 2>/dev/null || true)
    fi
fi

ARCH_FLAGS=()
for t in "${ARCH_TOKENS[@]}"; do
    ARCH_FLAGS+=(--offload-arch="$t")
done

STRIP_FLAGS=(-s)
if [[ -n "${NO_STRIP:-}" ]]; then STRIP_FLAGS=(); fi

echo "== ROCm root : $ROCM_ROOT"
echo "== hipcc     : $HIPCC_REAL"
if [[ -n "$HIP_VER" ]]; then
    echo "== HIP ver   : $HIP_VER"
fi
echo "== Lib dir   : $LIB_DIR"
echo "== Output    : $OUT_DIR/librocprim_shim.so"
if [[ ${#ARCH_TOKENS[@]} -gt 0 ]]; then
    echo "== Arch      : ${ARCH_TOKENS[*]}  (${#ARCH_TOKENS[@]} target[s])"
else
    echo "== Arch      : (defaults -no GFX_ARCH, no GPU enumerated)"
fi

if [[ -n "${DRY_RUN:-}" ]]; then
    echo "== DRY_RUN: would compile rocprim_shim.cpp + rocprim_shim_math_kernels.cpp"
    echo "==          hipcc flags: ${ARCH_FLAGS[*]:-<default>}"
    exit 0
fi

cd "$SRC_DIR"
# NOTE: HIP headers come only from the selected ROCm install ($ROCM_ROOT/include).
# Never put a local/copied hip/ header set on the include path: mixing header
# versions fails to compile, because the installed hip/amd_detail/*.h internals
# must match the runtime (ROCm 7.x reported it as 'unknown type name
# hipDriverEntryPointQueryResult').
"$HIPCC" \
    -O2 \
    -std=c++17 \
    --rocm-path="$ROCM_ROOT" \
    "${ARCH_FLAGS[@]}" \
    -shared -fPIC -Xlinker --no-undefined \
    -I"$ROCM_ROOT/include" \
    -L"$LIB_DIR" -lamdhip64 \
    -o "$OUT_DIR/librocprim_shim.so" \
    rocprim_shim.cpp rocprim_shim_math_kernels.cpp

strip "${STRIP_FLAGS[@]}" "$OUT_DIR/librocprim_shim.so"

echo "== Linked against:"
readelf -d "$OUT_DIR/librocprim_shim.so" | grep NEEDED || true
# Show where the NEEDED HIP entry actually resolves, i.e. which libamdhip64
# the shim will load at runtime under the baked-in rpath / LD_LIBRARY_PATH.
if command -v ldd >/dev/null 2>&1; then
    echo "== Loads at runtime:"
    ldd "$OUT_DIR/librocprim_shim.so" 2>/dev/null \
        | grep -E 'amdhip64|not found' || true
fi
echo "== Done."
