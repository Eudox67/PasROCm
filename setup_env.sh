#!/bin/bash
# ---------------------------------------------------------------------------
# PasROCm - environment setup (source this file, do not execute it)
#
#   source setup_env.sh
#
# To load it in every new shell:
#   echo "source /path/to/PasROCm/setup_env.sh" >> ~/.bashrc
#
# No path here is tied to a specific checkout: the ROCm root is auto-detected
# (ROCM_PATH/HOME, /opt/rocm, then the newest /opt/rocm-* that actually looks
# like an install), and the rocPRIM shim directory is derived from this
# script's own location.
# ---------------------------------------------------------------------------

# 1. ROCm installation (honour an existing ROCM_PATH / ROCM_HOME)
if [ -z "${ROCM_PATH:-}" ]; then
    if [ -n "${ROCM_HOME:-}" ]; then
        ROCM_PATH="$ROCM_HOME"
    elif [ -d /opt/rocm ]; then
        ROCM_PATH=/opt/rocm
    else
        # Prefer the newest versioned tree that can actually BUILD (bin/hipcc),
        # then one that can at least RUN (libamdhip64), so runtime-only installs
        # still work for tests. Header-only /opt/rocm-<ver> leftovers are never
        # selected: they have no compiler and no libraries.
        for _pasrocm_cand in $(ls -1d /opt/rocm-[0-9]* 2>/dev/null | sort -Vr); do
            if [ -x "$_pasrocm_cand/bin/hipcc" ]; then
                ROCM_PATH="$_pasrocm_cand"; break
            fi
        done
        if [ -z "${ROCM_PATH:-}" ]; then
            for _pasrocm_cand in $(ls -1d /opt/rocm-[0-9]* 2>/dev/null | sort -Vr); do
                if ls "$_pasrocm_cand"/lib*/libamdhip64.so* >/dev/null 2>&1; then
                    ROCM_PATH="$_pasrocm_cand"; break
                fi
            done
        fi
        unset _pasrocm_cand
    fi
fi
export ROCM_PATH
export ROCM_HOME="$ROCM_PATH"

if [ -z "$ROCM_PATH" ] || [ ! -d "$ROCM_PATH" ]; then
    echo "[PasROCm][ERROR] ROCm not found (\$ROCM_PATH, \$ROCM_HOME, /opt/rocm, /opt/rocm-* all unset/missing)." >&2
    echo "[PasROCm]         Install ROCm, then re-run: export ROCM_PATH=/path/to/rocm" >&2
    return 1 2>/dev/null || exit 1
fi

# 2. HIP / AMD platform variables
export HIP_PLATFORM=amd
export HIP_PATH="$ROCM_PATH"
export HIP_CLANG_PATH="$ROCM_PATH/llvm/bin"
export HIP_CC="${HIP_CC:-hipcc}"
if [ -d "$ROCM_PATH/include/hsa" ]; then
    export HSA_PATH="$ROCM_PATH/include/hsa"
fi

# 3. Tools on PATH (idempotent)
case ":$PATH:" in
    *":$ROCM_PATH/bin:"*) ;;
    *) export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH" ;;
esac

# 4. Device bitcode library (layout differs between ROCm releases)
for _pasrocm_cand in "$ROCM_PATH/amdgcn/bitcode" \
                       "$ROCM_PATH/lib/llvm/amdgcn/bitcode" \
                       "$ROCM_PATH/lib64/llvm/amdgcn/bitcode"; do
    if [ -d "$_pasrocm_cand" ]; then
        export HIP_DEVICE_LIB_PATH="$_pasrocm_cand"
        export DEVICE_LIB_PATH="$_pasrocm_cand"
        break
    fi
done
unset _pasrocm_cand

# 5. Shared libraries: ROCm + this checkout's rocPRIM shim
_pasrocm_add_ld() {
    case ":$LD_LIBRARY_PATH:" in
        *":$1:"*) ;;
        *) LD_LIBRARY_PATH="$1${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
}
_pasrocm_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_pasrocm_add_ld "$ROCM_PATH/lib"
_pasrocm_add_ld "$ROCM_PATH/lib64"
if [ -d "$_pasrocm_root/src/rocprim" ]; then
    _pasrocm_add_ld "$_pasrocm_root/src/rocprim"
fi
export LD_LIBRARY_PATH
unset -f _pasrocm_add_ld
unset _pasrocm_root

if ! command -v fpc >/dev/null 2>&1; then
    echo "[PasROCm][WARNING] 'fpc' is not on PATH - install Free Pascal 3.2.2 or newer." >&2
fi

echo "[PasROCm] ROCm at $ROCM_PATH | device bitcode: ${HIP_DEVICE_LIB_PATH:-<not found>}"

