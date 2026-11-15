#!/usr/bin/env bash
#
# Install the pinned ROCm SDK that one configuration in ci/supported-builds.conf
# requires, and nothing more.
#
#   ci/install-rocm.sh --config ubuntu-24.04-rocm7.2
#   ci/install-rocm.sh --config ubuntu-24.04-rocm10.1 --dry-run
#
# Only the packages needed to COMPILE and LINK the shim are installed: no kernel
# driver, no dkms, no graphics/OpenCL stack. Two repository families are
# supported, because AMD ships them differently:
#
#   radeon-apt       ROCm 6.x/7.x  https://repo.radeon.com/rocm/apt/<patch>
#                    The repository directory IS the patch, so pinning the repo
#                    pins the SDK (package versions carry ~22.04/~24.04 suffixes
#                    and must not be pinned literally).
#   amdrocm-stable   ROCm 10.x     https://stable.repo.amd.com/rocm/core/packages/ubuntuXXXX
#                    One rolling suite with minor-scoped package names, so this
#                    script pins exact versions from the manifest.
#
# The script is idempotent and safe to re-run. On a workstation it WILL modify
# system package sources and install packages, so use --dry-run to inspect.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG=""
DRY_RUN=""

usage() {
    cat <<'EOF'
usage: ci/install-rocm.sh --config <id> [--dry-run]

  --config <id>   configuration id from ci/supported-builds.conf
  --dry-run       print the commands without changing the system
  -h, --help      this text
EOF
}

die() { echo "install-rocm: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="${2:?--config needs a value}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown argument: $1" ;;
    esac
done

[[ -n "$CONFIG" ]] || { usage >&2; die "--config is required"; }

eval "$("$ROOT/ci/manifest.sh" env --id "$CONFIG")"

SUDO=()
[[ ${EUID} -eq 0 ]] || SUDO=(sudo)

run() {
    if [[ -n "$DRY_RUN" ]]; then printf '+ %s\n' "$*"; return 0; fi
    "$@"
}

write_file() {   # write_file <dest>; content on stdin
    local dest="$1"
    if [[ -n "$DRY_RUN" ]]; then
        printf '+ sudo tee %s <<< EOF\n' "$dest"
        sed 's/^/+   /'
        printf '+ EOF\n'
    else
        "${SUDO[@]}" tee "$dest" >/dev/null
    fi
}

install_key() {  # install_key <url> <dest>
    local url="$1" dest="$2"
    if [[ -n "$DRY_RUN" ]]; then
        printf '+ wget -qO- %s | gpg --dearmor | sudo tee %s\n' "$url" "$dest"
        return 0
    fi
    command -v wget >/dev/null 2>&1 || "${SUDO[@]}" apt-get install -y --no-install-recommends wget
    command -v gpg  >/dev/null 2>&1 || "${SUDO[@]}" apt-get install -y --no-install-recommends gpg
    "${SUDO[@]}" mkdir -p "$(dirname "$dest")"
    wget -qO- "$url" | gpg --dearmor | "${SUDO[@]}" tee "$dest" >/dev/null
    "${SUDO[@]}" chmod 0644 "$dest"
}

echo "== configuration : $CFG_ID ($CFG_MODE)"
echo "== ROCm SDK      : $CFG_SDK_LINE line, pinned $CFG_SDK_VERSION ($PIN_KIND)"
echo "== runner        : $CFG_OS"
echo "== covers        : $CFG_COVERS"

case "$PIN_KIND" in
    radeon-apt)
        KEYRING=/etc/apt/keyrings/rocm.gpg
        install_key "$PIN_KEY" "$KEYRING"
        printf 'deb [arch=amd64 signed-by=%s] %s %s main\n' \
            "$KEYRING" "$PIN_REPO" "$PIN_SUITE" \
            | write_file /etc/apt/sources.list.d/rocm.list
        # The distro repositories publish ROCm 5.0-era packages under the same
        # names as AMD's, and their version strings sort HIGHER: measured
        # 2026-11-14, jammy/universe ships rocm-device-libs 5.0.0-1 and
        # rocminfo 5.0.0-1, while AMD's 6.4.4 repo ships rocm-device-libs
        # 1.0.0.60404-129~22.04. Plain apt therefore installs the distro copy,
        # and the build dies with "cannot find ROCm device library" because
        # 5.0 bitcode does not live under --rocm-path. Prefer the AMD origin
        # for every name it publishes; the repo carries ROCm packages only.
        write_file /etc/apt/preferences.d/rocm.pref <<'PREF'
Package: *
Pin: origin repo.radeon.com
Pin-Priority: 1001
PREF
        ;;
    amdrocm-stable)
        KEYRING=/etc/apt/keyrings/amdrocm.gpg
        install_key "$PIN_KEY" "$KEYRING"
        {
            echo "X-Repo-Id: amdrocm-stable"
            echo "Types: deb"
            echo "URIs: ${PIN_REPO%/}/"
            echo "Suites: $PIN_SUITE"
            echo "Components: main"
            echo "Architectures: amd64"
            echo "Signed-By: $KEYRING"
            echo "Enabled: yes"
        } | write_file /etc/apt/sources.list.d/amdrocm-stable.sources
        ;;
    *)
        die "unknown pin kind '$PIN_KIND'"
        ;;
esac

run "${SUDO[@]}" apt-get update -qq

# --- install the pinned packages --------------------------------------
# rocPRIM headers come from rocprim-dev (6.x/7.x) or amdrocm-ccl-dev (10.x);
# libamdhip64 plus its dev symlink come from hip-dev / amdrocm-runtime-dev.
INSTALL_ARGS=()
if [[ -n "${PIN_EXACT:-}" ]]; then
    read -r -a INSTALL_ARGS <<<"$PIN_EXACT"
else
    read -r -a INSTALL_ARGS <<<"$PIN_PACKAGES"
fi

if [[ -n "${PASROCM_SKIP_ROCM_INSTALL:-}" ]]; then
    echo "== PASROCM_SKIP_ROCM_INSTALL set: not installing, verifying only"
else
    run "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive \
        apt-get install -y --no-install-recommends "${INSTALL_ARGS[@]}"
fi

# --- resolve the install root -----------------------------------------
# 10.x installs into /opt/rocm/core-<ver> behind the /opt/rocm symlink; 6.x/7.x
# use /opt/rocm-<ver>. Either way the compiler and headers must be reachable.
# One predicate for every branch: hipcc (6.x/7.x) or amdclang++ (10.x, where a
# minimal install ships no hipcc at all). The fallback loops used to check only
# hipcc, so a 10.x tree without the /opt/rocm alternatives symlinks was invisible.
has_compiler() {
    [[ -x "$1/bin/hipcc" ]] ||
    [[ -x "$1/lib/llvm/bin/amdclang++" ]] ||
    [[ -x "$1/llvm/bin/amdclang++" ]]
}

probe_rocm_root() {
    if [[ -d /opt/rocm ]] && has_compiler /opt/rocm; then
        echo /opt/rocm
        return 0
    fi
    local candidate
    # 10.x tree without the alternatives symlinks, then 6.x/7.x installs.
    while IFS= read -r candidate; do
        has_compiler "$candidate" && { echo "$candidate"; return 0; }
    done < <(compgen -G '/opt/rocm/core-[0-9]*' | sort -Vr)
    while IFS= read -r candidate; do
        has_compiler "$candidate" && { echo "$candidate"; return 0; }
    done < <(compgen -G '/opt/rocm-[0-9]*' | sort -Vr)
    return 1
}

if [[ -n "$DRY_RUN" ]]; then
    echo "== DRY_RUN: would resolve /opt/rocm and export ROCM_PATH"
    exit 0
fi

ROCM_ROOT="$(probe_rocm_root)" || die "no ROCm install found after installing $CFG_SDK_VERSION"
[[ -d "$ROCM_ROOT/include/hip" ]] || die "missing HIP headers under $ROCM_ROOT/include/hip"

# The compiler needs the device bitcode, and apt can silently hand back the
# distro's 5.0-era rocm-device-libs instead of AMD's 1.0.0.<sdk> (the
# radeon-apt pin above prevents it; this proves it). A bitcode tree without
# ockl.bc is not one clang can use for --rocm-path.
DEVICE_LIBS=""
for candidate in "$ROCM_ROOT/amdgcn/bitcode" \
                 "$ROCM_ROOT/lib/llvm/amdgcn/bitcode" \
                 "$ROCM_ROOT/llvm/amdgcn/bitcode"; do
    if [[ -f "$candidate/ockl.bc" ]]; then
        DEVICE_LIBS="$candidate"
        break
    fi
done
if [[ -z "$DEVICE_LIBS" ]]; then
    DEVLIBS_VER="$(dpkg-query -W -f='${Version}' rocm-device-libs 2>/dev/null || true)"
    die "no ROCm device bitcode (ockl.bc) under $ROCM_ROOT, but rocm-device-libs is '${DEVLIBS_VER:-<not installed>}'. The distro package (5.0.0-1) cannot serve a 6.x/7.x build; apt must take AMD's 1.0.0.<sdk> through /etc/apt/preferences.d/rocm.pref."
fi

export ROCM_PATH="$ROCM_ROOT"
export HIP_PLATFORM=amd
export HIP_PATH="$ROCM_ROOT"
LD_PATH="$ROCM_ROOT/lib:$ROCM_ROOT/lib64"
[[ -n "${LD_LIBRARY_PATH:-}" ]] && LD_PATH="$LD_PATH:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$LD_PATH"
case ":${PATH}:" in
    *":$ROCM_ROOT/bin:"*) ;;
    *) export PATH="$ROCM_ROOT/bin:$ROCM_ROOT/lib/llvm/bin:$PATH" ;;
esac

echo "== ROCm root   : $ROCM_ROOT"
echo "== device libs : $DEVICE_LIBS"
for pkg in $PIN_PACKAGES; do
    version="$(dpkg-query -W -f='${Version}' "$pkg" 2>/dev/null || true)"
    printf '== installed   : %-30s %s\n' "$pkg" "${version:-<missing>}"
done

# Publish the environment for subsequent workflow steps.
if [[ -n "${GITHUB_ENV:-}" ]]; then
    {
        echo "ROCM_PATH=$ROCM_PATH"
        echo "HIP_PLATFORM=amd"
        echo "HIP_PATH=$HIP_PATH"
        echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
        echo "PATH=$PATH"
    } >>"$GITHUB_ENV"
fi

echo "== ready: $CFG_SDK_LINE ($CFG_SDK_VERSION) at $ROCM_ROOT"

