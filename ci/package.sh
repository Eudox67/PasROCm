#!/usr/bin/env bash
#
# Package one built shim (DESIGN.md section 8).
#
#   ci/package.sh --config ubuntu-24.04-rocm10.1 --target gfx1100 \
#                 --lib dist/work/<id>-gfx1100/libpasrocm_shim_gfx1100.so \
#                 --build-info dist/work/<id>-rdna3/build-info.env \
#                 --out dist
#
# Produces
#   pasrocm-shim-<version>-linux-x86_64-rocm-<line>-<group>.tar.gz
#   ...tar.gz.sha256
#
# Layout inside the tarball
#   lib/libpasrocm_shim_<gfx>.so  the shim
#   include/rocprim_shim.h      public header (once DESIGN phase 1 lands)
#   pascal/*.pas                matching Pascal declarations
#   manifest.json               build provenance and claims
#   SHA256SUMS                  checksums of every file in the package
#   LICENSE, INSTALL.md
#
# The archive is deterministic for identical inputs (sorted entries, fixed
# mtime/owner), so two local builds of one configuration can be compared.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG=""
TARGET=""
LIB=""
BUILD_INFO=""
OUT="$ROOT/dist"

usage() {
    cat <<'EOF'
usage: ci/package.sh --config <id> --target <gfx> --lib <path> [options]

  --config <id>        configuration id from ci/supported-builds.json
  --target <gfx>       GPU target that was built
  --lib <path>         libpasrocm_shim_<gfx>.so to package
  --build-info <path>  build-info.env written by src/rocprim/build.sh
  --out <dir>          output directory (default: dist/)
  -h, --help           this text
EOF
}

die() { echo "package: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)     CONFIG="${2:?--config needs a value}"; shift 2 ;;
        --target)     TARGET="${2:?--target needs a value}"; shift 2 ;;
        --lib)        LIB="${2:?--lib needs a value}"; shift 2 ;;
        --build-info) BUILD_INFO="${2:?--build-info needs a value}"; shift 2 ;;
        --out)        OUT="${2:?--out needs a value}"; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        *)            usage >&2; die "unknown argument: $1" ;;
    esac
done

[[ -n "$CONFIG" && -n "$TARGET" && -n "$LIB" ]] || { usage >&2; die "--config, --target and --lib are required"; }
[[ -f "$LIB" ]] || die "no such library: $LIB"
[[ -n "$BUILD_INFO" && -f "$BUILD_INFO" ]] || die "no build-info file (expected from src/rocprim/build.sh)"

set -a
. "$BUILD_INFO"
eval "$(python3 "$ROOT/ci/manifest.py" env --id "$CONFIG")"
set +a

VERSION="${BUILD_VERSION:-0.0.0}"
PKG_STEM="pasrocm-shim-${VERSION}-linux-x86_64-rocm-${CFG_SDK_LINE}-${TARGET}"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/lib" "$STAGE/pascal"

LIB_FILE="${BUILD_LIBRARY:-libpasrocm_shim_${TARGET}.so}"
cp "$LIB" "$STAGE/lib/$LIB_FILE"
if [[ -f "$ROOT/src/rocprim/rocprim_shim.h" ]]; then
    mkdir -p "$STAGE/include"
    cp "$ROOT/src/rocprim/rocprim_shim.h" "$STAGE/include/"
fi
cp "$ROOT"/src/rocprim/*.pas "$STAGE/pascal/"
cp "$ROOT/LICENSE" "$STAGE/"

cat >"$STAGE/INSTALL.md" <<EOF
# PasROCm shim $(basename "$PKG_STEM" | cut -d- -f3)

Built against ROCm ${CFG_SDK_LINE} (${CFG_SDK_VERSION}) on ${CFG_OS}.
Covers: ${CFG_COVERS}
GPU target: ${TARGET} (wavefront ${BUILD_WAVEFRONT}, widths mask ${BUILD_WIDTHS_MASK})

## Use it

1. Install ROCm ${CFG_SDK_LINE} (or a newer release that still provides
   \`libamdhip64.so\`). The package never bundles AMD runtime or driver files.
2. Load \`lib/${LIB_FILE}\` for ${TARGET}: the Pascal loader picks the file that
   matches the detected GPU. Put \`lib/\` on \`LD_LIBRARY_PATH\` if the loader
   needs a search path.
3. Copy \`pascal/*.pas\` over \`src/rocprim/\` if they differ from your checkout.

The build is compile-verified and ABI-checked; it has **not** been run on a GPU
(\`validation.gpu_runtime\` is \`not-run\` in \`manifest.json\`).
EOF

export PKG_STEM TARGET LIB_FILE
python3 - "$STAGE/manifest.json" "$ROOT/ci/supported-builds.json" <<'PY'
import datetime
import json
import os
import subprocess
import sys

destination, manifest_path = sys.argv[1], sys.argv[2]
with open(manifest_path, encoding="utf-8") as handle:
    supported = json.load(handle)


def dpkg_version(package):
    try:
        result = subprocess.run(["dpkg-query", "-W", "-f=${Version}", package],
                                capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def rocprim_version(root):
    """Read the header, because package versions encode the SDK, not rocPRIM."""
    if not root:
        return None
    for filename in ("rocprim_version.hpp", "version.hpp"):
        try:
            with open(os.path.join(root, "include", "rocprim", filename),
                      encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            continue
        found = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "#define" and parts[1].startswith("ROCPRIM_VERSION"):
                found[parts[1]] = parts[2]
        keys = ("ROCPRIM_VERSION_MAJOR", "ROCPRIM_VERSION_MINOR", "ROCPRIM_VERSION_PATCH")
        if all(key in found for key in keys):
            return ".".join(found[key] for key in keys)
        packed = found.get("ROCPRIM_VERSION")
        if packed and packed.isdigit():
            value = int(packed)
            return "{}.{}.{}".format(value // 100000, value // 1000 % 100, value % 100)
    return None


env = os.environ
packages = {name: dpkg_version(name) for name in env.get("PIN_PACKAGES", "").split()}
shim = supported["shim"]

manifest = {
    "schema": 1,
    "package": env["PKG_STEM"] + ".tar.gz",
    "gpu_target": env["TARGET"],
    "wavefront_size": int(env.get("BUILD_WAVEFRONT") or 0),
    "widths_mask": int(env.get("BUILD_WIDTHS_MASK") or 0),
    "library": env["LIB_FILE"],
    "shim": {
        "name": shim["name"],
        "version": env.get("BUILD_VERSION") or shim["version"],
        "abi_version": shim["abi_version"],
    },
    "git_commit": env.get("BUILD_GIT_COMMIT") or None,
    "os": {"name": env["CFG_OS"], "codename": env["CFG_CODENAME"]},
    "host_architecture": "x86_64",
    "sdk": {
        "line": env["CFG_SDK_LINE"],
        "version": env["CFG_SDK_VERSION"],
        "packages": packages,
        "covers": env["CFG_COVERS"],
        "covers_evidence": env["CFG_COVERS_EVIDENCE"],
    },
    "rocprim_version": rocprim_version(env.get("BUILD_ROCM_ROOT")),
    "compiler": {
        "kind": env.get("BUILD_COMPILER_KIND") or None,
        "path": env.get("BUILD_COMPILER") or None,
        "version": env.get("BUILD_COMPILER_VERSION") or None,
    },
    "hip_version": env.get("BUILD_HIP_VERSION") or None,

    "validation": {"compile": "passed", "abi": "passed", "gpu_runtime": "not-run"},
    "built_at": env.get("BUILD_TIMESTAMP")
        or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}

with open(destination, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

(
    cd "$STAGE"
    find . -type f ! -name SHA256SUMS -printf '%P\n' \
        | LC_ALL=C sort \
        | xargs -r sha256sum >SHA256SUMS
)

mkdir -p "$OUT"
TARBALL="$OUT/${PKG_STEM}.tar.gz"
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='@0' \
    -C "$STAGE" -czf "$TARBALL" .
sha256sum "$TARBALL" >"$TARBALL.sha256"

echo "== package   : $TARBALL ($(du -h "$TARBALL" | cut -f1))"
echo "== checksum  : $(cut -d' ' -f1 "$TARBALL.sha256")"
echo "== contents  :"
tar -tzf "$TARBALL" | sed 's/^/   /'

