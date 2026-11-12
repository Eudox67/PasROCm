# Shim build and release pipeline

**Status:** proposed, not implemented. Everything below was measured against the
repository as it stands on branch `ci/shim-build-pipeline` (based on `8fab987`).

The shim is `src/rocprim/librocprim_shim.so`: the C++/HIP layer that the Pascal
units call into for the rocPRIM primitives that cannot be expressed as Pascal
generics. This document defines how it is built, identified, packaged and
validated.

## 1. Scope

In scope:

- Linux x86_64 builds of the shim against **selected, pinned ROCm SDK releases**.
- Reproducible local builds using the same scripts CI uses.
- Per-SDK packages carrying build metadata, checksums and notices.
- A closed, versioned C ABI with a matching public header.

Out of scope for now:

- **Windows.** ROCm 6.3 has no Windows release, and the shim's Pascal loader
  expects `rocprim_shim.dll`, which does not exist yet.
- **Runtime testing in CI.** No GPU-equipped runner is available; runtime
  validation stays manual until one exists.
- **Cross-SDK fat binaries.** See §3 — they are impossible across HIP majors.
- **fpdoc.** Public documentation, if any, uses pasdoc only. The fpdoc driver
  (`tools/dox2fpdoc.pas`, `tools/docs.sh`) is a local experiment, kept out of
  the public tree.

## 2. What exists today (measured)

| Fact | Value |
|---|---|
| Dynamic exported symbols | **204** (`nm -D`, `T`), of which **146** are `rocprim_*` host API |
| Visibility control | none — no `-fvisibility=hidden`, no export macro |
| SONAME | **absent** (`readelf -d` shows no SONAME) |
| Public C header | **none** — the ABI is declared only in `rocprim_bindings.pas` |
| Identity functions | **none** — nothing reports ABI version, git commit, SDK or targets |
| Runtime dependencies | `libamdhip64.so.7`, `libstdc++.so.6`, `libm.so.6`, `libgcc_s.so.1`, `libc.so.6` |
| Build entry points | `src/rocprim/build.sh` (hipcc direct) and `src/rocprim/CMakeLists.txt` (both maintained) |
| Explicit GPU targets | already supported: `GFX_ARCH=gfx1100;gfx942 ./build.sh` → repeated `--offload-arch=`; falls back to `rocminfo` detection when unset |
| Compiler selection | hard-coded `$ROCM_ROOT/bin/hipcc`; no override, so ROCm 7.2+ (hipcc deprecated) is blocked |
| Library name in Pascal | `rocprim_shim` (`rocprim_bindings.pas:45-47`), Windows `rocprim_shim.dll` |
| Layout | flat `src/rocprim/` (11 Pascal units + 2 `.cpp` + `build.sh` + `CMakeLists.txt`) |
| CI | none — no `.github/` directory |
| Local toolchain | `/opt/rocm` → HIP 7.16.26385, AMD clang 24.0.0 |

`/opt/rocm-6.3.2` and `/opt/rocm-7.0.2` exist as directories but are **not
installs** (stray `include/hip/` only, no `bin/hipcc`), so 6.x and 7.x support
can only be proven in CI containers.

## 3. Per-SDK packages are mandatory, not cosmetic

The shipped library declares:

```
NEEDED  libamdhip64.so.7
```

A shim built against one HIP **major** cannot load against another. Therefore:

- one package per (OS, SDK release) pair, always;
- a "fat" binary can span **GPU architectures only**, never ROCm versions;
- the package name must state its SDK, and the manifest must repeat it.

## 4. ABI prerequisites, before any stability claim

The current surface (204 exports, default visibility, no SONAME) cannot be
called stable because nothing constrains it. Required, in order:

1. `src/rocprim/rocprim_shim.h` — the public header: export macro, identity
   functions, host API declarations.
2. `rocprim_shim_abi_version()` and `rocprim_shim_build_info()` implemented in
   `rocprim_shim.cpp`, stamped via `-D` from git commit / SDK / compiler.
3. `-fvisibility=hidden` plus explicit `PASROCM_EXPORT` on the exported
   functions; `-Wl,-soname,librocprim_shim.so.1`.
4. `ci/abi-expected.txt` — the golden export list.
5. Two gates: `nm -D` output must equal the golden list, and every `external`
   name parsed from `rocprim_bindings.pas` must exist in the built library.

Names stay as they are: symbol prefix `rocprim_*`, library base name
`rocprim_shim`. Renaming would break the Pascal loader and every existing
consumer. If `libpasrocm_shim.so` is wanted later, add it as a SONAME plus a
`librocprim_shim.so` symlink inside the package — not as a rename.

## 5. Supported-configuration policy

- Pin **exact** SDK versions; never `latest`, never `7.x`.
- One selected release per family initially (6.x, 7.x, 10.x).
- Add patch releases only if PasROCm genuinely promises patch-by-patch coverage.
- Record unsupported combinations in the manifest rather than letting CI retry
  impossible builds.
- A new SDK release is a **reviewed manifest change**, not an automatic bump.
- GPU targets are an explicit list per configuration; CI must never infer them
  (`REQUIRE_ARCH=1`), because compiler acceptance of a `gfx…` target does not
  establish runtime support for that GPU.

## 6. Validation levels

Three distinct levels; a configuration must not be labelled above the level it
reached.

| Level | What is checked | What it does not prove |
|---|---|---|
| Compile | Headers, instantiated primitives, kernels, host code, linking | Correctness on hardware |
| ABI / package | Exports match the golden list, Pascal declarations resolve, package contents and metadata | GPU execution or performance |
| GPU runtime | Real operations against a CPU reference on a named OS/GPU/runtime | Untested devices or releases |

## 7. Phases and gates

| Phase | Work | Gate |
|---|---|---|
| 1. ABI identity | `rocprim_shim.h`, identity functions, hidden visibility, SONAME, golden export list | `ci/check-abi.sh` passes against a locally rebuilt library |
| 2. CI-callable build | `COMPILER=`, `OUT_DIR=`, `VERSION=`/`GIT_COMMIT=`, `REQUIRE_ARCH=1` in `build.sh` | `REQUIRE_ARCH=1 ./build.sh` without `GFX_ARCH` exits non-zero; `OUT_DIR` keeps `src/` clean |
| 3. Manifest + driver | `ci/supported-builds.json`, `ci/build-linux.sh --config <id>`, `ci/package.py` | Two consecutive local builds of one config produce identical package contents (modulo timestamps) |
| 4. Workflows | `shim-build.yml` (dynamic matrix from the manifest), `shim-release.yml` (tag → assets), `shim-gpu-tests.yml` (self-hosted, manual/scheduled) | A PR run produces one package per enabled configuration; a tag run attaches them to the release |
| 5. Documentation | Supported-configuration table derived from the manifest; local reproduction notes | Table matches the manifest exactly |

Actions in workflows are pinned by commit SHA. Substantive build logic lives in
`ci/*.sh` and `build.sh`, not in YAML, so a failed CI build can be reproduced
verbatim on a workstation.

## 8. Packaging

Name: `pasrocm-shim-<version>-linux-x86_64-rocm-<sdk>-<targets>.tar.gz`
(for a fat binary, `<targets>` is an architecture-set label and the full list is
repeated inside `manifest.json`).

Contents: the shared library, `rocprim_shim.h`, the matching Pascal declarations,
`manifest.json`, `SHA256SUMS`, `LICENSE`, and short install instructions.

The package never bundles AMD runtime or driver files; it depends on a ROCm
installation being present, exactly as `libamdhip64.so.N` implies.

`manifest.json` fields: `shim_version`, `shim_abi_version`, `git_commit`, `os`,
`host_architecture`, `sdk_version`, `rocprim_version`, `compiler_version`,
`gpu_targets[]`, and `validation: {compile, abi, gpu_runtime}`.

## 9. Risks

- **Container tags must be verified** before pinning; a `rocm/dev-*` tag that
does not exist fails every job in the matrix.
- **hipcc is deprecated from ROCm 7.2**, so the compiler is per-configuration
(`hipcc` vs `amdclang++`), not a global default.
- **Flaky batched tests.** `CTRMM_Batched_64` and `CTRSM_Batched_64` fail
intermittently on tolerance on this machine (proven run-to-run nondeterminism).
Release gates need an explicit policy: retry, widen tolerance, or exclude with a
recorded reason.
- **Fat binaries cost size** (the current single-arch library is ~12 MB before
stripping) and lengthen builds; start with one architecture per job.
- **Public API drift.** Until the export list is golden-tested, any internal
symbol can leak into the ABI unnoticed.

## 10. Open decisions

| # | Decision | Recommendation |
|---|---|---|
| 1 | Artifact name | keep `librocprim_shim.so`; add SONAME/symlink later if wanted |
| 2 | Exact SDK releases to pin (6.x, 7.x, 10.x) | verify image tags first, then pin one per family |
| 3 | GPU runners | none today: ship runtime tests as manual/self-hosted only |
| 4 | Symbol prefix | keep `rocprim_*` |
| 5 | Layout | keep flat `src/rocprim/`; add `ci/` beside it |

## 11. Reproduce a build locally

```bash
# one architecture, explicit, no rocminfo probing
GFX_ARCH=gfx1100 REQUIRE_ARCH=1 ./src/rocprim/build.sh

# several architectures in one fat binary
GFX_ARCH='gfx1100;gfx942' ./src/rocprim/build.sh

# alternate compiler / ROCm root
COMPILER=/opt/rocm/bin/amdclang++ ROCM_ROOT=/opt/rocm ./src/rocprim/build.sh

# show the resolved toolchain without building
DRY_RUN=1 ./src/rocprim/build.sh

# compile + run the test suite (builds the shim if missing)
./test.sh
```

The build already reports the resolved compiler (via `readlink -f`), the HIP
version, the library directory, and the runtime resolution of the result, so a
CI log answers "which toolchain produced this binary?" on its own.
