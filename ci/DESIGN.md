# Shim build and release pipeline

**Status:** phases 2, 3 and 5 of §7 are implemented; phase 1 (public header,
identity functions, closed visibility) is still open, so the ABI is checked but
not yet *closed*, and phase 4 exists only as the build workflow (no tag-driven
release and no GPU test workflow). Phase 7 (GPU runtime) has a decision and a
procedure — a one-off rented MI instance, §13.6 — but no recorded run yet.

The shim is `src/rocprim/libpasrocm_shim_<gfx>.so`: the C++/HIP layer that the
Pascal units call into for the rocPRIM primitives that cannot be expressed as
Pascal generics. This document defines how it is built, identified, packaged and
validated, and records the upstream facts the configuration policy rests on.

## 1. Scope

In scope:

- Linux x86_64 builds of the shim against **selected, pinned ROCm SDK lines**.
- A GitHub Actions matrix over Ubuntu 22.04 / 24.04 / 26.04, one artifact per
  (configuration, GPU architecture group).
- Reproducible local builds using the same scripts CI uses.
- Per-SDK packages carrying build metadata, checksums and notices.
- A versioned C ABI with a matching public header.

Out of scope for now:

- **Windows.** ROCm has no Windows release for the 6.x line, and the shim's
  Pascal loader expects `rocprim_shim.dll`, which does not exist yet.
- **Runtime testing in CI.** No AMD GPU runner is available; runtime validation
  stays manual until one exists (phase F of §7).
- **Cross-SDK fat binaries.** Impossible across HIP majors — see §3.
- **fpdoc.** Public documentation, if any, uses pasdoc only. The fpdoc driver
  (`tools/dox2fpdoc.pas`, `tools/docs.sh`) is a local experiment, kept out of
  the public tree.

## 2. What exists today (measured)

| Fact | Value |
|---|---|
| Dynamic exported symbols | 146 `rocprim_*` plus 5 `pasrocm_*` identity symbols |
| Declared in Pascal | **149** `external librocprimshim` functions (144 + 5 identity) |
| ABI gate | `ci/check-abi.sh`: 149/149 exported, 2 unexpected exports (see §12.4) |
| Visibility control | none — no `-fvisibility=hidden`, no export macro (phase 1) |
| SONAME | **absent** (`readelf -d` shows no SONAME) |
| Public C header | **none** — the ABI is declared only in `rocprim_bindings.pas` |
| Identity functions | `pasrocm_shim_target`, `_wavefront_size`, `_rocm_version`, `_wavefront_widths_mask` (GPU-free); ABI version/build-info still pending in phase 1 |
| Runtime dependencies | `libamdhip64.so.7`, `libstdc++.so.6`, `libm.so.6`, `libgcc_s.so.1`, `libc.so.6` |
| Build entry points | `src/rocprim/build.sh` (canonical), `src/rocprim/CMakeLists.txt` |
| Compiler selection | `COMPILER_KIND=hipcc` (6.x/7.x) or `COMPILER_KIND=clang` (10.x) |
| Compiler flags | `-x hip` for the `clang` kind; `--rocm-path` for both; `-mwavefrontsize64` only when the table says 64 |
| GPU target | exactly one per build: `GFX_ARCH=gfx1100 ./build.sh` → one `--offload-arch` and one `-DPASROCM_WAVEFRONT_SIZE` |
| Arch inference | `rocm_agent_enumerator`, **disabled** when `REQUIRE_ARCH` is set |
| Library name | `libpasrocm_shim_<gfx>.so` per target, opened at run time by `rocprim_shim_loader.pas`; nothing links it (the dev symlink is gone) |
| Pascal binding mechanism | `rocprim_shim_abi.inc` (hand-edited spec) → generated `rocprim_shim_api.pas` (pointer table) and `rocprim_bindings.pas` (149 wrappers); no symbol is resolved at link time |
| Layout | flat `src/rocprim/` (13 Pascal units incl. 2 generated + 1 spec `.inc` + 2 `.cpp` + `build.sh` + `CMakeLists.txt`) |
| CI | `.github/workflows/shim-build.yml`, matrix from `ci/supported-builds.json` |
| CI scripts | `ci/gfx-targets.json`, `ci/manifest.py`, `ci/install-rocm.sh`, `ci/build-linux.sh`, `ci/package.sh`, `ci/check-abi.sh` |
| Local toolchain | `/opt/rocm` → ROCm **10.1.0**, HIP 7.16.26385, AMD clang 24.0.0 |

`/opt/rocm-6.3.2` and `/opt/rocm-7.0.2` exist as directories but are **not
installs** (stray `include/hip/`, no compiler), so 6.x and 7.x support is proven
only in CI.

## 3. What a binary is bound to: one target, one library

The shipped libraries declare a HIP **major** — measured from the packages, not
assumed:

| SDK | HIP runtime | `libamdhip64.so` | rocPRIM headers |
|---|---|---|---|
| 6.4.0 (`apt/6.4`) | 6.4.43482 | `.so.6` | 3.4.0 |
| 6.4.4 (`apt/6.4.4`) | 6.4.43484 | `.so.6` | 3.4.1 |
| 7.2.0 (`apt/7.2`) | 7.2.26015 | `.so.7` | 4.2.0 |
| 7.2.4 (`apt/7.2.4`) | 7.2.53211 | `.so.7` | 4.2.0 |
| 10.1.0 (`stable`) | 7.16.26385 | `.so.7` | 4.6.0 |

Consequences:

- One package per **(OS, SDK line)**, always. A 6.x build cannot load against a
  7.x/10.x runtime, and vice versa; `DT_NEEDED` on `libamdhip64.so.<major>`
  makes a cross-major mix fail at `dlopen`, which is a loud enough failure that
  the SDK line does not need to appear in the on-disk file name.
- **No fat binaries.** One library per GPU target:
  `libpasrocm_shim_<gfx>.so`. The loader picks the file that matches the device,
  so a wrong choice is a file-not-found rather than a silently mis-sized kernel.
- 7.x and 10.x share `libamdhip64.so.7`, so a 7.2-built artifact will *load*
  under 10.1. Loading is not support: the device code came from a different
  toolchain and a different target set. Packages stay per-SDK.
- The wavefront width belongs to the target, not to the compiler's guess, and is
  injected at build time — see §12.3 for the mechanism and §9 for the measured
  reason it cannot be derived from a macro.

**Why 19 targets and not the 24 that AMD publishes for 10.1.** The table
(`ci/gfx-targets.json`) lists what all claimed SDK lines can actually build, plus
what 10.1 adds:

| Excluded | Reason |
|---|---|
| `gfx1010`, `gfx1011`, `gfx1012` | RDNA1; only 10.1 publishes packages, so `min_rocm` is 10.1 and they are absent from the 6.4/7.2 matrices |
| `gfx1031` … `gfx1036` | RDNA2 variants with no separate CI value over `gfx1030`; add them when a user reports one |
| `gfx940`, `gfx941` | CDNA3 parts that 10.1 does not publish (`gfx942` covers MI300) |
| `gfx951` | Not published by 10.1; `gfx950` covers MI350/MI355 |

The remaining 19 are: `gfx908`, `gfx90a`, `gfx942`, `gfx950`, `gfx1030`,
`gfx1100`–`gfx1103`, `gfx1150`–`gfx1153`, `gfx1200`, `gfx1201`, `gfx1250`, and
`gfx1010`–`gfx1012` (10.1 only). The table is the single source of truth: the
planner filters it by each target's `min_rocm`, so a configuration can never ask
a 6.4 build for a 10.1-only part.

## 4. ABI prerequisites, before any stability claim

The current surface (146 `rocprim_*` exports, default visibility, no SONAME)
cannot be called stable because nothing constrains it. Measured today:

- 144 names are declared in `rocprim_bindings.pas`; all 144 are exported.
- 2 `rocprim_*` symbols are exported without a declaration:
  `rocprim_fill_kernel_int` (a `__global__` kernel that leaked into the dynamic
  symbol table) and `rocprim_set_warp_size` (host helper nothing calls).

Required, in order (phase 1, still open):

1. `src/rocprim/rocprim_shim.h` — the public header: export macro, identity
   functions, host API declarations.
2. `rocprim_shim_abi_version()` and `rocprim_shim_build_info()` implemented in
   `rocprim_shim.cpp`, stamped via `-D` from git commit / SDK / compiler.
   (`build.sh` already passes `-DPASROCM_SHIM_VERSION` and
   `-DPASROCM_GIT_COMMIT`; nothing consumes them yet.)
3. `-fvisibility=hidden` plus explicit `PASROCM_EXPORT` on exported functions;
   `-Wl,-soname,librocprim_shim.so.1`.
4. `ci/abi-expected.txt` — the golden export list. **Done**: it is generated
   from the ABI specification (`src/rocprim/rocprim_shim_abi.inc`), so it is the
   contract rather than a snapshot of whatever the compiler emitted.
5. Two gates — **done** in `ci/check-abi.sh`: the declaration set must match the
   golden list, and every declaration must resolve in the built library. It also
   rejects a generated unit that is stale with respect to the specification.
   Extra exports are a warning today and become an error (`--strict`) with item 3.

Names stay as they are for the API: symbol prefix `rocprim_*`, library base name
`pasrocm_shim_<gfx>`. Two exceptions are now part of the ABI and are recorded
here deliberately:

- the file name carries the GPU target (`libpasrocm_shim_gfx1100.so`), because
  there is one library per target and the loader resolves by device;
- five `pasrocm_*` identity symbols exist beside the `rocprim_*` API:
  `pasrocm_shim_target`, `pasrocm_shim_wavefront_size`,
  `pasrocm_shim_rocm_version`, `pasrocm_shim_wavefront_widths_mask` and the pure
  predicate `pasrocm_wavefront_compatible`. They are GPU-free and exist so a
  loader (and CI) can verify it opened the right library.

## 5. Supported-configuration policy

AMD's repositories do not offer a "latest patch" channel for 6.x/7.x: a bare
minor directory **is the initial release** (`apt/6.4` ships 6.4.0; there is no
`apt/6.4.0`), and every patch has its own directory. ROCm 10.x is the opposite:
one rolling `stable` suite with minor-scoped package names. The policy therefore
separates what we *claim* from what we *compile against*:

- **`sdk_line`** — the line we claim: `6.4`, `7.2`, `10.1`.
- **`pin`** — the exact SDK a job installs: repository directory for 6.x/7.x,
  exact package versions for 10.x.
- **`covers`** — the patch range the artifact may claim, with
  **`covers_evidence`** recording why. 7.2 claims `7.2.0-7.2.4` because
  `rocprim-dev` is 4.2.0 in both channels. 6.4 claims `6.4.0-6.4.4` only
  because a **compile-probe** configuration builds against `apt/6.4`
  (rocPRIM 3.4.0) alongside the 6.4.4 build (3.4.1).
- A new SDK release is a **reviewed manifest change**, never an automatic bump.
- Distro repositories carry ROCm 5.0-era packages under AMD's names with higher
  version strings, so `install-rocm.sh` pins the AMD origin above them and then
  verifies the device-bitcode tree; see §12.5.
- One build produces one GPU target; CI sets `REQUIRE_ARCH=1` so inference can
  never run there, because compiler acceptance of a `gfx…` target does not
  establish runtime support. A family keyword that expands to several targets is
  rejected by the build rather than silently fattening the library.
- Impossible combinations are absent from the manifest rather than retried:
  ROCm 6.x/7.x publish no Ubuntu 26.04 repository, so 26.04 runs 10.1 only, and
  a target is never requested below its `min_rocm`.
- **Profiles are the cost valve.** Each (line, target) is built once, on the
  oldest supported OS (glibc is forward compatible); newer OSes rebuild a small
  smoke set to prove the newer userland still works.

Supported matrix (`ci/supported-builds.json`; `pr` = 2 jobs, `smoke` = 16,
`full`/`release` = 55):

| Runner | Ubuntu repo dir | ROCm line | Pin | Compiler | Targets built in `full` |
|---|---|---|---|---|---|
| `ubuntu-22.04` | `ubuntu2204` | 6.4 | 6.4.4 | `hipcc` | all 13 that 6.4 ships |
| `ubuntu-24.04` | `ubuntu2404` | 6.4 | 6.4.4 | `hipcc` | `gfx1100`, `gfx942` |
| `ubuntu-22.04` | `ubuntu2204` | 6.4 | **6.4.0 (probe)** | `hipcc` | `gfx1100` |
| `ubuntu-24.04` | `ubuntu2404` | 6.4 | **6.4.0 (probe)** | `hipcc` | `gfx1100` |
| `ubuntu-22.04` | `ubuntu2204` | 7.2 | 7.2.4 | `hipcc` | all 13 that 7.2 ships |
| `ubuntu-24.04` | `ubuntu2404` | 7.2 | 7.2.4 | `hipcc` | `gfx1100`, `gfx942` |
| `ubuntu-22.04` | `ubuntu2204` | 10.1 | 10.1.0 | `clang` | all 19 that 10.1 ships |
| `ubuntu-24.04` | `ubuntu2404` | 10.1 | 10.1.0 | `clang` | `gfx1100`, `gfx942` |
| `ubuntu-26.04` | `ubuntu2604` | 10.1 | 10.1.0 | `clang` | `gfx1100`, `gfx942` |

## 6. Validation levels

Three distinct levels; a configuration must not be labelled above the level it
reached, and `manifest.json` records exactly which ones ran.

| Level | What is checked | What it does not prove |
|---|---|---|
| Compile | Headers, instantiated primitives, kernels, host code, linking (`-Xlinker --no-undefined`) | Correctness on hardware |
| ABI / package | Declared names vs golden list vs exports; package contents and metadata | GPU execution or performance |
| GPU runtime | Real operations against a CPU reference on a named OS/GPU/runtime | Untested devices or releases |


## 7. Phases and gates

| Phase | Work | Gate | Status |
|---|---|---|---|
| 1. ABI identity | `rocprim_shim.h`, hidden visibility, SONAME | `ci/check-abi.sh` passes with `--strict` | **open** — golden list, gate and the five `pasrocm_*` identity symbols exist; the public header, `-fvisibility=hidden` and SONAME do not |
| 2. CI-callable build | `COMPILER_KIND`/`COMPILER`, `OUT_DIR`, `VERSION`/`GIT_COMMIT`/`ROCM_VERSION`, `REQUIRE_ARCH`, one target per build | `REQUIRE_ARCH=1 ./src/rocprim/build.sh` without `GFX_ARCH` exits non-zero; a multi-target `GFX_ARCH` exits non-zero | **done** — both compiler kinds and both wavefront widths built locally |
| 3. Manifest + driver | `ci/gfx-targets.json`, `ci/supported-builds.json`, `ci/manifest.py`, `ci/install-rocm.sh`, `ci/build-linux.sh`, `ci/package.sh` | two consecutive builds of one config produce identical package contents | **done** — 55-job `full` matrix; byte-identical packaging |
| 4. Workflows | `shim-build.yml` (matrix from the manifest), `shim-release.yml` (tag → assets), `shim-gpu-tests.yml` (self-hosted, manual) | a PR run produces one package per enabled configuration; a tag run attaches them | **partly done** — `shim-build.yml` with `pr`/`smoke`/`full`; no release or GPU workflow |
| 5. Documentation | Supported-configuration table derived from the manifest; local reproduction notes | table matches the manifest exactly | **done** — §5 mirrors the manifest; `validate` rejects structural drift |
| 6. Wavefront correctness | table-driven width injection, hardware guards, runtime gate, identity self-test | no CDNA build can be wave32, and a wrong library refuses to run | **done** — see §12.3 |
| 7. GPU runtime | one-off rented MI instance (§13.6), `./test.sh` against the gfx942 package | named OS/GPU/runtime passes and the result is written into `manifest.json` | **decided, not run** — rental procedure recorded; every gfx9xx package stays `validation.gpu_runtime = not-run` until the run is logged |

Actions in workflows are pinned by commit SHA. Substantive build logic lives in
`ci/*.sh` and `build.sh`, not in YAML, so a failed CI build can be reproduced
verbatim on a workstation.

## 8. Packaging

Name: `pasrocm-shim-<version>-linux-x86_64-rocm-<sdk-line>-<gfx>.tar.gz`.

Contents: `lib/libpasrocm_shim_<gfx>.so`, `include/rocprim_shim.h` (once §4 item
1 lands), `pascal/*.pas`, `manifest.json`, `SHA256SUMS`, `LICENSE`, `INSTALL.md`.
A `<tarball>.sha256` file sits next to the archive for release uploads. The SDK
line appears in the package name, never in the on-disk library name: the loader
resolves by GPU, and `DT_NEEDED libamdhip64.so.<major>` already rejects a
cross-major mix at `dlopen`.

`manifest.json` fields: `schema`, `package`, `gpu_target`, `wavefront_size`,
`widths_mask`, `library`, `shim{name,version,abi_version}`, `git_commit`,
`os{name,codename}`, `host_architecture`,
`sdk{line,version,packages,covers,covers_evidence}`, `rocprim_version`,
`compiler{kind,path,version}`, `hip_version`, and
`validation{compile,abi,gpu_runtime}`. `rocprim_version` is read from
`include/rocprim/rocprim_version.hpp`, because package versions encode the SDK,
not
rocPRIM.

The package never bundles AMD runtime or driver files; it depends on a ROCm
installation being present, exactly as `libamdhip64.so.N` implies.

## 9. Risks

- **The wavefront width cannot be inferred from a macro** (§12.3), and the
  runtime property does not follow the compiled width either. Measured
  2026-11-13 on gfx1100 (RDNA3) with a standalone probe:

  | build | host `props.warpSize` | device `__builtin_amdgcn_wavefrontsize()` | launch rc |
  |---|---|---|---|
  | default | 32 | 32 | 0 |
  | `-mwavefrontsize64` | **32** | **64** | 0 (no error) |

  **RDNA3 proxy; RDNA4 unverified.** Two consequences:

  1. `-mwavefrontsize64` is *silently accepted* on RDNA3 and changes what the
     device code believes without failing the launch. It must never be exposed
     as a user knob — only `ci/gfx-targets.json` may select wave64, and a
     wave32-on-`__GFX9__` or wave64-on-`__GFX10/11__` combination is a compile
     error.
  2. A future wave64-RDNA gate must **not** compare `props.warpSize` with the
     compiled width; it must test the reported width against the table's
     `widths` set. That is what `pasrocm_wavefront_compatible(compiled, reported,
     allowed_mask)` does: single-width hardware requires agreement, dual-width
     hardware (mask `3`) accepts either width. Today every RDNA target pins 32,
     so nothing shipping is affected — but enabling a 64-bit RDNA4 entry must not
     trip a false rejection.

  What remains unproven: whether RDNA4 reports 32 or 64 in `props.warpSize` when
  a wave64 kernel runs. The 7900 XTX cannot answer it. The pure predicate makes
  the question testable the moment such a device appears, and the CI self-test
  already exercises every branch with fixed inputs.
- **Flaky batched tests.** `CTRMM_Batched_64` and `CTRSM_Batched_64` fail
  intermittently on tolerance (proven run-to-run nondeterminism). Any release
  gate needs an explicit policy: retry, widen tolerance, or exclude with a
  recorded reason.
- **6.x header drift.** `apt/6.4` ships rocPRIM 3.4.0 and `apt/6.4.4` ships
  3.4.1, so a 6.4.4 build does not prove 6.4.0; the probe configurations exist
  for exactly that reason. 7.2 has no such drift (4.2.0 in both).
- **Ubuntu 22.04 runners retire** on 2027-04-17. The 22.04 configurations then
  move to `container: ubuntu:22.04` (or `rocm/dev-ubuntu-22.04`) on a newer
  host runner; the build logic does not change.
- **`ubuntu-latest` is migrating to 26.04** during Oct 19 – Nov 19, 2026. Every
  workflow pins an explicit label, so the migration cannot move a job.
- **A new ROCm patch can remove a pinned version** from the rolling 10.x suite.
  The build then fails loudly with an apt "version not found" and needs a
  reviewed manifest bump — intended behaviour, not a flake.
- **Private repository minutes.** The `full` profile is 55 jobs (13 targets on
  the oldest OS per line, 2 targets on the newer ones, plus 2 probes); at ~4
  minutes per job that is ~220 minutes per run against a 2,000 min/month
  allowance. `pr` is 2 jobs and `smoke` is 16, so everyday work stays cheap and
  `full` runs nightly and on tags.
- **Artifact count.** One library per (line, target) means ~55 packages per full
  run at ~1.1 MB each; retention is 14 days and release uploads are per tag.
  This is the deliberate cost of dropping fat binaries.
- **Public API drift.** Until §4 item 3 lands, `rocprim_fill_kernel_int` and
  `rocprim_set_warp_size` are exported without being part of the contract.


## 10. Open decisions

| # | Decision | Resolution |
|---|---|---|
| 1 | Artifact name | **resolved** — one library per target, `libpasrocm_shim_<gfx>.so`; the SDK line lives in the package name only |
| 2 | Exact SDK releases to pin | **resolved** — lines 6.4 / 7.2 / 10.1, pinned to 6.4.4 / 7.2.4 / 10.1.0, plus a 6.4.0 compile probe (§5) |
| 3 | GPU runners | **resolved for now** — none; runtime tests stay manual/self-hosted (phase 7) |
| 4 | Symbol prefix | **resolved** — keep `rocprim_*`; the five `pasrocm_*` identity symbols are a recorded, deliberate exception (§4) |
| 5 | Layout | **resolved** — keep flat `src/rocprim/`, `ci/` beside it, `ci/gfx-targets.json` as the shared table |
| 6 | CDNA runtime execution | **resolved: one-off rented MI instance** for the first execution proof, procedure in §13.6. A self-hosted MI runner is reconsidered only when CDNA becomes a release-gating tier. Until the run is logged, every gfx9xx package is labelled `validation.gpu_runtime = not-run` |
| 7 | One meaning for the SDK stamp | **open follow-up, issue #1** — a local `build.sh` still stamps the HIP version while CI stamps the SDK line; stamp the line when detectable, or require `ROCM_VERSION` locally. Until then the loader treats it as reported metadata (§13.3) |

## 11. Reproduce a build locally

```bash
# one target per build -- the only supported form
GFX_ARCH=gfx1100 REQUIRE_ARCH=1 ./src/rocprim/build.sh

# the wave64 path (CDNA); the table picks the width and the wave flag
GFX_ARCH=gfx942 REQUIRE_ARCH=1 ./src/rocprim/build.sh

# ROCm 10.x compiler path (clang++ -x hip, what modern hipcc wraps)
COMPILER_KIND=clang GFX_ARCH=gfx1100 ./src/rocprim/build.sh

# a family keyword fails on purpose: it expands to several targets
GFX_ARCH=rdna3 ./src/rocprim/build.sh      # exit 1, lists the expansion

# show the resolved toolchain without building
DRY_RUN=1 GFX_ARCH=gfx1100 ./src/rocprim/build.sh

# CMake path (same table, same defines, same output name)
cmake -S src/rocprim -B build -DGPU_TARGET=gfx1100 -DROCM_VERSION=10.1.0
cmake --build build

# exactly what CI runs (installs the pinned SDK, builds, self-tests, ABI-checks, packages)
ci/install-rocm.sh --config ubuntu-24.04-rocm10.1 --dry-run   # inspect first
ci/build-linux.sh  --config ubuntu-24.04-rocm10.1 --target gfx1100

# the host-only identity probe on its own (no GPU needed)
cc -std=c99 -o /tmp/info src/rocprim/pasrocm_shim_info.c \
   -Lsrc/rocprim -l:libpasrocm_shim_gfx1100.so -Wl,-rpath,$PWD/src/rocprim
/tmp/info

# compile + run the test suite (detects the GPU, builds that target, links)
SHIM_RESOLVE_ONLY=1 ./test.sh     # show which shim would be used
./test.sh
```

The build writes `$OUT_DIR/build-info.env` (compiler, target, wavefront, widths
mask, SDK, commit), and `ci/package.sh` turns it into `manifest.json`, so an
artifact answers "which toolchain produced this, for which GPU?" on its own.

## 12. Measured upstream facts and open defects

### 12.1 Repository layout (measured 2026-11-13)

| Query | Result |
|---|---|
| `repo.radeon.com/rocm/apt/6.4` | **exists** = 6.4.0 channel (`hipcc 1.1.1.60400-47`); `apt/6.4.0` does not exist |
| `apt/6.4.1` … `apt/6.4.4` | exist individually; `apt/6.4.5` does not |
| `apt/7.2` | = 7.2.0 channel (`hipcc 1.1.1.70200-43`) |
| `apt/7.2.1` … `apt/7.2.4` | exist individually; `apt/7.2.5` does not |
| `apt/7.14`, `apt/7.13` | **do not exist** — 7.14 is a docs preview and a Docker tag only |
| `stable.repo.amd.com/rocm/core/packages/` | `ubuntu2204`, `ubuntu2404`, `ubuntu2604`, debian12/13, rhel8/9/10, sles15/16, azl3 |
| `stable` suite packages | `amdrocm-base10.0`/`10.1` coexist; `amdrocm-llvm10.1` = 10.1.0-3 (823 MB installed), `amdrocm-ccl-dev10.1` = 27 MB |
| ROCm 10.1 gfx packages | gfx908, gfx90a, gfx942, gfx950, gfx1010–1012, gfx1030–1036, gfx1100–1103, gfx1150–1153, gfx1200/1201, gfx1250 |
| clang target probe | `gfx942` accepted, `gfx1250` accepted, **`gfx1300` rejected** ("unsupported HIP gpu architecture") |
| Docker fallback tags | `rocm/dev-ubuntu-{22.04,24.04}:{6.4.4,7.2.4}-complete`, `…:10.1.0-full` (5.5–8.9 GB compressed); `dev-ubuntu-26.04` starts at 7.14 |
| Runner labels | `ubuntu-22.04` (deprecating 2026-09-17, unsupported 2027-04-17), `ubuntu-24.04`, `ubuntu-26.04` (26.04.1 LTS, codename `resolute`) |

### 12.2 Compiler equivalence (measured locally, ROCm 10.1.0)

`amdclang++ -x hip --rocm-path=/opt/rocm --offload-arch=gfx1100` on the two
shim translation units: 127 s, 14 MB library, **146 `rocprim_*` exports**,
`NEEDED libamdhip64.so.7` — identical to the `hipcc` output. The clang path is
therefore a drop-in replacement for hipcc, which the ROCm 7.2+ deprecation
requires.

### 12.3 Wavefront size: mechanism and measurements

**Closed.** The width is a build input, never a guess:

- `ci/gfx-targets.json` holds `wavefront` and `widths` per target. It is the
  only copy of that mapping.
- `src/rocprim/build.sh` and `CMakeLists.txt` look it up and inject
  `PASROCM_WAVEFRONT_SIZE`, `PASROCM_WAVEFRONT_WIDTHS_MASK`, `PASROCM_GPU_TARGET`
  and `PASROCM_ROCM_VERSION`, and add `-mwavefrontsize64` **only** when the table
  says 64.
- `rocprim_shim.cpp` `#error`s when any definition is missing, `static_assert`s
  the width is 32 or 64, and refuses a wave32 build on `__GFX9__` or a wave64
  build on `__GFX10__`/`__GFX11__`. RDNA4 gets no width guard: it can run either
  width, and `gfx1250` rejects the wave64 flag outright, so the table decides.
- At first use the shim compares the device's reported width with the table's
  `widths` via the pure predicate `pasrocm_wavefront_compatible()`; a mismatch
  latches `hipErrorInvalidDevice` and no kernel is launched.
- `pasrocm_shim_target()`, `pasrocm_shim_wavefront_size()`,
  `pasrocm_shim_rocm_version()` and `pasrocm_shim_wavefront_widths_mask()` are
  GPU-free accessors, so `src/rocprim/pasrocm_shim_info.c` can assert in CI that
  a build carries the parameters the matrix requested. That is the difference
  between "it compiled" and "it compiled with the right parameters".

Why no macro is involved (measured on clang 24 / ROCm 10.1):

| Probe | Result |
|---|---|
| `__AMDGCN_TARGET__` | **not defined** — the original bug |
| `__AMDGCN__` | defined on gfx942 **and** gfx1100 |
| `__AMDGPU__` | defined on gfx942 **and** gfx1100 |
| `__AMDGCN_WAVEFRONT_SIZE` | **absent for every target** |
| `__builtin_amdgcn_wavefrontsize()` | correct, folds at `-O2`, but is **not an integral constant expression**, so it cannot size a template or feed `static_assert` |
| `__GFX9__` / `__GFX10__` / `__GFX11__` / `__GFX12__` | present and correct — used only as compile-time hardware guards |
| `ROCPRIM_NAVI` | 1 on gfx1100, 0 on gfx942 (upstream equivalent of the guard) |

The launch geometry is host code, compiled once, so no device-only signal can
reach it; the injected constant is what makes host and device agree.
**A third, independent guard exists upstream.** rocPRIM itself refuses a virtual
wavefront larger than the hardware's: `rocprim/detail/check_virtual_wave_size`
`static_assert`s `VirtualWaveSize <= size_from_target()`. Measured by instantiating
`rocprim::warp_reduce<int, 64>` for gfx1100 (RDNA3, wave32), even with
`-mwavefrontsize64`:

```text
rocprim/intrinsics/arch.hpp:268: static assertion failed
  in instantiation of 'check_virtual_wave_size<64, wavefront::target::size32>'
```

Two consequences: a wave64 build for RDNA cannot be produced by accident (the
compiler refuses before our guard is reached), and the "instantiate both widths
and dispatch at runtime" design is **impossible** with rocPRIM 4.6 — one library
per target is the only shape that works. The two guards are complementary:
rocPRIM rejects `virtual > hardware`; our `#error` rejects a wave32 build on CDNA,
which rocPRIM permits (`32 <= 64`) and which would silently halve the geometry we
launch.
 §9 records
the runtime measurement that makes an equality check wrong on dual-width parts.

### 12.4 Defects fixed alongside

1. **Stale architecture family map.** The previous `build.sh` mapped
   `cdna5 → gfx1010/gfx1011` (RDNA1 parts), `rdna5 → gfx1300/gfx1301` (rejected
   by the 10.1 toolchain) and put gfx940/941 in CDNA3 (not published for 10.x).
   The family map now only produces a helpful error, and the target list lives in
   `ci/gfx-targets.json`.
2. **Two exports outside the contract.** `rocprim_fill_kernel_int` (a
   `__global__` kernel in the dynamic symbol table) and `rocprim_set_warp_size`.
   Both disappear with `-fvisibility=hidden` plus the export macro in §4 item 3;
   until then `ci/check-abi.sh` reports them as warnings.

### 12.5 Provisioning defect found by the first CI run (2026-11-14)

The first `smoke` run built every 10.1 job and failed every 6.4 and 7.2 job —
on both Ubuntu releases and both GPU targets — with:

```text
clang++: error: cannot find ROCm device library; provide its path via
  '--rocm-path' or '--rocm-device-lib-path', or pass '-nogpulib'
```

`build.sh` does pass `--rocm-path=$ROCM_ROOT`; the device bitcode itself was
wrong. Measured from the run's apt log and the repository indexes:

| source | `rocm-device-libs` version |
|---|---|
| `repo.radeon.com/rocm/apt/6.4.4` | `1.0.0.60404-129~22.04` |
| `repo.radeon.com/rocm/apt/7.2.4` | `1.0.0.70204-93~22.04` |
| Ubuntu `jammy/universe` | `5.0.0-1` |

AMD versions the package `1.0.0.<rocm-build>`; the distro copy is `5.0.0-1`, so
**plain apt prefers the distro package** and installs ROCm 5.0-era bitcode that
does not live under `--rocm-path`. `rocminfo` collides the same way (both
`5.0.0-1`). The 10.1 configurations were unaffected because their packages are
named `amdrocm-*` and have no distro namesake.

Fixed in `ci/install-rocm.sh`, in the `radeon-apt` branch only:

1. `/etc/apt/preferences.d/rocm.pref` pins `origin repo.radeon.com` at priority
   1001 for every package name the AMD repository publishes, so the version
   scheme cannot lose the race again (this also fixes `rocminfo`);
2. after resolving `ROCM_ROOT`, the script requires a bitcode tree containing
   `ockl.bc` and fails with the installed `rocm-device-libs` version and the pin
   file's path if it is absent — the collision cannot silently come back.

This is an install/provisioning defect, not a target rejection: no `min_rocm`
entry changes. It is also not proof that 6.4/7.2 compile — that is what the next
`smoke` run measures.

## 13. The Pascal loader (runtime binding)

The shim is never linked. `rocprim_shim_loader.pas` opens the library for the
GPU that is present, on the first rocPRIM call, and `rocprim_bindings.pas`
dispatches through a pointer table that the loader fills.

| File | Role |
|---|---|
| `src/rocprim/rocprim_shim_abi.inc` | hand-edited ABI specification; never compiled, read by the generator and by `ci/check-abi.sh` |
| `src/rocprim/rocprim_shim_api.pas` | **generated**: 149 typed pointers, symbol-name table, slot table, `ROCPRIM_SHIM_SYMBOL_COUNT` |
| `src/rocprim/rocprim_bindings.pas` | **generated**: the 149 public wrappers with the original names and signatures |
| `src/rocprim/rocprim_shim_loader.pas` | hand-written: device detection, dlopen, identity gate, binding, `TShimLoadInfo` |
| `ci/gen-shim-bindings.py` | generates the two units; `--check` is a CI gate |

Dependency direction: `bindings` (interface: types + wrappers) ← `api`
(interface: pointers) and `bindings` (implementation) → `loader` → `api`. The
edge back from `api` to `bindings` is the only cycle and it is
implementation-only on the `bindings` side, which FPC accepts (verified on
3.3.1).

Why `rocprim_bindings.pas` is generated instead of a two-line re-export: **FPC
does not re-export identifiers from an `interface uses` clause** (measured on
3.3.1 with a three-unit probe; Delphi does). A unit that merely `uses` the
pointer table would not keep `uses rocprim_bindings; rocprim_reduce_int(...)`
compiling, so the 149 names have to be declared in the unit callers already
use. Generating them from one specification keeps the names, the pointer table
and the golden ABI list from drifting apart.

### 13.1 Load sequence

Fixed order; each step can refuse, and none of them launches a kernel.

1. **Device.** `hipGetDeviceProperties(prop, index)`; index from
   `PASROCM_DEVICE` (default 0). Gfx target = `prop.gcnArchName` normalized
   (strip from `:` on, lowercase). `PASROCM_EXPECT_GFX` replaces the query for
   host-only checks.
2. **File.** `PASROCM_SHIM` (authoritative, no fallback), then
   `PASROCM_SHIM_DIR`, then beside the executable / `../lib` / `../../lib`,
   `~/.local/lib/pasrocm`, `/usr/local/lib/pasrocm`, `/usr/lib{,64}/pasrocm`,
   `./src/rocprim`, then the bare file name (so `LD_LIBRARY_PATH` still works).
   Only `PASROCM_SHIM` failing to exist is an error before the search.
3. **Identity.** Bind the five `pasrocm_*` exports by index and read target,
   compiled width, widths mask and SDK stamp. This happens *before* any other
   pointer is written.
4. **Gate.** `PasROCmClassifyIdentity` in Pascal decides the verdict from the
   values plus the shim's own `pasrocm_wavefront_compatible()` result, which is
   passed in and never re-implemented (single source of truth, already covered
   by the C self-test in `pasrocm_shim_info.c`).
5. **Bind.** Resolve all 149 into a temporary array and publish only when none
   is missing: the table is never half-bound.
6. **Latch.** Success stores `TShimLoadInfo`; failure stores the message and
   every later call raises it again instead of retrying a broken load.

Every wrapper calls the loader first, so the first rocPRIM call is the one that
opens the library; a program that imports the unit but never calls rocPRIM pays
nothing.

### 13.2 What each identity export gates

| Export | Check | Failure |
|---|---|---|
| `pasrocm_shim_target` | equals the detected gfx (case-insensitive) | `EShimIdentityError`, both targets in the message |
| `pasrocm_shim_wavefront_size` | is 32 or 64, and its bit is in the mask | `EShimIdentityError` |
| `pasrocm_shim_wavefront_widths_mask` | 1..3, covers the compiled width | `EShimIdentityError` |
| `pasrocm_wavefront_compatible` | returns 1 for (compiled, reported, mask) | `EShimIdentityError` with the three numbers |
| `pasrocm_shim_rocm_version` | see 13.3 | warning by default |

`reported` is `prop.warpSize`; with `PASROCM_EXPECT_GFX` and no device it is
derived from the mask when the mask is single-valued (1 → 32, 2 → 64), the same
reasoning the C self-test uses. Dual-width hardware (mask 3) accepts either
width, per §9.

### 13.3 The SDK stamp is reported, not guessed

`pasrocm_shim_rocm_version()` is read into `TShimLoadInfo` and compared only
where a comparison is real. It is **not** a default hard gate, for four
measured reasons:

- ROCm 10.1 ships HIP 7.16 (`hipRuntimeGetVersion` → 71626385), so the HIP
  version does not identify the SDK line;
- 7.2 and 10.1 share `libamdhip64.so.7`, so a 7.2 artifact *loads* under 10.1;
- the ROCm 10.x install layout has no `/opt/rocm/.info/version` (checked on the
  development machine), so there is no reliable on-disk SDK line to compare to;
- a local `build.sh` without `ROCM_VERSION` stamps the **HIP** version
  (`7.16.26385-0000000`), while CI stamps the SDK line (`10.1.0`).

What the loader does instead: `PASROCM_EXPECT_ROCM=<prefix>` makes the stamp an
identity gate; otherwise a stamp whose major maps to a different HIP major than
the running one (`6.x → 6`, `7.x → 7`, `10.x → 7`) becomes a warning recorded in
`TShimLoadInfo.Warnings`, and `PASROCM_STRICT_ROCM=1` turns that warning into a
fatal error. The HIP major itself is still enforced by `DT_NEEDED`.

**Follow-up, tracked as issue #1 (not silently dropped):** make a local `build.sh`
stamp the SDK line when it can be detected, or require `ROCM_VERSION` for local
builds, so one export stops returning two kinds of value. Until then this
section is the contract. A package sidecar — `libpasrocm_shim_<gfx>.so.info`
with `rocm=<line>` — would let the loader compare against the artifact instead
of the install, and is the second option if a hard gate is wanted.

### 13.4 Memory ordering

The steady state is one test of `GState` per shim call. That is safe on x86_64:
an aligned 32-bit load is atomic and the store that publishes `ssLoaded` is
ordered after the slot writes (TSO), so a reader that sees `ssLoaded` sees a
complete table. On a weakly ordered target (aarch64) this must become a barrier
or an unconditional `PasROCmShimLoad`; an uncontended critical section is
negligible next to a kernel launch, so the simple fix is available. The fast
path is documented at the point of use as well.

### 13.5 Limits and tracked follow-ups

- **One library per process.** A process opens the shim for `PASROCM_DEVICE`
  (default 0). A machine with two GPUs of different architectures cannot use
  both through one process, which is inherent to one-library-per-target and is
  recorded here rather than discovered at run time.
- **The live environment is what counts.** The loader reads the environment with
  `getenv`, not FPC's `GetEnvironmentVariable`: measured on 3.3.1, the latter
  snapshots the environment at start-up, so a host program that sets
  `PASROCM_SHIM_DIR` after start would be ignored.
- **`rocprim_device.pas` cannot link.** It still declares 27 procedures with a
  bare `external;` (no library, no shim counterpart). Any program that uses the
  `rocprim` facade would fail to link. Tracked as issue #2: map those names onto
  the 149-symbol API or retire the unit. It deliberately does not ride along
  with this change.
- **Two exports outside the contract.** `rocprim_fill_kernel_int` and
  `rocprim_set_warp_size` remain exported-but-undeclared (§12.4). They fold into
  the phase-1 `-fvisibility=hidden` work; the public surface should end at
  exactly 149 symbols plus the SONAME.
- **Local SDK stamp.** §13.3 follow-up, issue #1: stamp the SDK line, or
  require `ROCM_VERSION`, so `pasrocm_shim_rocm_version()` has one meaning.

### 13.6 CDNA execution proof: rented, one-off (decided)

**Decision.** The first execution proof for gfx942 runs on a **one-off rented
MI instance**, not a self-hosted runner. There is no MI hardware in the project,
the gap is a single correctness question — do the wave64 kernels run and match
the CPU reference? — and a rental keeps the cost bounded without owning and
maintaining a runner. A self-hosted MI runner is reconsidered only if CDNA
becomes a release-gating tier, i.e. if every release must re-prove gfx9xx
instead of once per SDK line.

**Scope.** One thing only: turn `validation.gpu_runtime = not-run` into a
recorded `pass` (or a real defect) for the 10.1 / gfx942 package. Not a
benchmark, not a compatibility sweep.

**Procedure** — the checklist the run follows, so the next one is not an
adventure:

1. Rent an MI300X-class instance (gfx942) with Ubuntu 24.04 + ROCm 10.1.0,
   matching `ci/supported-builds.json` (`ubuntu-24.04-rocm10.1`).
2. Confirm the hardware before anything else: `rocm_agent_enumerator` prints
   `gfx942`, `hipconfig --version` reports HIP 7.16.
3. Take the CI artifact for that configuration, or build it on the instance:
   `COMPILER_KIND=clang GFX_ARCH=gfx942 REQUIRE_ARCH=1 ./src/rocprim/build.sh`.
4. Run the Pascal suite against the package's `lib/`:
   `PASROCM_SHIM_DIR=<pkg>/lib ./test.sh --all --format=plain`.
5. Freeze the evidence: test count, failures, `rocminfo` gfx target, SDK
   version, the shim's `TShimLoadInfo` (target, wavefront, mask) and the
   artifact's `manifest.json`.
6. Flip `validation.gpu_runtime` from `not-run` to `pass` in that manifest and
   record date + SDK + instance class in the package notes. Until step 6, the
   labels do not move and no README or release note claims CDNA support.

**If it fails**, it is a shim/wave64 defect, not a packaging one. File it with
the numbers from step 5 — `compiled` width, `props.warpSize`, widths mask — the
three values that distinguish "wrong library" from "wrong kernel".
