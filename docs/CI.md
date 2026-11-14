# Continuous integration

`.github/workflows/shim-build.yml` compiles the shim for every supported
(Ubuntu, ROCm) pair and GPU architecture group, and uploads one package per
(configuration, arch group). The matrix is not written in YAML — it is generated
from the two data files below, so adding or retiring a configuration is a
one-file change.

| layer | file |
|---|---|
| GPU targets, wavefront widths, minimum ROCm | `ci/gfx-targets.conf` |
| SDK pins, profiles | `ci/supported-builds.conf` |
| queries (matrix, env, wavefront, mask) | `ci/manifest.sh` |
| SDK provisioning | `ci/install-rocm.sh --config <id>` |
| build, ABI check, package | `ci/build-linux.sh --config <id> --arch-group <group>` |
| ABI gate | `ci/check-abi.sh` |
| design, measured upstream facts, risks | `ci/DESIGN.md` |

## Profiles

Profiles keep a private repository inside its Actions allowance, measured on
2026-11-14:

| trigger | profile | jobs | billed minutes |
|---|---|---|---|
| pull request | `pr` | 2 | ~5 |
| push to `main` | `smoke` | 16 | ~37 |
| nightly cron | `smoke` | 16 | ~37/night |
| Sunday cron | `full` | 54 | ~125 |

That is about 1,640 minutes a month against the 2,000-minute allowance, leaving
room for pull requests and manual dispatches. A manual run can request any
profile; nightly `full` would be ~3,800 minutes and needs a public repository or
a bigger plan (`ci/DESIGN.md` §9).

```bash
ci/manifest.sh validate                    # self-check the data files
ci/manifest.sh matrix --profile smoke      # what CI would run

# reproduce one CI job verbatim (installs the pinned SDK, so inspect first)
ci/install-rocm.sh --config ubuntu-24.04-rocm10.1 --dry-run
ci/build-linux.sh  --config ubuntu-24.04-rocm10.1 --target gfx1100
```

## Validation levels

Every package states how far it was proven, in `manifest.json`:

| field | meaning |
|---|---|
| `validation.build` | the shim compiled and passed the export/ABI gate |
| `validation.abi` | exported surface equals the specification (151) |
| `validation.load` | the loader found the library and bound all 151 entries |
| `validation.gpu_runtime` | kernels actually executed on the target GPU |

**CDNA artifacts are compile-verified and ABI-checked only.** Nothing in this
repository executes gfx9xx kernels, so every gfx908/gfx90a/gfx942/gfx950
package carries `validation.gpu_runtime: not-run`. Their wavefront handling is
proven at build time (guards, identity self-test) and the launch geometry is
proven by construction, but real kernel execution on MI hardware stays
unverified until the one-off rented-MI proof in `ci/DESIGN.md` §13.6 runs and
its result is written into the package manifest.

## Patch ranges

A configuration claims a range of patches it can back up: `7.2` claims
`7.2.0-7.2.4` because the rocPRIM headers are identical in both channels, while
`6.4` claims `6.4.0-6.4.4` only because dedicated probe jobs compile against the
6.4.0 channel as well. `min_rocm` in `ci/gfx-targets.conf` means "the oldest
SDK line that produces a working package for this target", not the oldest that
can be made to compile.

## Release flow

`.github/workflows/shim-release.yml` runs on a `v*` tag:

1. assert the tag equals the version in the project file (a tag is never the
   only place a version lives);
2. build every (configuration, target) pair from the data files;
3. refuse any package whose `abi` validation is 0;
4. emit per-target manifests, `SHA256SUMS`, and the library SHA256 of each
   `.so` in the release notes.

Nothing in a package depends on the build machine: no timestamps, no absolute
paths, and `built_at` is deliberately absent, so a rebuild of the same inputs
produces the same file names and the same library hash. That hash is the
invariant to compare across a rebuild — the tarball itself changes if you flip
`gpu_runtime` in the evidence file (`ci/DESIGN.md` §12.1).

## Local gates

| command | what it proves |
|---|---|
| `./test.sh` | 1,702 assertions, including the loader's refusal paths |
| `ci/check-abi.sh --strict` | 151 exports, nothing else |
| `ci/manifest.sh validate` | the data files parse and are consistent |
| `./src/rocprim/build.sh` | one target builds and links against the SDK |
