# Installing PasROCm

The Pascal units are ordinary FPC units; the only part with moving pieces is
the native shim they call into. This page covers the shim, the loader's search
order, and what a release package contains.

## What a package contains

```text
pasrocm-shim-<ubuntu>-<rocm>-<gfx>/
  lib/libpasrocm_shim_<gfx>.so   the shim; soname equals the file name
  include/rocprim_shim.h         the complete exported C surface (151 symbols)
  manifest.json                  provenance: toolchain, rocPRIM version, claims
  INSTALL.md                     this file
  LICENSE
```

`manifest.json` records the Ubuntu and ROCm coordinates, the GPU target and its
wavefront width, the rocPRIM version the shim was compiled against, the exact
compiler and version string, the library SHA256, and a `validation` block saying
how far the package was proven (see `docs/CI.md`).

## Where the loader looks

On the first rocPRIM call, `rocprim_shim_loader.pas` runs a fixed sequence and
refuses anything that does not belong to this machine:

1. read the device architecture from HIP (`hipGetDeviceProperties`, or
   `PASROCM_EXPECT_GFX` for host-only checks);
2. find and open `libpasrocm_shim_<gfx>.so`;
3. read `pasrocm_shim_target`, `_wavefront_size`, `_rocm_version`,
   `_wavefront_widths_mask`, `_abi_version` and `_build_info` — pure exports, no
   kernels involved;
4. refuse the library if its ABI revision differs from the one the Pascal units
   were generated for, or if the target or the wavefront width disagrees with
   the device;
5. resolve all 151 entry points, and publish them only if every one resolves.

A wrong library therefore raises `EShimIdentityError` naming both targets,
instead of launching kernels with the wrong geometry. There is no nil-function
window: a failure is latched, and every later call repeats it.

| Variable | Meaning |
|---|---|
| `PASROCM_SHIM` | exact file to open; overrides the search, no fallback |
| `PASROCM_SHIM_DIR` | directory holding `libpasrocm_shim_<gfx>.so` |
| `PASROCM_DEVICE` | device index to detect (default `0`) |
| `PASROCM_EXPECT_GFX` | trust this target instead of querying HIP (CI) |
| `PASROCM_EXPECT_ROCM` | require the shim's SDK stamp to start with this |
| `PASROCM_STRICT_ROCM` | turn an SDK-line warning into a fatal error |

Without `PASROCM_SHIM` or `PASROCM_SHIM_DIR` the loader also looks next to the
executable, in `lib/` above it, in `~/.local/lib/pasrocm`, in
`/usr/local/lib/pasrocm` and `/usr/lib{,64}/pasrocm`, in `./src/rocprim` (a
checkout), and finally by bare file name, so `LD_LIBRARY_PATH` keeps working as
it always did.

`PASROCM_EXPECT_GFX` is also how the loader's tests prove the refusal path
without a second GPU: point `PASROCM_SHIM` at the real library and claim a
different target.

## The ABI specification

The exported surface lives in `src/rocprim/rocprim_shim_abi.inc`. The two units
`rocprim_shim_api.pas` and `rocprim_bindings.pas`, the public header
`rocprim_shim.h` and the linker version script `rocprim_shim.map` are generated
from it and from the C++ definitions; CI rejects a stale copy.

```bash
cd src/rocprim && ./build.sh          # regenerates, then compiles
ci/check-abi.sh                       # exports vs. specification
ci/check-abi.sh --strict              # and nothing else in the table
```

Edit the specification, not the generated files.

## What each SDK line supports

| SDK line | rocPRIM | `rocprim_merge_inplace_int` |
|---|---|---|
| 6.4 | 3.4.x | returns `hipErrorNotSupported` — the primitive does not exist in that rocPRIM |
| 7.2 | 4.2.0 | supported |
| 10.1 | 4.6.0 | supported |

Everything else in the 151-function surface is available on all three lines.
The symbol is always exported, so the gap is visible to the caller at run time
rather than a missing symbol at load time (`ci/DESIGN.md` §12.6).

`gfx950` (CDNA4) requires ROCm 7.2 or newer: the 6.4 rocPRIM cannot compile for
that architecture (no 128-bit atomics), which is a rocPRIM floor, not a
compiler gap (`ci/DESIGN.md` §12.7).

## Reading a build log

Every build prints the toolchain it actually found, so the log answers "which
compiler produced this `.so`?" without following symlinks by hand:

```text
== ROCm root : /opt/rocm
== compiler  : /opt/rocm/core-10.1/bin/hipcc  [hipcc]
== compiler v: AMD clang version 24.0.0git (...) (via hipcc)
== HIP ver   : 7.16.26385-0000000
== lib dir   : /opt/rocm/lib
== target    : gfx1100  (wavefront 32)
== output    : /path/to/PasROCm/src/rocprim/libpasrocm_shim_gfx1100.so
...
== Linked against:
 (NEEDED)    Shared library: [libamdhip64.so.7]
== Loads at runtime:
    libamdhip64.so.7 => /opt/rocm/lib/libamdhip64.so.7
== rocprim_* exports: 151
== build info: /path/to/PasROCm/src/rocprim/build-info.env
```

`DRY_RUN=1 ./build.sh` prints that same header without compiling.

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `EShimIdentityError: library is gfx942, device is gfx1100` | one library per target; the wrong file was found | set `PASROCM_SHIM` or `PASROCM_SHIM_DIR` |
| `ABI revision mismatch` | Pascal units and shim from different revisions | rebuild both from one checkout |
| `unknown type name 'hipDriverEntryPointQueryResult'` | mixed HIP headers (a copied `hip/` next to the shim) | take the headers only from the selected ROCm install |
| `hipErrorInvalidDevice` on the first kernel | wavefront width mismatch: the library was built for the other width | rebuild with the machine's target from `ci/gfx-targets.conf` |
| a `6.4` package refuses `gfx950` | rocPRIM 3.4.x has no CDNA4 atomics | use a 7.2 or 10.1 package |

## Using the library without the shim sources

Point `PASROCM_SHIM_DIR` at the package's `lib/` and compile with the units:

```bash
fpc @pasrocm.cfg program.pas     # in a checkout
export PASROCM_SHIM_DIR=/opt/pasrocm/lib   # with an installed package
```

Nothing links the shim, so the same binary works with any package built for
its GPU target; replacing the package only requires restarting the process.
