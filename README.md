# PasROCm

Pascal bindings for [rocPRIM](https://github.com/ROCm/rocPRIM) — AMD's GPU
parallel primitives library — for reduction, scan, radix and merge sort, and
block operations, driven from Free Pascal on AMD Radeon and Instinct GPUs.

No reimplementation: Pascal calls rocPRIM's own primitives through a thin
generated shim, at the throughput rocPRIM's own tests measure.

**Status:** 0.1.0, pre-1.0. RDNA packages are exercised on real hardware; CDNA
(gfx9xx) packages are compile- and ABI-verified only. `docs/INSTALL.md` records
what 1.0.0 will freeze.

## Coverage

rocPRIM's primitives are exposed as plain Pascal calls. The verbs you would
expect from a parallel unit are all here:

| primitive | operation |
|---|---|
| `rocprim_reduce_int` | device-wide associative reduction |
| `rocprim_inclusive_scan_int`, `rocprim_exclusive_scan_int` | prefix sum |
| `rocprim_radix_sort_keys_int` | stable LSD key sort |
| `rocprim_radix_sort_pairs_int` | key/value sort |
| `rocprim_merge_sort_keys_int` | comparison sort |
| `rocprim_run_length_encode_int` | run detection |
| `rocprim_find_first_of_int` | intersection probe |
| `rocprim_merge_int` | sorted merge |
| `rocprim_binary_search_int` | lower bound per element |
| `rocprim_segmented_reduce_int` | per-segment reduction |
| `rocprim_segmented_inclusive_scan_int` | scan that restarts at each segment |
| `rocprim_histogram_even_int` | equal-width binning |
| `rocprim_block_reduce_int`, `rocprim_warp_reduce_int32` | block- and warp-level variants |

Device-wide entries take the **output pointer first**, then the input, the
element count, any initial value, and the stream, and return a `hipError_t`:

```pascal
function rocprim_reduce_int(d_out: PCint; d_in: PCint; num_items: csize_t;
  initial_value: cint; stream: hipStream_t): hipError_t;
```

You allocate the input and output buffers with `hipMalloc`; the shim allocates
and frees rocPRIM's temporary storage. Block and warp variants add their
geometry parameters. The full surface is **144 rocPRIM entry points plus 7
identity exports** -- `rocprim_*_double`, `_float`, `_uint`, `_uint8`, `_uint64`,
radix-digit and pairs variants among them -- fixed by
`src/rocprim/rocprim_shim_abi.inc`.

## Requirements

- Free Pascal 3.2.2 or newer (`fpc`)
- AMD GPU: RDNA 1–4 (including 3.5) or CDNA 1–4 — the 19 targets in
  `targets.conf`, from `gfx1010` and `gfx908` to `gfx1201` and `gfx950`
- ROCm 6.4, 7.2 or 10.1 — with the ROCm versions you install
  (`hipcc`/`amdclang++` and `libamdhip64`)
- The rocPRIM native shim (below) — one shared library per GPU target

CDNA (`gfx9xx`) packages are compile- and ABI-verified only: no MI hardware has
executed them yet. RDNA packages run on real GPUs.

`pacman -S fpc`, `apt install fpc`, or `dnf install fpc` is enough for FPC.
You need `hipcc` and the HIP runtime, not a full ROCm install; on most systems
`rocminfo` is the quickest check that a working runtime is present.

## Quick start (3 commands)

```bash
source setup_env.sh          # optional if ROCm is already on PATH
cd src/rocprim && ./build.sh  # detect ROCm and this machine's GPU and build
cd ../.. && ./test.sh        # run the test suite
```

`./build.sh` needs nothing but FPC and ROCm. It prints the toolchain it
selected, the detected target, and the library it produced.

No GPU in this machine? Name the target instead of detecting it:
`GFX_ARCH=gfx1100 ./build.sh`. Add `DRY_RUN=1` to print the resolved toolchain
and stop before compiling.

## Environment setup

ROCm is usually installed already (for PyTorch, or the distro's `rocm` package).
You need `hipcc` on `PATH` and `libamdhip64.so` findable:

```bash
export ROCM_PATH=/opt/rocm     # only if /opt/rocm is not the actual path
export PATH="$ROCM_PATH/bin:$PATH"
source setup_env.sh            # or let this do it
```

`setup_env.sh` detects `/opt/rocm` (and `/opt/rocm-<ver>`, and ROCm 10.x's
`/opt/rocm/core-<ver>`), verifies `hipcc` and the HIP library, then prints the
result. It does not modify files. `rocm-smi` or `rocminfo` confirms the GPU:
RDNA 3 is `gfx1100`, RDNA 4 `gfx1200`, CDNA 3 `gfx942`, CDNA 4 `gfx950`.

## How to use PasROCm

The shim library is not linked: it is opened at runtime and looked for like any
other shared library. So you need the search path, and nothing else. Pick one:

### Option 1 — native FPC package (`fpmake`)

Install once, then any project can `uses rocprim;`:

```bash
fpc fpmake.pp          # build the installer (once, from the repository root)
./fpmake install       # into this FPC's unit directory
```

That writes into the compiler's own unit directory, so it needs write access
there. To keep it user-local, choose the directory yourself:

```bash
./fpmake install -B ~/pasrocm
fpc -Mobjfpc -Fu~/pasrocm/units/x86_64-linux/pasrocm -Fl/opt/rocm/lib program.pas
```

`-Mobjfpc` is required (the units carry no `{$mode}` of their own) and `-Fl`
points the linker at the HIP runtime.

Usage — the shape of a call, from `examples/rocprim/test_reduce.pas`:

```pascal
uses ctypes, hip_bindings, rocprim_bindings;

hipStreamCreate(@stream);
hipMalloc(@d_in, num_items * sizeof(cint));
hipMalloc(@d_out, sizeof(cint));
hipMemcpy(d_in, @h_in[0], num_items * sizeof(cint), hipMemcpyHostToDevice);

rocprim_reduce_int(d_out, d_in, num_items, 0, stream);   // 0 = initial value

hipMemcpy(@h_out, d_out, sizeof(cint), hipMemcpyDeviceToHost);
```

Every call returns a `hipError_t`; the example checks each one and compares the
GPU result with a CPU computation.

### Option 2 — response file, no install (`pasrocm.cfg`)

A checked-in `pasrocm.cfg` on the command line gives the paths without copying
anything or touching `~/.fpc.cfg`:

```bash
fpc @pasrocm.cfg program.pas
```

`pasrocm.cfg` sets `-Fusrc -Fusrc/rocprim -Isrc` for the units and
`-Fl/opt/rocm/lib -Fl/opt/rocm/lib64` for the HIP runtime. The shim is never
linked: the loader opens it at run time.

### Option 3 — Lazarus package (`pasrocm.lpk`)

Open `pasrocm.lpk` in Lazarus and click *Use → Add to Project*. The package
registers `src` and `src/rocprim` as unit paths and sets `-Mobjfpc -Sh`, so the
IDE build compiles the units. The shim is found at run time, not linked.

### Can I use more than one?

Yes, they all point at the same directory. Installing through `fpmake` and then
using `pasrocm.cfg` in another project is fine; the loader finds one library,
and each process opens it once.

## Getting the rocPRIM native shim

`libpasrocm_shim_<gfx>.so` is the native library that makes rocPRIM's
primitives callable from Pascal. Two ways to get it:

### Option A — use a pre-built package

Download `pasrocm-shim-0.1.0-linux-x86_64-rocm-7.2-gfx1100.tar.gz` from the
releases page — the name carries the version, platform, SDK line and GPU target —
unpack it into a directory, and point the loader at its `lib/`:

```bash
mkdir pasrocm-shim
tar xzf pasrocm-shim-0.1.0-linux-x86_64-rocm-7.2-gfx1100.tar.gz -C pasrocm-shim
export PASROCM_SHIM_DIR=$PWD/pasrocm-shim/lib
```

The archive has no top-level directory of its own, so unpack it into one.

### Option B — build it locally

```bash
cd src/rocprim && ./build.sh
```

`build.sh` finds ROCm, detects this machine's GPU, and builds only that target.
For cross-building, pass `GFX_ARCH=gfx942` (one target per library, by design).

### What each SDK line supports

The three SDK lines are built from different rocPRIM versions, and one primitive
depends on a rocPRIM feature that only arrived after 3.4:

| SDK line | rocPRIM | `rocprim_merge_inplace_int` |
|---|---|---|
| 6.4 | 3.4.x | returns `hipErrorNotSupported` — the primitive does not exist in that rocPRIM |
| 7.2 | 4.2.0 | supported |
| 10.1 | 4.6.0 | supported |

Everything else in the 151-function surface is available on all three. The
symbol is always exported, so this is a capability gap the caller can see and
test, not a missing symbol at load time.

Target coverage follows `targets.conf`. `gfx950` (CDNA4) needs ROCm 7.2 or
newer: the 6.4 rocPRIM cannot compile it (no 128-bit atomics for that
architecture) -- a rocPRIM capability floor, not a compiler one.

**One target per build.** `GFX_ARCH` names exactly one GPU target, for example
`gfx1100` (RDNA 3 — Radeon RX 7000) or `gfx942` (CDNA 3 — MI300). A family
keyword such as `rdna3` is rejected on purpose: it expands to several targets,
and this project ships one small library per architecture instead of a fat
binary. The library is named after the target it was built for, and the Pascal
loader picks the file that matches the detected GPU.

`ROCM_PATH` accepts both install layouts. ROCm 10.x keeps the versioned tree at
`/opt/rocm/core-<ver>` and points `/opt/rocm` at it through alternatives, while
6.x/7.x releases installed `/opt/rocm-<ver>` directly. Detection prefers
`/opt/rocm` when it holds a compiler, then takes the newest `/opt/rocm-<ver>`
that also has one. A stray header-only directory can never be selected. If
nothing usable exists, the error tells you to set `ROCM_PATH`.

`COMPILER_KIND` selects the driver: `hipcc` is the 6.x/7.x entry point, while
`clang` runs `amdclang++ -x hip`, which is what ROCm 10.x expects (hipcc is
deprecated there, and modern hipcc is only a wrapper around that command line).
`REQUIRE_ARCH=1` disables GPU auto-detection and fails when `GFX_ARCH` is
missing — CI always sets it, because the machine a build lands on must not
decide which target the artifact contains.

## Examples

64 compile-ready programs live in `examples/rocprim/`, one per primitive. Most
cross-check the GPU result against a CPU computation. Each is a complete
program; build one with `pasrocm.cfg` and run it with the shim path set:

```bash
fpc @pasrocm.cfg -FEbin examples/rocprim/test_reduce.pas
PASROCM_SHIM_DIR=src/rocprim ./bin/test_reduce
```

## Running the test suite

```bash
./test.sh
```

The suite compiles the units and runs 1,702 FPCUnit tests covering the
primitives, the loader's refusal paths, and the generated bindings. Failures
print per-test; the exit code is the failure count.

### Leak checking

Heaptrc is enabled by default for test builds. Because `cmem` and heaptrc
disagree about who owns the heap, leak checking runs without `cmem` — the
`cfgmgr`/`-gh` workflow: build and run without `cmem` (`cthreads` alone is
enough for the pool), fix the leaks, then add `cmem` back for release builds.

## Project layout

```text
src/                    Pascal units (hip_*.pas, rocprim.pas, roc*.pas)
src/rocprim/            the C++ shim: sources, build.sh, generated ABI units
tests/                  1,702 FPCUnit tests
test.sh                 builds and runs them
examples/rocprim/       64 standalone programs, one per primitive
targets.conf            GPU target table: wavefront width, usable widths
VERSION                 the project version, read by build.sh and fpmake
ci/                     build matrix, ABI gate, packages (maintainer tooling)
docs/                   installation and CI detail
```

## More documentation

- `docs/INSTALL.md` — shim loader search order and overrides, what is inside a
  package, per-SDK capability notes, diagnostics.
- `docs/CI.md` — the build matrix, job profiles and billed minutes, validation
  levels, and the release flow.

## Contributing

Pull requests are welcome. Before sending one, run `./test.sh` and
`ci/check-abi.sh`; CI runs both, plus a check that the generated bindings match
`rocprim_shim_abi.inc`.

## License

MIT
