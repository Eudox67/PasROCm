# PasROCm

Pascal bindings for [rocPRIM](https://github.com/ROCm/rocPRIM) — AMD's GPU
parallel primitives library — for counting, radix, merge sort and block
operations, driven from Free Pascal on AMD Radeon and Instinct GPUs.

No C++ translation layer: Pascal calls the same multi-TB/s primitives that
rocPRIM's own tests measure.

## Coverage

rocPRIM's primitives are exposed as plain Pascal calls. The verbs you would
expect from a parallel unit are all here:

| primitive | Pascal caller | underlying operation |
|---|---|---|
| `rocprim_reduce_int` | `RocPRIMReduce` | device-wide associative reduction |
| `rocprim_scan_int` | `RocPRIMScan` | exclusive / inclusive prefix sum |
| `rocprim_radix_sort_int` | `RocPRIMRadixSort` | stable LSD key sort |
| `rocprim_radix_sort_pairs_int` | `RocPRIMRadixSortPairs` | key/value sort |
| `rocprim_merge_sort_int` | `RocPRIMMergeSort` | comparison sort |
| `rocprim_run_length_encode_int` | `RocPRIMRunLengthEncode` | run detection |
| `rocprim_count_int` | `RocPRIMCount` | predicate count |
| `rocprim_find_first_of_int` | `RocPRIMFindFirstOf` | intersection probe |
| `rocprim_merge_int` | `RocPRIMMerge` | sorted merge |
| `rocprim_binary_search_int` | `RocPRIMBinarySearch` | lower bound per element |
| `rocprim_segmented_reduce_int` | `RocPRIMSegmentedReduce` | per-segment reduction |
| `rocprim_inclusive_scan_per_segment_int` | `RocPRIMInclusiveScanPerSegment` | scan that restarts at each segment |
| `rocprim_histogram_even_int` | `RocPRIMHistogramEven` | equal-width binning |

Every entry follows the same shape: `(input, count, output, stream)` returns a
HIP status; you allocate nothing, the shim does the GPU planning. The other 138
functions of the 151-symbol surface — `rocprim_*_double`,
`_float`, `_uint`, `_uint8`, `_uint64`, radix-digit variants, pairs variants —
are available with the same signature convention; the set is fixed by
`src/rocprim/rocprim_shim_abi.inc`.

## Requirements

- Free Pascal 3.2.2 or newer (`fpc`)
- AMD GPU: RDNA 3/4 or CDNA 3/4 — MI300-class Instinct, or Radeon RX 7000 and
  up, or newer
- ROCm 6.4, 7.2 or 10.1 — with the ROCm versions you install
  (`hipcc`/`amdclang++` and `libamdhip64`)
- The rocPRIM native shim (below) — one shared library per GPU target

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

Only one thing can fail without a GPU: the step that checks the build. Skip it
with `DRY_RUN=1` and the script stops before compiling.

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
cd src && fpmake install
```

```bash
fpc -Fu$(fpc -iTP)-$(fpc -iTS) program.pas
```

Usage:

```pascal
uses rocprim;
begin
  RocPRIMReduce(@Data, Count, @Total, nil);
end.
```

### Option 2 — response file, no install (`pasrocm.cfg`)

A checked-in `pasrocm.cfg` on the command line gives the paths without copying
anything or touching `~/.fpc.cfg`:

```bash
fpc @pasrocm.cfg program.pas
```

`pasrocm.cfg` contains a `-Fu` line for the units plus `-Fl` for
`src/rocprim`, which is where `build.sh` puts `libpasrocm_shim_<gfx>.so`.

### Option 3 — Lazarus package (`pasrocm.lpk`)

Open `pasrocm.lpk` in Lazarus and click *Use → Add to Project*. The package
registers the units and the `-Fl` path, so the IDE build finds the shim.

### Can I use more than one?

Yes, they all point at the same directory. Installing through `fpmake` and then
using `pasrocm.cfg` in another project is fine; the loader finds one library,
and each process opens it once.

## Getting the rocPRIM native shim

`libpasrocm_shim_<gfx>.so` is the native library that makes rocPRIM's
primitives callable from Pascal. Two ways to get it:

### Option A — use a pre-built package

Download `pasrocm-shim-ubuntu-24.04-rocm7.2-gfx1100.tar.gz` (the asset names
list all four coordinates) from the releases page, unpack it, and put its `lib/`
on the search path:

```bash
tar xzf pasrocm-shim-ubuntu-24.04-rocm7.2-gfx1100.tar.gz
export PASROCM_SHIM_DIR=$PWD/pasrocm-shim-ubuntu-24.04-rocm7.2-gfx1100/lib
```

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
test, not a missing symbol at load time (`ci/DESIGN.md` §12.6).

Target coverage follows `ci/gfx-targets.conf`. `gfx950` (CDNA4) needs ROCm 7.2
or newer: the 6.4 rocPRIM cannot compile it (no 128-bit atomics for that
architecture), which is a rocPRIM capability floor rather than compiler support
(`ci/DESIGN.md` §12.7).

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

Compile-ready programs are in `examples/` — reduction, sort, and a GPU vs CPU
cross-check. Each is a complete program; run it with the shim path set.

## Running the test suite

```bash
./test.sh
```

The suite compiles the units, the shim-aware tests, and the 1,702 assertions
that cover the primitives, the loader's refusal paths, and the generated
bindings. Failures print per-test; the exit code is the count.

### Leak checking

Heaptrc is enabled by default for test builds. Because `cmem` and heaptrc
disagree about who owns the heap, leak checking runs without `cmem` — the
`cfgmgr`/`-gh` workflow is described in `ci/DESIGN.md` §13.4.

## Project layout

```text
src/                    Pascal units (rocprim.pas, rocprim_*.pas)
src/rocprim/            the C++ shim: sources, build.sh, generated ABI units
src/test/pascal/        1,702 assertions (FPCUnit)
examples/               standalone programs
ci/                     build matrix, ABI gate, packages, design notes
docs/                   installation and CI detail
```

## More documentation

- `docs/INSTALL.md` — shim loader search order and overrides, what is inside a
  package, per-SDK capability notes, diagnostics.
- `docs/CI.md` — the build matrix, job profiles and billed minutes, validation
  levels, and the release flow.
- `ci/DESIGN.md` — design decisions, measured upstream facts, risk register.

## Contributing

Pull requests are welcome. Before sending one, run `./test.sh` and
`ci/check-abi.sh`; CI runs both, plus a check that the generated bindings match
`rocprim_shim_abi.inc`.

## License

MIT
