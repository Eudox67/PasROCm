# PasROCm

[![shim CI](https://img.shields.io/github/actions/workflow/status/Eudox67/PasROCm/shim-build.yml?branch=main&label=shim%20CI)](https://github.com/Eudox67/PasROCm/actions/workflows/shim-build.yml)
[![release](https://img.shields.io/github/v/release/Eudox67/PasROCm)](https://github.com/Eudox67/PasROCm/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Free Pascal 3.2.2+](https://img.shields.io/badge/Free%20Pascal-3.2.2%2B-blue)](https://www.freepascal.org/)
[![Lazarus package](https://img.shields.io/badge/Lazarus-package-blueviolet)](pasrocm.lpk)
[![Platform: Linux x86_64](https://img.shields.io/badge/platform-Linux%20x86__64-lightgrey)](#requirements)
[![ROCm 6.4 | 7.2 | 10.1](https://img.shields.io/badge/ROCm-6.4%20%7C%207.2%20%7C%2010.1-red)](#what-each-sdk-line-supports)
[![shim CI](https://img.shields.io/github/actions/workflow/status/Eudox67/PasROCm/shim-build.yml?branch=main&label=shim%20CI)](https://github.com/Eudox67/PasROCm/actions/workflows/shim-build.yml)
[![release](https://img.shields.io/github/v/release/Eudox67/PasROCm)](https://github.com/Eudox67/PasROCm/releases)

Pascal bindings for the **AMD ROCm libraries** — HIP, rocBLAS, MIOpen,
rocSPARSE, rocSOLVER, rocFFT, rocRAND, RCCL and rocPRIM — for Free Pascal on AMD
Radeon and Instinct GPUs.

These are direct bindings to AMD's own libraries: no reimplementation, no CPU
fallback. You call `rocblas_sgemm` or `miopenConvolutionForward` and the GPU
library does the work.

**Status:** 0.1.1, pre-1.0. `docs/INSTALL.md` records what 1.0.0 will freeze.

## Libraries

| library | unit | entry points | what it covers |
|---|---|---|---|
| **rocBLAS** | `rocblas` | 866 | BLAS levels 1–3, batched and strided-batched, 64-bit index variants, `gemm_ex` mixed precision |
| **rocSOLVER** | `rocsolver` | 737 | dense linear algebra: LU, Cholesky, QR, inversion, least squares, SVD |
| **rocSPARSE** | `rocsparse` | 321 | sparse matrices (CSR/COO), SpMV, SpMM, conversion and analysis |
| **MIOpen** | `miopen` | 253 | convolution, pooling, activation, softmax, batch norm, RNN, fusion, find/solver selection |
| **HIP runtime** | `hip_runtime`, `hip_types`, `hip_vector_types` | 383 + 40 | devices, streams, events, memory, graphs, kernel launch, vector types |
| **RCCL** | `rccl` | 38 | multi-GPU collectives (`ncclAllReduce`, `ncclAllGather`, …) |
| **rocRAND** | `rocrand` | 33 | host and device random-number generation |
| **rocFFT** | `rocfft` | 28 | FFT plan creation and execution |
| **hipRTC** | `hiprtc` | 18 | runtime compilation of HIP kernel source |
| **rocPRIM** | `rocprim_bindings` | 144 | device, block and warp primitives — see below |

About 2,800 entry points in total. Every unit is plain Pascal with `cdecl`
declarations; nothing is generated at run time and nothing needs a wrapper
object.

### Two ways the bindings reach the GPU

- **Most libraries link directly.** `rocblas`, `rocsolver`, `rocsparse`,
  `miopen`, `rccl`, `rocrand`, `rocfft`, `hip_runtime` and `hiprtc` carry
  `{$linklib …}` and bind to the shared library ROCm already installed. If you
  have ROCm, you have them — there is nothing extra to build.
- **rocPRIM needs a shim.** It is C++ templates with no C ABI, so PasROCm ships
  a small native library per GPU target that exposes 144 primitives, and a
  Pascal loader that opens the one matching your GPU at run time. That is the
  only part of PasROCm you have to build or download.

## Requirements

- Free Pascal 3.2.2 or newer (`fpc`)
- ROCm 6.4, 7.2 or 10.1, with the libraries you intend to call
  (`librocblas`, `libMIOpen`, …) and `libamdhip64`
- An AMD GPU: RDNA 1–4 (including 3.5) or CDNA 1–4 for the rocPRIM shim — the
  19 targets in `targets.conf`. The other libraries support whatever your ROCm
  install supports.
- Only for rocPRIM: the native shim, built locally or taken from a release

`pacman -S fpc`, `apt install fpc`, or `dnf install fpc` is enough for FPC.
`rocminfo` is the quickest check that a working ROCm runtime is present.

CDNA (`gfx9xx`) **shim** packages are compile- and ABI-verified only: no MI
hardware has executed them yet. RDNA shim packages run on real GPUs.

## Quick start

For rocBLAS, MIOpen and the rest, there is nothing to build — install ROCm, point
FPC at the units, and call them:

```bash
source setup_env.sh                       # optional if ROCm is already on PATH
fpc @pasrocm.cfg -FEbin my_program.pas    # units + HIP library path
./bin/my_program
```

Only rocPRIM needs its native library first:

```bash
cd src/rocprim && ./build.sh   # detect ROCm and this machine's GPU, build the shim
cd ../.. && ./test.sh          # run the test suite
```

`./build.sh` needs nothing but FPC and ROCm. It prints the toolchain it selected,
the detected target, the version, and the library it produced.

No GPU in this machine? Name the target instead of detecting it:
`GFX_ARCH=gfx1100 ./build.sh`. Add `DRY_RUN=1` to print the resolved toolchain
and stop before compiling.

## Environment setup

ROCm is usually installed already (for PyTorch, or the distro's `rocm` package).
You need a HIP compiler on `PATH` — `hipcc` on ROCm 6.x/7.x, `amdclang++` on
ROCm 10.x, where `hipcc` is deprecated and a minimal install may not ship it.
Both live in `$ROCM_PATH/bin` (10.x also puts `amdclang++` in `lib/llvm/bin`).
The ROCm libraries must be findable:

```bash
export ROCM_PATH=/opt/rocm     # only if /opt/rocm is not the actual path
export PATH="$ROCM_PATH/bin:$PATH"
source setup_env.sh            # or let this do it
```

`setup_env.sh` detects `/opt/rocm`, ROCm 10.x's `/opt/rocm/core-<ver>`, and
`/opt/rocm-<ver>`, accepting a tree that has **either** `hipcc` or `amdclang++`.
It sets `HIP_CC` to whichever exists, warns if neither does (a runtime-only
install can still run the tests), and prints what it found. It does not modify
files. `rocm-smi` or `rocminfo` confirms the GPU:
RDNA 3 is `gfx1100`, RDNA 4 `gfx1200`, CDNA 3 `gfx942`, CDNA 4 `gfx950`.

## How to use PasROCm

Pick one of three ways to give FPC the unit paths. They are interchangeable.

### Option 1 — native FPC package (`fpmake`)

Install once, then any project can `uses rocblas;`:

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
points the linker at the ROCm libraries.

### Option 2 — response file, no install (`pasrocm.cfg`)

A checked-in `pasrocm.cfg` on the command line gives the paths without copying
anything or touching `~/.fpc.cfg`:

```bash
fpc @pasrocm.cfg program.pas
```

It sets `-Fusrc -Fusrc/rocprim -Isrc` for the units and
`-Fl/opt/rocm/lib -Fl/opt/rocm/lib64` for the ROCm libraries.

### Option 3 — Lazarus package (`pasrocm.lpk`)

Open `pasrocm.lpk` in Lazarus and click *Use → Add to Project*. The package
registers `src` and `src/rocprim` as unit paths and sets `-Mobjfpc -Sh`.

### Can I use more than one?

Yes, they all point at the same directory. Installing through `fpmake` and then
using `pasrocm.cfg` in another project is fine.

## A first program: rocBLAS `sgemm`

```pascal
program blas_gemm;
{$mode objfpc}{$H+}
uses ctypes, hip_types, hip_runtime, rocblas;

const
  N = 4;
var
  hA, hB, hC: array[0..N*N-1] of Single;
  dA, dB, dC: Pointer;
  handle: rocblas_handle;
  alpha, beta: Single;
  i: Integer;
begin
  for i := 0 to N*N-1 do begin hA[i] := 1; hB[i] := 2; end;
  alpha := 1; beta := 0;

  hipMalloc(dA, N*N*SizeOf(Single));
  hipMalloc(dB, N*N*SizeOf(Single));
  hipMalloc(dC, N*N*SizeOf(Single));
  hipMemcpy(dA, @hA[0], N*N*SizeOf(Single), hipMemcpyHostToDevice);
  hipMemcpy(dB, @hB[0], N*N*SizeOf(Single), hipMemcpyHostToDevice);

  rocblas_create_handle(handle);
  rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                N, N, N, @alpha, dA, N, dB, N, @beta, dC, N);
  hipMemcpy(@hC[0], dC, N*N*SizeOf(Single), hipMemcpyDeviceToHost);

  WriteLn('C[0] = ', hC[0]:0:0);   // 8

  rocblas_destroy_handle(handle);
  hipFree(dA); hipFree(dB); hipFree(dC);
end.
```

```bash
fpc @pasrocm.cfg -FEbin blas_gemm.pas && ./bin/blas_gemm
```

Every call returns a status (`rocblas_status`, `THipError`, `miopenStatus_t`);
the tests check each one. MIOpen, rocSPARSE, rocSOLVER and the others follow the
same pattern: create a handle or descriptor, allocate with HIP, call, copy back.

## rocPRIM and the native shim

rocPRIM is the one library with no C ABI, so it needs
`libpasrocm_shim_<gfx>.so`. Its 144 primitives are ordinary Pascal calls once the
loader has opened the right file:

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

Device-wide entries take the **output pointer first**, then input, count, any
initial value, and the stream, and return a `hipError_t`:

```pascal
rocprim_reduce_int(d_out, d_in, num_items, 0, stream);   // 0 = initial value
```

You allocate the buffers with `hipMalloc`; the shim allocates and frees
rocPRIM's temporary storage. The surface is 144 entry points plus 7 identity
exports, fixed by `src/rocprim/rocprim_shim_abi.inc`.

### Getting the shim

**Option A — a pre-built package.** Download
`pasrocm-shim-0.1.1-linux-x86_64-rocm-7.2-gfx1100.tar.gz` from the releases page
(the name carries version, platform, SDK line and GPU target), unpack it into a
directory, and point the loader at its `lib/`:

```bash
mkdir pasrocm-shim
tar xzf pasrocm-shim-0.1.1-linux-x86_64-rocm-7.2-gfx1100.tar.gz -C pasrocm-shim
export PASROCM_SHIM_DIR=$PWD/pasrocm-shim/lib
```

The archive has no top-level directory of its own, so unpack it into one.

**Option B — build it locally.**

```bash
cd src/rocprim && ./build.sh
```

For cross-building, pass `GFX_ARCH=gfx942` (one target per library, by design).

### What each SDK line supports

The three SDK lines are built from different rocPRIM versions, and one primitive
depends on a feature that only arrived after 3.4:

| SDK line | rocPRIM | `rocprim_merge_inplace_int` |
|---|---|---|
| 6.4 | 3.4.x | returns `hipErrorNotSupported` — the primitive does not exist in that rocPRIM |
| 7.2 | 4.2.0 | supported |
| 10.1 | 4.6.0 | supported |

Everything else in the shim's surface is available on all three. The symbol is
always exported, so this is a capability gap the caller can see and test, not a
missing symbol at load time.

Target coverage follows `targets.conf`. `gfx950` (CDNA4) needs ROCm 7.2 or
newer: the 6.4 rocPRIM cannot compile it (no 128-bit atomics for that
architecture) — a rocPRIM capability floor, not a compiler one.

**One target per build.** `GFX_ARCH` names exactly one GPU target, for example
`gfx1100` (RDNA 3 — Radeon RX 7000) or `gfx942` (CDNA 3 — MI300). A family
keyword such as `rdna3` is rejected on purpose: it expands to several targets,
and this project ships one small library per architecture instead of a fat
binary. The loader picks the file that matches the detected GPU.

`ROCM_PATH` accepts both install layouts. ROCm 10.x keeps the versioned tree at
`/opt/rocm/core-<ver>` and points `/opt/rocm` at it through alternatives, while
6.x/7.x releases installed `/opt/rocm-<ver>` directly. Detection prefers
`/opt/rocm` when it holds a compiler, then takes the newest `/opt/rocm-<ver>`
that also has one. If nothing usable exists, the error tells you to set
`ROCM_PATH`.

`COMPILER_KIND` selects the driver: `hipcc` is the 6.x/7.x entry point, while
`clang` runs `amdclang++ -x hip`, which is what ROCm 10.x expects.
`REQUIRE_ARCH=1` disables GPU auto-detection and fails when `GFX_ARCH` is
missing — CI always sets it, because the machine a build lands on must not
decide which target the artifact contains.

## Examples

64 compile-ready programs live in `examples/rocprim/`, one per primitive. Most
cross-check the GPU result against a CPU computation:

```bash
fpc @pasrocm.cfg -FEbin examples/rocprim/test_reduce.pas
PASROCM_SHIM_DIR=src/rocprim ./bin/test_reduce
```

For the other libraries, the test suite is the worked example: each test sets up
handles and buffers, calls the library, and checks the result.

## Running the test suite

```bash
./test.sh
```

1,702 FPCUnit tests across the whole binding set:

| area | tests |
|---|---|
| rocBLAS | levels 1–3, batched, strided-batched and 64-bit variants (19 suites) |
| rocSOLVER | dense factorisations and solvers |
| MIOpen | convolution, pooling, activation, normalisation, RNN, fusion |
| rocSPARSE, rocRAND, rocFFT | formats and kernels, generators, plans |
| HIP runtime | device, stream and memory basics |
| rocPRIM shim loader | identity, ABI and refusal paths |

Failures print per-test; the exit code is the failure count.

### Leak checking

Heaptrc is enabled by default for test builds. Because `cmem` and heaptrc
disagree about who owns the heap, leak checking runs without `cmem`: build and
run with `cthreads` alone, fix the leaks, then add `cmem` back for release
builds.

## Project layout

```text
src/                    the bindings: hip_*, rocblas, rocsolver, rocsparse,
                        miopen, rccl, rocrand, rocfft, hiprtc, rocprim
src/rocprim/            the rocPRIM shim: C++ sources, build.sh, loader, ABI units
tests/                  1,702 FPCUnit tests across all libraries
test.sh                 builds and runs them
examples/rocprim/       64 standalone programs, one per primitive
targets.conf            GPU target table for the shim
VERSION                 the project version, read by build.sh and fpmake
ci/                     build matrix, ABI gate, packages (maintainer tooling)
docs/                   installation and CI detail
```

## More documentation

- `docs/INSTALL.md` — the shim loader's search order and overrides, what is
  inside a package, per-SDK capability notes, versioning, diagnostics.
- `docs/CI.md` — the build matrix, job profiles, validation levels and the
  release flow.
- `docs/VERIFY.md` — how to check that a download is genuine: checksums, build
  provenance, the checksum signature, and the loader's own integrity pin.

## Contributing

Pull requests are welcome. Before sending one, run `./test.sh` and
`ci/check-abi.sh`; CI runs both, plus a check that the generated bindings match
`rocprim_shim_abi.inc`.

## License

MIT. This is an unofficial project and is not affiliated with or endorsed by AMD.
