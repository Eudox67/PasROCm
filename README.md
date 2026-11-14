# PasROCm

PasROCm provides AMD's ROCm GPU stack to Free Pascal developers through Pascal
bindings. A thin Pascal wrapper covers the rocPRIM primitives, which exist only
as C++ templates, compiled into a small C++ shim with AMD's `hipcc` compiler.

ROCm is AMD's open platform for GPU computing — what CUDA is for NVIDIA GPUs.
HIP is its C/C++ programming interface, and `hipcc` is its C/C++ compiler.
PasROCm wraps these parts for you, so you stay in Pascal.

The units work with the plain Free Pascal compiler (CLI) and with Lazarus. The
Pascal side needs no external build tool. If you use the rocPRIM units, you
also build the small C++ shim once with `hipcc` — `hipcc` ships with ROCm, so
your ROCm install already covers it.

> **Disclaimer** — This is an independent, community project. It is **not**
> affiliated with, sponsored by, or approved by AMD. ROCm, HIP, rocBLAS,
> rocSOLVER, rocSPARSE, rocFFT, rocRAND, MIOpen, RCCL and rocPRIM are
> trademarks or registered trademarks of Advanced Micro Devices, Inc.

**Quick start — three steps:**

```bash
source setup_env.sh                    # 1. set every ROCm path for this shell
fpc @pasrocm.cfg my_rocm_program.pas   # 2. compile your program (from the PasROCm root)
./test.sh                              # 3. confirm that your GPU setup works (~12 s)
```

## Coverage

Each unit wraps one ROCm library, so you import only the parts you need.

| Unit | ROCm library | Notes |
|---|---|---|
| `hip_runtime.pas` | HIP runtime (`libamdhip64`) | core API, memory, streams, events |
| `hip_types.pas` | HIP runtime | vector/`half`/`bfloat16` types, enums |
| `hip_vector_types.pas` | HIP runtime | `int2`/`float4`/… record types |
| `hip_error.pas` | HIP runtime | error-code to message helpers |
| `hip_common.pas` | HIP runtime | shared helpers used by the wrappers |
| `hiprtc.pas` | hipRTC (`libhiprtc`) | compiles GPU kernels while your program runs |
| `rocblas.pas` | rocBLAS (`librocblas`) | levels 1-3, batched + strided-batched, `_64` variants |
| `rocsolver.pas` | rocSOLVER (`librocsolver`) | dense/sparse solvers, factorizations |
| `rocsparse.pas` | rocSPARSE (`librocsparse`) | sparse matrix products (SpMV/SpMM), conversions, solvers, preconditioners |
| `rocfft.pas` | rocFFT (`librocfft`) | plans, real/complex transforms |
| `rocrand.pas` | rocRAND (`librocrand`) | generators, distributions |
| `miopen.pas` | MIOpen (`libMIOpen`) | convolutions, pooling, activations, RNN |
| `rccl.pas` | RCCL (`librccl`) | collective communication between GPUs |
| `rocprim.pas` | rocPRIM | Pascal layer over the native shim (see below) |

Under `src/rocprim/`, the rocPRIM Pascal layer is split into small units, one
topic each: `rocprim_types`, `rocprim_core`, `rocprim_block`, `rocprim_device`,
`rocprim_warp`, `rocprim_thread`, `rocprim_intrinsics` and `rocprim_iterator`.
Three low-level units — `rocprim_bindings`, `hip_bindings` and
`hip_specialized` — link against `librocprim_shim.so`.

The rocPRIM algorithms are C++ templates, and Free Pascal cannot call C++
templates directly. A small C++ shim in `src/rocprim/` solves this. The shim
wraps every template as a plain C function (`extern "C"`), and the Pascal units
call those functions. You build `librocprim_shim.so` once and keep it on the
library search path; the shim section below shows the steps.

## Requirements

Your system needs these parts. If you have a working ROCm install and FPC 3.2.2
or newer, you are ready to start.

| Component | Version / notes |
|---|---|
| Linux x86-64 | developed and tested on Linux only |
| AMD ROCm | 6.x – 10.x (developed and tested on 10.1); expected at `/opt/rocm`, `/opt/rocm-<ver>`, or `$ROCM_PATH` |
| AMD GPU | any ROCm-supported CDNA/RDNA device; if `rocm-smi` lists your GPU, the driver is ready |
| Free Pascal | 3.2.2 or newer (developed against FPC 3.3.1) |
| `hipcc` | ships with ROCm; you need it only to build the rocPRIM shim |
| Lazarus | optional — only for the `.lpk` installation route |

Two quick words for readers new to ROCm: CDNA is AMD's data-center GPU family
(Instinct MI cards), and RDNA is the consumer family (Radeon RX cards).
`rocm-smi` is ROCm's small tool that lists your GPUs.

The units link directly to the ROCm shared libraries (`{$linklib}`). The
compiler needs the library folders at link time (`-Fl`); `pasrocm.cfg` and
`test.sh` already set them for you. The system needs them again at run time
(`LD_LIBRARY_PATH`); `setup_env.sh` sets that for you too.

## Environment setup

The ROCm tools and libraries live outside the standard system paths, and Unix
shells keep environment variables inside the session that sets them. So each
new shell needs the ROCm settings again. `setup_env.sh` sets every path for
you, and you have two ways to load it:

**One-time setup (recommended):** add one line to your `~/.bashrc`. You do this
once, and every future bash shell picks up the settings automatically:

```bash
echo "source /path/to/PasROCm/setup_env.sh" >> ~/.bashrc
```

(In CI jobs and non-bash shells, run the `source` line directly instead;
`test.sh` already sets up its own environment.)

**Per shell (manual):** or run this line in each new shell, before you compile
or run anything:

```bash
source setup_env.sh
```

The script looks at its own location and at your ROCm install, then sets every
path for you:

- `ROCM_PATH` / `ROCM_HOME` — the script uses your current `ROCM_PATH` if it is
  set, then `ROCM_HOME`, then `/opt/rocm`, then the newest `/opt/rocm-*` that
  holds a real install. A real install has `bin/hipcc`, or at least
  `lib*/libamdhip64.so*` for runtime-only setups. Directories with only headers
  never count.
- The usual HIP variables: `HIP_PLATFORM=amd`, `HIP_PATH`, `HIP_CLANG_PATH`,
  `HIP_CC`, `HSA_PATH`.
- `PATH` gains `$ROCM_PATH/bin` and `$ROCM_PATH/llvm/bin`, so `hipcc` and the
  other ROCm tools are ready to use.
- `HIP_DEVICE_LIB_PATH` / `DEVICE_LIB_PATH` point to the device bitcode
  (ready-made GPU code that the compiler links in).
- `LD_LIBRARY_PATH` gains the ROCm `lib`/`lib64` folders and `src/rocprim` (the
  shim folder). It is safe to source the script more than once.

## How to use PasROCm

Choose the option that fits your project. All three already carry the ROCm
`-Fl` library paths for you.

### Option 1 — native FPC package (`fpmake`)

You compile the package once and install the units into your FPC unit tree.
After that, no project needs to carry PasROCm source paths.

```bash
# 1. Build the installer (once)
fpc fpmake.pp

# 2. Compile the package -> units/x86_64-linux/*.ppu
./fpmake compile

# 3. Install into a per-user prefix (no sudo, predictable path)
./fpmake install --prefix="$HOME/.local"

# ... or into the compiler's own tree:
#     <fpc-prefix>/lib/fpc/<version>/units/<target>/pasrocm
#     (needs write access -> sudo when FPC comes from the distro)
./fpmake install
```

Now make the new unit directory visible to the compiler. What you do depends on
your `fpc.cfg`:

- **Stock wildcard:** many `fpc.cfg` files contain
  `-Fu<prefix>/lib/fpc/$fpcversion/units/$fpctarget/*`. If you installed under
  the same prefix, FPC finds `pasrocm` by itself — nothing to add.
- **Explicit package list:** some configs name each package directory. Then the
  compiler does not see the new package until you add one line — even when it
  sits inside the compiler's own unit root.

Adding the line is always safe. FPC does not expand environment variables in
config files, so write the absolute path:

```bash
UNITDIR="$HOME/.local/lib/fpc/$(fpc -iV)/units/$(fpc -iTP)-$(fpc -iTO)/pasrocm"
echo "-Fu$UNITDIR" >> ~/.fpc.cfg
```

After that, any project can simply say:

```pascal
uses hip_runtime, rocsparse;
```

(If you chose a different prefix, use the same `-Fu` line with your own install
path. `find "$HOME/.local" /usr -type d -name pasrocm` prints it.) To
uninstall, delete the `pasrocm` unit directory, plus the matching
`<prefix>/lib/fpc/<version>/fpmkinst/<target>/pasrocm.fpm` manifest if present.

### Option 2 — response file, no install (`pasrocm.cfg`)

Nothing is written outside the checkout. This option fits well when PasROCm is
a git submodule of your project.

```bash
cd /path/to/PasROCm
fpc @pasrocm.cfg my_rocm_program.pas
```

`pasrocm.cfg` supplies `-Mobjfpc -Sh`, `-Fu src`, `-Fu src/rocprim`,
`-Fl src/rocprim`, `-Fl /opt/rocm/lib`, `-Fl /opt/rocm/lib64` and `-I src`.
Its paths are relative, so run `fpc` from the PasROCm root (as above), or add
your own absolute `-Fu`/`-Fl` paths. Later flags override earlier flags, so you
can correct a non-standard ROCm prefix easily:

```bash
fpc @pasrocm.cfg -Fl"$ROCM_PATH/lib" -Fl"$ROCM_PATH/lib64" my_rocm_program.pas
```

Before you run the program, load `setup_env.sh` (or set `LD_LIBRARY_PATH`
yourself). The program then finds `libamdhip64.so`, `librocblas.so` and the
other ROCm libraries at run time.

### Option 3 — Lazarus package (`pasrocm.lpk`)

This option gives you the Lazarus IDE with full source-level tools, and it also
serves headless CI builds.

**IDE:** *Package → Open Package File (.lpk)* → `pasrocm.lpk` → **Compile** →
*Use → Add to Project*.

**Command line:**

```bash
lazbuild pasrocm.lpk                        # -> lib/$(TargetCPU)-$(TargetOS)/*.ppu
lazbuild --add-package pasrocm.lpk my_project.lpi
lazbuild my_project.lpi
```

The package shares its unit paths (`src`, `src/rocprim`) with every project
that uses it. So `uses hip_runtime, rocprim;` needs no further configuration.

### Can I use more than one?

Yes, and a popular choice is Option 1 **and** Option 3 together: the installed
package serves plain `fpc`/`lazbuild` builds, and the Lazarus package gives the
IDE its source-level tools.

| Setup | `fpc` CLI | `lazbuild` | IDE Code Tools (completion, *Ctrl+Click*) |
| --- | :-: | :-: | :-: |
| Only Option 1 (installed + `-Fu`) | yes | yes | **no** — only `.ppu`/`.o` are installed |
| Only Option 2 (`@pasrocm.cfg`) | yes | with the same `-Fu` flags in the project | no |
| Only Option 3 (`pasrocm.lpk`) | with `-Fu src -Fu src/rocprim` | yes | **yes** |
| Option 1 + Option 3 | yes | yes | yes |

Why do the columns differ? `lazbuild` is a wrapper around `fpc`, so anything
that resolves on the command line also compiles. The IDE Code Tools are
different: they read the `.pas` sources, and a directory of compiled units
gives them nothing to complete from. That is the main reason `pasrocm.lpk`
exists.

When both are installed, the command-line paths come first: `-Fu` flags take
precedence over the paths in `fpc.cfg`. So a project that uses the `.lpk`
builds against the fresh output in `lib/$(TargetCPU)-$(TargetOS)/`, not against
the installed copy. The reverse also applies: if you edit PasROCm and rebuild
only through the `.lpk`, an unrelated CLI project keeps using the units that
`fpmake install` last wrote.

## Building the rocPRIM native shim

You build `librocprim_shim.so` yourself, once per ROCm/GPU combination. None of
the three install options installs it, because the correct binary depends on
your GPU and your ROCm version.

```bash
cd src/rocprim
./build.sh                                  # detect ROCm and this machine's GPU
GFX_ARCH=gfx1100 ./build.sh                 # one target -> libpasrocm_shim_gfx1100.so
GFX_ARCH=gfx942 ./build.sh                  # CDNA: wave64, adds -mwavefrontsize64
COMPILER_KIND=hipcc ./build.sh              # ROCm 6.x/7.x: hipcc
COMPILER_KIND=clang ./build.sh              # ROCm 10.x: amdclang++ -x hip
OUT_DIR=/tmp/shim ./build.sh                # build outside the source tree
DRY_RUN=1 ./build.sh                        # print the resolved toolchain only
```

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

**The wavefront size is injected, never guessed.** It comes from
`ci/gfx-targets.json` (the one mapping table) and reaches the compiler as
`-DPASROCM_WAVEFRONT_SIZE`; the build refuses to start without it, and refuses a
wave32 build on CDNA or a wave64 build on RDNA 1–3.5. At first use the shim also
compares that width with what the device reports and returns
`hipErrorInvalidDevice` instead of running kernels with the wrong geometry.

Every build prints the toolchain it actually found. So a build log answers
"which compiler produced this `.so`?" — and you do not have to follow symlinks
by hand:

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
== rocprim_* exports: 146
== build info: /path/to/PasROCm/src/rocprim/build-info.env
```

`DRY_RUN=1 ./build.sh` prints that same header without compiling. It is the
quickest way to confirm which install a build would use.

The shim sources and `build.sh` live next to the Pascal units they serve, and
the build writes `librocprim_shim.so` into that same directory. Your tools
already know this place: `pasrocm.cfg` points there with `-Fl src/rocprim`,
`test.sh` with `-Fl` plus an rpath (a library search path stored inside the
executable), and `setup_env.sh` with `LD_LIBRARY_PATH`. A CMake path
(`CMakeLists.txt`) is also there for IDE integration, and it builds the same
sources.

Take the HIP headers only from your selected ROCm install. A copied `hip/`
header set next to the shim does not work well: mixed header versions do not
compile together with the installed `hip/amd_detail/*.h` internals. On ROCm 7.x
this appeared as `unknown type name 'hipDriverEntryPointQueryResult'`.

## Continuous integration

`.github/workflows/shim-build.yml` compiles the shim for every supported
(Ubuntu, ROCm) pair and GPU architecture group, and uploads one package per
(configuration, arch group). The matrix is not written in YAML — it is generated
from `ci/supported-builds.json` by `ci/manifest.py`, so adding or retiring a
configuration is a one-file change.

| layer | file |
|---|---|
| GPU targets, wavefront widths | `ci/gfx-targets.json` |
| SDK pins, profiles | `ci/supported-builds.json` |
| SDK provisioning | `ci/install-rocm.sh --config <id>` |
| build, ABI check, package | `ci/build-linux.sh --config <id> --arch-group <group>` |
| ABI gate | `ci/check-abi.sh` |
| design, measured upstream facts, risks | `ci/DESIGN.md` |

Profiles keep a private repository inside its Actions allowance:
`pull_request` runs 2 jobs, a push to `main` runs 16, a nightly or manual run
covers all 55 jobs (nine configurations × the targets each one builds).

```bash
python3 ci/manifest.py validate                    # self-check the manifest
python3 ci/manifest.py matrix --profile smoke      # what CI would run

# reproduce one CI job verbatim (installs the pinned SDK, so inspect first)
ci/install-rocm.sh --config ubuntu-24.04-rocm10.1 --dry-run
ci/build-linux.sh  --config ubuntu-24.04-rocm10.1 --target gfx1100
```

**CDNA artifacts are compile-verified and ABI-checked only.** Nothing in this
repository can execute gfx9xx kernels, so every gfx908/gfx90a/gfx942/gfx950
package carries `validation.gpu_runtime: not-run` in its `manifest.json`. Their
wavefront handling is proven at build time (guards, identity self-test) and the
launch geometry is proven by construction, but real kernel execution on MI
hardware is unverified until a self-hosted or rented runner exists.

A configuration claims a range of patches it can back up: `7.2` claims
`7.2.0-7.2.4` because the rocPRIM headers are identical in both channels, while
`6.4` claims `6.4.0-6.4.4` only because dedicated probe jobs compile against the
6.4.0 channel as well. `manifest.json` inside each package records the claim,
the evidence, and the exact toolchain that produced the binary.

## Examples

The folder `examples/rocprim/` holds one small, complete program for each
wrapped primitive: `test_reduce.pas`, `test_radix_sort_keys_asc.pas`,
`test_segmented_radix_sort.pas`, `test_sort.pas`, `test_scan.pas` and more.
Compile the one you need:

```bash
mkdir -p bin/examples
fpc -Mobjfpc \
    -Fusrc -Fusrc/rocprim -Flsrc/rocprim \
    -Fl"$ROCM_PATH/lib" -Fl"$ROCM_PATH/lib64" \
    -FUbin/examples -FEbin/examples \
    examples/rocprim/test_reduce.pas

LD_LIBRARY_PATH="src/rocprim:$ROCM_PATH/lib:$ROCM_PATH/lib64" \
  ./bin/examples/test_reduce
# -> SUCCESS: GPU result matches CPU result.
```

Always pass `-FUbin/examples` (the unit output folder). Without it, FPC writes
`.ppu` and `.o` files next to the `.pas` sources; with it, your source tree
stays clean. Git ignores the compiled example binaries.

## Running the test suite

The test suite uses **FPCUnit** and lives in `tests/`; `TestRunner.pas`
registers the test unit of every subsystem.

```bash
./test.sh                              # full suite
./test.sh --suite=TTest_rocSPARSE      # a single suite
./test.sh --list                       # list registered suites
```

`test.sh` does the housekeeping for you: it builds the shim when it is missing,
compiles `tests/TestRunner.pas` into `bin/` (units in `bin/units/`), exports
`LD_LIBRARY_PATH` for ROCm and the shim, embeds an rpath, and runs the binary
with your extra flags.

The equivalent manual build:

```bash
fpc @pasrocm.cfg tests/TestRunner.pas -FEbin -FUbin/units
LD_LIBRARY_PATH="src/rocprim:$ROCM_PATH/lib:$ROCM_PATH/lib64" \
  ./bin/TestRunner --all --format=plain
```

Current status on ROCm 10.1 (HIP 7.16, MIOpen 3.6.1, gfx1100): **1683 tests
run, 0 errors, 0 failures, 16 ignores**, in about 12 s. Each ignore is an
expected "not supported by this ROCm build" case and quotes the library's real
return status. So a real regression shows up as a failure — it never hides as a
silent skip.

> The tests allocate real GPU memory. So they need a working device and — for
> the rocPRIM suites — a matching `librocprim_shim.so`.

### Leak checking

Add `-gh` to the `fpc` command line (or to the flags in `test.sh`), and FPC
builds with the heaptrc leak tracker. One caution: the `cmem` unit replaces the
memory manager and silently switches leak tracking off. While you look for
leaks, build without `cmem`.

## Project layout

```
PasROCm/
├── src/                  # library units (14) + rocprim/ sub-units (11)
│   └── rocprim/          # rocPRIM layer + C++ shim
│                         #   rocprim_shim*.cpp, build.sh, CMakeLists.txt
│                         #   librocprim_shim.so   <- build output
├── tests/                # FPCUnit tests + TestRunner.pas
├── examples/rocprim/     # standalone programs, one per rocPRIM primitive
├── fpmake.pp             # Tier 1: native FPC package installer
├── pasrocm.cfg           # Tier 2: FPC response file
├── pasrocm.lpk           # Tier 3: Lazarus package
├── test.sh               # compile + run the test suite
├── setup_env.sh          # ROCm environment helper (source it)
└── LICENSE               # MIT
```

## Contributing

Contributions are welcome. A few simple rules:

- Pascal style: `{$mode objfpc}{$H+}`, one unit per ROCm module.
- Every new unit needs the MIT header with `Copyright (c) 2024-2026 Eudox67`.
  Files derived from AMD sources keep AMD's notice as well.
- Register a new unit in `fpmake.pp` (`P.Targets.AddUnit(...)`) and in
  `pasrocm.lpk` (`<Files>`). `pasrocm.cfg` needs no change, because it adds
  whole directories.
- Run `./test.sh` and confirm 0 failures. Then open your pull request — it is
  welcome.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2024-2026 **Eudox67**.

Portions of this project come from AMD ROCm headers and reference
implementations. Those files keep AMD's original MIT/LLVM notices next to the
project copyright. No AMD code ships in binary form.

