# PasROCm

Unofficial **Free Pascal** bindings for the AMD ROCm compute stack, plus a thin
Pascal wrapper layer for the rocPRIM primitives that only exist as C++ templates.

Works with the plain Free Pascal compiler (CLI) and with Lazarus — no external
build tool.

> **Disclaimer** — This is an independent, community project. It is **not**
> affiliated with, sponsored by, or endorsed by AMD. ROCm, HIP, rocBLAS,
> rocSOLVER, rocSPARSE, rocFFT, rocRAND, MIOpen, RCCL and rocPRIM are
> trademarks or registered trademarks of Advanced Micro Devices, Inc.

## Coverage

| Unit | ROCm library | Notes |
|---|---|---|
| `hip_runtime.pas` | HIP runtime (`libamdhip64`) | core API, memory, streams, events |
| `hip_types.pas` | HIP runtime | vector/`half`/`bfloat16` types, enums |
| `hip_vector_types.pas` | HIP runtime | `int2`/`float4`/… record types |
| `hip_error.pas` | HIP runtime | error-code to message mapping helpers |
| `hip_common.pas` | HIP runtime | shared helpers used by the wrappers |
| `hiprtc.pas` | hipRTC (`libhiprtc`) | runtime kernel compilation |
| `rocblas.pas` | rocBLAS (`librocblas`) | levels 1-3, batched + strided-batched, `_64` variants |
| `rocsolver.pas` | rocSOLVER (`librocsolver`) | dense/sparse solvers, factorizations |
| `rocsparse.pas` | rocSPARSE (`librocsparse`) | SpMV, SpMM, conversions, solver/precond |
| `rocfft.pas` | rocFFT (`librocfft`) | plans, real/complex transforms |
| `rocrand.pas` | rocRAND (`librocrand`) | generators, distributions |
| `miopen.pas` | MIOpen (`libMIOpen`) | convolutions, pooling, activations, RNN |
| `rccl.pas` | RCCL (`librccl`) | collective communication |
| `rocprim.pas` | rocPRIM | Pascal facade over the native shim (see below) |

Under `src/rocprim/` the rocPRIM Pascal layer is split by concern:
`rocprim_types`, `rocprim_core`, `rocprim_block`, `rocprim_device`,
`rocprim_warp`, `rocprim_thread`, `rocprim_intrinsics`, `rocprim_iterator`,
plus the low-level `rocprim_bindings` / `hip_bindings` / `hip_specialized`
units that link against `librocprim_shim.so`.

rocPRIM's actual algorithms are C++ templates, which cannot be instantiated from
Free Pascal. They are therefore compiled into a small C++ shim
(`src/rocprim/`) that exposes a flat `extern "C"` ABI. The Pascal units call
that ABI, so `librocprim_shim.so` must be built and present on the library
search path before any rocPRIM wrapper can be used.

## Requirements

| Component | Version / notes |
|---|---|
| Linux x86-64 | developed and tested on Linux only |
| AMD ROCm | 6.x – 10.x (developed and tested on 10.1); expected at `/opt/rocm`, `/opt/rocm-<ver>`, or `$ROCM_PATH` |
| AMD GPU | any ROCm-supported CDNA/RDNA device; a working `rocm-smi` is a good sanity check |
| Free Pascal | 3.2.2 or newer (developed against FPC 3.3.1) |
| `hipcc` | ships with ROCm; only needed to build the rocPRIM shim |
| Lazarus | optional — only for the `.lpk` installation route |

The bindings link the ROCm shared objects through `{$linklib}`, so they must be
findable **at link time** (`-Fl`, already handled by `pasrocm.cfg` and
`test.sh`) and **at run time** (`LD_LIBRARY_PATH`, set by `setup_env.sh`).

## Environment setup

Load the environment once per shell — before compiling *or* running anything:

```bash
source setup_env.sh
```

`setup_env.sh` derives everything from its own location and the detected ROCm
install:

- `ROCM_PATH` / `ROCM_HOME` — honours an existing value, else `$ROCM_HOME`, else
  `/opt/rocm`, else the newest `/opt/rocm-*` that looks like a real install (one
  with `bin/hipcc`, or at least `lib*/libamdhip64.so*` for runtime-only setups;
  header-only leftovers are never picked).
- `HIP_PLATFORM=amd`, `HIP_PATH`, `HIP_CLANG_PATH`, `HIP_CC`, `HSA_PATH`.
- `PATH` gains `$ROCM_PATH/bin` and `$ROCM_PATH/llvm/bin`.
- `HIP_DEVICE_LIB_PATH` / `DEVICE_LIB_PATH` for the device bitcode.
- `LD_LIBRARY_PATH` gains the ROCm `lib`/`lib64` dirs **and** `src/rocprim` (the shim). It is safe to source repeatedly.

To load it automatically in every new shell:

```bash
echo "source /path/to/PasROCm/setup_env.sh" >> ~/.bashrc
```

## How to use PasROCm

Pick the integration that fits your workflow; all three cover ROCm's `-Fl`
directories already.

### Option 1 — native FPC package (`fpmake`)

Compiles the package once and installs the units into an FPC unit directory, so
projects never carry source paths.

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

Then make the unit directory visible. Whether that needs an explicit entry
depends on how your `fpc.cfg` was generated:

- **Stock wildcard:** if it still contains
  `-Fu<prefix>/lib/fpc/$fpcversion/units/$fpctarget/*` and you installed under
  that same prefix, FPC discovers `pasrocm` by itself — nothing to add.
- **Explicit package list:** if it names each package directory individually
  the package stays invisible until you add it — even when it sits inside the
  compiler's own unit root.

Registering it once is always safe. FPC does not expand environment variables
inside config files, so the path must be absolute:

```bash
UNITDIR="$HOME/.local/lib/fpc/$(fpc -iV)/units/$(fpc -iTP)-$(fpc -iTO)/pasrocm"
echo "-Fu$UNITDIR" >> ~/.fpc.cfg
```

After that, any project can simply say

```pascal
uses hip_runtime, rocsparse;
```

(Use the same `-Fu` line with your own install path if you chose a different
prefix — `find "$HOME/.local" /usr -type d -name pasrocm` prints it.) To
uninstall, delete that `pasrocm` directory plus the matching
`<prefix>/lib/fpc/<version>/fpmkinst/<target>/pasrocm.fpm` manifest, if present.


### Option 2 — zero-install response file (`pasrocm.cfg`)

Nothing is written outside the checkout — ideal when PasROCm is a git submodule.

```bash
cd /path/to/PasROCm
fpc @pasrocm.cfg my_rocm_program.pas
```

`pasrocm.cfg` supplies `-Mobjfpc -Sh`, `-Fu src`, `-Fu src/rocprim`,
`-Fl src/rocprim`, `-Fl /opt/rocm/lib`, `-Fl /opt/rocm/lib64` and `-I src`.
Its paths are relative, so either run `fpc` from the PasROCm root (as above) or
add the absolute `-Fu`/`-Fl` paths yourself. With a non-standard ROCm prefix,
override afterwards — later flags win:

```bash
fpc @pasrocm.cfg -Fl"$ROCM_PATH/lib" -Fl"$ROCM_PATH/lib64" my_rocm_program.pas
```

Remember to `source setup_env.sh` (or set `LD_LIBRARY_PATH`) so the resulting
binary can find `libamdhip64.so`, `librocblas.so`, … at run time.

### Option 3 — Lazarus package (`pasrocm.lpk`)

For the Lazarus IDE and for headless CI builds.

**IDE:** *Package → Open Package File (.lpk)* → `pasrocm.lpk` → **Compile** →
*Use → Add to Project*.

**Command line:**

```bash
lazbuild pasrocm.lpk                        # -> lib/$(TargetCPU)-$(TargetOS)/*.ppu
lazbuild --add-package pasrocm.lpk my_project.lpi
lazbuild my_project.lpi
```

The package exports its unit paths (`src`, `src/rocprim`) to every project that
uses it, so `uses hip_runtime, rocprim;` needs no further configuration.

### Can I use more than one?

Yes — the options are not mutually exclusive. Option 1 **and** Option 3 together
is the common combination: the installed package serves plain `fpc`/`lazbuild`
builds, the Lazarus package gives the IDE its source-level tooling.

| Setup | `fpc` CLI | `lazbuild` | IDE Code Tools (completion, *Ctrl+Click*) |
| --- | :-: | :-: | :-: |
| Only Option 1 (installed + `-Fu`) | yes | yes | **no** — only `.ppu`/`.o` are installed |
| Only Option 2 (`@pasrocm.cfg`) | yes | with the same `-Fu` flags in the project | no |
| Only Option 3 (`pasrocm.lpk`) | with `-Fu src -Fu src/rocprim` | yes | **yes** |
| Option 1 + Option 3 | yes | yes | yes |

Why the columns differ: `lazbuild` is a wrapper around `fpc`, so anything that
resolves on the command line compiles. The IDE's Code Tools, however, parse
`.pas` sources — a directory of compiled units gives them nothing to complete
from. That is the main reason `pasrocm.lpk` exists.

When both are installed, the command line wins: paths passed with `-Fu` take
precedence over the ones in `fpc.cfg` (the compiler searches command-line paths
first), so a project requiring the `.lpk` builds against the package output in
`lib/$(TargetCPU)-$(TargetOS)/`, not against the installed copy. The flip side:
if you edit PasROCm and rebuild only through the `.lpk`, an unrelated CLI
project keeps seeing whatever `fpmake install` last wrote.

## Building the rocPRIM native shim

`librocprim_shim.so` is a build artifact that is **not** installed
by any of the three options above because it's build depends on the specific GPU
and ROCm version. Build it once per ROCm/GPU combination:

```bash
cd src/rocprim
./build.sh                                  # auto-detect ROCm + GPU arch
ROCM_PATH=/opt/rocm/core-10.1 ./build.sh    # pin a specific ROCm install
GFX_ARCH=gfx1100 ./build.sh                 # specific GPU
GFX_ARCH=rdna3;cdna3 ./build.sh             # fat binary across families
DRY_RUN=1 ./build.sh                        # print resolved toolchain/archs only
```

`ROCM_PATH` accepts either layout: ROCm 10.x installs the versioned tree at
`/opt/rocm/core-<ver>` and points `/opt/rocm` at it through alternatives, while
6.x/7.x releases installed `/opt/rocm-<ver>` directly. Auto-detection prefers
`/opt/rocm` whenever it contains an executable `bin/hipcc`, then falls back to
the newest `/opt/rocm-<ver>` **that also has one** — a stray header-only
`/opt/rocm-<ver>` directory can never be selected, and if nothing usable exists
the error tells you to set `ROCM_PATH`.

Every build prints the toolchain it actually resolved, so a log answers "which
compiler produced this `.so`?" without tracing symlinks by hand:

```text
== ROCm root : /opt/rocm
== hipcc     : /opt/rocm/core-10.1/bin/hipcc     <- resolved through alternatives
== HIP ver   : 7.16.26385-0000000
== Lib dir   : /opt/rocm/lib
== Arch      : gfx1100  (1 target[s])
...
== Linked against:
 (NEEDED)    Shared library: [libamdhip64.so.7]
== Loads at runtime:
    libamdhip64.so.7 => /opt/rocm/lib/libamdhip64.so.7
```

`DRY_RUN=1 ./build.sh` prints that same header without compiling — the quickest
way to confirm which install a build would use.

The shim sources and `build.sh` live next to the Pascal units they serve, and
the build emits `librocprim_shim.so` into that same directory — exactly where
`pasrocm.cfg` (`-Fl src/rocprim`), `test.sh` (both `-Fl` and an rpath) and
`setup_env.sh` (`LD_LIBRARY_PATH`) point at it. A CMake path (`CMakeLists.txt`)
exists for IDE integration and builds the same sources.

HIP headers are taken exclusively from the selected ROCm install. Do not drop a
copied `hip/` header set next to the shim and add it to the include path: mixing
header versions with the installed `hip/amd_detail/*.h` internals does not
compile — on ROCm 7.x this surfaced as `unknown type name
'hipDriverEntryPointQueryResult'`.

## Examples

`examples/rocprim/` holds one self-contained program per wrapped primitive
(`test_reduce.pas`, `test_radix_sort_keys_asc.pas`, `test_segmented_radix_sort.pas`,
`test_sort.pas`, `test_scan.pas`, …). Compile whichever one you need:

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

Always pass `-FUbin/examples` (unit output) — without it FPC writes `.ppu` and
`.o` files next to the `.pas` sources and pollutes the tree. Compiled example
binaries are git-ignored.

## Running the test suite

The suite uses **FPCUnit** and lives in `tests/`; `TestRunner.pas` registers
every subsystem's test unit.

```bash
./test.sh                              # full suite
./test.sh --suite=TTest_rocSPARSE      # a single suite
./test.sh --list                       # list registered suites
```

`test.sh` builds the shim when it is missing, compiles `tests/TestRunner.pas`
into `bin/` (units in `bin/units/`), exports `LD_LIBRARY_PATH` for ROCm + the
shim, embeds an rpath, and runs the binary with whatever flags you pass.

The equivalent manual build:

```bash
fpc @pasrocm.cfg tests/TestRunner.pas -FEbin -FUbin/units
LD_LIBRARY_PATH="src/rocprim:$ROCM_PATH/lib:$ROCM_PATH/lib64" \
  ./bin/TestRunner --all --format=plain
```

Current status on ROCm 10.1 (HIP 7.16, MIOpen 3.6.1, gfx1100): **1683 tests
run, 0 errors, 0 failures, 16 ignores**, in about 12 s. Every ignore is an
expected "not supported by this ROCm build" path and quotes the library's real
return status, so an unexpected regression surfaces as a failure rather than as
a silently skipped test.

> Tests allocate real GPU memory, so they need a working device and — for the
> rocPRIM suites — a matching `librocprim_shim.so`.

### Leak checking

Add `-gh` to the `fpc` command line (or to the flags in `test.sh`) to build with
FPC's heaptrc. Note that if the program also links `cmem`, heaptrc's hooks are
replaced and leak tracking is silently disabled — build without `cmem` while
chasing leaks.

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

- Pascal style: `{$mode objfpc}{$H+}`, one unit per ROCm module.
- Every new unit needs the MIT header with `Copyright (c) 2024-2026 Eudox67`;
  files derived from AMD sources keep AMD's notice as well.
- A new unit must be registered in `fpmake.pp` (`P.Targets.AddUnit(...)`) and in
  `pasrocm.lpk` (`<Files>`); `pasrocm.cfg` needs no change because it adds whole
  directories.
- Run `./test.sh` and confirm 0 failures before opening a pull request.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2024-2026 **Eudox67**.

Portions of this project are derived from AMD ROCm headers and reference
implementations; those files retain AMD's original MIT/LLVM notices alongside
the project copyright. No AMD code is redistributed in binary form.



