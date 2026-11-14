# ci/ — build matrix, ABI gate, packaging

Free Pascal is the only interpreter-free prerequisite; nothing here needs
Python. Every script is shell and reads its data from the two `.conf` files.

| file | job |
|---|---|
| `../targets.conf` | GPU target → wavefront width, runnable widths, minimum SDK line (at the repo root: `build.sh` reads it too) |
| `supported-builds.conf` | Ubuntu/ROCm pins, profiles (`pr`/`smoke`/`full`), patch claims |
| `manifest.sh` | queries: `validate`, `matrix`, `env`, `wave`, `mask`, `targets`, `meta` |
| `install-rocm.sh` | install a pinned SDK (`--config <id>`, `--dry-run` first) |
| `build-linux.sh` | one CI job: install, build, ABI check, package |
| `check-abi.sh` | exports vs. `rocprim_shim_abi.inc` (`--strict` also rejects extras) |
| `package.sh` | tarball + `manifest.json` for one coordinate |
| `gpu-runtime-evidence.json` | what was actually executed on real GPU hardware |
| `gen-shim-bindings.py` | generates `rocprim_shim_api.pas`, `rocprim_bindings.pas` |
| `gen-shim-header.py` | generates `rocprim_shim.h`, `rocprim_shim.map` |
| `DESIGN.md` | stub: the design record lives in the private [PasROCm-design](https://github.com/Eudox67/PasROCm-design) repository |

Data files are authoritative. The workflow matrices are produced by
`manifest.sh`, so adding a configuration is a one-line edit plus a
`./ci/manifest.sh validate`.

This directory is maintainer tooling. `.gitattributes` marks it
`export-ignore`, so it is absent from the `pasrocm-<version>-src.tar.gz`
release asset and from `git archive` output; a `git clone` still contains it.

```bash
ci/manifest.sh validate                   # self-check
ci/manifest.sh matrix --profile smoke     # what CI would run
ci/install-rocm.sh --config ubuntu-24.04-rocm10.1 --dry-run
ci/build-linux.sh --config ubuntu-24.04-rocm10.1 --target gfx1100
```
