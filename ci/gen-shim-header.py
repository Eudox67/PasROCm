#!/usr/bin/env python3
"""Generate the public C header for the rocPRIM shim.

    python3 ci/gen-shim-header.py            # rewrite src/rocprim/rocprim_shim.h
    python3 ci/gen-shim-header.py --check    # fail if stale (CI)

Why the prototypes come from the definitions and not from
rocprim_shim_abi.inc: a public C header must match the definitions exactly,
including const-ness, and the Pascal declarations do not encode const. The
specification still owns the *names*: this script refuses to write a header
unless the definitions cover exactly the 149 names in the spec, in spec order,
so a name can never appear in one place and not the other.

The header is also the visibility contract: every prototype carries
PASROCM_EXPORT, and build.sh/CMake compile with -fvisibility=hidden, so the
library exports exactly these symbols, and nothing else.

Exit codes: 0 ok, 1 spec/definition mismatch, 2 stale in --check mode.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC = ROOT / "src" / "rocprim" / "rocprim_shim_abi.inc"
SOURCES = [
    ROOT / "src" / "rocprim" / "rocprim_shim.cpp",
    ROOT / "src" / "rocprim" / "rocprim_shim_math_kernels.cpp",
]
HEADER = ROOT / "src" / "rocprim" / "rocprim_shim.h"
VERSION_SCRIPT = ROOT / "src" / "rocprim" / "rocprim_shim.map"

DECL_NAME_RE = re.compile(r"^function\s+([A-Za-z_]\w*)", re.M)
# Exported definitions are either at column 0 or inside an indented
# `extern "C" { ... }` block. `definitions()` additionally requires the
# parameter list to be followed by '{', so a call can never be mistaken for a
# definition.
DEFN_RE = re.compile(
    r'^[ \t]*(?:extern\s+"C"\s+)?'
    r"(?:const\s+char\s*\*|hipError_t|int|unsigned|float|double|void|size_t)"
    r"\s+([A-Za-z_]\w*)\s*\(",
    re.M,
)


class HeaderError(Exception):
    """The spec and the definitions disagree, or the source is unparseable."""


def spec_names() -> list[str]:
    text = SPEC.read_text(encoding="utf-8")
    marker = "{@@DECLARATIONS}"
    if marker not in text:
        raise HeaderError("missing %s in %s" % (marker, SPEC.name))
    declarations = text.split(marker, 1)[1]
    names = DECL_NAME_RE.findall(declarations)
    if not names:
        raise HeaderError("no declarations found in %s" % SPEC.name)
    if len(set(names)) != len(names):
        raise HeaderError("duplicate declarations in %s" % SPEC.name)
    return names


def abi_enum() -> tuple[str, list[str]]:
    """The ABI enum declared in the spec's TYPES section."""
    text = SPEC.read_text(encoding="utf-8")
    if "{@@TYPES}" not in text or "{@@CONST}" not in text:
        raise HeaderError("missing TYPES/CONST markers in %s" % SPEC.name)
    types = text.split("{@@TYPES}", 1)[1].split("{@@CONST}", 1)[0]
    match = re.search(r"(\w+)\s*=\s*\(\s*([\w\s,]+?)\s*\);", types, re.S)
    if not match:
        raise HeaderError("no enum declaration in the TYPES section")
    name = match.group(1)
    members = [item.strip() for item in match.group(2).split(",") if item.strip()]
    if not members:
        raise HeaderError("enum %s has no members" % name)
    return name, members


def abi_version() -> int:
    """The ABI revision declared in the spec's CONST section."""
    text = SPEC.read_text(encoding="utf-8")
    match = re.search(r"PASROCM_ABI_REVISION\s*=\s*(\d+)\s*;", text)
    if not match:
        raise HeaderError("missing PASROCM_ABI_REVISION in the CONST section")
    value = int(match.group(1))
    if value < 1:
        raise HeaderError("PASROCM_ABI_REVISION must be >= 1")
    return value


def license_text() -> str:
    """The spec's Pascal-style licence block, rewrapped as a C comment."""
    text = SPEC.read_text(encoding="utf-8")
    marker = "{@@SPEC}"
    if marker not in text:
        raise HeaderError("missing %s" % marker)
    block = text.split(marker, 1)[0].rstrip()
    body = block.strip()
    if body.startswith("{"):
        body = body[1:]
    if body.rstrip().endswith("}"):
        body = body.rstrip()[:-1]
    lines = [line.rstrip() for line in body.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "/*\n" + "\n".join(lines) + "\n*/"


def signature_at(source: str, start: int, open_paren: int) -> tuple[str, int]:
    """Return (signature text, index of the closing paren)."""
    depth = 0
    index = open_paren
    while index < len(source):
        char = source[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                text = source[start:index + 1].strip()
                text = re.sub(r'^extern\s+"C"\s+', "", text).strip()
                return text, index
        index += 1
    raise HeaderError("unbalanced parentheses at offset %d" % start)


def definitions() -> dict[str, str]:
    found: dict[str, str] = {}
    for path in SOURCES:
        source = path.read_text(encoding="utf-8")
        for match in DEFN_RE.finditer(source):
            name = match.group(1)
            text, close = signature_at(source, match.start(), match.end() - 1)
            if "__global__" in text or "__restrict__" in text:
                continue  # kernels and device-only qualifiers are not public ABI
            # Only a definition, never a declaration or a call.
            if not source[close + 1:].lstrip().startswith("{"):
                continue
            if name in found:
                raise HeaderError("definition of %s appears twice" % name)
            found[name] = text
    return found


def render_map() -> str:
    """The linker version script: the same 149 names, everything else local.

    -fvisibility=hidden alone is not enough. Measured 2026-11-14 on gfx1100:
    every __global__ kernel stayed in the dynamic symbol table (30+ symbols)
    plus rocprim_fill_kernel_int, because HIP emits kernel host stubs with
    default visibility so the runtime can register them. A version script is
    the only thing that closes the table exactly.
    """
    names = spec_names()
    lines = [
        "# GENERATED FILE -- do not edit.",
        "# Exported surface of the rocPRIM shim; every other symbol is local.",
        "# Generated from rocprim_shim_abi.inc by ci/gen-shim-header.py.",
        "# See render_map() for why -fvisibility=hidden is not sufficient.",
        "{",
        "  global:",
    ]
    lines += ["    %s;" % name for name in names]
    lines += ["  local:", "    *;", "};", ""]
    return "\n".join(lines)


def render() -> str:
    names = spec_names()
    defs = definitions()

    missing = [name for name in names if name not in defs]
    if missing:
        raise HeaderError(
            "declared in the spec but not defined: %s" % ", ".join(missing)
        )
    extra = sorted(set(defs) - set(names))
    if extra:
        # Not fatal by itself: these are the internals visibility is meant to
        # hide. They simply do not go into the public header.
        print("== not public (hidden by -fvisibility=hidden): %s" % ", ".join(extra))

    out: list[str] = [license_text(), "", """/*
 * GENERATED FILE -- do not edit.
 *   Source:    the definitions in rocprim_shim.cpp and
 *              rocprim_shim_math_kernels.cpp, gated by the 149 names in
 *              rocprim_shim_abi.inc (which owns the ABI contract).
 *   Generator: ci/gen-shim-header.py
 *   Regenerate: python3 ci/gen-shim-header.py
 *
 * Every prototype carries PASROCM_EXPORT because the library is compiled with
 * -fvisibility=hidden: this header is the complete list of exported symbols.
 * The SONAME is the on-disk file name (libpasrocm_shim_<gfx>.so), because the
 * Pascal loader resolves the library by exact file name; ABI breaks are gated
 * by ci/check-abi.sh --strict and the package version, not by a soname suffix.
 */
#ifndef PASROCM_ROCprim_SHIM_H
#define PASROCM_ROCprim_SHIM_H

#include <hip/hip_runtime.h> /* hipError_t, hipStream_t */
#include <hip/hip_fp16.h>     /* __half */
#include <hip/hip_bfloat16.h> /* hip_bfloat16 */
#include <stdbool.h>
#include <stddef.h> /* size_t */

#if defined(_WIN32)
#  define PASROCM_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#  define PASROCM_EXPORT __attribute__((visibility("default")))
#else
#  define PASROCM_EXPORT
#endif

/* ABI revision, generated from the spec. A consumer compares this with
 * pasrocm_shim_abi_version() before trusting the prototypes below. */
#define PASROCM_SHIM_ABI_VERSION ABI_VERSION_PLACEHOLDER

#ifdef __cplusplus
extern "C" {
#endif

/* ABI enum, generated from the spec's TYPES section so the C and Pascal sides
 * cannot drift. Same order means the same values. */
typedef enum ENUM_NAME {
ENUM_MEMBERS
} ENUM_NAME;
"""]
    enum_name, enum_members = abi_enum()
    out[-1] = out[-1].replace("ENUM_NAME", enum_name).replace(
        "ENUM_MEMBERS",
        "\n".join(
            "    %s%s" % (member, "," if index < len(enum_members) - 1 else "")
            for index, member in enumerate(enum_members)
        ),
    )
    out[-1] = out[-1].replace("ABI_VERSION_PLACEHOLDER", str(abi_version()))
    for name in names:
        # rocprim::int128_t is `using int128_t = __int128_t` (rocprim/types.hpp),
        # so the builtin spelling is the same type and keeps the header free of
        # rocPRIM includes -- it is a public C header, HIP only.
        text = defs[name].replace("rocprim::int128_t", "__int128_t")
        out.append("PASROCM_EXPORT %s;" % text)
    out += ["", """#ifdef __cplusplus
}
#endif

#endif /* PASROCM_ROCprim_SHIM_H */"""]
    return "\n".join(out) + "\n"

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="only verify that the committed header and map match the sources",
    )
    options = parser.parse_args(argv)

    try:
        outputs = [
            (HEADER, render()),
            (VERSION_SCRIPT, render_map()),
        ]
    except HeaderError as error:
        print("gen-shim-header: %s" % error, file=sys.stderr)
        return 1

    if options.check:
        stale: list[pathlib.Path] = []
        for path, expected in outputs:
            try:
                current = path.read_text(encoding="utf-8")
            except OSError:
                current = None
            if current != expected:
                stale.append(path)
        if stale:
            for path in stale:
                print("gen-shim-header: stale: %s" % path.relative_to(ROOT), file=sys.stderr)
            print(
                "gen-shim-header: run 'python3 ci/gen-shim-header.py' and commit the result",
                file=sys.stderr,
            )
            return 2
        print("== public header and version script are current")
        return 0

    for path, text in outputs:
        path.write_text(text, encoding="utf-8")
        print("== wrote %s" % path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
