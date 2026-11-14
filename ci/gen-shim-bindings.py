#!/usr/bin/env python3
"""Generate the Pascal shim binding units from the ABI specification.

    python3 ci/gen-shim-bindings.py            # rewrite the generated units
    python3 ci/gen-shim-bindings.py --check    # fail if they are stale (CI)

Input:  src/rocprim/rocprim_shim_abi.inc   (hand-edited, never compiled)
Output: src/rocprim/rocprim_shim_api.pas   (internal pointer table)
        src/rocprim/rocprim_bindings.pas   (the public 149-function surface)

The generated units never link the shim. `rocprim_bindings` wraps each entry
point so that a call cannot reach a nil pointer: the wrapper asks
`rocprim_shim_loader` to open the library that matches the detected GPU first.

Every parse runs the full validation (count, uniqueness, name shape, identity
exports first, parameter names extractable) in *both* modes, so a bad
specification can never produce a clean diff.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC = ROOT / "src" / "rocprim" / "rocprim_shim_abi.inc"
API_OUT = ROOT / "src" / "rocprim" / "rocprim_shim_api.pas"
BINDINGS_OUT = ROOT / "src" / "rocprim" / "rocprim_bindings.pas"

# The loader must call these before it can trust anything else, so they are
# required to be the first declarations in the specification and the generated
# table is indexed by the constants below.
IDENTITY = [
    ("pasrocm_shim_target", "TARGET"),
    ("pasrocm_shim_wavefront_size", "WAVEFRONT_SIZE"),
    ("pasrocm_shim_rocm_version", "ROCM_VERSION"),
    ("pasrocm_shim_wavefront_widths_mask", "WIDTHS_MASK"),
    ("pasrocm_wavefront_compatible", "WAVEFRONT_COMPATIBLE"),
    ("pasrocm_shim_abi_version", "ABI_VERSION"),
    ("pasrocm_shim_build_info", "BUILD_INFO"),
]

NAME_RE = re.compile(r"^(?:rocprim|pasrocm)_[a-z0-9_]+$")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DECL_RE = re.compile(
    r"^function\s+(?P<name>[A-Za-z_]\w*)\s*(?P<args>\(.*?\))?\s*"
    r":\s*(?P<ret>[^;]+?)\s*;\s*cdecl;\s*external\s+librocprimshim\s*;$",
    re.S,
)
EXTERNAL_RE = re.compile(r";\s*cdecl;\s*external\s+librocprimshim\s*;$")


class SpecError(Exception):
    """The specification cannot be turned into bindings."""


def read_section(text: str, start: str, end: str | None) -> str:
    """Return the text after `start`, up to `end` (or end of file)."""
    index = text.find(start)
    if index < 0:
        raise SpecError("missing marker %s" % start)
    index += len(start)
    if end is None:
        return text[index:]
    stop = text.find(end, index)
    if stop < 0:
        raise SpecError("missing marker %s after %s" % (end, start))
    return text[index:stop]


def strip_comments(text: str) -> str:
    """Remove Pascal comments so parameter parsing sees only declarations."""
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"\{[^}]*\}", "", text, flags=re.S)
    text = re.sub(r"\(\*.*?\*\)", "", text, flags=re.S)
    return text


def param_names(args: str | None) -> list[str]:
    """Names of the parameters in a `(a: T; b: U)` list, for the wrapper call."""
    if not args:
        return []
    inner = strip_comments(args.strip()[1:-1]).strip()
    if not inner:
        return []

    parts: list[str] = []
    current = ""
    depth = 0
    for char in inner:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        if char == ";" and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)

    names: list[str] = []
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        for qualifier in ("const ", "var ", "out "):
            if piece.lower().startswith(qualifier):
                piece = piece[len(qualifier):].strip()
        if ":" not in piece:
            raise SpecError("parameter without a type: %r" % part)
        for name in piece.split(":", 1)[0].split(","):
            name = name.strip()
            if not IDENT_RE.match(name):
                raise SpecError("not a parameter name: %r" % name)
            names.append(name)
    if not names:
        raise SpecError("parameter list produced no names: %r" % args)
    return names


def parse_blocks(text: str) -> list[dict]:
    """Split the declarations section into comment blocks + one declaration each."""
    items: list[dict] = []
    pending: list[str] = []
    lines = text.splitlines()
    in_brace = False
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if in_brace:
            pending.append(line.rstrip())
            if "}" in stripped:
                in_brace = False
        elif stripped.startswith("{"):
            pending.append(line.rstrip())
            if "}" not in stripped:
                in_brace = True
        elif stripped == "" or stripped.startswith("//"):
            pending.append(line.rstrip())
        elif stripped.startswith("function"):
            chunk = [line]
            while not EXTERNAL_RE.search(" ".join(chunk).strip()):
                index += 1
                if index >= len(lines):
                    raise SpecError("declaration never ends: %r" % stripped)
                chunk.append(lines[index])
            declaration = "\n".join(chunk).strip()
            match = DECL_RE.match(declaration)
            if not match:
                raise SpecError("cannot parse declaration:\n%s" % declaration)
            items.append(
                {
                    "comments": pending,
                    "decl": declaration,
                    "name": match.group("name"),
                    "args": match.group("args"),
                    "ret": match.group("ret").strip(),
                }
            )
            pending = []
        else:
            raise SpecError(
                "%s: unexpected text in the declarations section: %r"
                % (SPEC.name, line)
            )
        index += 1
    return items


def abi_version() -> int:
    """The ABI revision declared in the spec's CONST section."""
    text = SPEC.read_text(encoding="utf-8")
    match = re.search(r"PASROCM_ABI_REVISION\s*=\s*(\d+)\s*;", text)
    if not match:
        raise SpecError("missing PASROCM_ABI_REVISION in the CONST section")
    value = int(match.group(1))
    if value < 1:
        raise SpecError("PASROCM_ABI_REVISION must be >= 1")
    return value


def validate(items: list[dict], text: str) -> None:
    """Fail before anything is written. Runs in --check mode as well."""
    if not items:
        raise SpecError("no declarations found")

    declared_clauses = text.count("external librocprimshim")
    if declared_clauses != len(items):
        raise SpecError(
            "parsed %d declarations but found %d external clauses"
            % (len(items), declared_clauses)
        )

    seen: set[str] = set()
    for item in items:
        name = item["name"]
        if name in seen:
            raise SpecError("duplicate declaration: %s" % name)
        seen.add(name)
        if not NAME_RE.match(name):
            raise SpecError("name does not match %s: %s" % (NAME_RE.pattern, name))
        if not item["ret"]:
            raise SpecError("no return type for %s" % name)
        if item["args"]:
            param_names(item["args"])

    expected = [name for name, _ in IDENTITY]
    got = [item["name"] for item in items[: len(expected)]]
    if got != expected:
        raise SpecError(
            "the identity exports must come first, in this order: %s (found %s)"
            % (", ".join(expected), ", ".join(got))
        )

    rocprim_count = sum(1 for item in items if item["name"].startswith("rocprim_"))
    pasrocm_count = len(items) - rocprim_count
    if pasrocm_count != len(expected):
        raise SpecError(
            "expected exactly %d pasrocm_* identity exports, found %d"
            % (len(expected), pasrocm_count)
        )
    print(
        "== spec ok: %d declarations (%d rocprim_*, %d pasrocm_*)"
        % (len(items), rocprim_count, pasrocm_count)
    )


GENERATED_NOTICE = """{ GENERATED FILE -- do not edit.
  Source:     src/rocprim/rocprim_shim_abi.inc
  Generator:  ci/gen-shim-bindings.py
  Regenerate: python3 ci/gen-shim-bindings.py
"""


def pointer_decl(item: dict) -> str:
    """`__name: function(...): ret; cdecl;` -- signature identical to the spec."""
    if item["args"]:
        head = "  __%s: function%s" % (item["name"], item["args"])
    else:
        head = "  __%s: function" % item["name"]
    return "%s: %s; cdecl;" % (head, item["ret"])


def render_api(license_text: str, uses_text: str, items: list[dict]) -> str:
    count = len(items)
    out: list[str] = [license_text.rstrip(), "", GENERATED_NOTICE.rstrip(), "}", ""]
    out += [
        "{ Internal raw entry points into the rocPRIM shim.",
        "",
        "  Nothing here opens the library or checks the GPU: rocprim_shim_loader.pas",
        "  owns that policy and writes these slots only after the identity exports of",
        "  the opened library agree with the device. The names are prefixed with __ so",
        "  that the public API stays in rocprim_bindings.pas, which is what callers use.",
        "",
        "  The ABI types are declared once, in rocprim_bindings (which owns the public",
        "  surface); this unit imports them. FPC allows that direction because the",
        "  dependency back from rocprim_bindings is implementation-only.",
        "}",
        "",
        "unit rocprim_shim_api;",
        "",
        "{$mode objfpc}{$H+}",
        "",
        "interface",
        "",
        "uses",
        "  %s," % uses_text,
        "  rocprim_bindings;",
        "",
        "const",
        "  { Symbols the shim must export: 7 identity + 144 rocPRIM entry points. }",
        "  ROCPRIM_SHIM_SYMBOL_COUNT = %d;" % count,
        "",
        "  { ABI revision this unit was generated for; the loader refuses a",
        "    library whose pasrocm_shim_abi_version() differs. }",
        "  ROCPRIM_SHIM_ABI_VERSION = %d;" % abi_version(),
        "",
        "  { Indexes of the identity exports. They are the first declarations in the",
        "    specification because the loader must call them before it can trust the",
        "    rest of the table. }",
    ]
    for offset, (_name, label) in enumerate(IDENTITY):
        out.append("  ROCPRIM_SHIM_IDX_%s = %d;" % (label, offset))
    out += [
        "",
        "var",
        "  { Raw entry points, assigned by rocprim_shim_loader. Never call these",
        "    directly -- the wrappers in rocprim_bindings.pas load the library first. }",
    ]
    for item in items:
        if item["comments"]:
            out.extend(item["comments"])
        out.append(pointer_decl(item))
        out.append("")

    out += [
        "{ Table view used by the loader: one name and one slot per symbol. }",
        "function rocprim_shim_symbol_name(AIndex: Integer): PAnsiChar;",
        "function rocprim_shim_symbol_slot(AIndex: Integer): PPointer;",
        "",
        "implementation",
        "",
        "const",
        "  SHIM_SYMBOL_NAMES: array[0..ROCPRIM_SHIM_SYMBOL_COUNT - 1] of PAnsiChar = (",
    ]
    for offset, item in enumerate(items):
        suffix = "," if offset < len(items) - 1 else ""
        out.append("    '%s'%s" % (item["name"], suffix))
    out += [
        "  );",
        "",
        "var",
        "  SHIM_SYMBOL_SLOTS: array[0..ROCPRIM_SHIM_SYMBOL_COUNT - 1] of PPointer;",
        "",
        "function rocprim_shim_symbol_name(AIndex: Integer): PAnsiChar;",
        "begin",
        "  if (AIndex < 0) or (AIndex >= ROCPRIM_SHIM_SYMBOL_COUNT) then",
        "    Exit(nil);",
        "  Result := SHIM_SYMBOL_NAMES[AIndex];",
        "end;",
        "",
        "function rocprim_shim_symbol_slot(AIndex: Integer): PPointer;",
        "begin",
        "  if (AIndex < 0) or (AIndex >= ROCPRIM_SHIM_SYMBOL_COUNT) then",
        "    Exit(nil);",
        "  Result := SHIM_SYMBOL_SLOTS[AIndex];",
        "end;",
        "",
        "initialization",
    ]
    for index, item in enumerate(items):
        out.append(
            "  SHIM_SYMBOL_SLOTS[%d] := PPointer(@__%s);" % (index, item["name"])
        )
    out.append("")
    out.append("end.")
    out.append("")
    return "\n".join(out)


def render_bindings(license_text: str, uses_text: str, types_text: str,
                    const_text: str, items: list[dict]) -> str:
    out: list[str] = [license_text.rstrip(), "", GENERATED_NOTICE.rstrip(), "}", ""]
    out += [
        "{ The public rocPRIM shim surface. Each function is a thin wrapper: the call",
        "  asks rocprim_shim_loader.pas for the shim that matches the detected GPU, and",
        "  only then dispatches through the pointer table in rocprim_shim_api.pas. A",
        "  missing library, a library built for another GPU, or a missing symbol raises",
        "  an exception naming the problem -- it never calls through a nil pointer.",
        "",
        "  This unit keeps the original names and signatures, so existing programs need",
        "  no change; nothing here is resolved at link time. }",
        "",
        "unit rocprim_bindings;",
        "",
        "{$mode objfpc}{$H+}",
        "",
        "interface",
        "",
        "uses",
        "  %s;" % uses_text,
        "",
        types_text,
        "",
        const_text,
        "",
    ]
    for item in items:
        if item["comments"]:
            out.extend(item["comments"])
        out.append(EXTERNAL_RE.sub("; cdecl;", item["decl"]))
        out.append("")

    out += [
        "implementation",
        "",
        "uses",
        "  rocprim_shim_api,",
        "  rocprim_shim_loader;",
        "",
    ]
    for item in items:
        names = param_names(item["args"])
        out.append(EXTERNAL_RE.sub("; cdecl;", item["decl"]))
        out.append("begin")
        out.append("  PasROCmShimEnsureLoaded;")
        out.append(
            "  Result := rocprim_shim_api.__%s(%s);"
            % (item["name"], ", ".join(names))
        )
        out.append("end;")
        out.append("")
    out.append("end.")
    out.append("")
    return "\n".join(out)


def render_all() -> tuple[str, str, int]:
    text = SPEC.read_text(encoding="utf-8")
    if "{@@SPEC}" not in text:
        raise SpecError("missing marker {@@SPEC}")
    license_text = text.split("{@@SPEC}", 1)[0].rstrip()
    uses_text = read_section(text, "{@@USES}", "{@@TYPES}").strip()
    types_text = read_section(text, "{@@TYPES}", "{@@CONST}").strip()
    const_text = read_section(text, "{@@CONST}", "{@@DECLARATIONS}").strip()
    declarations = read_section(text, "{@@DECLARATIONS}", None)
    items = parse_blocks(declarations)
    validate(items, declarations)
    return (
        render_api(license_text, uses_text, items),
        render_bindings(license_text, uses_text, types_text, const_text, items),
        len(items),
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="only verify that the generated units match the specification",
    )
    options = parser.parse_args(argv)

    try:
        api_text, bindings_text, count = render_all()
    except SpecError as error:
        print("gen-shim-bindings: %s" % error, file=sys.stderr)
        return 1

    outputs = [(API_OUT, api_text), (BINDINGS_OUT, bindings_text)]

    if options.check:
        stale = []
        for path, expected in outputs:
            try:
                current = path.read_text(encoding="utf-8")
            except OSError:
                stale.append(path)
                continue
            if current != expected:
                stale.append(path)
        if stale:
            for path in stale:
                print("gen-shim-bindings: stale: %s" % path, file=sys.stderr)
            print(
                "gen-shim-bindings: run 'python3 ci/gen-shim-bindings.py' and "
                "commit the result",
                file=sys.stderr,
            )
            return 1
        print("== generated units are current (%d declarations)" % count)
        return 0

    for path, text in outputs:
        path.write_text(text, encoding="utf-8")
        print("== wrote %s" % path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

