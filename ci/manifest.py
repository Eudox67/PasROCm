#!/usr/bin/env python3
"""Read ci/supported-builds.json and ci/gfx-targets.json for CI.

Subcommands
    validate                       check manifest + target table for consistency
    matrix --profile P             emit {"include": [...]} for a GitHub job matrix
    env --id ID                    emit shell assignments for one configuration
    wave --target gfxNNNN          emit the wavefront width of one target
    targets --config ID            emit the applicable targets, ';' separated

One build produces exactly one gfx target (DESIGN.md section 3), so a matrix
entry is a (configuration, target) pair, never a list.
"""

import argparse
import json
import os
import shlex
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "ci", "supported-builds.json")
TARGETS = os.path.join(ROOT, "ci", "gfx-targets.json")

VALID_MODES = {"build", "compile-probe"}
VALID_OS = {"ubuntu-22.04", "ubuntu-24.04", "ubuntu-26.04"}
VALID_PIN_KINDS = {"radeon-apt", "amdrocm-stable"}
VALID_WIDTHS = {32, 64}
# ROCm 6.x/7.x publish jammy+noble repositories only; 10.x adds ubuntu2604.
OS_TO_UBUNTU = {
    "ubuntu-22.04": "ubuntu2204",
    "ubuntu-24.04": "ubuntu2404",
    "ubuntu-26.04": "ubuntu2604",
}
LEGACY_LINES = ("6.", "7.")


def fail(message):
    print("manifest: " + message, file=sys.stderr)
    raise SystemExit(2)


def read_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        fail("cannot read {}: {}".format(path, exc))
    except ValueError as exc:
        fail("{} is not valid JSON: {}".format(path, exc))


def load():
    return read_json(MANIFEST), read_json(TARGETS)


def width_bit(width):
    """Bit per hardware-supported wavefront width: 32 -> 1, 64 -> 2."""
    return {32: 1, 64: 2}.get(int(width), 0)


def version_key(line):
    """'10.1' -> (10, 1); used only to compare SDK lines."""
    parts = []
    for piece in str(line).split("."):
        try:
            parts.append(int(piece))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def configuration(manifest, config_id):
    for config in manifest["configurations"]:
        if config["id"] == config_id:
            return config
    fail("unknown configuration id '{}'".format(config_id))


def resolve(manifest, config):
    """Return (line, version, compiler, pin) with the pin fully expanded."""
    sdk_name = config.get("sdk")
    if sdk_name not in manifest.get("sdks", {}):
        fail("configuration '{}' references unknown sdk '{}'".format(config["id"], sdk_name))
    sdk = manifest["sdks"][sdk_name]
    pin = dict(sdk["pin"])
    pin["repo"] = pin["repo"].format(ubuntu=config["ubuntu"])
    pin.setdefault("suite", "stable" if pin["kind"] == "amdrocm-stable" else config["codename"])
    return sdk["line"], sdk["version"], sdk["compiler"], pin


def supported_targets(table, line):
    """Targets the given SDK line ships, in table order."""
    key = version_key(line)
    return [name for name, info in table["targets"].items()
            if version_key(info["min_rocm"]) <= key]


def config_targets(manifest, table, config):
    """Every target this configuration may build."""
    declared = config.get("targets", "auto")
    available = supported_targets(table, resolve(manifest, config)[0])
    if declared == "auto":
        return available
    unknown = [target for target in declared if target not in table["targets"]]
    if unknown:
        fail("configuration '{}' names unknown target(s): {}".format(config["id"], ", ".join(unknown)))
    too_new = [target for target in declared if target not in available]
    if too_new:
        fail("configuration '{}' requests target(s) the SDK line does not ship: {}".format(
            config["id"], ", ".join(too_new)))
    return list(declared)


def effective_targets(manifest, table, config, profile):
    """Targets one profile actually builds for one configuration."""
    allowed = config_targets(manifest, table, config)
    if profile.get("all_targets"):
        ceiling = config.get("full_targets", "all")
        if ceiling == "all":
            return allowed
        missing = [target for target in ceiling if target not in allowed]
        if missing:
            fail("configuration '{}' full_targets contains unsupported target(s): {}".format(
                config["id"], ", ".join(missing)))
        return [target for target in allowed if target in ceiling]
    wanted = profile.get("targets", [])
    return [target for target in allowed if target in wanted]


def validate(manifest, table):
    problems = []

    for name, info in table.get("targets", {}).items():
        wavefront = info.get("wavefront")
        widths = info.get("widths")
        if wavefront not in VALID_WIDTHS:
            problems.append("gfx-targets.json:{}: wavefront must be 32 or 64".format(name))
        if not widths:
            problems.append("gfx-targets.json:{}: widths is empty".format(name))
        elif any(width not in VALID_WIDTHS for width in widths):
            problems.append("gfx-targets.json:{}: widths must only contain 32/64".format(name))
        elif wavefront not in widths:
            problems.append("gfx-targets.json:{}: wavefront {} not in widths {}".format(
                name, wavefront, widths))
        if not info.get("family"):
            problems.append("gfx-targets.json:{}: missing family".format(name))
        if not info.get("min_rocm"):
            problems.append("gfx-targets.json:{}: missing min_rocm".format(name))

    seen = set()
    for config in manifest.get("configurations", []):
        where = "configuration[{}]".format(config.get("id", "?"))
        if config.get("id") in seen:
            problems.append(where + ": duplicate id")
        seen.add(config.get("id"))
        for key in ("id", "mode", "os", "ubuntu", "codename", "sdk", "covers", "covers_evidence"):
            if not config.get(key):
                problems.append("{}: missing '{}'".format(where, key))
        if config.get("mode") not in VALID_MODES:
            problems.append(where + ": mode must be one of " + ", ".join(sorted(VALID_MODES)))
        if config.get("os") not in VALID_OS:
            problems.append(where + ": unsupported runner label '{}'".format(config.get("os")))
        if OS_TO_UBUNTU.get(config.get("os")) != config.get("ubuntu"):
            problems.append(where + ": 'ubuntu' must be the repo directory for the runner os")
        if config.get("mode") == "compile-probe" and config.get("artifact", False):
            problems.append(where + ": compile-probe configurations must not publish artifacts")
        try:
            line, _version, compiler, pin = resolve(manifest, config)
            targets = config_targets(manifest, table, config)
        except SystemExit:
            problems.append(where + ": unresolved sdk or targets")
            continue
        if not targets:
            problems.append(where + ": no targets resolve for this SDK line")
        if pin["kind"] not in VALID_PIN_KINDS:
            problems.append(where + ": unknown pin kind '{}'".format(pin["kind"]))
        if compiler not in ("hipcc", "clang"):
            problems.append(where + ": compiler must be hipcc or clang")
        if config["os"] == "ubuntu-26.04" and not line.startswith("10."):
            problems.append(where + ": ROCm {} has no Ubuntu 26.04 repository".format(line))
        if line.startswith(LEGACY_LINES) and pin["kind"] != "radeon-apt":
            problems.append(where + ": ROCm {} must use the radeon-apt repository".format(line))
        if line.startswith("10.") and pin["kind"] != "amdrocm-stable":
            problems.append(where + ": ROCm {} must use the amdrocm-stable repository".format(line))

    ids = {config.get("id") for config in manifest.get("configurations", [])}
    for name, profile in manifest.get("profiles", {}).items():
        for config_id in profile.get("configs", []):
            if config_id not in ids:
                problems.append("profiles.{}: unknown configuration '{}'".format(name, config_id))
        for target in profile.get("targets", []):
            if target not in table.get("targets", {}):
                problems.append("profiles.{}: unknown target '{}'".format(name, target))
        if not profile.get("all_configs") and not profile.get("configs"):
            problems.append("profiles.{}: needs all_configs or configs".format(name))
        if not profile.get("all_targets") and not profile.get("targets"):
            problems.append("profiles.{}: needs all_targets or targets".format(name))

    if problems:
        for problem in problems:
            print("manifest: " + problem, file=sys.stderr)
        raise SystemExit(2)

    summary = ", ".join("{}={}".format(name, len(build_matrix(manifest, table, name)))
                        for name in manifest["profiles"])
    print("manifest OK: {} configurations, {} targets, {} SDK pins; jobs: {}".format(
        len(manifest["configurations"]), len(table["targets"]), len(manifest["sdks"]), summary))



def selected_configurations(manifest, profile_name):
    profile = manifest["profiles"].get(profile_name)
    if profile is None:
        fail("unknown profile '{}' (have: {})".format(
            profile_name, ", ".join(sorted(manifest["profiles"]))))
    if profile.get("all_configs"):
        return list(manifest["configurations"])
    return [configuration(manifest, config_id) for config_id in profile["configs"]]


def build_matrix(manifest, table, profile_name):
    profile = manifest["profiles"][profile_name]
    entries = []
    for config in selected_configurations(manifest, profile_name):
        line, version, compiler, _pin = resolve(manifest, config)
        for target in effective_targets(manifest, table, config, profile):
            entries.append({
                "name": "{}__{}".format(config["id"], target),
                "config": config["id"],
                "os": config["os"],
                "mode": config["mode"],
                "target": target,
                "wavefront": table["targets"][target]["wavefront"],
                "widths": ",".join(str(width) for width in table["targets"][target]["widths"]),
                "sdk_line": line,
                "sdk_version": version,
                "compiler": compiler,
                "artifact": config.get("artifact", config["mode"] == "build"),
            })
    entries.sort(key=lambda entry: (entry["config"], entry["target"]))
    return entries


def emit_env(manifest, table, config_id):
    config = configuration(manifest, config_id)
    line, version, compiler, pin = resolve(manifest, config)
    exact = pin.get("exact_versions", {})
    targets = config_targets(manifest, table, config)
    values = {
        "CFG_ID": config["id"],
        "CFG_MODE": config["mode"],
        "CFG_OS": config["os"],
        "CFG_UBUNTU": config["ubuntu"],
        "CFG_CODENAME": config["codename"],
        "CFG_SDK_LINE": line,
        "CFG_SDK_VERSION": version,
        "CFG_COMPILER_KIND": compiler,
        "CFG_COVERS": config["covers"],
        "CFG_COVERS_EVIDENCE": config["covers_evidence"],
        "CFG_TARGETS": " ".join(targets),
        "CFG_LIBRARY": manifest["shim"]["library"],
        "CFG_SHIM_VERSION": manifest["shim"]["version"],
        "CFG_ARTIFACT": "1" if config.get("artifact", config["mode"] == "build") else "0",
        "PIN_KIND": pin["kind"],
        "PIN_REPO": pin["repo"],
        "PIN_SUITE": pin["suite"],
        "PIN_KEY": pin["key"],
        "PIN_PACKAGES": " ".join(pin["packages"]),
        "PIN_EXACT": " ".join("{}={}".format(name, exact[name]) for name in pin["packages"]
                              if name in exact),
    }
    for key in sorted(values):
        print("{}={}".format(key, shlex.quote(values[key])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="check manifest and target table")

    matrix_parser = subparsers.add_parser("matrix", help="emit a job matrix")
    matrix_parser.add_argument("--profile", required=True)
    matrix_parser.add_argument("--github-output", action="store_true",
                               help="append matrix/count to $GITHUB_OUTPUT")

    env_parser = subparsers.add_parser("env", help="emit shell assignments for one configuration")
    env_parser.add_argument("--id", required=True)

    wave_parser = subparsers.add_parser("wave", help="emit the wavefront width of a target")
    wave_parser.add_argument("--target", required=True)

    mask_parser = subparsers.add_parser("mask", help="emit the width mask of a target (1=32, 2=64, 3=both)")
    mask_parser.add_argument("--target", required=True)

    targets_parser = subparsers.add_parser("targets", help="emit a configuration's targets")
    targets_parser.add_argument("--config", required=True)

    args = parser.parse_args(argv)
    manifest, table = load()

    if args.command == "validate":
        validate(manifest, table)
    elif args.command == "matrix":
        entries = build_matrix(manifest, table, args.profile)
        payload = json.dumps({"include": entries}, separators=(",", ":"))
        if args.github_output:
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
                handle.write("matrix={}\n".format(payload))
                handle.write("count={}\n".format(len(entries)))
        else:
            print(payload)
    elif args.command == "env":
        emit_env(manifest, table, args.id)
    elif args.command == "wave":
        info = table["targets"].get(args.target)
        if info is None:
            fail("unknown target '{}'".format(args.target))
        print(info["wavefront"])
    elif args.command == "mask":
        info = table["targets"].get(args.target)
        if info is None:
            fail("unknown target '{}'".format(args.target))
        mask = 0
        for width in info["widths"]:
            mask |= width_bit(width)
        print(mask)
    elif args.command == "targets":
        print(";".join(config_targets(manifest, table, configuration(manifest, args.config))))
    return 0


if __name__ == "__main__":
    sys.exit(main())

