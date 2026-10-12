#!/usr/bin/env python3
"""Sign and pin an experiment's preregistration for the freeze, all or nothing.

    python3 scripts/freeze/freeze_sign.py --id F007 --dir experiments/feedback/F007-vcw-loop \
        [--sign must-be-signed-primary-margin=30 ...] [--dry-run] [--root <worktree>]

What it does, in memory first:
  * replaces each `--sign must-be-signed-<name>=<int>` placeholder in config.toml (exactly one
    assignment per placeholder);
  * pins `[preregistration].criteria_sha256` and `.preregistration_text_sha256` when they are still
    `must-be-...` placeholders (scripts/experiment_records.py preregistered_file_digest of
    criteria.toml and PREREGISTRATION.md); an already pinned value must equal the file digest;
  * refuses when any string value of the parsed config still starts with `must-be` (comments never
    count, neither to trip nor to hide a check);
  * refuses an id that experiments/preregistration.toml does not list;
  * turns experiment.toml `status = "planned"` into `prepared` and appends
    `preregistration_sha256` (digest of the [preregistration] table) and
    `preregistration_rules_sha256` (canonical digest of the id's entry in
    experiments/preregistration.toml), exactly as the prepared experiments carry them;
  * turns the experiment's experiments/registry.toml status `planned` into `prepared`.

Files are written only if every check passes and --dry-run is not given; if a write fails midway the
files already replaced are restored. Exit 0 ok, 4 refusal (FREEZE-SIGN ERROR on stderr).
The repository is the one holding this script unless --root names another tree (tests, worktrees).
"""

import argparse
import re
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atomic_write import atomic_write_all  # noqa: E402

PLACEHOLDER = "must-be"
PIN_KEYS = {
    "preregistration_text_sha256": "PREREGISTRATION.md",
    "criteria_sha256": "criteria.toml",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class FreezeError(Exception):
    pass


def placeholder_values(value, path=""):
    """Yield (dotted path, value) of every string value that still is a placeholder."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from placeholder_values(item, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from placeholder_values(item, f"{path}[{index}]")
    elif isinstance(value, str) and value.startswith(PLACEHOLDER):
        yield path, value


def replace_key_value(text, key, value, replacement):
    """Replace the one assignment of `key` whose value is the quoted placeholder `value`."""
    pattern = re.compile(
        r'^(\s*%s\s*=\s*)"%s"' % (re.escape(key), re.escape(value)), re.MULTILINE
    )
    found = pattern.findall(text)
    if len(found) != 1:
        raise FreezeError(f'{key} = "{value}" is assigned {len(found)} times, expected 1')
    return pattern.sub(lambda m: m.group(1) + replacement, text, count=1)


def replace_assignment_value(text, value, replacement):
    """Replace exactly one assignment (any key) whose value is the quoted placeholder `value`."""
    pattern = re.compile(r'^(\s*[A-Za-z0-9_]+\s*=\s*)"%s"' % re.escape(value), re.MULTILINE)
    found = pattern.findall(text)
    if len(found) != 1:
        raise FreezeError(f'placeholder "{value}" is assigned {len(found)} times, expected 1')
    return pattern.sub(lambda m: m.group(1) + replacement, text, count=1)


def load_records(root: Path):
    scripts = root / "scripts"
    if not (scripts / "experiment_records.py").is_file():
        scripts = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(scripts))
    import experiment_records as records  # noqa: E402  (repository module)

    return records


def sign(args):
    root = Path(args.root).resolve()
    records = load_records(root)
    directory = root / args.dir
    config_path = directory / "config.toml"
    experiment_path = directory / "experiment.toml"
    registry_path = root / "experiments/registry.toml"
    list_path = root / "experiments/preregistration.toml"
    if not list_path.is_file():
        raise FreezeError("missing file experiments/preregistration.toml")
    entry = tomllib.loads(list_path.read_text(encoding="utf-8")).get("experiment", {}).get(args.id)
    if entry is None:
        raise FreezeError(f"{args.id} is not listed in experiments/preregistration.toml")

    for path in (config_path, experiment_path, registry_path,
                 directory / "PREREGISTRATION.md", directory / "criteria.toml"):
        if not path.is_file():
            raise FreezeError(f"missing file {path.relative_to(root)}")

    config = config_path.read_text(encoding="utf-8")
    for item in args.sign:
        placeholder, _, raw = item.partition("=")
        if not placeholder.startswith(PLACEHOLDER) or not re.fullmatch(r"-?[0-9]+", raw):
            raise FreezeError(f"bad --sign {item!r} (expected must-be-...=<int>)")
        config = replace_assignment_value(config, placeholder, raw)

    table = tomllib.loads(config).get("preregistration")
    if not isinstance(table, dict):
        raise FreezeError("config has no [preregistration] table")
    for key, filename in PIN_KEYS.items():
        current = table.get(key)
        if not isinstance(current, str):
            raise FreezeError(f"[preregistration].{key} missing")
        digest = records.preregistered_file_digest(directory / filename)
        if current.startswith(PLACEHOLDER):
            config = replace_key_value(config, key, current, '"%s"' % digest)
        elif current != digest:
            raise FreezeError(f"[preregistration].{key} is pinned to {current}, {filename} has {digest}")

    parsed = tomllib.loads(config)
    left = list(placeholder_values(parsed))
    if left:
        raise FreezeError("placeholder values left after signing: %s" % left)
    table = parsed["preregistration"]

    experiment = experiment_path.read_text(encoding="utf-8")
    current = tomllib.loads(experiment)
    if current.get("id") != args.id:
        raise FreezeError(f"experiment.toml id is {current.get('id')!r}, not {args.id!r}")
    if current.get("status") != "planned":
        raise FreezeError(f"experiment.toml status is {current.get('status')!r}, expected 'planned'")
    if "preregistration_sha256" in current or "preregistration_rules_sha256" in current:
        raise FreezeError("experiment.toml already carries preregistration digests")
    experiment = re.sub(r'^status = "planned"', 'status = "prepared"', experiment, count=1,
                        flags=re.MULTILINE).rstrip("\n")
    experiment += '\npreregistration_sha256 = "%s"\npreregistration_rules_sha256 = "%s"\n' % (
        records.preregistration_digest(table), records.canonical_digest(entry))
    if tomllib.loads(experiment).get("status") != "prepared":
        raise FreezeError("experiment.toml status did not become 'prepared'")

    registry = registry_path.read_text(encoding="utf-8")
    new_registry, count = re.subn(
        r'(id = "%s"\npath = "[^"]+"\nstatus = )"planned"' % re.escape(args.id),
        r'\1"prepared"', registry)
    if count != 1:
        raise FreezeError(f"registry entry for {args.id} with status planned found {count} times")
    tomllib.loads(new_registry)

    if args.dry_run:
        print(f"DRY-RUN OK {args.id}: signed {len(args.sign)}, pinned {sorted(PIN_KEYS)}, "
              "experiment and registry would become prepared; nothing written")
        return 0
    atomic_write_all({config_path: config, experiment_path: experiment, registry_path: new_registry})
    print(f"SIGNED {args.id}: config, experiment.toml and registry written")
    return 0


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--id", required=True)
    parser.add_argument("--dir", required=True, help="experiment directory, repo-relative")
    parser.add_argument("--sign", action="append", default=[], metavar="PLACEHOLDER=INT")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]),
                        help="tree to freeze (default: the repository holding this script)")
    return sign(parser.parse_args(argv))


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except FreezeError as error:
        print(f"FREEZE-SIGN ERROR: {error}", file=sys.stderr)
        sys.exit(4)
