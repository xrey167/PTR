"""Scaffold a new experiment that passes `run_experiment.py validate` as written.

The previous scaffolder wrote only `id`, `status` and `hypothesis`, so every
manifest it produced was missing six of the nine keys `experiments/schema.toml`
requires, had no `config.toml` or `tests/`, and was never added to
`experiments/registry.toml` — the runner could not resolve it and the validator
never saw it. This one asks for every required field up front, refuses before it
writes anything, and registers the experiment as its last step.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (17, 29, 43, 71, 101)
DEFAULT_HARDWARE = "hardware/default.toml"
EXPERIMENT_ID = re.compile(r"^[A-Z][0-9]{3}$")
SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
# An area is one of the category directories under experiments/, named in plain
# lowercase. Checking the name, and not only that the directory exists, is what
# keeps `..` (which exists) from placing an experiment outside experiments/.
AREA = re.compile(r"^[a-z]+$")
HARDWARE_PROFILE = re.compile(r"^hardware/([a-z0-9]+(?:-[a-z0-9]+)*)\.toml$")
MAX_SEED = 2**63 - 1


def toml_string(value: str) -> str:
    """A TOML basic string. JSON's escapes are a subset of TOML's, except that
    TOML also forbids a raw DEL, which JSON leaves unescaped."""
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007F")


def check_seeds(seeds: list[int]) -> list[int]:
    """Seeds are passed to entrypoints as decimal text and parsed there, typically
    as a u64, so they must be distinct non-negative integers that fit one."""
    if not seeds:
        raise ValueError("at least one seed is required")
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
            raise ValueError(f"each seed must be an integer from 0 to {MAX_SEED}, got {seed!r}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be distinct, got {seeds}")
    return list(seeds)


def parse_seeds(text: str) -> list[int]:
    try:
        seeds = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise ValueError(f"seeds must be comma-separated integers, got {text!r}") from None
    return check_seeds(seeds)


def manifest(
    exp_id: str,
    *,
    hypothesis: str,
    metrics: str,
    baseline: str,
    falsification: str,
    hardware_profile: str,
    seeds: list[int],
) -> str:
    # Same keys, in the same order, as the manifests already in the tree.
    lines = [
        "version = 1",
        f"id = {toml_string(exp_id)}",
        'status = "planned"',
        f"hypothesis = {toml_string(hypothesis)}",
        f"metrics = {toml_string(metrics)}",
        f"baseline = {toml_string(baseline)}",
        f"falsification = {toml_string(falsification)}",
        f"hardware_profile = {toml_string(hardware_profile)}",
        'entrypoint = ""',
        f"seeds = [{', '.join(str(seed) for seed in seeds)}]",
        'results_dir = "results"',
        'required_artifacts = ["run.json", "metrics.json"]',
    ]
    return "\n".join(lines) + "\n"


def create(
    root: Path,
    path: str,
    *,
    hypothesis: str,
    metrics: str,
    baseline: str,
    falsification: str,
    hardware_profile: str = DEFAULT_HARDWARE,
    seeds: list[int] | None = None,
) -> Path:
    """Create and register `experiments/<path>` under `root`.

    Every refusal happens before the first write, and a write that fails anyway
    (a full disk, a permission) removes what was created and restores the
    registry, so a failed call leaves the tree exactly as it was."""
    seeds = check_seeds(list(DEFAULT_SEEDS) if seeds is None else seeds)
    experiments = root / "experiments"
    registry_path = experiments / "registry.toml"

    parts = path.strip("/").split("/")
    if len(parts) != 2 or "\\" in path:
        raise ValueError(f"path must be <area>/<ID>-<name>, got {path!r}")
    area, name = parts
    if not AREA.match(area) or area == "tests" or not (experiments / area).is_dir():
        raise ValueError(f"unknown experiment area {area!r}")
    exp_id, _, slug = name.partition("-")
    if not EXPERIMENT_ID.match(exp_id):
        raise ValueError(f"experiment id must be a capital letter and three digits, got {exp_id!r}")
    if not SLUG.match(slug):
        raise ValueError(f"experiment name must be lowercase words joined by '-', got {slug!r}")
    for field, value in [
        ("hypothesis", hypothesis),
        ("metrics", metrics),
        ("baseline", baseline),
        ("falsification", falsification),
    ]:
        if not value.strip():
            raise ValueError(f"{field} must not be empty")
        # Text that cannot be written would otherwise fail halfway through the
        # writes below (argv bytes that are not UTF-8 arrive as lone surrogates).
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError(f"{field} is not valid UTF-8 text") from None
    profile = HARDWARE_PROFILE.match(hardware_profile)
    if not profile or profile.group(1) == "config":
        raise ValueError(
            f"hardware profile must be hardware/<name>.toml (not the area's config.toml), "
            f"got {hardware_profile!r}"
        )
    if not (root / hardware_profile).is_file():
        raise ValueError(f"hardware profile {hardware_profile!r} does not exist")

    registry_bytes = registry_path.read_bytes()
    registry_text = registry_bytes.decode("utf-8")
    registered = tomllib.loads(registry_text).get("experiment", [])
    if any(entry["id"] == exp_id for entry in registered):
        raise ValueError(f"experiment id {exp_id} is already registered")
    target = experiments / area / name
    if target.exists():
        raise ValueError(f"experiments/{area}/{name} already exists")

    text = manifest(
        exp_id,
        hypothesis=hypothesis,
        metrics=metrics,
        baseline=baseline,
        falsification=falsification,
        hardware_profile=hardware_profile,
        seeds=seeds,
    )
    # Read back what would be written, so an escaping mistake is a refusal
    # rather than a manifest that no TOML reader accepts.
    parsed = tomllib.loads(text)
    expected = {
        "hypothesis": hypothesis,
        "metrics": metrics,
        "baseline": baseline,
        "falsification": falsification,
        "hardware_profile": hardware_profile,
        "seeds": seeds,
    }
    if any(parsed[key] != value for key, value in expected.items()):
        raise ValueError("manifest does not round-trip through TOML")

    relative = f"experiments/{area}/{name}"
    try:
        write_experiment(target, relative, name, text, hypothesis, metrics, baseline, falsification)
        separator = "" if registry_text.endswith("\n") else "\n"
        with registry_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{separator}\n[[experiment]]\n"
                f"id = {toml_string(exp_id)}\n"
                f"path = {toml_string(f'{area}/{name}')}\n"
                'status = "planned"\n'
            )
    except OSError as error:
        shutil.rmtree(target, ignore_errors=True)
        registry_path.write_bytes(registry_bytes)
        raise ValueError(f"could not write {relative}, nothing was kept: {error}") from None
    return target


def write_experiment(
    target: Path,
    relative: str,
    name: str,
    text: str,
    hypothesis: str,
    metrics: str,
    baseline: str,
    falsification: str,
) -> None:
    target.mkdir(parents=True)
    (target / "experiment.toml").write_text(text, encoding="utf-8")
    (target / "config.toml").write_text(
        "version = 1\n"
        'kind = "workspace-area"\n'
        f"name = {toml_string(name)}\n"
        f"path = {toml_string(relative)}\n"
        'tests_dir = "tests"\n',
        encoding="utf-8",
    )
    (target / "README.md").write_text(
        f"# {name}\n\n"
        f"## Hypothesis\n{hypothesis}\n\n"
        f"## Primary metrics\n{metrics}\n\n"
        f"## Baseline\n{baseline}\n\n"
        f"## Falsification\n{falsification}\n\n"
        "## Rule\n"
        "Record matched baselines, hardware, seeds and negative results. "
        "Do not change success criteria after observing results.\n",
        encoding="utf-8",
    )
    (target / "tests").mkdir()
    (target / "tests" / "README.md").write_text(
        f"# Tests for {relative}\n\n"
        f"This directory holds tests, fixtures, validation cases, or executable checks "
        f"owned by `{relative}`. Add real tests here as the area becomes executable; "
        "do not use this directory as evidence of implementation by itself.\n",
        encoding="utf-8",
    )
    (target / "results").mkdir()
    (target / "results" / ".gitkeep").write_text("", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="<area>/<ID>-<name>, e.g. model/M008-slot-dropout")
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--falsification", required=True)
    parser.add_argument("--hardware-profile", default=DEFAULT_HARDWARE)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="comma-separated, e.g. 17,29,43",
    )
    args = parser.parse_args(argv)
    try:
        target = create(
            ROOT,
            args.path,
            hypothesis=args.hypothesis,
            metrics=args.metrics,
            baseline=args.baseline,
            falsification=args.falsification,
            hardware_profile=args.hardware_profile,
            seeds=parse_seeds(args.seeds),
        )
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(target.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
