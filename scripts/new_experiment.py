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
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (17, 29, 43, 71, 101)
DEFAULT_HARDWARE = "hardware/default.toml"
EXPERIMENT_ID = re.compile(r"^[A-Z][0-9]{3}$")
SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def toml_string(value: str) -> str:
    """A TOML basic string. JSON's escapes are a subset of TOML's, except that
    TOML also forbids a raw DEL, which JSON leaves unescaped."""
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007F")


def parse_seeds(text: str) -> list[int]:
    try:
        seeds = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise ValueError(f"seeds must be comma-separated integers, got {text!r}") from None
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be distinct, got {seeds}")
    return seeds


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

    Every refusal happens before the first write, so a refused call leaves the
    tree exactly as it was."""
    seeds = list(DEFAULT_SEEDS) if seeds is None else seeds
    experiments = root / "experiments"
    registry_path = experiments / "registry.toml"

    parts = path.strip("/").split("/")
    if len(parts) != 2:
        raise ValueError(f"path must be <area>/<ID>-<name>, got {path!r}")
    area, name = parts
    if area == "tests" or not (experiments / area).is_dir():
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
    if not (root / hardware_profile).is_file():
        raise ValueError(f"hardware profile {hardware_profile!r} does not exist")

    registry_text = registry_path.read_text(encoding="utf-8")
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

    separator = "" if registry_text.endswith("\n") else "\n"
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{separator}\n[[experiment]]\n"
            f"id = {toml_string(exp_id)}\n"
            f"path = {toml_string(f'{area}/{name}')}\n"
            'status = "planned"\n'
        )
    return target


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
