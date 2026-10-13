"""Identities of the A0 mechanism ablation studies, shared by the driver, the
aggregator and the report.

v1 (research/falsification/A0-ablations-v1) ran through the exploratory runner
under the M001-M004 manifests with separate `a0_*` entrypoints. It is archival.

v2 (research/falsification/A0-ablations-v2) is its successor. Every process of
it runs through the hardened runner as a *listed* experiment of its own
(scripts/run_experiment.py, experiments/preregistration.toml): one frozen
`entrypoint` per experiment, one run per seed, so the sweep (one learning rate
per experiment), the evaluation (one experiment per family), the determinism
rerun and any contingency are separate experiments. This module names them,
classifies their records and renders their registration. It performs no I/O
except `write_experiment` and `append_registration`, which take their root as an
argument so tests can use a temporary tree.
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_records  # noqa: E402

FAMILIES = ("M001", "M002", "M003", "M004")
SEED = 17
KINDS = ("sweep", "eval", "rerun", "contingency")
V1_ENTRYPOINTS = {
    "sweep": "a0_sweep_entrypoint",
    "eval": "a0_ablation_entrypoint",
    "rerun": "a0_rerun_entrypoint",
    "contingency": "a0_contingency_entrypoint",
}
V1_FAMILY_PATHS = {
    "M001": "model/M001-semantic-slots",
    "M002": "model/M002-typed-attention",
    "M003": "model/M003-latent-recurrence",
    "M004": "model/M004-operator-router",
}
HYPOTHESES = {
    "M001": ("Do typed semantic slots improve the A0 composite over content-free slots?",
             "composite A; test_iid accuracy"),
    "M002": ("Does the typed attention bias contribute to A0's constraint retention?",
             "composite B; test_iid accuracy"),
    "M003": ("Does a second tied latent refinement step help A0?", "composite C; test_iid accuracy"),
    "M004": ("Does a learned operator router beat a frozen one inside A0?", "test_iid accuracy"),
}
V2_ID = re.compile(
    r"^(?P<family>M00[1-4])-a0v2-(?:(?P<sweep>sweep-lr(?P<lr>[0-9p]+))|(?P<eval>eval)|(?P<rerun>rerun)"
    r"|cont-(?P<arm>[a-z0-9-]+))$"
)


@dataclasses.dataclass(frozen=True)
class Study:
    """One study's identity: where its files live and how its runs are named."""

    key: str
    name: str
    directory: str
    config: str
    prereg_tag: str
    prereg_commit: str | None
    hardened: bool

    @property
    def path(self) -> Path:
        return ROOT / self.directory

    # ---- v2 experiment naming -------------------------------------------------
    def sweep_id(self, family: str, lr: float) -> str:
        return f"{family}-a0v2-sweep-lr{str(lr).replace('.', 'p')}"

    def eval_id(self, family: str) -> str:
        return f"{family}-a0v2-eval"

    def rerun_id(self) -> str:
        return "M001-a0v2-rerun"

    def contingency_id(self, family: str, arm: str) -> str:
        return f"{family}-a0v2-cont-{arm}"

    def experiment_path(self, exp_id: str) -> str:
        """The experiment's directory below experiments/."""
        return f"model/{exp_id}"

    def classify(self, exp_id: str) -> tuple[str, str] | None:
        """(kind, family) of a v2 experiment id, or None for any other id."""
        match = V2_ID.match(exp_id)
        if not self.hardened or not match:
            return None
        kind = ("sweep" if match["sweep"] else "eval" if match["eval"]
                else "rerun" if match["rerun"] else "contingency")
        return kind, match["family"]

    def learning_rate(self, exp_id: str) -> float | None:
        match = V2_ID.match(exp_id)
        return float(match["lr"].replace("p", ".")) if match and match["lr"] else None


V1 = Study("v1", "A0-ablations-v1", "research/falsification/A0-ablations-v1",
           "model/configs/a0_ablation_study.toml", "a0-ablation-prereg-v1",
           "45b2d5491e40171dd242d0bac6cf7d78d9c70746", False)
V2 = Study("v2", "A0-ablations-v2", "research/falsification/A0-ablations-v2",
           "model/configs/a0_ablation_study_v2.toml", "a0-ablations-v2-prereg", None, True)
STUDIES = {study.key: study for study in (V1, V2)}


# --------------------------------------------------------------------------- registration

COMMAND_HEAD = ("cargo +1.95.0 run --release --locked --quiet --manifest-path model/burn-a0/Cargo.toml "
                "--example a0_ablation --")


def entrypoint_template(kind: str, family: str) -> str:
    """The frozen command of a v2 experiment. Everything but the seed is a
    placeholder the experiment's [preregistration] table fills; the table's
    values, not the command line, decide the run."""
    common = "--data <data> --data-fnv64 <data_fnv64> --d-model <d_model>"
    if kind == "sweep":
        return (f"{COMMAND_HEAD} --phase sweep --experiment {family} --arms <arms> --seed <seed> "
                f"--steps <steps> --lr <lr> {common}")
    return (f"{COMMAND_HEAD} --phase eval --experiment {family} --arms <arms> --seed <seed> "
            f"--steps <steps> --lr-file <lr_file> {common}")


@dataclasses.dataclass(frozen=True)
class Registration:
    """Everything a listed v2 experiment freezes."""

    exp_id: str
    path: str
    kind: str
    family: str
    seeds: list[int]
    table: dict
    required: dict
    entrypoint: str


def registration(study: Study, kind: str, family: str, *, arms: list[str], seeds: list[int], steps: int,
                 d_model: int, data: str, data_fnv64: str, lr: float | None = None,
                 lr_file: str | None = None, arm: str | None = None) -> Registration:
    """The registration of one v2 experiment (`kind` of `family`)."""
    if kind == "sweep":
        exp_id = study.sweep_id(family, lr)
    elif kind == "eval":
        exp_id = study.eval_id(family)
    elif kind == "rerun":
        exp_id = study.rerun_id()
    else:
        exp_id = study.contingency_id(family, arm)
    table = {"seeds": seeds, "steps": steps, "arms": ",".join(arms), "data": data,
             "data_fnv64": data_fnv64, "d_model": d_model}
    required = {"seeds": "int-list", "steps": "int", "arms": "str", "data": "str", "data_fnv64": "str",
                "d_model": "int"}
    if kind == "sweep":
        table["lr"] = str(lr)
        required["lr"] = "str"
    else:
        table["lr_file"] = lr_file
        required["lr_file"] = "file"
    return Registration(exp_id, study.experiment_path(exp_id), kind, family, seeds, table, required,
                        entrypoint_template(kind, family))


def manifest_text(study: Study, reg: Registration, table_digest: str, rules_digest: str,
                  hardware_profile: str) -> str:
    """The experiment's frozen manifest, status prepared (it is created frozen,
    as M001-v2 was)."""
    hypothesis, metrics = HYPOTHESES[reg.family]
    toml = experiment_records_toml
    lines = [
        "version = 1",
        f"id = {toml(reg.exp_id)}",
        'status = "prepared"',
        f"hypothesis = {toml(hypothesis + ' (' + study.name + ', ' + reg.kind + ')')}",
        f"metrics = {toml(metrics)}",
        f"baseline = {toml('the full A0 arm of the same study; no external baseline')}",
        f"falsification = {toml('Decided only by research/falsification/' + study.name + '/criteria.toml')}",
        f"hardware_profile = {toml(hardware_profile)}",
        f"entrypoint = {toml(reg.entrypoint)}",
        f"seeds = [{', '.join(str(s) for s in reg.seeds)}]",
        'results_dir = "results"',
        'required_artifacts = ["run.json", "metrics.json"]',
        f"preregistration_sha256 = {toml(table_digest)}",
        f"preregistration_rules_sha256 = {toml(rules_digest)}",
    ]
    return "\n".join(lines) + "\n"


def experiment_records_toml(value: str) -> str:
    """A TOML basic string."""
    return json.dumps(value, ensure_ascii=False)


def config_text(reg: Registration, file_digests: dict[str, str]) -> str:
    """The experiment's config.toml: the workspace-area header and the frozen
    [preregistration] table, with the SHA-256 of each `file` key."""
    toml = experiment_records_toml
    lines = ["version = 1", 'kind = "workspace-area"', f"name = {toml(reg_name(reg))}",
             f"path = {toml('experiments/' + reg.path)}", 'tests_dir = "tests"', "", "[preregistration]"]
    for key, value in reg.table.items():
        lines.append(f"{key} = " + (f"[{', '.join(str(v) for v in value)}]" if isinstance(value, list)
                                    else toml(value) if isinstance(value, str) else str(value)))
    for key, digest in file_digests.items():
        lines.append(f"{key}_sha256 = {toml(digest)}")
    return "\n".join(lines) + "\n"


def reg_name(reg: Registration) -> str:
    return reg.exp_id


def preregistration_entry(reg: Registration) -> str:
    """The experiment's entry in experiments/preregistration.toml."""
    toml = experiment_records_toml
    body = ", ".join(f"{key} = {toml(kind)}" for key, kind in reg.required.items())
    return f"[experiment.{reg.exp_id}]\nrequired = {{ {body} }}\n"


def freeze_digests(root: Path, reg: Registration) -> tuple[dict[str, str], str, str]:
    """(file digests, table digest, rules digest) of a registration; reads the
    `file` keys from `root`."""
    digests = {}
    table = dict(reg.table)
    for key, kind in reg.required.items():
        if kind == "file":
            digests[key] = experiment_records.preregistered_file_digest(root / table[key])
    frozen = dict(table)
    for key, digest in digests.items():
        frozen[f"{key}_sha256"] = digest
    return (digests, experiment_records.preregistration_digest(frozen),
            experiment_records.canonical_digest({"required": reg.required}))


def write_experiment(root: Path, study: Study, reg: Registration, hardware_profile: str) -> None:
    """Create the experiment's directory (manifest, config, README, tests and
    results placeholders) below `root`/experiments and register it in
    experiments/registry.toml and experiments/preregistration.toml. Refuses an
    id that is already registered or a directory that exists."""
    experiments = root / "experiments"
    target = experiments / reg.path
    registry = experiments / "registry.toml"
    prereg = experiments / "preregistration.toml"
    if target.exists():
        raise ValueError(f"{reg.path} already exists")
    if f'id = "{reg.exp_id}"' in registry.read_text(encoding="utf-8"):
        raise ValueError(f"{reg.exp_id} is already registered")
    digests, table_digest, rules_digest = freeze_digests(root, reg)
    target.mkdir(parents=True)
    (target / "experiment.toml").write_text(
        manifest_text(study, reg, table_digest, rules_digest, hardware_profile), encoding="utf-8")
    (target / "config.toml").write_text(config_text(reg, digests), encoding="utf-8")
    (target / "README.md").write_text(
        f"# {reg.exp_id}\n\nOne {reg.kind} experiment of the A0 mechanism ablation study "
        f"`{study.name}` (family {reg.family}). It is a listed experiment: its command, values and "
        f"seeds are frozen in `config.toml`, and each seed runs once through the hardened runner. "
        f"Design: `{study.directory}/DESIGN.md`. Its records are A0-internal evidence only.\n",
        encoding="utf-8")
    (target / "tests").mkdir()
    (target / "tests" / "README.md").write_text(
        f"# Tests for experiments/{reg.path}\n\nThis experiment's checks are the study's gate tests "
        f"(`scripts/tests/test_a0_study.py`, `scripts/tests/test_a0_ablation_aggregate.py`).\n",
        encoding="utf-8")
    (target / "results").mkdir()
    (target / "results" / ".gitkeep").write_text("", encoding="utf-8")
    with registry.open("a", encoding="utf-8") as handle:
        handle.write(f'\n[[experiment]]\nid = "{reg.exp_id}"\npath = "{reg.path}"\nstatus = "prepared"\n')
    with prereg.open("a", encoding="utf-8") as handle:
        handle.write("\n" + preregistration_entry(reg))


# --------------------------------------------------------------------------- gate G6


def python_check_tails(study: Study) -> dict[str, list[str]]:
    """The arguments (after the interpreter) of G6's Python suites. v2 also runs
    the suites of the v2 mechanism itself, so its evidence covers the code that
    registers, runs and aggregates it."""
    aggregator = ["-m", "unittest", "scripts/tests/test_a0_ablation_aggregate.py"]
    if study.hardened:
        aggregator += ["scripts/tests/test_a0_study.py", "scripts/tests/test_run_a0_ablation_study.py"]
    return {
        "benchmark_tests": ["-m", "unittest", "discover", "-s", "benchmarks/operator-routing/tests"],
        "aggregator_tests": aggregator,
        "config_tests": ["-m", "unittest", "scripts/tests/test_a0_ablation_config.py"],
    }
