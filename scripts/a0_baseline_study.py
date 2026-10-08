"""Registration of the A0 baseline study v1 (research/falsification/A0-baseline-v1).

The study has three confirmatory experiments (M001-v5, M002-v10, M003-v2), one
validation-only learning-rate sweep experiment per study (`<id>-sweep`) and the
deferred M004-v2. Like the A0 ablation study v2 (`scripts/a0_study.py`), every
process of it runs through the hardened runner (`scripts/run_experiment.py`) as
a *listed* experiment (`experiments/preregistration.toml`): one frozen
`entrypoint` per experiment whose placeholders the `[preregistration]` table of
its `config.toml` fills, one run per seed.

This module is the single source of the registration. `criteria.toml` of the
study owns every protocol value and threshold; the tables below are derived from
it, never typed twice. It

- renders each experiment (manifest, configuration, README, registry and
  preregistration entries) with status `planned` (`render`),
- extracts the committed sweep records into the row file the runner's select
  phase reads (`extract_sweep`), and
- freezes: computes every digest from the repository and moves an experiment to
  `prepared` (`freeze`). The freeze is a separate, explicit step of the head
  (see scratchpad `m-freeze-plan.md`); nothing here runs on import.

Values that can only be known at the freeze stay `must-be-pinned-before-prepared`
in a planned configuration (`pending_keys`): the digest of a learning-rate file
before its sweep ran, and the digests that depend on the final `model/burn-a0`
tree.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_research_gates as gates  # noqa: E402
import experiment_records  # noqa: E402

STUDY_DIR = "research/falsification/A0-baseline-v1"
CRITERIA = f"{STUDY_DIR}/criteria.toml"
PREREGISTRATION_TEXT = f"{STUDY_DIR}/PREREGISTRATION.md"
DECISION_SCRIPT = "scripts/aggregate_a0_baseline.py"
HARDWARE_PROFILE = "hardware/a0-cpu-4core.toml"
PLAIN_BASELINE = "research/baselines/plain_model/config.toml"
IMPLEMENTATION_TREE = "model/burn-a0"
RUNNER_SOURCES = {
    "runner_arm_table": "model/burn-a0/examples/a0_ablation/arms.rs",
    "runner_main": "model/burn-a0/examples/a0_ablation/main.rs",
    "runner_train": "model/burn-a0/examples/a0_ablation/train.rs",
    "runner_baseline": "model/burn-a0/examples/a0_ablation/baseline.rs",
}
SEEDS = [17, 29, 43, 71, 101]
SWEEP_SEEDS = [17]
PENDING = "must-be-pinned-before-prepared"
COMMAND_HEAD = (
    "cargo +1.95.0 run --release --locked --quiet --manifest-path model/burn-a0/Cargo.toml "
    "--example a0_ablation --"
)
# Flags every training command passes through its placeholders, so the runner
# never falls back to a default of its own (`m002_versioned_runner_binding_errors`
# has the same rule for M002-v9).
COMMON_FLAGS = (
    "--study <study> --arms <arms> --seed <seed> --steps <steps> --data <data> --folds <folds> "
    "--d-model <d_model> --rank <rank> --bias-limit <bias_limit> --metadata-dropout <metadata_dropout>"
)
SWEEP_COMMAND = f"{COMMAND_HEAD} --phase baseline-v1-sweep {COMMON_FLAGS} --out -"
CONFIRMATORY_COMMAND = f"{COMMAND_HEAD} --phase baseline-v1 {COMMON_FLAGS} --lr-file <lr_file> --out -"
BASELINE_ENTRY = (
    'name = "plain_model", path = "research/baselines/plain_model/config.toml", keys = '
    '["model.backend", "model.model", "model.implementation", "model.architecture", "model.d_model", '
    '"model.latent_slots", "model.sequence_length", "model.batch_size", "model.optimizer"], '
    'status_key = "status"'
)

# Per study: directory slug, hypothesis, primary contrast wording, extra metrics.
TEXTS = {
    "M001-v5": {
        "slug": "semantic-slots",
        "hypothesis": (
            "Do typed semantic slots improve out-of-distribution routing accuracy by at least 0.02 over "
            "index-tagged raw binding (no-semantic-slots-masked) in A0 on operator-routing v2, without a "
            "per-fold OOD calibration regression? The full A0 bundle against the pinned plain baseline is "
            "decided separately and labelled bundle vs baseline, non-isolating."
        ),
        "baseline": (
            "no-semantic-slots-masked (index-tagged raw binding; isolating contrast); the pinned plain "
            "baseline research/baselines/plain_model (ptr-burn-a0@matched-cross-attention-v1) for the "
            "bundle contrast"
        ),
        "extra_metrics": "mask share (masked minus unmasked accuracy, diagnostic arm no-semantic-slots)",
    },
    "M002-v10": {
        "slug": "typed-attention",
        "hypothesis": (
            "Does factorized typed pair attention improve out-of-distribution routing accuracy by at least "
            "0.02 over the identical computation with the pair bias multiplied by zero (factorized-v2-off) "
            "in A0 on operator-routing v2, without a per-fold OOD calibration regression? The A0-ablations-v2 "
            "study (d_model 32) found no necessity of typed attention, so a null is plausible. The full A0 "
            "bundle against the pinned plain baseline is labelled bundle vs baseline, non-isolating."
        ),
        "baseline": (
            "factorized-v2-off (identical compute, pair bias zeroed; isolating contrast); the pinned plain "
            "baseline research/baselines/plain_model (ptr-burn-a0@matched-cross-attention-v1) for the "
            "bundle contrast"
        ),
        "extra_metrics": "equal total_params and flops_per_example of the factorized pair",
    },
    "M003-v2": {
        "slug": "latent-recurrence",
        "hypothesis": (
            "Does the tied latent recurrence (k=2) improve out-of-distribution routing accuracy by at least "
            "0.02 over no recurrence (latent-0) at equal parameters in A0 on operator-routing v2, without a "
            "per-fold OOD calibration regression, and does the per-slot nonlinearity (full against "
            "latent-linear at matched parameters and FLOPs) carry the benefit? Token savings are not "
            "tested. The full A0 bundle against the pinned plain baseline is labelled bundle vs baseline, "
            "non-isolating."
        ),
        "baseline": (
            "latent-0 (no recurrence; isolating primary contrast) and latent-linear (affine-only "
            "recurrence; isolating secondary contrast); latent-1 is descriptive; the pinned plain baseline "
            "research/baselines/plain_model (ptr-burn-a0@matched-cross-attention-v1) for the bundle contrast"
        ),
        "extra_metrics": "latent_steps, explicit_token_count (non-PAD input tokens, identical across arms)",
    },
}
FALSIFICATION = (
    "Decided only by research/falsification/A0-baseline-v1/criteria.toml and PREREGISTRATION.md. SUPPORTS "
    "needs every gate: OOD accuracy delta >= 0.02 with the paired 95% interval lower bound > 0 and positive "
    "in every fold; per fold on test_ood and raw logits the NLL delta upper bound <= 0 and the ECE15 mean "
    "delta <= 0 with upper bound <= 0.02; IID accuracy lower bound >= -0.01; coverage and selective-error "
    "gates; matched compute. Accuracy up with any calibration gate failed is NO-GO-CALIBRATION (INCONCLUSIVE; "
    "no quality claim). FALSIFIES needs the OOD accuracy interval upper bound below 0.02 with both arms "
    "learnable and compute matched. A missing seed, fold, arm or digest is a refusal, not a verdict."
)
SWEEP_FALSIFICATION = (
    "Not a hypothesis test: the select phase picks one rate per arm by the rule fixed before the sweep (best "
    "mean validation accuracy over folds, tie lower mean validation NLL, tie smaller lr, a diverged grid point "
    "ineligible). An incomplete or inconsistent sweep is refused by the select phase, not patched; a "
    "changed sweep is a successor experiment."
)


class RegistrationError(ValueError):
    """A registration, render or freeze that cannot proceed."""


@dataclass(frozen=True)
class Registration:
    """One experiment of the study: its directory, manifest, configuration
    table, preregistration entry and registry status."""

    exp_id: str
    path: str  # below experiments/
    kind: str  # "confirmatory", "sweep" or "deferred"
    study_id: str
    manifest: dict
    table: dict
    required: dict
    pending: tuple[str, ...]

    @property
    def listed(self) -> bool:
        return self.kind != "deferred"


def load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007F")


def toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return toml_string(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    raise RegistrationError(f"no TOML form for {value!r}")


def grid_text(grid: list[float]) -> str:
    return ",".join(str(value) for value in grid)


def lr_file_path(study_id: str) -> str:
    return f"{STUDY_DIR}/lr_{study_id}.tsv"


def file_digest(root: Path, relative: str) -> str:
    path = root / relative
    if not path.is_file():
        raise RegistrationError(f"{relative} is not a file below {root}")
    return experiment_records.preregistered_file_digest(path)


def baseline_digest(root: Path) -> str:
    problems, digest = gates.directory_digest(root, gates.baseline_directory(PLAIN_BASELINE))
    if problems or digest is None:
        raise RegistrationError(f"plain baseline directory cannot be frozen: {problems}")
    return digest


def tree_digest(root: Path) -> str:
    digest = gates.directory_digest_at(root, "HEAD", IMPLEMENTATION_TREE)
    if digest is None:
        raise RegistrationError(f"{IMPLEMENTATION_TREE} at HEAD cannot be frozen (not a plain tree of regular files)")
    return digest


def registrations(criteria: dict) -> list[Registration]:
    """Every experiment of the study, derived from `criteria` (the parsed
    criteria.toml)."""
    protocol = criteria["protocol"]
    data = criteria["data"]
    thresholds = criteria["thresholds"]
    bias = protocol["bias_limit"]
    if float(bias) != int(bias):
        raise RegistrationError("bias_limit must be integral to be an int preregistration key")
    shared = {
        "steps": protocol["steps"],
        "batch": protocol["batch_size"],
        "d_model": protocol["d_model"],
        "rank": protocol["rank"],
        "bias_limit": int(bias),
        "metadata_dropout": f"{protocol['metadata_dropout']:g}",
        "router_mode": protocol["router_mode"],
        "data": data["dataset"],
        "folds": data["folds"],
        "dataset_lock": data["lock"],
    }
    file_keys = {
        "criteria": CRITERIA,
        "preregistration_text": PREREGISTRATION_TEXT,
        "decision_script": DECISION_SCRIPT,
        "decision_core": criteria["decision"]["core"],
        **RUNNER_SOURCES,
    }
    out = []
    for study in criteria["study"]:
        study_id = study["id"]
        text = TEXTS[study_id]
        arms = ",".join(study["arms"])
        sweep_id = f"{study_id}-sweep"
        primary = next(c for c in study["contrast"] if c["role"] == "primary")
        pins = {
            "study": study["runner_study"],
            "arms": arms,
            **shared,
            "lr_grid": grid_text(protocol["lr_grid"]),
            "lr_selection_rule": protocol["lr_selection_rule"],
            "sweep_seed": protocol["sweep_seed"],
        }
        common_required = {
            "seeds": "int-list", "study": "str", "arms": "str", "steps": "int", "batch": "int",
            "d_model": "int", "rank": "int", "bias_limit": "int", "metadata_dropout": "str",
            "router_mode": "str", "data": "str", "folds": "str", "dataset_lock": "file",
            "lr_grid": "str", "lr_selection_rule": "str", "sweep_seed": "int",
            "criteria": "file", "preregistration_text": "file", "decision_script": "file",
            "decision_core": "file", "runner_arm_table": "file", "runner_main": "file",
            "runner_train": "file", "runner_baseline": "file", "implementation_tree": "str",
            "implementation_tree_git_digest": "str",
        }
        tail_pins = {
            **file_keys,
            "implementation_tree": IMPLEMENTATION_TREE,
            "implementation_tree_git_digest": PENDING,
        }
        pending_runner = ("runner_main_sha256", "runner_baseline_sha256", "implementation_tree_git_digest")
        # Sweep: validation only, seed 17, one record per study.
        sweep_table = {
            "seeds": SWEEP_SEEDS, **pins, "lr_file_target": lr_file_path(study_id), **tail_pins,
        }
        sweep_required = {**common_required, "lr_file_target": "str"}
        out.append(Registration(
            exp_id=sweep_id,
            path=f"model/{sweep_id}",
            kind="sweep",
            study_id=study_id,
            manifest={
                "hypothesis": (
                    f"Which learning rate of the fixed grid {grid_text(protocol['lr_grid'])} trains each arm of "
                    f"{study_id}, the plain baseline included, best on validation data (seed "
                    f"{protocol['sweep_seed']}, validation only)? Sweep of the A0 baseline study v1."
                ),
                "metrics": "validation accuracy, validation NLL and divergence per fold, arm and learning rate",
                "baseline": "no external baseline: the plain baseline is swept as an arm with its own rate",
                "falsification": SWEEP_FALSIFICATION,
                "entrypoint": SWEEP_COMMAND,
                "seeds": SWEEP_SEEDS,
            },
            table=sweep_table,
            required=sweep_required,
            pending=pending_runner,
        ))
        confirmatory_table = {
            "seeds": SEEDS, **pins,
            "lr_file": lr_file_path(study_id),
            "sweep_experiment": sweep_id,
            "primary_contrast": primary["id"],
            "delta_min": f"{thresholds['delta_min']:g}",
            "ood_nll_ci_upper_le": f"{thresholds['ood_nll_ci_upper_le']:g}",
            "ood_ece_mean_le": f"{thresholds['ood_ece_mean_le']:g}",
            "ood_ece_ci_upper_le": f"{thresholds['ood_ece_ci_upper_le']:g}",
            "iid_accuracy_ci_lower_ge": f"{thresholds['iid_accuracy_ci_lower_ge']:g}",
            "plain_param_tolerance": f"{thresholds['plain_param_tolerance']:g}",
            **tail_pins,
        }
        confirmatory_required = {
            **common_required,
            "lr_file": "file", "sweep_experiment": "str", "primary_contrast": "str", "delta_min": "str",
            "ood_nll_ci_upper_le": "str", "ood_ece_mean_le": "str", "ood_ece_ci_upper_le": "str",
            "iid_accuracy_ci_lower_ge": "str", "plain_param_tolerance": "str",
        }
        out.append(Registration(
            exp_id=study_id,
            path=f"model/{study_id}-{text['slug']}",
            kind="confirmatory",
            study_id=study_id,
            manifest={
                "hypothesis": text["hypothesis"],
                "metrics": (
                    "OOD accuracy (sole primary endpoint: mean over three folds, paired over five seeds, "
                    "delta_min 0.02); gated per fold on test_ood with raw logits: NLL, ECE15; IID accuracy; "
                    "raw coverage and selective error; matched parameters; descriptive: total_params, "
                    "flops_per_example, latent_steps, wall time, explicit_token_count, temperature-fitted "
                    "diagnostics; " + text["extra_metrics"]
                ),
                "baseline": text["baseline"],
                "falsification": FALSIFICATION,
                "entrypoint": CONFIRMATORY_COMMAND,
                "seeds": SEEDS,
            },
            table=confirmatory_table,
            required=confirmatory_required,
            pending=("lr_file_sha256", *pending_runner),
        ))
    out.append(Registration(
        exp_id="M004-v2",
        path="model/M004-v2-operator-router",
        kind="deferred",
        study_id="M004-v2",
        manifest={
            "hypothesis": (
                "Does a learned operator router beat the frozen router and the pinned baseline? Planned and "
                "deferred: not run at A0 (ADR-0021 item 2); the comparison needs the pretrained LM baseline "
                "of the GPU phase."
            ),
            "metrics": "regret; quality-cost utility; route accuracy (to be preregistered with the GPU-phase study)",
            "baseline": "LLM-only + deterministic router (to be pinned in the GPU phase)",
            "falsification": (
                "Not run at A0 (ADR-0021 item 2). No criterion is preregistered here and none may be inferred "
                "from M001-v5, M002-v10 or M003-v2; a successor is preregistered before any run."
            ),
            "entrypoint": "",
            "seeds": SEEDS,
        },
        table={},
        required={},
        pending=(),
    ))
    return out


# ---------------------------------------------------------------- rendering


def pinned_table(reg: Registration, root: Path, pending: bool) -> dict:
    """The `[preregistration]` table of `reg`: every `file` key followed by its
    `<key>_sha256` (computed from `root`; with `pending`, the placeholder for
    the digests `reg.pending` names), the tree digest of `model/burn-a0` and the
    plain baseline's digest last."""
    table = {}
    for key, value in reg.table.items():
        if key == "implementation_tree_git_digest":
            table[key] = PENDING if pending else tree_digest(root)
            continue
        table[key] = value
        if reg.required.get(key) == "file":
            name = f"{key}_sha256"
            table[name] = PENDING if pending and name in reg.pending else file_digest(root, value)
    table["baseline_plain_model_sha256"] = baseline_digest(root)
    return table


def manifest_text(reg: Registration, extra: dict | None = None) -> str:
    lines = [
        "version = 1",
        f"id = {toml_string(reg.exp_id)}",
        'status = "planned"',
        f"hypothesis = {toml_string(reg.manifest['hypothesis'])}",
        f"metrics = {toml_string(reg.manifest['metrics'])}",
        f"baseline = {toml_string(reg.manifest['baseline'])}",
        f"falsification = {toml_string(reg.manifest['falsification'])}",
        f"hardware_profile = {toml_string(HARDWARE_PROFILE)}",
        f"entrypoint = {toml_string(reg.manifest['entrypoint'])}",
        f"seeds = {toml_value(reg.manifest['seeds'])}",
        'results_dir = "results"',
        'required_artifacts = ["run.json", "metrics.json"]',
    ]
    for key, value in (extra or {}).items():
        lines.append(f"{key} = {toml_string(value)}")
    return "\n".join(lines) + "\n"


def config_text(reg: Registration, table: dict) -> str:
    lines = [
        "version = 1",
        'kind = "workspace-area"',
        f"name = {toml_string(Path(reg.path).name)}",
        f"path = {toml_string('experiments/' + reg.path)}",
        'tests_dir = "tests"',
    ]
    if table:
        lines += [
            "",
            "# Frozen at the freeze commit (scripts/a0_baseline_study.py); every value is derived from",
            f"# {CRITERIA}. A `must-be-pinned-before-prepared` value can only be known at the freeze.",
            "[preregistration]",
        ]
        lines += [f"{key} = {toml_value(value)}" for key, value in table.items()]
    return "\n".join(lines) + "\n"


def readme_text(reg: Registration) -> str:
    if reg.kind == "deferred":
        return (
            f"# {reg.exp_id}\n\nPlanned and deferred: **not run at A0 (ADR-0021 item 2)**. The learned "
            "operator router needs the pretrained LM baseline of the GPU phase. It is registered only so that "
            "the successor id of M004 is reserved by the A0 baseline study v1 "
            f"(`{STUDY_DIR}/`). It has no entrypoint, no preregistered criteria and no results; "
            "M001-v5, M002-v10 and M003-v2 say nothing about it.\n"
        )
    what = (
        "One learning-rate sweep (validation only, seed 17) of study "
        f"`{reg.study_id}`" if reg.kind == "sweep" else
        f"The confirmatory experiment of study `{reg.study_id}` (five seeds, three folds)"
    )
    return (
        f"# {reg.exp_id}\n\n{what} of the A0 baseline study v1. It is a listed experiment: its command, "
        "values and seeds are frozen in `config.toml` at the freeze commit, and each seed runs once through "
        f"the hardened runner. Design and frozen text: `{STUDY_DIR}/`. Its records are A0-internal, "
        "synthetic-data evidence only (ADR-0021).\n\nStatus `planned` until the freeze: placeholders "
        f"(`{PENDING}`) in `config.toml` are the values only known then.\n"
    )


def preregistration_entry(reg: Registration) -> str:
    body = ", ".join(f"{key} = {toml_string(kind)}" for key, kind in reg.required.items())
    return (
        f"[experiment.{reg.exp_id}]\nrequired = {{ {body} }}\n\n"
        f"[[experiment.{reg.exp_id}.baseline]]\n"
        'name = "plain_model"\npath = "research/baselines/plain_model/config.toml"\n'
        'keys = ["model.backend", "model.model", "model.implementation", "model.architecture", '
        '"model.d_model", "model.latent_slots", "model.sequence_length", "model.batch_size", '
        '"model.optimizer"]\nstatus_key = "status"\n'
    )


def render_files(reg: Registration, root: Path) -> dict[str, str]:
    """The files of `reg`'s directory (relative to the directory) at status
    planned."""
    table = pinned_table(reg, root, pending=True) if reg.listed else {}
    return {
        "experiment.toml": manifest_text(reg),
        "config.toml": config_text(reg, table),
        "README.md": readme_text(reg),
        "tests/README.md": (
            f"# Tests for experiments/{reg.path}\n\nThe checks of this experiment are the study's "
            "registration and gate tests (`scripts/tests/test_a0_baseline_study.py`), the decider's tests "
            "(`scripts/tests/test_aggregate_a0_baseline.py`) and the runner's tests "
            "(`model/burn-a0/tests/baseline_v1.rs`).\n"
        ),
        "results/.gitkeep": "",
    }


def render(root: Path = ROOT) -> list[str]:
    """Create every experiment of the study below `root`, register it
    (planned) in registry.toml and list it in preregistration.toml. Refuses an
    id or directory that exists; nothing is written then."""
    criteria = load_toml(root / CRITERIA)
    regs = registrations(criteria)
    registry = root / "experiments/registry.toml"
    prereg = root / "experiments/preregistration.toml"
    known = {item["id"] for item in load_toml(registry).get("experiment", [])}
    for reg in regs:
        if reg.exp_id in known:
            raise RegistrationError(f"{reg.exp_id} is already registered")
        if (root / "experiments" / reg.path).exists():
            raise RegistrationError(f"experiments/{reg.path} already exists")
    rendered = {reg.exp_id: render_files(reg, root) for reg in regs}
    for reg in regs:
        target = root / "experiments" / reg.path
        for name, text in rendered[reg.exp_id].items():
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
    with registry.open("a", encoding="utf-8", newline="\n") as handle:
        for reg in regs:
            handle.write(f'\n[[experiment]]\nid = "{reg.exp_id}"\npath = "{reg.path}"\nstatus = "planned"\n')
    with prereg.open("a", encoding="utf-8", newline="\n") as handle:
        for reg in regs:
            if reg.listed:
                handle.write("\n" + preregistration_entry(reg))
    return [reg.exp_id for reg in regs]


# ---------------------------------------------------------------- freezing


def set_registry_status(root: Path, exp_id: str, status: str) -> None:
    path = root / "experiments/registry.toml"
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(rf'(\[\[experiment\]\]\nid = "{re.escape(exp_id)}"\npath = "[^"]*"\nstatus = )"[a-z]+"')
    new, count = pattern.subn(rf'\1"{status}"', text)
    if count != 1:
        raise RegistrationError(f"{exp_id} has no unique entry in experiments/registry.toml")
    path.write_text(new, encoding="utf-8", newline="\n")


def freeze(root: Path, exp_id: str, *, write: bool) -> dict:
    """Compute the frozen form of the listed experiment `exp_id`: the fully
    pinned configuration, the manifest at status `prepared` with the digests of
    its table and of its preregistration entry. With `write`, replace the
    experiment's `config.toml` and `experiment.toml` and its registry status;
    without, only return what would be written. Refuses a deferred
    experiment, one that is not `planned`, a missing learning-rate file and a
    HEAD whose `model/burn-a0` is not the committed tree to pin."""
    criteria = load_toml(root / CRITERIA)
    reg = next((r for r in registrations(criteria) if r.exp_id == exp_id), None)
    if reg is None or not reg.listed:
        raise RegistrationError(f"{exp_id} is not a listed experiment of the study")
    directory = root / "experiments" / reg.path
    manifest = load_toml(directory / "experiment.toml")
    if manifest.get("status") != "planned":
        raise RegistrationError(f"{exp_id} is {manifest.get('status')!r}; only a planned experiment freezes")
    if reg.kind == "confirmatory" and not (root / reg.table["lr_file"]).is_file():
        raise RegistrationError(f"{reg.table['lr_file']} does not exist: run and commit the sweep first")
    table = pinned_table(reg, root, pending=False)
    entry = load_toml(root / "experiments/preregistration.toml")["experiment"][exp_id]
    digest = experiment_records.preregistration_digest(table)
    rules = experiment_records.canonical_digest(entry)
    text = manifest_text(reg, {"preregistration_sha256": digest, "preregistration_rules_sha256": rules}).replace(
        'status = "planned"', 'status = "prepared"', 1
    )
    result = {"exp_id": exp_id, "table": table, "preregistration_sha256": digest, "rules_sha256": rules}
    if write:
        (directory / "config.toml").write_text(config_text(reg, table), encoding="utf-8", newline="\n")
        (directory / "experiment.toml").write_text(text, encoding="utf-8", newline="\n")
        set_registry_status(root, exp_id, "prepared")
    return result


# ---------------------------------------------------------------- sweep rows


def record_rows(record: dict, name: str) -> list[str]:
    stdout = record.get("stdout")
    if not isinstance(stdout, str):
        raise RegistrationError(f"{name}: a run record without stdout")
    return [line for line in stdout.splitlines() if line.strip()]


def extract_sweep(records: list[Path], out: Path) -> int:
    """Write the stdout rows of the committed sweep run records `records` to
    `out` (created exclusively), the file the runner's select phase reads
    (`--sweep`). Refuses a record that did not complete with exit code 0, a
    record that holds no sweep row and a record of another experiment kind than
    a sweep. Returns the row count."""
    lines = []
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "completed" or record.get("exit_code") != 0:
            raise RegistrationError(f"{path}: not a completed run with exit code 0")
        if not str(record.get("experiment_id", "")).endswith("-sweep"):
            raise RegistrationError(f"{path}: not a record of a sweep experiment")
        rows = record_rows(record, str(path))
        if not any('"row":"sweep-baseline-v1"' in row.replace(" ", "") for row in rows):
            raise RegistrationError(f"{path}: holds no sweep-baseline-v1 row")
        lines.extend(rows)
    with out.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    return len(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("render", help="create the planned experiments (refuses existing ones)")
    plan = sub.add_parser("freeze", help="pin and move one planned experiment to prepared")
    plan.add_argument("exp_id")
    plan.add_argument("--write", action="store_true", help="write the files (default: print the frozen digests)")
    extract = sub.add_parser("extract-sweep", help="stdout rows of committed sweep records for the select phase")
    extract.add_argument("records", nargs="+", type=Path)
    extract.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "render":
            for exp_id in render():
                print(exp_id)
        elif args.command == "freeze":
            result = freeze(ROOT, args.exp_id, write=args.write)
            print(json.dumps({k: v for k, v in result.items() if k != "table"}, indent=2))
            print(json.dumps(result["table"], indent=2))
        else:
            print(extract_sweep(args.records, args.out))
    except (RegistrationError, OSError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
