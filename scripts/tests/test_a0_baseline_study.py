"""Registration of the A0 baseline study v1: the planned experiments are what
`scripts/a0_baseline_study.py` renders from criteria.toml, they are bound to
it by the research gate once frozen, and the freeze of a fixture copy passes
the repository's own gate."""

import contextlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


study = load_module("a0_baseline_study_under_test", ROOT / "scripts/a0_baseline_study.py")
gates = study.gates
CRITERIA = tomllib.loads((ROOT / study.CRITERIA).read_text(encoding="utf-8"))
REGS = {reg.exp_id: reg for reg in study.registrations(CRITERIA)}
LISTED = [exp_id for exp_id, reg in REGS.items() if reg.listed]
CONFIRMATORY = ["M001-v5", "M002-v10", "M003-v2"]
# The sweeps that stand: each study's current one. A superseded sweep keeps its
# frozen files and records as history and is no longer rendered or frozen.
CURRENT_SWEEPS = [study.current_sweep(exp_id) for exp_id in CONFIRMATORY]
SUPERSEDED = [exp_id for exp_id in REGS if study.superseded_by(exp_id) is not None]


def toml_at(root, reg, name):
    return tomllib.loads((root / "experiments" / reg.path / name).read_text(encoding="utf-8"))


@contextlib.contextmanager
def mock_successors(table):
    saved = dict(study.SWEEP_SUCCESSORS)
    study.SWEEP_SUCCESSORS.clear()
    study.SWEEP_SUCCESSORS.update(table)
    try:
        yield
    finally:
        study.SWEEP_SUCCESSORS.clear()
        study.SWEEP_SUCCESSORS.update(saved)


def registry_status(root, exp_id):
    items = tomllib.loads((root / "experiments/registry.toml").read_text(encoding="utf-8"))["experiment"]
    return next(item["status"] for item in items if item["id"] == exp_id)


def git(root, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=root, check=True, capture_output=True,
    )


class PlannedRegistrationTests(unittest.TestCase):
    """The committed experiments are what the generator renders for their
    status: planned, or (after the head's freeze) prepared, or superseded."""

    def test_the_registered_ids_and_statuses(self):
        self.assertEqual(
            sorted(REGS),
            sorted(["M001-v5", "M002-v10", "M003-v2", "M001-v5-sweep", "M002-v10-sweep", "M002-v10-sweep-r2",
                    "M003-v2-sweep", "M004-v2"]),
        )
        registry = {item["id"]: item for item in tomllib.loads((ROOT / "experiments/registry.toml").read_text())["experiment"]}
        for exp_id, reg in REGS.items():
            status = toml_at(ROOT, reg, "experiment.toml")["status"]
            with self.subTest(exp_id=exp_id):
                self.assertEqual(registry[exp_id], {"id": exp_id, "path": reg.path, "status": status})
                self.assertIn(status, {"planned", "prepared", "superseded"})
                self.assertEqual(status == "superseded", exp_id in SUPERSEDED)

    def test_committed_files_are_what_the_generator_renders(self):
        # A criteria.toml, decision script or runner source that changed since the
        # registration makes its pinned digest stale: re-render before the freeze.
        for exp_id, reg in REGS.items():
            status = toml_at(ROOT, reg, "experiment.toml")["status"]
            if status == "superseded":
                continue
            files = study.render_files(reg, ROOT)
            if status == "prepared":
                result, manifest = study.frozen_form(ROOT, reg)
                files["config.toml"] = study.config_text(reg, result["table"])
                files["experiment.toml"] = manifest
            for name, text in files.items():
                with self.subTest(exp_id=exp_id, file=name):
                    self.assertEqual((ROOT / "experiments" / reg.path / name).read_text(encoding="utf-8"), text)

    def test_the_listing_matches_the_generator_and_m004_is_not_listed(self):
        listed = tomllib.loads((ROOT / "experiments/preregistration.toml").read_text(encoding="utf-8"))["experiment"]
        text = (ROOT / "experiments/preregistration.toml").read_text(encoding="utf-8")
        for exp_id in LISTED:
            with self.subTest(exp_id=exp_id):
                self.assertEqual(listed[exp_id]["required"], REGS[exp_id].required)
                self.assertIn(study.preregistration_entry(REGS[exp_id]), text)
                self.assertEqual([b["name"] for b in listed[exp_id]["baseline"]], ["plain_model"])
                self.assertEqual(listed[exp_id]["baseline"][0]["path"], "research/baselines/plain_model/config.toml")
        self.assertNotIn("M004-v2", listed)
        deferred = toml_at(ROOT, REGS["M004-v2"], "experiment.toml")
        self.assertEqual(deferred["entrypoint"], "")
        self.assertIn("not run at A0 (ADR-0021 item 2)", deferred["hypothesis"])
        self.assertIn("not run at a0 (adr-0021 item 2)", deferred["falsification"].lower())

    def test_only_values_known_at_the_freeze_are_placeholders(self):
        for exp_id in LISTED:
            if toml_at(ROOT, REGS[exp_id], "experiment.toml")["status"] != "planned":
                continue
            table = toml_at(ROOT, REGS[exp_id], "config.toml")["preregistration"]
            placeholders = sorted(k for k, v in table.items() if v == study.PENDING)
            expected = ["implementation_tree_git_digest", "runner_baseline_sha256", "runner_main_sha256"]
            if exp_id in CONFIRMATORY:
                expected.append("lr_file_sha256")
            with self.subTest(exp_id=exp_id):
                self.assertEqual(placeholders, sorted(expected))
                self.assertEqual(sorted(REGS[exp_id].pending), sorted(expected))

    def test_every_table_value_is_the_criteria_value(self):
        protocol, data = CRITERIA["protocol"], CRITERIA["data"]
        lock = json.loads((ROOT / data["lock"]).read_text(encoding="utf-8"))
        folds = ",".join(f"{f}={lock['folds'][f]['data_fnv1a64']}" for f in gates.M002_V6_FOLDS)
        self.assertEqual(data["folds"], folds)
        plain = study.baseline_digest(ROOT)
        for exp_id in LISTED:
            table = toml_at(ROOT, REGS[exp_id], "config.toml")["preregistration"]
            with self.subTest(exp_id=exp_id):
                self.assertEqual(table["folds"], folds)
                self.assertEqual((table["steps"], table["batch"], table["d_model"], table["rank"]),
                                 (protocol["steps"], protocol["batch_size"], protocol["d_model"], protocol["rank"]))
                self.assertEqual(float(table["bias_limit"]), protocol["bias_limit"])
                self.assertEqual(float(table["metadata_dropout"]), protocol["metadata_dropout"])
                self.assertEqual(table["baseline_plain_model_sha256"], plain)
                self.assertEqual(table["dataset_lock_sha256"], data["lock_sha256"])
                self.assertEqual(table["decision_core_sha256"], CRITERIA["decision"]["core_sha256"])
                self.assertEqual(table["seeds"], CRITERIA["seeds"] if REGS[exp_id].kind == "confirmatory" else [17])
                self.assertEqual(table["arms"].split(","), next(s for s in CRITERIA["study"] if s["id"] == REGS[exp_id].study_id)["arms"])
                self.assertIn("plain", table["arms"].split(","))

    def test_the_commands_are_ones_the_runner_can_build_from_the_table(self):
        for exp_id in LISTED:
            manifest = toml_at(ROOT, REGS[exp_id], "experiment.toml")
            table = toml_at(ROOT, REGS[exp_id], "config.toml")["preregistration"]
            with self.subTest(exp_id=exp_id):
                self.assertEqual(gates.command_errors(exp_id, manifest, table), [])
                self.assertEqual(manifest["seeds"], table["seeds"])
                self.assertEqual(manifest["hardware_profile"], "hardware/a0-cpu-4core.toml")
                self.assertIn("--out -", manifest["entrypoint"])
                self.assertEqual("--lr-file" in manifest["entrypoint"], exp_id in CONFIRMATORY)

    def test_the_binding_is_green_for_every_experiment_as_committed(self):
        # Inactive while planned or superseded; a prepared one is held to criteria.toml.
        for exp_id in LISTED:
            reg = REGS[exp_id]
            manifest = toml_at(ROOT, reg, "experiment.toml")
            self.assertEqual(gates.a0_baseline_v1_freeze_errors(ROOT, ROOT / "experiments" / reg.path, manifest, exp_id), [])

    def test_the_interrupted_sweep_is_kept_and_superseded_by_an_identical_successor(self):
        self.assertEqual(SUPERSEDED, ["M002-v10-sweep"])
        old, new = REGS["M002-v10-sweep"], REGS["M002-v10-sweep-r2"]
        self.assertEqual(study.superseded_by("M002-v10-sweep"), "M002-v10-sweep-r2")
        self.assertEqual(toml_at(ROOT, old, "experiment.toml")["status"], "superseded")
        self.assertTrue((ROOT / "experiments" / old.path / "SUPERSEDED.md").is_file())
        # The interrupted record is evidence as the runner wrote it: started, no output, no exit code.
        records = sorted((ROOT / "experiments" / old.path / "results").glob("run-*.json"))
        self.assertEqual(len(records), 1)
        record = json.loads(records[0].read_text(encoding="utf-8"))
        self.assertEqual((record["status"], record["seed"], record["experiment_id"]), ("started", 17, "M002-v10-sweep"))
        self.assertNotIn("stdout", record)
        self.assertNotIn("exit_code", record)
        # Identical experiment under a new id: every pinned value equal, the predecessor named.
        self.assertEqual({k: v for k, v in new.table.items() if k != "supersedes"}, old.table)
        self.assertEqual(new.table["supersedes"], "M002-v10-sweep")
        self.assertEqual(new.required, {**old.required, "supersedes": "str"})
        for key in ("metrics", "baseline", "falsification", "entrypoint", "seeds"):
            self.assertEqual(new.manifest[key], old.manifest[key])
        self.assertIn("M002-v10-sweep", new.manifest["hypothesis"])
        # The confirmatory experiment takes its lr file from the successor; the file name is the study's.
        table = REGS["M002-v10"].table
        self.assertEqual(table["sweep_experiment"], "M002-v10-sweep-r2")
        self.assertEqual(table["lr_file"], "research/falsification/A0-baseline-v1/lr_M002-v10.tsv")
        self.assertEqual(new.table["lr_file_target"], table["lr_file"])
        for exp_id in ("M001-v5", "M003-v2"):
            self.assertEqual(REGS[exp_id].table["sweep_experiment"], f"{exp_id}-sweep")

    def test_sweep_chain_and_ids(self):
        self.assertEqual(study.sweep_chain("M001-v5"), ("M001-v5-sweep",))
        self.assertEqual(study.sweep_chain("M002-v10"), ("M002-v10-sweep", "M002-v10-sweep-r2"))
        self.assertIsNone(study.superseded_by("M002-v10-sweep-r2"))
        self.assertIsNone(study.superseded_by("M002-v10"))
        self.assertEqual(gates.a0_baseline_v1_parts("M002-v10-sweep-r2"), ("M002-v10", True, 2))
        self.assertEqual(gates.a0_baseline_v1_parts("M002-v10-sweep"), ("M002-v10", True, 1))
        self.assertEqual(gates.a0_baseline_v1_parts("M002-v10"), ("M002-v10", False, 1))
        for bad in ("M002-v10-sweep-r1", "M002-v10-sweep-r02", "M002-v10-r2", "M004-v2", "M002-v10-sweep-r2x"):
            self.assertIsNone(gates.a0_baseline_v1_parts(bad), bad)
        for chain in (("M002-v10-sweep-r3",), ("M002-v10-sweep-r2", "M002-v10-sweep-r2"), ("M001-v5-sweep-r2",)):
            with self.subTest(chain=chain), mock_successors({"M002-v10": chain}):
                with self.assertRaises(study.RegistrationError):
                    study.sweep_chain("M002-v10")
        with mock_successors({"M002-v10": ("M002-v10-sweep-r2", "M002-v10-sweep-r3")}):
            self.assertEqual(study.current_sweep("M002-v10"), "M002-v10-sweep-r3")


class FrozenFixtureTests(unittest.TestCase):
    """A copy of the study's files in a temporary repository, frozen by the
    generator, is accepted by the repository's own gate; each tamper is not."""

    COPIED = [
        "research/falsification/A0-baseline-v1", "scripts/aggregate_a0_baseline.py", "scripts/aggregate_m002_v5.py",
        "benchmarks/operator-routing-v2/splits.lock.json", "model/burn-a0", "research/baselines/plain_model",
        "hardware/a0-cpu-4core.toml",
    ]

    @classmethod
    def setUpClass(cls):
        scratch = tempfile.TemporaryDirectory(prefix="a0-baseline-freeze-")
        cls.addClassCleanup(scratch.cleanup)
        cls.template = Path(scratch.name) / "template"
        root = cls.template
        for relative in cls.COPIED:
            source = ROOT / relative
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target, ignore=shutil.ignore_patterns("target*", "__pycache__"))
            else:
                shutil.copy2(source, target)
        (root / "experiments").mkdir()
        git(root, "init", "-q")
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", "inputs")
        registry = "version = 1\n"
        prereg = (ROOT / "experiments/preregistration.toml").read_text(encoding="utf-8").split("[experiment.M001-v5-sweep]", 1)
        text = "version = 1\n\n[experiment.M001-v5-sweep]" + prereg[1]
        (root / "experiments/preregistration.toml").write_text(text, encoding="utf-8")
        for reg in REGS.values():
            for name, text in study.render_files(reg, root).items():
                target = root / "experiments" / reg.path / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8", newline="\n")
            registry += f'\n[[experiment]]\nid = "{reg.exp_id}"\npath = "{reg.path}"\nstatus = "planned"\n'
        (root / "experiments/registry.toml").write_text(registry, encoding="utf-8")
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", "registered, planned")
        # A sweep that was frozen and then superseded (its only seed spent without an outcome).
        for old in SUPERSEDED:
            study.freeze(root, old, write=True)
            git(root, "add", "-A")
            git(root, "commit", "-q", "-m", f"{old} frozen")
            cls.supersede(root, old)
            git(root, "add", "-A")
            git(root, "commit", "-q", "-m", f"{old} superseded")

    @staticmethod
    def supersede(root, exp_id):
        """What the head does by hand: only the status of the manifest and the registry move."""
        manifest = root / "experiments" / REGS[exp_id].path / "experiment.toml"
        manifest.write_text(manifest.read_text().replace('status = "prepared"', 'status = "superseded"', 1), encoding="utf-8")
        registry = root / "experiments/registry.toml"
        text = registry.read_text()
        entry = f'id = "{exp_id}"\npath = "{REGS[exp_id].path}"\nstatus = "prepared"'
        assert entry in text
        registry.write_text(text.replace(entry, entry.replace('"prepared"', '"superseded"')), encoding="utf-8")

    def fresh(self):
        copy = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="a0-baseline-run-"))) / "repo"
        shutil.copytree(self.template, copy)
        return copy

    def frozen(self, edit=None):
        """The fixture with the sweeps frozen, lr files committed, then every
        experiment frozen and committed; `edit(root)` runs before the last commit."""
        root = self.fresh()
        for exp_id in CONFIRMATORY:
            study.freeze(root, study.current_sweep(exp_id), write=True)
            (root / study.lr_file_path(exp_id)).write_text(
                "# fixture\narm\tlr\tval_accuracy\tval_nll\nfull\t0.005\t0.5\t1.0\n", encoding="utf-8")
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", "sweeps frozen, lr files")
        for exp_id in CONFIRMATORY:
            study.freeze(root, exp_id, write=True)
        if edit:
            edit(root)
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", "freeze")
        return root

    def study_errors(self, root):
        ids = tuple(LISTED)
        return sorted(e for e in gates.gate_errors(root) if e.startswith(tuple(f"{i}:" for i in ids)) or e.split(" ", 1)[0] in ids)

    def test_a_frozen_study_passes_the_gate_and_is_prepared(self):
        root = self.frozen()
        self.assertEqual(self.study_errors(root), [])
        for exp_id in LISTED:
            manifest = toml_at(root, REGS[exp_id], "experiment.toml")
            if exp_id in SUPERSEDED:
                self.assertEqual(manifest["status"], "superseded")
                self.assertTrue(gates.launch_errors(root, exp_id), exp_id)
                continue
            self.assertEqual(gates.launch_errors(root, exp_id), [], exp_id)
            self.assertEqual(manifest["status"], "prepared")
            table = toml_at(root, REGS[exp_id], "config.toml")["preregistration"]
            self.assertNotIn(study.PENDING, table.values())
            self.assertEqual(table["implementation_tree_git_digest"], gates.directory_digest_at(root, "HEAD", "model/burn-a0"))
        registry = tomllib.loads((root / "experiments/registry.toml").read_text())["experiment"]
        self.assertEqual({i["id"]: i["status"] for i in registry}["M004-v2"], "planned")

    def test_freezing_needs_the_lr_file_and_a_planned_experiment(self):
        root = self.fresh()
        with self.assertRaisesRegex(study.RegistrationError, "run and commit the sweep first"):
            study.freeze(root, "M001-v5", write=True)
        study.freeze(root, "M001-v5-sweep", write=True)
        with self.assertRaisesRegex(study.RegistrationError, "only a planned experiment freezes"):
            study.freeze(root, "M001-v5-sweep", write=True)
        with self.assertRaisesRegex(study.RegistrationError, "not a listed experiment"):
            study.freeze(root, "M004-v2", write=True)
        with self.assertRaisesRegex(study.RegistrationError, "only a planned experiment freezes"):
            study.freeze(root, "M002-v10-sweep", write=True)

    def test_a_successor_sweep_freezes_only_after_its_predecessor_is_superseded(self):
        root = self.fresh()
        manifest = root / "experiments" / REGS["M002-v10-sweep"].path / "experiment.toml"
        manifest.write_text(manifest.read_text().replace('status = "superseded"', 'status = "prepared"', 1), encoding="utf-8")
        with self.assertRaisesRegex(study.RegistrationError, "must be superseded before M002-v10-sweep-r2 freezes"):
            study.freeze(root, "M002-v10-sweep-r2", write=True)
        with self.assertRaisesRegex(study.RegistrationError, "is not an experiment of the study"):
            study.render(root, ids=["M002-v10-sweep-r3"])

    def test_a_successor_sweep_renders_after_the_fact_and_only_once_its_predecessor_is_superseded(self):
        root = self.fresh()
        successor = REGS["M002-v10-sweep-r2"]
        registered = {name: (root / "experiments" / successor.path / name).read_text(encoding="utf-8")
                      for name in study.render_files(successor, root)}
        listing = (root / "experiments/preregistration.toml").read_text(encoding="utf-8")
        registry = (root / "experiments/registry.toml").read_text(encoding="utf-8")
        shutil.rmtree(root / "experiments" / successor.path)
        (root / "experiments/preregistration.toml").write_text(listing.split("\n[experiment.M002-v10-sweep-r2]")[0], encoding="utf-8")
        block = '\n[[experiment]]\nid = "M002-v10-sweep-r2"\npath = "model/M002-v10-sweep-r2"\nstatus = "planned"\n'
        self.assertIn(block, registry)
        without = registry.replace(block, "")
        (root / "experiments/registry.toml").write_text(without, encoding="utf-8")
        manifest = root / "experiments" / REGS["M002-v10-sweep"].path / "experiment.toml"
        text = manifest.read_text(encoding="utf-8")
        manifest.write_text(text.replace('"superseded"', '"prepared"', 1), encoding="utf-8")
        with self.assertRaisesRegex(study.RegistrationError, "must be superseded before its successor"):
            study.render(root, ids=["M002-v10-sweep-r2"])
        self.assertFalse((root / "experiments" / successor.path).exists())
        manifest.write_text(text, encoding="utf-8")
        self.assertEqual(study.render(root, ids=["M002-v10-sweep-r2"]), ["M002-v10-sweep-r2"])
        for name, content in registered.items():
            self.assertEqual((root / "experiments" / successor.path / name).read_text(encoding="utf-8"), content)
        self.assertEqual((root / "experiments/preregistration.toml").read_text(encoding="utf-8"), listing)
        self.assertEqual((root / "experiments/registry.toml").read_text(encoding="utf-8"), without + block)
        with self.assertRaisesRegex(study.RegistrationError, "already registered"):
            study.render(root, ids=["M002-v10-sweep-r2"])

    def test_the_gate_binds_a_frozen_successor_to_its_predecessor(self):
        root = self.frozen()
        directory = root / "experiments" / REGS["M002-v10-sweep-r2"].path
        manifest = toml_at(root, REGS["M002-v10-sweep-r2"], "experiment.toml")
        self.assertEqual(gates.a0_baseline_v1_freeze_errors(root, directory, manifest, "M002-v10-sweep-r2"), [])
        config = (directory / "config.toml").read_text()
        self.assertIn('supersedes = "M002-v10-sweep"', config)
        for old, new in (('supersedes = "M002-v10-sweep"', 'supersedes = "M002-v10-sweep-r2"'), ('supersedes = "M002-v10-sweep"\n', "")):
            with self.subTest(change=new):
                (directory / "config.toml").write_text(config.replace(old, new, 1), encoding="utf-8")
                self.assertTrue(gates.a0_baseline_v1_freeze_errors(root, directory, manifest, "M002-v10-sweep-r2"))
        (directory / "config.toml").write_text(config, encoding="utf-8")
        predecessor = root / "experiments" / REGS["M002-v10-sweep"].path / "experiment.toml"
        predecessor.write_text(predecessor.read_text().replace('"superseded"', '"prepared"', 1), encoding="utf-8")
        errors = gates.a0_baseline_v1_freeze_errors(root, directory, manifest, "M002-v10-sweep-r2")
        self.assertTrue(any("not superseded" in e for e in errors), errors)
        # `supersedes` belongs to a successor only.
        other = root / "experiments" / REGS["M001-v5-sweep"].path
        (other / "config.toml").write_text((other / "config.toml").read_text() + 'supersedes = "M001-v5-sweep"\n', encoding="utf-8")
        errors = gates.a0_baseline_v1_freeze_errors(root, other, toml_at(root, REGS["M001-v5-sweep"], "experiment.toml"), "M001-v5-sweep")
        self.assertTrue(any("only for a successor sweep" in e for e in errors), errors)

    def test_a_dry_freeze_writes_nothing(self):
        root = self.fresh()
        before = (root / "experiments/registry.toml").read_text()
        result = study.freeze(root, "M002-v10-sweep-r2", write=False)
        self.assertEqual(len(result["preregistration_sha256"]), 64)
        self.assertEqual((root / "experiments/registry.toml").read_text(), before)
        self.assertEqual(toml_at(root, REGS["M002-v10-sweep-r2"], "experiment.toml")["status"], "planned")
        # Same pinned values as the superseded sweep's freeze, apart from the predecessor it names.
        old = toml_at(root, REGS["M002-v10-sweep"], "config.toml")["preregistration"]
        self.assertEqual({k: v for k, v in result["table"].items() if k != "supersedes"}, old)

    def test_a_changed_criteria_or_runner_after_the_freeze_is_caught(self):
        def criteria(root):
            path = root / study.CRITERIA
            path.write_text(path.read_text().replace("delta_min = 0.02", "delta_min = 0.01"), encoding="utf-8")

        def runner(root):
            path = root / "model/burn-a0/examples/a0_ablation/arms.rs"
            path.write_text(path.read_text() + "\n// edited\n", encoding="utf-8")

        self.assertTrue(any("criteria_sha256" in e for e in self.study_errors(self.frozen(criteria))))
        errors = self.study_errors(self.frozen(runner))
        self.assertTrue(any("runner_arm_table_sha256" in e for e in errors))
        self.assertTrue(any("implementation tree differs" in e for e in errors))

    def test_the_binding_rejects_a_protocol_that_differs_from_criteria(self):
        root = self.frozen()
        reg = REGS["M002-v10"]
        directory = root / "experiments" / reg.path
        manifest = toml_at(root, reg, "experiment.toml")
        good = gates.a0_baseline_v1_freeze_errors(root, directory, manifest, "M002-v10")
        self.assertEqual(good, [])
        config = (directory / "config.toml").read_text()
        for old, new in (("rank = 8", "rank = 16"), ('metadata_dropout = "0"', 'metadata_dropout = "0.1"'),
                         ("evidence-temporal=8c3bc2324e0eb770", "evidence-temporal=0000000000000000"),
                         ('study = "m002-v10"', 'study = "m001-v5"'), ("steps = 1500", "steps = 15")):
            with self.subTest(change=new):
                (directory / "config.toml").write_text(config.replace(old, new, 1), encoding="utf-8")
                self.assertTrue(gates.a0_baseline_v1_freeze_errors(root, directory, manifest, "M002-v10"))
        (directory / "config.toml").write_text(config, encoding="utf-8")
        for flag in ("--rank <rank>", "--folds <folds>", "--lr-file <lr_file>", "--out -", "--metadata-dropout <metadata_dropout>"):
            with self.subTest(flag=flag):
                dropped = {**manifest, "entrypoint": manifest["entrypoint"].replace(flag, "")}
                self.assertTrue(gates.a0_baseline_v1_freeze_errors(root, directory, dropped, "M002-v10"))
        wrong_phase = {**manifest, "entrypoint": manifest["entrypoint"].replace("--phase baseline-v1 ", "--phase baseline-v1-sweep ")}
        self.assertTrue(gates.a0_baseline_v1_freeze_errors(root, directory, wrong_phase, "M002-v10"))

    def test_extract_sweep_refuses_what_is_not_a_completed_sweep_record(self):
        scratch = Path(self.enterContext(tempfile.TemporaryDirectory()))
        row = '{"row":"sweep-baseline-v1","study":"m001-v5"}'

        def record(name, **fields):
            path = scratch / name
            path.write_text(json.dumps({"status": "completed", "exit_code": 0, "experiment_id": "M001-v5-sweep",
                                        "stdout": row + "\n", **fields}), encoding="utf-8")
            return path

        out = scratch / "rows"
        self.assertEqual(study.extract_sweep([record("a.json")], out), 1)
        self.assertEqual(out.read_text(), row + "\n")
        with self.assertRaises(FileExistsError):
            study.extract_sweep([record("b.json")], out)
        for name, fields in (("c.json", {"exit_code": 1}), ("d.json", {"status": "failed"}),
                             ("e.json", {"experiment_id": "M001-v5"}), ("f.json", {"stdout": "{}\n"}),
                             ("g.json", {"experiment_id": "M001-v5-sweep-r2"})):
            with self.subTest(name=name), self.assertRaises(study.RegistrationError):
                study.extract_sweep([record(name, **fields)], scratch / f"out-{name}")

    def test_the_lr_file_of_m002_v10_comes_from_the_successor_sweep_only(self):
        scratch = Path(self.enterContext(tempfile.TemporaryDirectory()))
        row = '{"row":"sweep-baseline-v1","study":"m002-v10"}'

        def record(name, experiment_id, **fields):
            path = scratch / name
            path.write_text(json.dumps({"status": "completed", "exit_code": 0, "experiment_id": experiment_id,
                                        "stdout": row + "\n", **fields}), encoding="utf-8")
            return path

        self.assertEqual(study.extract_sweep([record("r2.json", "M002-v10-sweep-r2")], scratch / "rows"), 1)
        with self.assertRaisesRegex(study.RegistrationError, "superseded; the learning rates of M002-v10 come from M002-v10-sweep-r2"):
            study.extract_sweep([record("old.json", "M002-v10-sweep")], scratch / "rows-old")
        # The interrupted record itself is no completed run and is refused on that ground too.
        interrupted = next((ROOT / "experiments" / REGS["M002-v10-sweep"].path / "results").glob("run-*.json"))
        with self.assertRaisesRegex(study.RegistrationError, "not a completed run"):
            study.extract_sweep([interrupted], scratch / "rows-interrupted")
        self.assertEqual(study.lr_file_path("M002-v10"), "research/falsification/A0-baseline-v1/lr_M002-v10.tsv")


if __name__ == "__main__":
    unittest.main()
