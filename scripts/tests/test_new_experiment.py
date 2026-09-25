"""The experiment scaffolder must produce something the runner accepts as written.

It used to write three of the nine required manifest keys, no config.toml or
tests/, and no registry entry, so its output was invisible to
`run_experiment.py validate` and unresolvable by `run_experiment.py run`. These
tests scaffold into fixture trees and hand the result to the real validator.
"""

import contextlib
import importlib.util
import io
import shutil
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scaffold = load("new_experiment")
runner = load("run_experiment")

FIELDS = {
    "hypothesis": "Slot dropout improves OOD route accuracy.",
    "metrics": "route accuracy; cost-adjusted regret",
    "baseline": "A0 without slot dropout, same seeds",
    "falsification": "Reject if OOD route accuracy does not improve on every seed.",
}


def fixture():
    """A tree with one registered experiment and a registry lacking a final newline."""
    directory = tempfile.TemporaryDirectory()
    root = Path(directory.name)
    (root / "experiments/model/M001-existing/tests").mkdir(parents=True)
    shutil.copy(ROOT / "experiments/schema.toml", root / "experiments/schema.toml")
    (root / "experiments/registry.toml").write_text(
        'version = 1\n\n[[experiment]]\nid = "M001"\npath = "model/M001-existing"\nstatus = "planned"',
        encoding="utf-8",
    )
    existing = root / "experiments/model/M001-existing"
    (existing / "config.toml").write_text("version = 1\n", encoding="utf-8")
    (existing / "experiment.toml").write_text(
        scaffold.manifest(
            "M001", hardware_profile="hardware/default.toml", seeds=[17], **FIELDS
        ),
        encoding="utf-8",
    )
    (root / "hardware").mkdir()
    (root / "hardware/default.toml").write_text("version = 1\n", encoding="utf-8")
    return directory, root


def validate(root: Path) -> tuple[int, str]:
    saved = runner.ROOT, runner.REGISTRY
    runner.ROOT, runner.REGISTRY = root, root / "experiments/registry.toml"
    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = runner.validate()
        return code, out.getvalue()
    finally:
        runner.ROOT, runner.REGISTRY = saved


def snapshot(root: Path) -> dict[str, bytes | None]:
    """Every file's bytes and every directory, so a leftover empty directory counts."""
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in sorted(root.rglob("*"))
    }


class Scaffold(unittest.TestCase):
    def test_a_scaffolded_experiment_passes_the_real_validator(self):
        directory, root = fixture()
        with directory:
            self.assertEqual(validate(root)[0], 0)
            target = scaffold.create(root, "model/M008-slot-dropout", **FIELDS)
            code, output = validate(root)

            self.assertEqual(code, 0, output)
            self.assertIn("OK: validated 2 experiments", output)
            data = tomllib.loads((target / "experiment.toml").read_text(encoding="utf-8"))
            required = tomllib.loads(
                (root / "experiments/schema.toml").read_text(encoding="utf-8")
            )["required"]
            self.assertEqual(set(required) - set(data), set())
            self.assertEqual(data["id"], "M008")
            self.assertEqual(data["status"], "planned")
            self.assertEqual(data["seeds"], [17, 29, 43, 71, 101])
            self.assertEqual(data["hardware_profile"], "hardware/default.toml")
            self.assertTrue((target / "tests/README.md").is_file())
            self.assertTrue((target / "results/.gitkeep").is_file())
            registry = tomllib.loads(
                (root / "experiments/registry.toml").read_text(encoding="utf-8")
            )["experiment"]
            self.assertEqual(
                registry[-1],
                {"id": "M008", "path": "model/M008-slot-dropout", "status": "planned"},
            )
            readme = (target / "README.md").read_text(encoding="utf-8")
            for value in FIELDS.values():
                self.assertIn(value, readme)

    def test_text_that_needs_escaping_round_trips(self):
        directory, root = fixture()
        awkward = 'It\'s "typed" \\ not\tplain\nsecond line \x7f ünïcode'
        with directory:
            target = scaffold.create(
                root,
                "model/M009-escaping",
                **{**FIELDS, "hypothesis": awkward},
                seeds=[3, 1],
            )
            data = tomllib.loads((target / "experiment.toml").read_text(encoding="utf-8"))
            self.assertEqual(data["hypothesis"], awkward)
            self.assertEqual(data["seeds"], [3, 1])
            self.assertEqual(validate(root)[0], 0)

    def test_every_refusal_leaves_the_tree_untouched(self):
        cases = [
            ("model/M001-duplicate-id", {}, "already registered"),
            ("model/M010", {}, "name must be lowercase"),
            ("model/m010-lower-id", {}, "capital letter and three digits"),
            ("M010-no-area", {}, "<area>/<ID>-<name>"),
            ("model/sub/M010-too-deep", {}, "<area>/<ID>-<name>"),
            ("nowhere/M010-unknown-area", {}, "unknown experiment area"),
            ("tests/M010-tests-is-not-an-area", {}, "unknown experiment area"),
            ("model/M010-Upper-Slug", {}, "name must be lowercase"),
            ("model/M010-no-hardware", {"hardware_profile": "hardware/absent.toml"}, "does not exist"),
            ("model/M010-empty-field", {"baseline": "  "}, "baseline must not be empty"),
            # `..` and `.` exist as directories; the area must be a real category name.
            ("../M010-escape", {}, "unknown experiment area"),
            ("./M010-dot", {}, "unknown experiment area"),
            ("model\\M010-backslash", {}, "<area>/<ID>-<name>"),
            ("model/M010-profile-outside", {"hardware_profile": "/etc/hostname"}, "hardware/<name>.toml"),
            ("model/M010-profile-config", {"hardware_profile": "hardware/config.toml"}, "not the area's config.toml"),
            ("model/M010-profile-escape", {"hardware_profile": "hardware/../x.toml"}, "hardware/<name>.toml"),
            # Undecodable argv bytes arrive as lone surrogates.
            ("model/M010-not-utf8", {"hypothesis": "caf\udce9"}, "hypothesis is not valid UTF-8"),
            ("model/M010-no-seeds", {"seeds": []}, "at least one seed"),
            ("model/M010-huge-seed", {"seeds": [2**70]}, "integer from 0 to"),
            ("model/M010-negative-seed", {"seeds": [-1]}, "integer from 0 to"),
            ("model/M010-bool-seed", {"seeds": [True]}, "integer from 0 to"),
        ]
        for path, overrides, message in cases:
            with self.subTest(path=path):
                directory, root = fixture()
                with directory:
                    before = snapshot(root)
                    with self.assertRaisesRegex(ValueError, message):
                        scaffold.create(root, path, **{**FIELDS, **overrides})
                    self.assertEqual(snapshot(root), before)

    def test_a_write_that_fails_midway_leaves_nothing_behind(self):
        directory, root = fixture()
        real = Path.write_text

        def failing(path, *args, **kwargs):
            if path.name == "README.md" and "M014" in str(path):
                raise OSError(28, "No space left on device")
            return real(path, *args, **kwargs)

        with directory:
            before = snapshot(root)
            Path.write_text = failing
            try:
                with self.assertRaisesRegex(ValueError, "nothing was kept"):
                    scaffold.create(root, "model/M014-disk-full", **FIELDS)
            finally:
                Path.write_text = real
            self.assertEqual(snapshot(root), before)
            # And the same call succeeds once the disk has room again.
            scaffold.create(root, "model/M014-disk-full", **FIELDS)
            self.assertEqual(validate(root)[0], 0)

    def test_an_existing_directory_is_never_overwritten(self):
        directory, root = fixture()
        with directory:
            (root / "experiments/model/M011-taken").mkdir()
            before = snapshot(root)
            with self.assertRaisesRegex(ValueError, "already exists"):
                scaffold.create(root, "model/M011-taken", **FIELDS)
            self.assertEqual(snapshot(root), before)

    def test_seed_lists_are_parsed_strictly(self):
        self.assertEqual(scaffold.parse_seeds("17, 29,43"), [17, 29, 43])
        for text, message in [("", "at least one"), ("1,x", "integers"), ("5,5", "distinct")]:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                scaffold.parse_seeds(text)

    def test_the_cli_requires_every_schema_field(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            scaffold.main(["model/M012-cli", "--hypothesis", "h"])
        self.assertEqual(raised.exception.code, 2)


class Validator(unittest.TestCase):
    def test_an_unregistered_manifest_is_reported(self):
        directory, root = fixture()
        with directory:
            stray = root / "experiments/model/M013-stray"
            (stray / "tests").mkdir(parents=True)
            (stray / "experiment.toml").write_text(
                scaffold.manifest(
                    "M013", hardware_profile="hardware/default.toml", seeds=[17], **FIELDS
                ),
                encoding="utf-8",
            )
            code, output = validate(root)
        self.assertEqual(code, 1)
        self.assertIn(
            "ERROR: experiments/model/M013-stray: not listed in experiments/registry.toml",
            output,
        )

    def test_the_real_tree_registers_every_manifest(self):
        code, output = validate(ROOT)
        self.assertEqual(code, 0, output)


if __name__ == "__main__":
    unittest.main()
