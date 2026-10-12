"""Tests of scripts/freeze/*: the freeze tooling, on temporary fixture trees (no network, no cargo)."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "scripts/freeze"
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
       "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, FREEZE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records_module():
    spec = importlib.util.spec_from_file_location("experiment_records_t", ROOT / "scripts/experiment_records.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["experiment_records_t"] = module
    spec.loader.exec_module(module)
    return module


def run(script, *args, cwd=None):
    return subprocess.run([sys.executable, str(FREEZE / script), *args], cwd=cwd, env=ENV,
                          text=True, capture_output=True)


def git(root, *args):
    done = subprocess.run(["git", "-C", str(root), *args], env=ENV, text=True, capture_output=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


PREREG_LIST = """version = 1

[experiment.X001.required]
seeds = "int-list"
margin = "int"
"""

CONFIG = """# a comment that says must-be-signed-primary and must-be-pinned-nothing
[preregistration]
seeds = [1, 2]
margin = "must-be-signed-margin"
preregistration_text = "experiments/area/X001-demo/PREREGISTRATION.md"
preregistration_text_sha256 = "must-be-pinned-text"
criteria_sha256 = "must-be-pinned-criteria"
"""

MANIFEST = """version = 1
id = "X001"
status = "planned"
seeds = [1, 2]
"""

REGISTRY = """version = 1

[[experiment]]
id = "X001"
path = "area/X001-demo"
status = "planned"
"""


def make_tree(root: Path, config=CONFIG, registry=REGISTRY):
    d = root / "experiments/area/X001-demo"
    d.mkdir(parents=True)
    (root / "experiments/preregistration.toml").write_text(PREREG_LIST)
    (root / "experiments/registry.toml").write_text(registry)
    (d / "config.toml").write_text(config)
    (d / "experiment.toml").write_text(MANIFEST)
    (d / "PREREGISTRATION.md").write_text("text\n")
    (d / "criteria.toml").write_text("version = 1\n")
    return d


def snapshot(root: Path):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


class FreezeSignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def sign(self, *extra, ident="X001"):
        return run("freeze_sign.py", "--id", ident, "--dir", "experiments/area/X001-demo",
                   "--root", str(self.root), *extra)

    def test_dry_run_writes_nothing(self):
        make_tree(self.root)
        before = snapshot(self.root)
        done = self.sign("--sign", "must-be-signed-margin=30", "--dry-run")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("DRY-RUN OK", done.stdout)
        self.assertEqual(before, snapshot(self.root))

    def test_sign_writes_pins_and_prepared_like_a_frozen_experiment(self):
        d = make_tree(self.root)
        done = self.sign("--sign", "must-be-signed-margin=30")
        self.assertEqual(done.returncode, 0, done.stderr)
        records = records_module()
        config = tomllib.loads((d / "config.toml").read_text())
        table = config["preregistration"]
        self.assertEqual(table["margin"], 30)
        self.assertEqual(table["criteria_sha256"], records.preregistered_file_digest(d / "criteria.toml"))
        self.assertEqual(table["preregistration_text_sha256"],
                         records.preregistered_file_digest(d / "PREREGISTRATION.md"))
        manifest = tomllib.loads((d / "experiment.toml").read_text())
        self.assertEqual(manifest["status"], "prepared")
        self.assertEqual(manifest["preregistration_sha256"], records.preregistration_digest(table))
        entry = tomllib.loads(PREREG_LIST)["experiment"]["X001"]
        self.assertEqual(manifest["preregistration_rules_sha256"], records.canonical_digest(entry))
        self.assertIn('status = "prepared"', (self.root / "experiments/registry.toml").read_text())
        # comments are kept and never counted
        self.assertIn("# a comment that says must-be-signed-primary", (d / "config.toml").read_text())

    def test_placeholder_only_in_a_comment_is_refused_and_comments_never_trip(self):
        comment_only = CONFIG.replace('margin = "must-be-signed-margin"', "margin = 30\n# margin = \"must-be-signed-margin\"")
        make_tree(self.root, config=comment_only)
        before = snapshot(self.root)
        done = self.sign("--sign", "must-be-signed-margin=30")
        self.assertEqual(done.returncode, 4)
        self.assertIn("assigned 0 times", done.stderr)
        self.assertEqual(before, snapshot(self.root))
        # without --sign the comment does not block: every parsed value is pinned or a number
        done = self.sign("--dry-run")
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_unsigned_placeholder_value_blocks(self):
        make_tree(self.root)
        done = self.sign("--dry-run")
        self.assertEqual(done.returncode, 4)
        self.assertIn("placeholder values left", done.stderr)

    def test_unlisted_id_refused(self):
        make_tree(self.root)
        done = self.sign("--sign", "must-be-signed-margin=30", ident="X404")
        self.assertEqual(done.returncode, 4)
        self.assertIn("not listed", done.stderr)

    def test_late_refusal_leaves_every_file_untouched(self):
        d = make_tree(self.root, registry=REGISTRY.replace('"planned"', '"prepared"'))
        before = snapshot(self.root)
        done = self.sign("--sign", "must-be-signed-margin=30")
        self.assertEqual(done.returncode, 4)
        self.assertEqual(before, snapshot(self.root))
        self.assertIn("must-be-signed-margin", (d / "config.toml").read_text())

    def test_atomic_write_all_rolls_back(self):
        module = load("freeze_sign_t", "freeze_sign.py")
        a, b = self.root / "a.txt", self.root / "b.txt"
        a.write_text("A")
        b.write_text("B")
        with self.assertRaises(Exception):
            module.atomic_write_all({a: "A2", b: "B2", self.root / "missing/c.txt": "C"})
        self.assertEqual((a.read_text(), b.read_text()), ("A", "B"))
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["a.txt", "b.txt"])
        module.atomic_write_all({a: "A2", b: "B2"})
        self.assertEqual((a.read_text(), b.read_text()), ("A2", "B2"))


class DigestParentsTests(unittest.TestCase):
    BASE = "version = 1\n\n[experiment.E1.required]\nseeds = \"int-list\"\nmargin = \"int\"\n\n[experiment.E2.required]\nseeds = \"int-list\"\n"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        git(self.root, "init", "-q", "-b", "main")
        (self.root / "experiments").mkdir()
        self.path = self.root / "experiments/preregistration.toml"
        self.path.write_text(self.BASE)
        git(self.root, "add", "--", "experiments")
        git(self.root, "commit", "-q", "-m", "base")
        git(self.root, "checkout", "-q", "-b", "other")
        self.path.write_text(self.BASE + "\n[experiment.E3.required]\nseeds = \"int-list\"\n")
        git(self.root, "commit", "-q", "-am", "other adds E3")
        git(self.root, "checkout", "-q", "main")

    def check(self, text):
        self.path.write_text(text)
        return run("digest_parents.py", "main", "other", "--root", str(self.root))

    def test_clean_merge_has_no_problems(self):
        done = self.check(self.BASE + "\n[experiment.E3.required]\nseeds = \"int-list\"\n")
        self.assertEqual(done.returncode, 0, done.stdout)
        self.assertIn("0 problems", done.stdout)

    def test_split_rule_line_is_a_mismatch(self):
        split = self.BASE.replace("margin = \"int\"\n\n[experiment.E2.required]", "\n[experiment.E2.required]\nmargin = \"int\"")
        split += "\n[experiment.E3.required]\nseeds = \"int-list\"\n"
        done = self.check(split)
        self.assertEqual(done.returncode, 1, done.stdout)
        self.assertIn("MISMATCH E1", done.stdout)
        self.assertIn("MISMATCH E2", done.stdout)

    def test_dropped_id_is_an_error_and_differing_parents_are_info(self):
        done = self.check("version = 1\n\n[experiment.E1.required]\nseeds = \"int-list\"\nmargin = \"int\"\n")
        self.assertEqual(done.returncode, 1)
        self.assertIn("DROPPED E2", done.stdout)
        # parents that disagree on E2, merge takes the first parent's
        git(self.root, "checkout", "-q", "--", ".")
        git(self.root, "checkout", "-q", "other")
        text = self.path.read_text().replace('[experiment.E2.required]\nseeds = "int-list"', '[experiment.E2.required]\nseeds = "int-list"\nextra = "int"')
        self.path.write_text(text)
        git(self.root, "commit", "-q", "-am", "other changes E2")
        git(self.root, "checkout", "-q", "main")
        self.path.write_text(self.BASE + "\n[experiment.E3.required]\nseeds = \"int-list\"\n")
        done = run("digest_parents.py", "main", "other", "--root", str(self.root))
        self.assertIn("DIFFERS-BETWEEN-PARENTS E2", done.stdout)
        self.assertEqual(done.returncode, 0)


class FinishExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.d = make_tree(self.root, registry=REGISTRY.replace('"planned"', '"prepared"'))
        (self.d / "experiment.toml").write_text(MANIFEST.replace("planned", "prepared"))
        (self.d / "results").mkdir()
        (self.root / "STATUS.md").write_text("# Status\n- **Anchor line one.** text\n- **Next.**\n")
        (self.root / "line.md").write_text("- **X001 completed.** the result\n")

    def finish(self, recommended, *extra):
        (self.d / "results/run.json").write_text(json.dumps({"recommended_status": recommended}))
        return run("finish_experiment.py", "--root", str(self.root), "status", "--id", "X001",
                   "--dir", "experiments/area/X001-demo", "--status-line-file", str(self.root / "line.md"),
                   "--anchor", "- **Anchor line", *extra)

    def test_status_completed_and_failed(self):
        for recommended in ("completed", "failed"):
            self.tearDown_files()
            done = self.finish(recommended)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(tomllib.loads((self.d / "experiment.toml").read_text())["status"], recommended)
            self.assertIn(f'status = "{recommended}"', (self.root / "experiments/registry.toml").read_text())
            lines = (self.root / "STATUS.md").read_text().splitlines()
            self.assertEqual(lines[lines.index("- **Anchor line one.** text") + 1], "- **X001 completed.** the result")

    def tearDown_files(self):
        (self.d / "experiment.toml").write_text(MANIFEST.replace("planned", "prepared"))
        (self.root / "experiments/registry.toml").write_text(REGISTRY.replace('"planned"', '"prepared"'))
        (self.root / "STATUS.md").write_text("# Status\n- **Anchor line one.** text\n- **Next.**\n")

    def test_no_recommended_status_or_missing_anchor_writes_nothing(self):
        self.assertEqual(self.finish(None).returncode, 4)
        before = snapshot(self.root)
        done = self.finish(None)
        self.assertEqual(done.returncode, 4)
        self.assertEqual(before, snapshot(self.root))
        (self.root / "STATUS.md").write_text("# Status\n")
        self.finish("completed")
        before = snapshot(self.root)
        done = self.finish("completed")
        self.assertEqual(done.returncode, 4)
        self.assertIn("anchor", done.stderr)
        self.assertEqual(before, snapshot(self.root))

    def test_anchor_must_be_unique_line_start(self):
        (self.root / "STATUS.md").write_text("# S\n- **Anchor line one.** a\n- **Anchor line two.** b\ntext - **Anchor line one.** mid\n")
        before = None
        self.finish("completed")
        before = snapshot(self.root)
        done = self.finish("completed")
        self.assertEqual(done.returncode, 4)
        self.assertIn("2 lines", done.stderr)
        self.assertEqual(before, snapshot(self.root))
        (self.root / "STATUS.md").write_text("# S\ntext - **Anchor line one.** mid\n- **Anchor line one.** a\n")
        done = self.finish("completed")
        self.assertEqual(done.returncode, 0, done.stderr)
        lines = (self.root / "STATUS.md").read_text().splitlines()
        self.assertEqual(lines[-1], "- **X001 completed.** the result")
        self.assertEqual(lines[-2], "- **Anchor line one.** a")
        self.assertEqual(lines[1], "text - **Anchor line one.** mid")

    def test_stale_marker(self):
        git(self.root, "init", "-q", "-b", "main")
        git(self.root, "add", "--", "experiments", "STATUS.md")
        git(self.root, "commit", "-q", "-m", "one")
        first = git(self.root, "rev-parse", "HEAD")
        git(self.root, "tag", "x001-prereg")
        (self.root / "STATUS.md").write_text("changed\n")
        git(self.root, "commit", "-q", "-am", "two")
        second = git(self.root, "rev-parse", "HEAD")
        done = run("finish_experiment.py", "--root", str(self.root), "stale", "--id", "X001",
                   "--dir", "experiments/area/X001-demo", "--results-sha", first, "--stale-since", second,
                   "--tag", "x001-prereg")
        self.assertEqual(done.returncode, 0, done.stderr)
        stale = tomllib.loads((self.d / "results/STALE.toml").read_text())
        self.assertEqual((stale["results_git_sha"], stale["stale_since"]), (first, second))
        self.assertIn("x001-prereg", stale["reason"])
        done = run("finish_experiment.py", "--root", str(self.root), "stale", "--id", "X001",
                   "--dir", "experiments/area/X001-demo", "--results-sha", first, "--stale-since", second,
                   "--tag", "no-such-tag")
        self.assertEqual(done.returncode, 4)


class SmallToolsTests(unittest.TestCase):
    def test_config_seeds_and_compare_pilot(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "c.toml").write_text("[a]\nx = 1\n[preregistration]\nseeds = [17, 29]\n")
            done = run("config_seeds.py", str(tmp / "c.toml"))
            self.assertEqual(done.stdout.strip(), "17 29")
            (tmp / "a.json").write_text(json.dumps({"k": 1, "t": 5, "n": [{"t": 1, "v": 2}]}))
            (tmp / "b.json").write_text(json.dumps({"k": 1, "t": 9, "n": [{"t": 7, "v": 2}]}))
            same = run("compare_pilot.py", str(tmp / "a.json"), str(tmp / "b.json"), "--timing-key", "t")
            self.assertEqual(same.returncode, 0, same.stdout)
            differ = run("compare_pilot.py", str(tmp / "a.json"), str(tmp / "b.json"))
            self.assertEqual(differ.returncode, 1)
            (tmp / "c.json").write_text(json.dumps({"k": True}))
            (tmp / "d.json").write_text(json.dumps({"k": 1}))
            (tmp / "e.json").write_text(json.dumps({"k": 1.0}))
            self.assertEqual(run("compare_pilot.py", str(tmp / "c.json"), str(tmp / "d.json")).returncode, 1)
            self.assertEqual(run("compare_pilot.py", str(tmp / "d.json"), str(tmp / "e.json")).returncode, 1)


class MakeFreezeScriptTests(unittest.TestCase):
    SPEC = ["--id", "X001", "--dir", "experiments/area/X001-demo", "--tag", "x001-prereg",
            "--pilot-ref", "experiments/area/X001-demo/pilots/p.json", "--pilot-seed", "1002",
            "--pilot-command", "echo {seed}", "--timing-key", "elapsed_ms",
            "--pre-sign-check", "true", "--unittest-module", "scripts.tests.test_x",
            "--crate-test", "ptr-types", "--cargo-package", "ptr-bench", "--cargo-feature", "f1",
            "--sign", "must-be-signed-margin=30"]

    def generate(self, *extra):
        done = run("make_freeze_script.py", *self.SPEC, *extra)
        self.assertEqual(done.returncode, 0, done.stderr)
        return done.stdout

    def test_syntax_order_and_no_blanket_add(self):
        script = self.generate()
        check = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assertNotIn("git add -A", script)
        self.assertNotIn("git add .", script)
        self.assertNotIn("--all", script.replace("--all-targets", "").replace("--workspace", "").replace("fmt --all", ""))
        steps = ["freeze_sign.py\" --id \"$ID\" --dir \"$D\" --sign must-be-signed-margin=30 --root \"$W\" --dry-run",
                 "only_paths 'experiments", "add_existing experiments crates", "check_research_gates.py",
                 "check_repo.py", "update_component_docs.py --check", "mutation_check.py \"$ID\" --check",
                 "run_experiment.py validate", "EnrolledExperimentTests scripts.tests.test_x",
                 "cargo fmt --all", "cargo clippy --workspace", "cargo test -p ptr-types", "bash -c true",
                 "guard_on", "freeze_sign.py\" --id \"$ID\" --dir \"$D\" --sign must-be-signed-margin=30 --root \"$W\" ||",
                 "update_component_docs.py --write", "git tag \"$TAG\"", "guard_off",
                 "unset CARGO_TARGET_DIR", "mutation_check.py \"$ID\" >", "compare_pilot.py",
                 "run_experiment.py preflight", "run_experiment.py prepare", "config_seeds.py",
                 "run_experiment.py run", "aggregate.py"]
        position = -1
        for step in steps:
            found = script.find(step, position + 1)
            self.assertGreater(found, position, f"step out of order or missing: {step}")
            position = found
        self.assertIn("x001-prereg", script)
        self.assertIn("--timing-key elapsed_ms", script)
        self.assertNotRegex(script, r"(?i)claude|anthropic|co-authored")
        self.assertIn("FREEZE_TRAILER", script)

    def test_gpu_export_only_when_set(self):
        self.assertNotIn("CUDA_VISIBLE_DEVICES", self.generate())
        script = self.generate("--gpu", "1")
        self.assertIn("export CUDA_VISIBLE_DEVICES=$GPU", script)
        self.assertIn("GPU=1", script)
        self.assertIn("gpu=$GPU", script)
        self.assertEqual(subprocess.run(["bash", "-n"], input=script, text=True).returncode, 0)

    def test_tag_is_mandatory_and_inputs_validated(self):
        done = run("make_freeze_script.py", "--id", "X001", "--dir", "experiments/a")
        self.assertEqual(done.returncode, 4)
        self.assertIn("tag", done.stderr)
        done = run("make_freeze_script.py", "--id", "X001", "--dir", "/abs", "--tag", "t")
        self.assertEqual(done.returncode, 4)
        done = run("make_freeze_script.py", "--id", "X001", "--dir", "a", "--tag", "t", "--pre-sign-check", "if then")
        self.assertEqual(done.returncode, 4)

    def test_spec_file_and_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp) / "s.toml"
            spec.write_text('id = "X001"\ndir = "experiments/a"\ntag = "x1-prereg"\nenv_file = "/etc/ptr.env"\n[sign]\nmust-be-signed-a = 3\n')
            done = run("make_freeze_script.py", "--spec", str(spec))
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn("ENV_FILE=/etc/ptr.env", done.stdout)
            self.assertIn(">/dev/null 2>&1", done.stdout)
            self.assertIn("--sign must-be-signed-a=3", done.stdout)
            self.assertNotIn("STEP 7", done.stdout)

    def test_unknown_keys_and_wrong_types_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = 'id = "X001"\ndir = "experiments/a"\ntag = "t"\n'
            for extra, needle in (('crate_test = ["ptr-types"]\n', "unknown spec keys"),
                                  ('unittest_modules = "x"\n', "list of strings"),
                                  ('gpu = true\n', "integer"),
                                  ('pilot_seed = "1"\n', "integer"),
                                  ('tag2 = "t"\n', "unknown spec keys"),
                                  ('[sign]\nmust-be-signed-a = true\n', "placeholder = int")):
                spec = Path(tmp) / "s.toml"
                spec.write_text(base + extra)
                done = run("make_freeze_script.py", "--spec", str(spec))
                self.assertEqual(done.returncode, 4, extra)
                self.assertIn(needle, done.stderr, extra)

    def test_resume_step_validated_and_commands_checked(self):
        script = self.generate()
        self.assertIn("1..10", script)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fz.sh"
            path.write_text(script)
            for bad in ("0", "11", "x", ""):
                done = subprocess.run(["bash", str(path), "--resume-from", bad], cwd=tmp, env=ENV,
                                      text=True, capture_output=True)
                self.assertEqual(done.returncode, 4, bad)
                self.assertNotIn("STEP", done.stdout)
        self.assertIn('PIPESTATUS[0]}" -eq 0 ] || stop 4 "docs write failed"', script)
        self.assertIn('stop 10 "aggregate failed"', script)
        self.assertIn('stop 9 "seed $s commit failed"', script)
        self.assertEqual(script.count('--root "$W"'), 2)

    def test_work_paths_match_on_path_boundaries(self):
        script = self.generate()
        self.assertIn("docs(/|$)", script)
        with tempfile.TemporaryDirectory() as tmp:
            git(tmp, "init", "-q", "-b", "main")
            (Path(tmp) / "docs-x").write_text("x")
            (Path(tmp) / "docs").mkdir()
            (Path(tmp) / "docs/ok.md").write_text("x")
            regex = r"experiments(/|$)|docs(/|$)"
            body = f'. {FREEZE}/freeze_lib.sh; only_paths {regex!r}'
            done = subprocess.run(["bash", "-c", body], cwd=tmp, text=True, capture_output=True)
            self.assertEqual(done.returncode, 1)
            self.assertIn("docs-x", done.stdout)
            self.assertNotIn("docs/ok.md", done.stdout)

    def test_shell_helpers_parse(self):
        for name in ("freeze_guard.sh", "freeze_lib.sh"):
            self.assertEqual(subprocess.run(["bash", "-n", str(FREEZE / name)]).returncode, 0, name)

    def test_staged_change_before_failed_commit_is_fully_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            git(tmp, "init", "-q", "-b", "main")
            (Path(tmp) / "f.txt").write_text("orig")
            git(tmp, "add", "--", "f.txt")
            git(tmp, "commit", "-q", "-m", "x")
            (Path(tmp) / "untracked.txt").write_text("keep")
            body = (f'. {FREEZE}/freeze_guard.sh; guard_on; echo signed > f.txt; git add -- f.txt; '
                    'git commit -q -m fail --no-verify --author="bad" 2>/dev/null; exit 5')
            done = subprocess.run(["bash", "-c", body], cwd=tmp, env=ENV, text=True, capture_output=True)
            self.assertEqual(done.returncode, 5)
            self.assertEqual((Path(tmp) / "f.txt").read_text(), "orig")
            self.assertEqual(git(tmp, "status", "--porcelain", "--untracked-files=no"), "")
            self.assertEqual((Path(tmp) / "untracked.txt").read_text(), "keep")
            self.assertIn("restored", done.stdout)
            self.assertNotIn("FAILED", done.stderr)

    def test_guard_restores_tracked_files_on_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            git(tmp, "init", "-q", "-b", "main")
            (Path(tmp) / "f.txt").write_text("orig")
            git(tmp, "add", "--", "f.txt")
            git(tmp, "commit", "-q", "-m", "x")
            body = f'. {FREEZE}/freeze_guard.sh; guard_on; echo changed > f.txt; exit 3'
            done = subprocess.run(["bash", "-c", body], cwd=tmp, env=ENV, text=True, capture_output=True)
            self.assertEqual(done.returncode, 3)
            self.assertEqual((Path(tmp) / "f.txt").read_text(), "orig")
            body = f'. {FREEZE}/freeze_guard.sh; guard_on; echo changed > f.txt; guard_off; exit 0'
            subprocess.run(["bash", "-c", body], cwd=tmp, env=ENV, text=True, capture_output=True)
            self.assertEqual((Path(tmp) / "f.txt").read_text().strip(), "changed")


if __name__ == "__main__":
    unittest.main()


class StepEightCleansTheCheckoutTests(unittest.TestCase):
    """Regression (E007 replay, 2026-10-12): the runner refuses a checkout holding target/, so the
    generated script must remove it before preflight whatever FREEZE_CLEAN_TARGET says."""

    def test_target_is_removed_before_preflight(self):
        mod = load("make_freeze_script_t8", "make_freeze_script.py")
        text = mod.generate({"id": "E9", "dir": "experiments/system/E9-x", "tag": "e9-prereg"})
        step8 = text.index('STEP 8: preflight, prepare')
        preflight = text.index('run_experiment.py preflight', step8)
        block = text[step8:preflight]
        self.assertIn("rm -rf target", block)
        self.assertIn("git clean -ndX", block)

