"""`make ci-local` and `make a0` must keep running what CI runs.

The documented local checks were a subset of CI: no vendor, notices, codebook or
research-gate check and no feature backend, so a contributor could pass every
one and still get a red CI. And `make a0` ran a bare `cargo`, which the root
`rust-toolchain.toml` resolves to 1.85.0, for a crate that requires 1.95. These
tests read the workflows and the Makefile and fail when a CI step has no
counterpart in the matching target, so the mirror cannot fall behind silently.

The workflow is read step by step rather than line by line. A line-level reading
passed vacuously in three ways, each shown by a review: a new multi-line `run: |`
step was reduced to `|` and skipped, a step's `env:` was ignored, and a
commented-out Makefile line still matched.
"""

import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TARGET = re.compile(r"^([A-Za-z0-9_.-]+):(?!=)(.*)$")

# Steps the Makefile deliberately does not reproduce, each with its reason.
NOT_REPRODUCED = {
    "python -m pip install -e training": (
        "installs into the contributor's environment; the targets set "
        "PYTHONPATH=training/src instead"
    ),
}

# Multi-line shell steps, by (job, step name), and the recipe line that stands
# for each. A block step not listed here fails the test: it has to be looked at.
BLOCK_STEPS = {
    ("repository-invariants", "Verify metadata across the complete change range"): (
        "$(PYTHON) scripts/check_component_metadata.py --base $(BASE)"
    ),
}


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def parse_workflow(path: Path) -> dict:
    """{"env": bool, "jobs": {job: {"env": bool, "steps": [step]}}}.

    A step is {"name", "run", "block", "env"}: `run` for a one-line command,
    `block` for the lines of a `run: |` script, `env` for the step's variables.
    Only keys at a step's own level are read, so nothing under `with:` is taken
    for a command."""
    lines = path.read_text(encoding="utf-8").splitlines()
    workflow = {"env": False, "jobs": {}}
    in_jobs = False
    job = None
    in_steps = False
    step = None
    step_indent = None
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = indent_of(line)
        if indent == 0:
            in_jobs = stripped == "jobs:"
            workflow["env"] = workflow["env"] or stripped.startswith("env:")
            job = None
            continue
        if not in_jobs:
            continue
        if indent == 2:
            job = stripped.rstrip(":")
            workflow["jobs"][job] = {"env": False, "steps": []}
            in_steps = False
            step = None
            continue
        if job is None:
            continue
        if indent == 4:
            in_steps = stripped == "steps:"
            if stripped.startswith("env:"):
                workflow["jobs"][job]["env"] = True
            step = None
            continue
        if not in_steps:
            continue
        if stripped.startswith("- "):
            step = {"name": None, "run": None, "block": None, "env": {}}
            workflow["jobs"][job]["steps"].append(step)
            step_indent = indent + 2
            content = stripped[2:]
        elif step is not None and indent == step_indent:
            content = stripped
        else:
            continue
        key, _, value = content.partition(":")
        value = value.strip()
        if key == "name":
            step["name"] = value
        elif key == "run" and value in ("|", "|-", ">", ">-"):
            block = []
            while i < len(lines) and (not lines[i].strip() or indent_of(lines[i]) > step_indent):
                if lines[i].strip():
                    block.append(lines[i].strip())
                i += 1
            step["block"] = block
        elif key == "run":
            step["run"] = value
        elif key == "env":
            while i < len(lines) and lines[i].strip() and indent_of(lines[i]) > step_indent:
                name, _, variable = lines[i].strip().partition(":")
                step["env"][name.strip()] = variable.strip().strip('"').strip("'")
                i += 1
    return workflow


def make_targets(path: Path) -> dict[str, tuple[str, list[str]]]:
    """Each target's prerequisites (continuation lines joined) and the recipe
    lines the shell will execute: comments are dropped, and make's `@`/`-`
    prefixes are removed before comparing."""
    lines = path.read_text(encoding="utf-8").replace("\\\n", " ").splitlines()
    targets: dict[str, tuple[str, list[str]]] = {}
    current = None
    for line in lines:
        if line.startswith("\t") and current:
            command = line.strip().lstrip("@-").strip()
            if command and not command.startswith("#"):
                targets[current][1].append(command)
            continue
        match = TARGET.match(line)
        if match and not line.startswith(("#", ".")):
            current = match.group(1)
            targets[current] = (match.group(2), [])
        elif not line.strip() or line.startswith("#"):
            current = None
    return targets


def normalize(command: str) -> str:
    return re.sub(r"^python ", "$(PYTHON) ", command)


def carries(line: str, env: dict[str, str]) -> bool:
    return all(f'{k}="{v}"' in line or f"{k}={v}" in line for k, v in env.items())


class TheParserReadsWhatTheWorkflowSays(unittest.TestCase):
    # A parse that silently found nothing would make every other test pass.
    def test_it_sees_every_job_a_step_env_and_a_block(self):
        workflow = parse_workflow(ROOT / ".github/workflows/ci.yml")
        self.assertGreaterEqual(len(workflow["jobs"]), 13, sorted(workflow["jobs"]))
        quality = workflow["jobs"]["quality"]["steps"]
        doc = next(s for s in quality if s["run"] and " doc " in s["run"])
        self.assertEqual(doc["env"], {"RUSTDOCFLAGS": "-D warnings"})
        invariants = workflow["jobs"]["repository-invariants"]["steps"]
        block = next(s for s in invariants if s["block"])
        self.assertTrue(any("check_component_metadata.py" in line for line in block["block"]))

    def test_it_is_not_fooled_by_the_shapes_that_fooled_the_line_reader(self):
        text = (
            "jobs:\n"
            "  j:\n"
            "    strategy:\n"
            "      matrix:\n"
            "        os:\n"
            "          - a\n"
            "    steps:\n"
            "      - uses: x\n"
            "        with:\n"
            "          run: not-a-command\n"
            "      - run: cargo test\n"
            "        env:\n"
            "          RUSTFLAGS: \"-D warnings\"\n"
            "      - name: script\n"
            "        run: |\n"
            "          echo one\n"
            "\n"
            "          echo two\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workflow.yml"
            path.write_text(text, encoding="utf-8")
            steps = parse_workflow(path)["jobs"]["j"]["steps"]
        self.assertEqual([s["run"] for s in steps], [None, "cargo test", None])
        self.assertEqual(steps[1]["env"], {"RUSTFLAGS": "-D warnings"})
        self.assertEqual(steps[2]["block"], ["echo one", "echo two"])

    def test_commented_recipe_lines_are_not_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Makefile"
            path.write_text("t:\n\t# cargo test\n\t@echo run\n", encoding="utf-8")
            self.assertEqual(make_targets(path)["t"][1], ["echo run"])


class CiLocalMirrorsCi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = parse_workflow(ROOT / ".github/workflows/ci.yml")
        cls.jobs = cls.workflow["jobs"]
        cls.targets = make_targets(ROOT / "Makefile")

    def test_no_workflow_or_job_environment_goes_unmirrored(self):
        # Only step-level env is carried into the recipes; anything wider would
        # change every command without appearing in any of them.
        self.assertFalse(self.workflow["env"])
        self.assertEqual([job for job, data in self.jobs.items() if data["env"]], [])

    def test_every_job_has_a_target_and_ci_local_runs_them_all(self):
        expected = {f"ci-{job}" for job in self.jobs}
        self.assertEqual(expected - set(self.targets), set())
        self.assertEqual(set(self.targets["ci-local"][0].split()), expected)

    def test_every_step_is_reproduced_in_its_jobs_target(self):
        for job, data in self.jobs.items():
            recipe = self.targets[f"ci-{job}"][1]
            for step in data["steps"]:
                if step["block"] is not None:
                    with self.subTest(job=job, block=step["name"]):
                        self.assertIn((job, step["name"]), BLOCK_STEPS, "a new multi-line step")
                        self.assertIn(BLOCK_STEPS[(job, step["name"])], recipe)
                    continue
                if step["run"] is None or step["run"] in NOT_REPRODUCED:
                    continue
                with self.subTest(job=job, step=step["run"]):
                    wanted = normalize(step["run"])
                    lines = [line for line in recipe if wanted in line]
                    self.assertTrue(lines, f"ci-{job} has no line running: {wanted}")
                    self.assertTrue(
                        any(carries(line, step["env"]) for line in lines),
                        f"ci-{job} runs {wanted} without {step['env']}",
                    )


class A0MirrorsTheBurnWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs = parse_workflow(ROOT / ".github/workflows/burn-a0.yml")["jobs"]
        cls.targets = make_targets(ROOT / "Makefile")
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        cls.defaults = dict(re.findall(r"^(A0_\w+) \?= (\S+)$", makefile, re.M))

    def test_a0_runs_one_target_per_workflow_job(self):
        expected = {f"a0-{job}" for job in self.jobs}
        self.assertEqual(expected, {"a0-stable", "a0-msrv"})
        self.assertEqual(set(self.targets["a0"][0].split()), expected)

    def test_every_step_runs_on_the_toolchain_its_job_uses(self):
        msrv = self.defaults["A0_MSRV"]
        self.assertEqual(self.defaults["A0_STABLE"], "stable")
        for job, data in self.jobs.items():
            recipe = self.targets[f"a0-{job}"][1]
            for line in recipe:
                if line.startswith("cargo "):
                    with self.subTest(line=line):
                        self.assertRegex(line, r"^cargo \+\$\(A0_(STABLE|MSRV)\) ")
            for step in data["steps"]:
                self.assertIsNone(step["block"], "a0 has no multi-line steps to mirror")
                if step["run"] is None:
                    continue
                with self.subTest(job=job, step=step["run"]):
                    wanted = step["run"].replace("+stable ", "+$(A0_STABLE) ").replace(
                        msrv, "$(A0_MSRV)"
                    )
                    self.assertNotIn("+stable", wanted.replace("$(A0_STABLE)", ""))
                    self.assertIn(wanted, recipe)

    def test_the_default_msrv_is_the_one_the_workflow_installs(self):
        installs = [
            s["run"] for s in self.jobs["msrv"]["steps"]
            if s["run"] and s["run"].startswith("rustup toolchain install ")
        ]
        self.assertEqual(installs, [f"rustup toolchain install {self.defaults['A0_MSRV']} --profile minimal"])


if __name__ == "__main__":
    unittest.main()
