"""`make ci-local` and `make a0` must keep running what CI runs.

The documented local checks were a subset of CI: no vendor, notices, codebook or
research-gate check and no feature backend, so a contributor could pass every
one and still get a red CI. And `make a0` ran a bare `cargo`, which the root
`rust-toolchain.toml` resolves to 1.85.0, for a crate that requires 1.95. These
tests read the workflows and the Makefile and fail when a CI step has no
counterpart in the matching target, so the mirror cannot fall behind silently.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

JOB = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")
STEP = re.compile(r"^\s+(?:- )?run: (.+)$")
TARGET = re.compile(r"^([A-Za-z0-9_.-]+):(?!=)(.*)$")

# Steps the Makefile deliberately does not reproduce, each with its reason.
NOT_REPRODUCED = {
    "python -m pip install -e training": (
        "installs into the contributor's environment; the targets set "
        "PYTHONPATH=training/src instead"
    ),
}


def workflow_steps(path: Path) -> dict[str, list[str]]:
    """Every job's `run:` commands, in order. A block (`run: |`) is kept as `|`."""
    jobs: dict[str, list[str]] = {}
    current = None
    in_jobs = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("jobs:"):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        job = JOB.match(line)
        if job:
            current = job.group(1)
            jobs[current] = []
            continue
        step = STEP.match(line)
        if step and current:
            jobs[current].append(step.group(1).strip())
    return jobs


def make_targets(path: Path) -> dict[str, tuple[str, list[str]]]:
    """Each target's prerequisites (continuation lines joined) and recipe lines."""
    lines = path.read_text(encoding="utf-8").replace("\\\n", " ").splitlines()
    targets: dict[str, tuple[str, list[str]]] = {}
    current = None
    for line in lines:
        if line.startswith("\t") and current:
            targets[current][1].append(line.strip())
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


class CiLocalMirrorsCi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs = workflow_steps(ROOT / ".github/workflows/ci.yml")
        cls.targets = make_targets(ROOT / "Makefile")

    def test_the_workflow_parse_sees_every_job(self):
        # A parse that silently found nothing would make every other test pass.
        self.assertGreaterEqual(len(self.jobs), 13, sorted(self.jobs))
        self.assertIn("cargo +stable fmt --all -- --check", self.jobs["quality"])

    def test_every_job_has_a_target_and_ci_local_runs_them_all(self):
        expected = {f"ci-{job}" for job in self.jobs}
        self.assertEqual(expected - set(self.targets), set())
        self.assertEqual(set(self.targets["ci-local"][0].split()), expected)

    def test_every_step_is_reproduced_in_its_jobs_target(self):
        for job, steps in self.jobs.items():
            recipe = self.targets[f"ci-{job}"][1]
            for step in steps:
                if step in NOT_REPRODUCED or step == "|":
                    continue
                with self.subTest(job=job, step=step):
                    wanted = normalize(step)
                    self.assertTrue(
                        any(wanted in line for line in recipe),
                        f"ci-{job} has no line running: {wanted}",
                    )

    def test_steps_that_need_more_than_their_command_keep_it(self):
        # The doc step's warnings are only errors because of its env block.
        self.assertIn(
            'RUSTDOCFLAGS="-D warnings" cargo +stable doc --workspace --no-deps --locked',
            self.targets["ci-quality"][1],
        )
        # The metadata step is a shell block that picks the base commit.
        self.assertIn("|", self.jobs["repository-invariants"])
        self.assertIn(
            "$(PYTHON) scripts/check_component_metadata.py --base $(BASE)",
            self.targets["ci-repository-invariants"][1],
        )


class A0MirrorsTheBurnWorkflow(unittest.TestCase):
    def test_every_burn_a0_step_runs_on_an_explicit_toolchain(self):
        jobs = workflow_steps(ROOT / ".github/workflows/burn-a0.yml")
        recipe = make_targets(ROOT / "Makefile")["a0"][1]
        self.assertTrue(recipe)
        for line in recipe:
            with self.subTest(line=line):
                self.assertTrue(line.startswith("cargo +$(A0_TOOLCHAIN) "), line)
        for job, steps in jobs.items():
            for step in steps:
                if step.startswith("rustup toolchain install"):
                    continue
                with self.subTest(job=job, step=step):
                    wanted = re.sub(r"^cargo \+\S+ ", "cargo +$(A0_TOOLCHAIN) ", step)
                    self.assertIn(wanted, recipe)

    def test_the_default_a0_toolchain_is_the_one_its_msrv_job_installs(self):
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        default = re.search(r"^A0_TOOLCHAIN \?= (\S+)$", makefile, re.M).group(1)
        workflow = (ROOT / ".github/workflows/burn-a0.yml").read_text(encoding="utf-8")
        self.assertIn(f"rustup toolchain install {default} ", workflow)


if __name__ == "__main__":
    unittest.main()
