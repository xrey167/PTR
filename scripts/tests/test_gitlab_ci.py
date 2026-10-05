"""`.gitlab-ci.yml` must stay a thin layer over the Makefile's `ci-*` targets.

Every job in `.github/workflows/ci.yml` needs a GitLab job of the same name that runs
`make ci-<job>`, and every `make <target>` the GitLab file names must exist. The core
checks are stdlib only (CI may not have PyYAML): the GitHub workflow is read with
test_ci_local's parser and the GitLab file with a small line-based reader of top-level
keys. One extra test loads the file with PyYAML when it is importable.
"""

import importlib.util
import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Top-level GitLab keywords that are not jobs.
RESERVED = {
    "stages", "variables", "workflow", "default", "include", "cache", "image", "services",
    "before_script", "after_script", "spec", "inherit",
}

TOP_KEY = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9_.-]*):(?:\s|$)")
MAKE_CALL = re.compile(r"(?<![\w-])make\s+(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*([A-Za-z][A-Za-z0-9_.-]*)")


def _load_ci_local():
    """Load test_ci_local by path so its parsers are reused without a package import."""
    path = ROOT / "scripts" / "tests" / "test_ci_local.py"
    spec = importlib.util.spec_from_file_location("ptr_test_ci_local", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CI_LOCAL = _load_ci_local()


def github_jobs(path: Path) -> list[str]:
    """Job names of a GitHub workflow, read the way test_ci_local reads them."""
    return list(CI_LOCAL.parse_workflow(path)["jobs"])


def gitlab_jobs(path: Path) -> dict[str, str]:
    """{job: its text} for every top-level key that is neither hidden nor reserved."""
    jobs: dict[str, list[str]] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = TOP_KEY.match(line)
        if match:
            name = match.group(1)
            current = None if name.startswith(".") or name in RESERVED else name
            if current:
                jobs[current] = []
        if current:
            jobs[current].append(line)
    return {name: "\n".join(lines) for name, lines in jobs.items()}


def code_lines(text: str) -> list[str]:
    """Lines with whole-line and trailing ` # ...` comments removed."""
    out = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        out.append(re.sub(r"\s#.*$", "", line))
    return out


def makes_target(text: str, target: str) -> bool:
    """Whether the job text runs `make <target>` exactly (not a longer target name)."""
    return target in MAKE_CALL.findall("\n".join(code_lines(text)))


def parity_problems(ci_path: Path, gitlab_path: Path) -> list[str]:
    """Problems with GitLab parity: missing jobs and jobs not running `make ci-<job>`."""
    jobs = gitlab_jobs(gitlab_path)
    problems = []
    for job in github_jobs(ci_path):
        if job not in jobs:
            problems.append(f"missing GitLab job: {job}")
        elif not makes_target(jobs[job], f"ci-{job}"):
            problems.append(f"GitLab job {job} does not run `make ci-{job}`")
    return problems


def unknown_make_targets(gitlab_path: Path, makefile_path: Path) -> list[str]:
    """`make <target>` names in the GitLab file that the Makefile does not define."""
    targets = set(CI_LOCAL.make_targets(makefile_path))
    mentioned = set(MAKE_CALL.findall("\n".join(code_lines(gitlab_path.read_text(encoding="utf-8")))))
    return sorted(mentioned - targets)


def remove_job(text: str, name: str) -> str:
    """The file text without the top-level job `name`."""
    out, skipping = [], False
    for line in text.splitlines():
        match = TOP_KEY.match(line)
        if match:
            skipping = match.group(1) == name
        if not skipping:
            out.append(line)
    return "\n".join(out) + "\n"


CI = ROOT / ".github/workflows/ci.yml"
GITLAB = ROOT / ".gitlab-ci.yml"
MAKEFILE = ROOT / "Makefile"


class GitLabMirrorsGitHub(unittest.TestCase):
    def test_the_readers_see_real_jobs(self):
        """A parse that found nothing would make every other test pass."""
        self.assertGreaterEqual(len(github_jobs(CI)), 13)
        jobs = gitlab_jobs(GITLAB)
        self.assertGreaterEqual(len(jobs), 17)
        for hidden in (".rust", ".python", "stages", "workflow", "default"):
            self.assertNotIn(hidden, jobs)

    def test_every_github_job_runs_its_make_target_on_gitlab(self):
        """Each ci.yml job has a same-named GitLab job running `make ci-<job>`."""
        self.assertEqual(parity_problems(CI, GITLAB), [])

    def test_every_make_target_named_exists(self):
        """No GitLab job calls a Makefile target that is not defined."""
        self.assertEqual(unknown_make_targets(GITLAB, MAKEFILE), [])
        self.assertIn("a0", MAKE_CALL.findall("script:\n - make a0"))

    def test_extra_jobs(self):
        """burn-a0, security, gpu-smoke and py-bindings exist and run their targets."""
        jobs = gitlab_jobs(GITLAB)
        for name in ("burn-a0", "security", "gpu-smoke", "py-bindings"):
            self.assertIn(name, jobs)
        self.assertTrue(makes_target(jobs["burn-a0"], "a0"))
        self.assertTrue(makes_target(jobs["gpu-smoke"], "gpu-doctor"))
        self.assertTrue(makes_target(jobs["gpu-smoke"], "smoke-gpu"))
        self.assertTrue(makes_target(jobs["py-bindings"], "py-bindings"))
        self.assertIn("cargo-audit", jobs["security"])
        self.assertIn("cargo-deny", jobs["security"])

    def test_gpu_smoke_is_manual_and_on_the_gpu_runner(self):
        """The GPU job needs the self-hosted runner tag and never runs automatically."""
        text = gitlab_jobs(GITLAB)["gpu-smoke"]
        code = "\n".join(code_lines(text))
        self.assertRegex(code, r"tags:\s*\n\s*-\s*ptr-gpu")
        self.assertIn("*rules-manual", code)
        manual = re.search(r"^\.rules-manual:.*?(?=^\S)", GITLAB.read_text(encoding="utf-8"),
                           re.M | re.S)
        self.assertIsNotNone(manual)
        self.assertIn("when: manual", manual.group(0))
        self.assertNotIn("when: on_success", manual.group(0))

    def test_postgres_service_and_dsns(self):
        """state-postgres has the pgvector service under the `postgres` alias and its DSNs."""
        text = gitlab_jobs(GITLAB)["state-postgres"]
        self.assertIn("pgvector/pgvector:pg18", text)
        self.assertIn("alias: postgres", text)
        for name in ("PTR_PG_TEST_DSN", "PTR_PG_EXPERIMENT_DSN"):
            self.assertIn(
                f"{name}: host=postgres port=5432 user=postgres password=postgres dbname=postgres",
                text,
            )

    def test_repository_invariants_has_history(self):
        """The metadata and research-gate checks read history, so the clone is full."""
        self.assertRegex(gitlab_jobs(GITLAB)["repository-invariants"], r'GIT_DEPTH:\s*"0"')


class ParityCheckCatchesDrift(unittest.TestCase):
    def test_a_removed_job_is_reported(self):
        """A copy of .gitlab-ci.yml without one job fails the parity check."""
        text = GITLAB.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / ".gitlab-ci.yml"
            copy.write_text(remove_job(text, "pod-wire"), encoding="utf-8")
            self.assertEqual(parity_problems(CI, copy), ["missing GitLab job: pod-wire"])

    def test_a_job_with_the_wrong_target_is_reported(self):
        """A job that exists but runs another target (or a longer name) is reported."""
        text = GITLAB.read_text(encoding="utf-8").replace(
            "make ci-state-turso", "make ci-state-turso-extra"
        )
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / ".gitlab-ci.yml"
            copy.write_text(text, encoding="utf-8")
            self.assertEqual(
                parity_problems(CI, copy),
                ["GitLab job state-turso does not run `make ci-state-turso`"],
            )
            self.assertEqual(unknown_make_targets(copy, MAKEFILE), ["ci-state-turso-extra"])

    def test_commented_make_calls_do_not_count(self):
        """A `make ci-x` that only appears in a comment is not a command."""
        self.assertFalse(makes_target("script:\n  # make ci-x\n  - echo hi  # make ci-x", "ci-x"))


class GitLabFileIsValidYaml(unittest.TestCase):
    def test_every_job_has_a_script_and_a_declared_stage(self):
        """Full load with PyYAML: script present, stage among `stages`."""
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML is not installed")
        data = yaml.safe_load(GITLAB.read_text(encoding="utf-8"))
        stages = data["stages"]
        jobs = {k: v for k, v in data.items() if not k.startswith(".") and k not in RESERVED}
        self.assertEqual(set(jobs), set(gitlab_jobs(GITLAB)))
        for name, job in jobs.items():
            with self.subTest(job=name):
                self.assertTrue(job.get("script"), "no script")
                self.assertIn(job.get("stage"), stages)
        self.assertEqual(jobs["gpu-smoke"]["tags"], ["ptr-gpu"])
        matrix = jobs["python-training"]["parallel"]["matrix"]
        self.assertEqual(matrix, [{"PYTHON_VERSION": ["3.11", "3.13"]}])


if __name__ == "__main__":
    unittest.main()
