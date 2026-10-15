#!/usr/bin/env python3
"""Build and run the Pod worker tests that need root (doc 42 stage S1).

The supervisor tests give real child processes a user id of their own, a cgroup-v2 group, a
network namespace and a seccomp allowlist, and reap them. Those mechanisms need root, but a
CI runner is an ordinary user, so the test binaries are built as that user and only the
finished binaries run under `sudo -E`. (`sudo cargo test` would build the whole workspace as
root, leave root-owned files in `target/` and need the toolchain on root's PATH.)

Each binary runs up to twice:

1. under `sudo -E` (or directly when already root): strict. A test that finds a mechanism
   missing fails under `CI=true`, unless `PTR_WORKER_UNPRIVILEGED_CI=1` is set, because the
   runner is supposed to provide root;
2. as the invoking user with `PTR_WORKER_UNPRIVILEGED_CI=1`, when that user is not root: the
   unprivileged path (seccomp only, tensors by `Bytes`, no `CudaIpc`) must work as well.

A runner that cannot give root (a container without namespaces, for example the GitLab docker
executor) sets `PTR_WORKER_UNPRIVILEGED_CI=1` itself; the privileged tests then skip with a
printed reason, and the unprivileged path is what runs.

usage: run_pod_worker_tests.py [--toolchain stable|1.99.0]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

# (package, features, integration test target)
TARGETS = (
    ("ptr-fabric", "worker-supervisor", "worker_supervisor"),
    ("ptr-worker", "data-plane,fault-injection", "pod_data_plane"),
    # fault-injection: the guardian-loss tests kill the guardian and fail worker reaps on purpose.
    ("ptr-worker", "data-plane,fault-injection", "staging_guardian"),
    ("ptr-worker", "data-plane,fault-injection", "cuda_ipc_flow"),
)


def build(toolchain: str, package: str, features: str, target: str) -> str:
    """Build one integration test and return the path of its executable."""
    command = [
        "cargo",
        f"+{toolchain}",
        "test",
        "--no-run",
        "--locked",
        "--message-format=json",
        "-p",
        package,
        "--features",
        features,
        "--test",
        target,
    ]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, text=True, encoding="utf-8", check=False
    )
    if result.returncode != 0:
        raise SystemExit(f"{' '.join(command)} failed with {result.returncode}")
    for line in result.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        artifact = message.get("target", {})
        if (
            message.get("reason") == "compiler-artifact"
            and message.get("executable")
            and artifact.get("name") == target
            and "test" in artifact.get("kind", [])
        ):
            return message["executable"]
    raise SystemExit(f"no test executable named {target} in the cargo output")


def strict_ci() -> bool:
    """CI declares root available unless the runner says otherwise."""
    return os.environ.get("CI") == "true" and os.environ.get("PTR_WORKER_UNPRIVILEGED_CI") != "1"


def can_sudo() -> bool:
    sudo = shutil.which("sudo")
    if sudo is None:
        return False
    return subprocess.run([sudo, "-n", "true"], check=False).returncode == 0


def run(command: list[str], extra_env: dict[str, str] | None = None) -> int:
    env = dict(os.environ)
    env.update(extra_env or {})
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, env=env, check=False).returncode


def remove_scratch(path: str) -> None:
    """Delete the scratch directory; files a root run left in it need sudo."""
    shutil.rmtree(path, ignore_errors=True)
    if os.path.exists(path) and shutil.which("sudo"):
        subprocess.run(["sudo", "-n", "rm", "-rf", path], check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--toolchain", default="stable")
    args = parser.parse_args()

    # The tests copy the worker binary into a directory of their own under TMPDIR; a scratch
    # TMPDIR that this script removes afterwards keeps /tmp clean. Mode 0711: the worker runs
    # under a user id of its own and must be able to enter it, but not list it.
    scratch = tempfile.mkdtemp(prefix="ptr-pod-worker-")
    os.chmod(scratch, 0o711)
    os.environ["TMPDIR"] = scratch
    try:
        return run_all(args.toolchain)
    finally:
        remove_scratch(scratch)


def run_all(toolchain: str) -> int:
    failures: list[str] = []
    for package, features, target in TARGETS:
        executable = build(toolchain, package, features, target)
        if os.geteuid() == 0:
            if run([executable]) != 0:
                failures.append(f"{target} (as root)")
            continue
        if can_sudo():
            if run(["sudo", "-n", "-E", executable]) != 0:
                failures.append(f"{target} (under sudo)")
        elif strict_ci():
            failures.append(f"{target}: CI=true needs root and passwordless sudo is missing")
        else:
            print(f"NOTE: no passwordless sudo; the root run of {target} is skipped", flush=True)
        if run([executable], {"PTR_WORKER_UNPRIVILEGED_CI": "1"}) != 0:
            failures.append(f"{target} (unprivileged)")
    if failures:
        print("FAILED:", "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
