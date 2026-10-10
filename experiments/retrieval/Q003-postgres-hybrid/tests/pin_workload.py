"""Re-pin the Q003 workload digests: the one scripted way to change them.

Every edit of the workload generator (`bins/ptr-bench/src/experiments/q003/workload.rs`, `params.rs`'s shape) changes the
generated workload and so its SHA-256, which the harness compares with the pin (`workload_digest_mismatches`).
This script runs the harness binary for each declared seed with `--print-workload-digest` (which generates the workload and
prints its digest without touching PostgreSQL or any driver) and writes the digest into the two places that hold the pin:
`WORKLOAD_DIGESTS` in `params.rs` and `workload_digest_seed_<seed>` in `config.toml`. It records nothing under `results/`.

    cargo build -p ptr-bench --features postgres-experiments --locked   # or --release; the digest does not depend on the profile
    python3 experiments/retrieval/Q003-postgres-hybrid/tests/pin_workload.py [path/to/ptr-bench]
    python3 experiments/retrieval/Q003-postgres-hybrid/tests/pin_workload.py --check [path/to/ptr-bench]

`--check` writes nothing and exits 1 when a pin differs from what the binary reports. Re-pin once, after the last
generator edit and before the freeze commit; a reviewer reproduces the pin with `--check` on the frozen commit.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PARAMS = ROOT / "bins/ptr-bench/src/experiments/q003/params.rs"
CONFIG = HERE.parent / "config.toml"
SEEDS = (17, 29, 43, 71, 101)
DIGEST = re.compile(r"[0-9a-f]{64}")


def pin_params(text: str, digests: dict[int, str]) -> str:
    """`params.rs` text with the `(seed, "digest")` entries of `WORKLOAD_DIGESTS` replaced."""
    for seed, digest in digests.items():
        text, count = re.subn(r'(\(\s*%d,\s*)"[0-9a-f]{64}"' % seed, r'\g<1>"%s"' % digest, text)
        if count != 1:
            raise SystemExit(f"params.rs: {count} entries for seed {seed}, expected 1")
    return text


def pin_config(text: str, digests: dict[int, str]) -> str:
    """`config.toml` text with the `workload_digest_seed_<seed>` keys replaced."""
    for seed, digest in digests.items():
        text, count = re.subn(r'(workload_digest_seed_%d = )"[0-9a-f]{64}"' % seed, r'\g<1>"%s"' % digest, text)
        if count != 2:
            raise SystemExit(f"config.toml: {count} keys for seed {seed}, expected 2 ([preregistration] and [pinned_thresholds])")
    return text


def measure(binary: Path) -> dict[int, str]:
    """The workload digest each declared seed reports."""
    digests: dict[int, str] = {}
    for seed in SEEDS:
        run = subprocess.run(
            [str(binary), "postgres-hybrid", "--print-workload-digest", str(seed)], capture_output=True, text=True
        )
        lines = run.stdout.splitlines()
        if not lines or run.returncode != 0:
            raise SystemExit(f"seed {seed}: the harness exited {run.returncode}: {run.stderr[-500:]}")
        digest = json.loads(lines[-1])["workload_digest"]
        if not DIGEST.fullmatch(digest):
            raise SystemExit(f"seed {seed}: {digest!r} is not a SHA-256")
        digests[seed] = digest
    return digests


def default_binary() -> Path:
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    return target / "debug" / "ptr-bench"


def main(argv: list[str]) -> int:
    check = "--check" in argv
    paths = [arg for arg in argv if arg != "--check"]
    binary = Path(paths[0]) if paths else default_binary()
    digests = measure(binary)
    pinned_params = pin_params(PARAMS.read_text(encoding="utf-8"), digests)
    pinned_config = pin_config(CONFIG.read_text(encoding="utf-8"), digests)
    stale = pinned_params != PARAMS.read_text(encoding="utf-8") or pinned_config != CONFIG.read_text(encoding="utf-8")
    if check:
        print("the pins are stale" if stale else "the pins are current")
        return 1 if stale else 0
    PARAMS.write_text(pinned_params, encoding="utf-8")
    CONFIG.write_text(pinned_config, encoding="utf-8")
    print(json.dumps(digests, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
