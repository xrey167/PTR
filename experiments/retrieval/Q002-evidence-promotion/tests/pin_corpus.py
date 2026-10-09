"""Re-pin the Q002 corpus digests: the one scripted way to change them.

Every edit of the world generator (`bins/ptr-bench/src/experiments/q002/world.rs`, `params.rs`'s shape) changes the
generated corpus and so its SHA-256, which the harness compares with the pin (`corpus_selection_digest_mismatches`).
This script runs the harness binary for each declared seed, reads the digest the run reports
(`procedure.corpus_digest`) and writes it into the two places that hold the pin: `CORPUS_DIGESTS` in `params.rs` and
`corpus_digest_seed_<seed>` in `config.toml`. It records nothing under `results/`; the run is a dry measurement.

    cargo build -p ptr-bench --locked          # or --release; the digest does not depend on the profile
    python3 experiments/retrieval/Q002-evidence-promotion/tests/pin_corpus.py [path/to/ptr-bench]
    python3 experiments/retrieval/Q002-evidence-promotion/tests/pin_corpus.py --check [path/to/ptr-bench]

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
PARAMS = ROOT / "bins/ptr-bench/src/experiments/q002/params.rs"
CONFIG = HERE.parent / "config.toml"
SEEDS = (17, 29, 43, 71, 101)
DIGEST = re.compile(r"[0-9a-f]{64}")


def pin_params(text: str, digests: dict[int, str]) -> str:
    """`params.rs` text with the `(seed, "digest")` entries of `CORPUS_DIGESTS` replaced."""
    for seed, digest in digests.items():
        text, count = re.subn(r'(\(\s*%d,\s*)"[0-9a-f]{64}"' % seed, r'\g<1>"%s"' % digest, text)
        if count != 1:
            raise SystemExit(f"params.rs: {count} entries for seed {seed}, expected 1")
    return text


def pin_config(text: str, digests: dict[int, str]) -> str:
    """`config.toml` text with the `corpus_digest_seed_<seed>` keys replaced."""
    for seed, digest in digests.items():
        text, count = re.subn(r'(corpus_digest_seed_%d = )"[0-9a-f]{64}"' % seed, r'\g<1>"%s"' % digest, text)
        if count != 1:
            raise SystemExit(f"config.toml: {count} keys for seed {seed}, expected 1")
    return text


def measure(binary: Path) -> dict[int, str]:
    """The corpus digest each declared seed reports (the harness exits 1 on a stale pin; that is fine here)."""
    digests: dict[int, str] = {}
    for seed in SEEDS:
        run = subprocess.run([str(binary), "evidence-promotion", str(seed)], capture_output=True, text=True)
        lines = run.stdout.splitlines()
        if not lines or run.returncode not in (0, 1):
            raise SystemExit(f"seed {seed}: the harness exited {run.returncode}: {run.stderr[-500:]}")
        digest = json.loads(lines[-1])["procedure"]["corpus_digest"]
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
