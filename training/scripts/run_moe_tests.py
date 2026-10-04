"""Run the `trl-moe` unit tests from a checkout without installing the package.

Used by the `trl-peft` candidate in evaluations/components/training-backend/candidates.toml
(executed from the repo root, argv-split, no shell).

Default: torch-dependent tests skip themselves when the `moe` extra is absent (plain unit run).
`--require-backend`: fail with a non-zero exit if the extra is missing or any test is skipped,
so a run that never exercised the torch/TRL backend cannot be reported as success.
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--require-backend", action="store_true")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(TRAINING / "src"))
    if args.require_backend:
        from ptr_training.moe import pipeline

        try:
            pipeline.require_extra()
        except pipeline.MissingExtraError as error:
            print(f"--require-backend: {error}", file=sys.stderr)
            return 2

    suite = unittest.defaultTestLoader.discover(
        str(TRAINING / "tests"), pattern="test_moe_*.py"
    )
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        return 1
    if args.require_backend and result.skipped:
        print(
            f"--require-backend: {len(result.skipped)} test(s) skipped; the backend was not "
            "fully exercised",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
