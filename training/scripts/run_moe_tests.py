"""Run the `trl-moe` unit tests from a checkout without installing the package.

Used by the `trl-peft` candidate in evaluations/components/training-backend/candidates.toml
(executed from the repo root, argv-split, no shell). Torch-dependent tests skip themselves
when the `moe` extra is absent.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]


def main() -> int:
    sys.path.insert(0, str(TRAINING / "src"))
    suite = unittest.defaultTestLoader.discover(
        str(TRAINING / "tests"), pattern="test_moe_*.py"
    )
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
