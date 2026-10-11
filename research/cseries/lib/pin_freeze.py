"""Step 3 of the freeze procedure of a stage-A experiment (C004, C007, C008): pin the two digests of `config.toml`'s [preregistration] table, write
the manifest digests and set `status = "prepared"` in `experiment.toml` and `experiments/registry.toml`.

    python3 research/cseries/lib/pin_freeze.py C007           # writes the three files
    python3 research/cseries/lib/pin_freeze.py C007 --dry-run  # prints what it would change, writes nothing

Standard library only (it reuses scripts/experiment_records.py, the code the runner itself uses to verify the digests). It refuses a tree in which the
placeholders are already replaced or the experiment is not `planned`, so it cannot be run twice. Run it from a clean pre-freeze commit (see section 8
of the experiment's PREREGISTRATION.md). It is not a freeze by itself: the commit, the tag and everything after are the head's."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment_records as records  # noqa: E402

DIRS = {
    "C004": "experiments/consciousness/C004-recurrent-integration",
    "C007": "experiments/consciousness/C007-metacognitive-self-model-cpu",
    "C008": "experiments/consciousness/C008-attention-schema-routing-cpu",
}
PLACEHOLDER = "must-be-pinned-before-prepared"


def pinned_texts(eid: str, root: Path = ROOT) -> dict[str, str]:
    """The new contents of config.toml, experiment.toml and experiments/registry.toml, by path relative to `root`. Raises ValueError when the tree is not
    a planned, unpinned one."""
    d = DIRS[eid] + "/"
    config = (root / d / "config.toml").read_text(encoding="utf-8")
    if config.count(PLACEHOLDER) != 2:
        raise ValueError(f"{d}config.toml must hold exactly two placeholders, it holds {config.count(PLACEHOLDER)}")
    config = config.replace(f'preregistration_text_sha256 = "{PLACEHOLDER}"', f'preregistration_text_sha256 = "{records.preregistered_file_digest(root / d / "PREREGISTRATION.md")}"')
    config = config.replace(f'criteria_sha256 = "{PLACEHOLDER}"', f'criteria_sha256 = "{records.preregistered_file_digest(root / d / "criteria.toml")}"')
    if PLACEHOLDER in config:
        raise ValueError("a placeholder is left in config.toml")
    table = tomllib.loads(config)["preregistration"]
    entry = tomllib.loads((root / "experiments/preregistration.toml").read_text(encoding="utf-8"))["experiment"][eid]
    manifest = (root / d / "experiment.toml").read_text(encoding="utf-8")
    if 'status = "planned"' not in manifest or "preregistration_sha256" in manifest:
        raise ValueError("experiment.toml must be planned and carry no digests yet")
    manifest = manifest.replace('status = "planned"', 'status = "prepared"', 1).rstrip("\n")
    manifest += f'\npreregistration_sha256 = "{records.preregistration_digest(table)}"\npreregistration_rules_sha256 = "{records.canonical_digest(entry)}"\n'
    registry = (root / "experiments/registry.toml").read_text(encoding="utf-8")
    new_registry, count = re.subn(rf'(id = "{eid}"\npath = "[^"]+"\nstatus = )"planned"', r'\1"prepared"', registry)
    if count != 1:
        raise ValueError(f"experiments/registry.toml must list {eid} once as planned")
    return {d + "config.toml": config, d + "experiment.toml": manifest, "experiments/registry.toml": new_registry}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=sorted(DIRS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        texts = pinned_texts(args.experiment)
    except (ValueError, KeyError) as error:
        print(f"refusing: {error}", file=sys.stderr)
        return 2
    for rel, text in texts.items():
        if args.dry_run:
            print(f"would write {rel} ({len(text)} bytes)")
        else:
            (ROOT / rel).write_text(text, encoding="utf-8")
            print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
