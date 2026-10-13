#!/usr/bin/env python3
"""Print the confirmatory seeds of an experiment's config.toml, space separated.

The seeds are read from `[preregistration].seeds` only. Other tables may carry their own `seeds`
lists (for example `[pinned_thresholds]`, pilots or tuning), and those are never confirmatory.
Exit 4 when `[preregistration].seeds` is missing, empty or not a list of integers.

    python3 scripts/freeze/config_seeds.py experiments/<area>/<ID>-<name>/config.toml
"""

import sys
import tomllib


def preregistered_seeds(config):
    table = config.get("preregistration")
    if not isinstance(table, dict):
        return None
    seeds = table.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        return None
    return seeds


def main(argv):
    if len(argv) != 1:
        print("usage: config_seeds.py <config.toml>", file=sys.stderr)
        return 2
    with open(argv[0], "rb") as handle:
        seeds = preregistered_seeds(tomllib.load(handle))
    if not seeds or not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
        print(
            f"config_seeds: no integer [preregistration].seeds list in {argv[0]}", file=sys.stderr
        )
        return 4
    print(" ".join(str(seed) for seed in seeds))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
