#!/usr/bin/env python3
"""Print the seeds of an experiment's config.toml, space separated.

The first `seeds` key found (depth first, document order of the parsed tables) is used, as in the
[preregistration] table and the manifest. Exit 4 when there is none.

    python3 scripts/freeze/config_seeds.py experiments/<area>/<ID>-<name>/config.toml
"""

import sys
import tomllib


def find_seeds(table):
    for key, value in table.items():
        if key == "seeds" and isinstance(value, list) and value:
            return value
        if isinstance(value, dict):
            found = find_seeds(value)
            if found:
                return found
    return None


def main(argv):
    if len(argv) != 1:
        print("usage: config_seeds.py <config.toml>", file=sys.stderr)
        return 2
    with open(argv[0], "rb") as handle:
        seeds = find_seeds(tomllib.load(handle))
    if not seeds or not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
        print(f"config_seeds: no integer `seeds` list in {argv[0]}", file=sys.stderr)
        return 4
    print(" ".join(str(seed) for seed in seeds))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
