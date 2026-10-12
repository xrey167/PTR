#!/usr/bin/env python3
"""Compare two pilot records recursively with timing keys stripped.

    python3 scripts/freeze/compare_pilot.py REFERENCE.json RERUN.json --timing-key elapsed_ms ...

A timing key is removed at every depth of both documents before comparing. Exit 0 equal, 1 different
(the first differing paths are printed), 2 unreadable input.
"""

import argparse
import json
import sys


def strip(value, timing):
    if isinstance(value, dict):
        return {key: strip(item, timing) for key, item in value.items() if key not in timing}
    if isinstance(value, list):
        return [strip(item, timing) for item in value]
    return value


def same(a, b):
    """Equality that keeps types apart: True != 1, 1 != 1.0, at every depth."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def differences(a, b, path="$", limit=8):
    found = []
    if type(a) is not type(b):
        found.append(f"{path} differs in type ({type(a).__name__} vs {type(b).__name__})")
    elif isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                found.append(f"{path}.{key} only in {'rerun' if key in b else 'reference'}")
            else:
                found.extend(differences(a[key], b[key], f"{path}.{key}", limit))
    elif isinstance(a, list) and len(a) == len(b):
        for index, (x, y) in enumerate(zip(a, b)):
            found.extend(differences(x, y, f"{path}[{index}]", limit))
    elif not same(a, b):
        found.append(f"{path} differs")
    return found[:limit]


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("rerun")
    parser.add_argument("--timing-key", action="append", default=[], dest="timing")
    args = parser.parse_args(argv)
    try:
        with open(args.reference, encoding="utf-8") as handle:
            reference = json.load(handle)
        with open(args.rerun, encoding="utf-8") as handle:
            rerun = json.load(handle)
    except (OSError, ValueError) as error:
        print(f"compare_pilot: unreadable input: {error}", file=sys.stderr)
        return 2
    timing = set(args.timing)
    a, b = strip(reference, timing), strip(rerun, timing)
    if not same(a, b):
        print("pilot outcome DIFFERS:", *differences(a, b), sep="\n  ")
        return 1
    size = len(a) if isinstance(a, (dict, list)) else 1
    print(f"pilot outcome equal on {size} top-level entries (timing keys stripped: {sorted(timing)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
