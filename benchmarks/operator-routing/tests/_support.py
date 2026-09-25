"""Shared helpers for the operator-routing tests.

generator.py and score.py are loaded from their files under distinct module names,
so the tests never let one of them see the other through sys.modules.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parents[1]
ROOT = SUITE.parents[1]
DATA_DIR = ROOT / "datasets/generated/operator_routing_v1"
LOCK_PATH = SUITE / "splits.lock.json"
REFERENCES_PATH = ROOT / "research/falsification/A0-ablations-v1/references.json"
SCHEMA_PATH = ROOT / "datasets/schemas/operator_route.schema.json"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gen = _load("operator_routing_generator", SUITE / "generator.py")
score = _load("operator_routing_score", SUITE / "score.py")


def data_on_disk() -> bool:
    return all((DATA_DIR / f"{s}.{k}").is_file() for s in gen.SPLITS for k in ("tsv", "jsonl"))


def lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


_PREFIX_CACHE: dict[tuple[str, int], tuple[bytes, bytes, str]] = {}


def prefix(split: str, n: int) -> tuple[bytes, bytes, str]:
    """The first n examples of a split, rendered in memory (cached per process)."""
    key = (split, n)
    if key not in _PREFIX_CACHE:
        _PREFIX_CACHE[key] = gen.render_split(split, n)
    return _PREFIX_CACHE[key]


def score_items_from_tsv(split: str, tsv: bytes) -> list:
    items = []
    for line in tsv.decode("utf-8").splitlines(keepends=True):
        item = score.parse_tsv_line(line, split)
        score.derive(item)
        items.append(item)
    return items


# --------------------------------------------------------- a small JSON Schema check

_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}
_SUPPORTED = {
    "$schema", "description", "type", "required", "properties", "additionalProperties",
    "enum", "minimum", "maximum", "minItems", "maxItems", "items", "minLength",
    "pattern", "uniqueItems",
}


def schema_errors(schema: dict, value, path: str = "$") -> list[str]:
    """The subset of JSON Schema 2020-12 that operator_route.schema.json uses.

    An unsupported keyword is itself an error, so the schema cannot grow a
    constraint this check silently ignores.
    """
    errors = []
    unknown = set(schema) - _SUPPORTED
    if unknown:
        return [f"{path}: unsupported schema keywords {sorted(unknown)}"]
    if "type" in schema and not _TYPES[schema["type"]](value):
        return [f"{path}: expected {schema['type']}, found {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum")
    if "minimum" in schema and _TYPES["number"](value) and value < schema["minimum"]:
        errors.append(f"{path}: {value} < minimum {schema['minimum']}")
    if "maximum" in schema and _TYPES["number"](value) and value > schema["maximum"]:
        errors.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match {schema['pattern']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if schema.get("uniqueItems") and len({json.dumps(v, sort_keys=True) for v in value}) != len(value):
            errors.append(f"{path}: items are not unique")
        if "items" in schema:
            for index, item in enumerate(value):
                errors.extend(schema_errors(schema["items"], item, f"{path}[{index}]"))
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required {key!r}")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                errors.extend(schema_errors(properties[key], item, f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: additional property {key!r}")
    return errors
