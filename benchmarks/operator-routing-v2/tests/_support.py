from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parents[1]
ROOT = SUITE.parents[1]
LOCK_PATH = SUITE / "splits.lock.json"
SCHEMA_PATH = ROOT / "datasets/schemas/operator_route_v2.schema.json"


def load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gen = load("operator_routing_v2_generator", SUITE / "generator.py")
score = load("operator_routing_v2_score", SUITE / "score.py")


def lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))
