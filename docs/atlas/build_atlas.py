#!/usr/bin/env python3
"""Merge extracted structure and verified descriptions into the atlas page.

Regenerate from the repository root:
  python3 docs/atlas/extract_atlas.py . docs/atlas/structure.json
  python3 docs/atlas/build_atlas.py
"""
import datetime
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPOSITORY = os.path.dirname(os.path.dirname(HERE))
OUTPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "atlas.html")

LAYERS = [
    {"id": "entry", "name": "Binaries and harness", "slot": 8,
     "note": "Programs a person runs: daemon, command line, benchmarks, fuzz target.",
     "members": ["ptrd", "ptrctl", "ptr-worker", "ptr-bench", "ptr-fuzz"]},
    {"id": "wires", "name": "Network wires", "slot": 7,
     "note": "Authenticated peer-to-peer protocols.",
     "members": ["ptr-podwire", "ptr-execwire", "ptr-cluster", "ptr-net"]},
    {"id": "runtime", "name": "Runtime and serving", "slot": 6,
     "note": "The orchestrator and what serves it.",
     "members": ["ptr-server", "ptr-runtime", "ptr-exec", "ptr-events", "ptr-model-api", "ptr-router"]},
    {"id": "platform", "name": "Platform", "slot": 5,
     "note": "Database substrate, telemetry, inspection.",
     "members": ["ptr-pg", "ptr-observe", "ptr-inspect"]},
    {"id": "learning", "name": "Learning and agents", "slot": 4,
     "note": "Branches, fast memory, adapter lineage, labels, statistics.",
     "members": ["ptr-branch", "ptr-fastmem", "ptr-lineage", "ptr-labeling", "ptr-analytics"]},
    {"id": "cognitive", "name": "Cognitive layer", "slot": 3,
     "note": "Pods, memory, search, model core.",
     "members": ["ptr-core", "ptr-pods", "ptr-ingress", "ptr-memory", "ptr-search", "ptr-feedback", "ptr-burn-a0"]},
    {"id": "authority", "name": "Authority and state", "slot": 2,
     "note": "Ledger, semantic database, verifiers.",
     "members": ["ptr-ledger", "ptr-state", "ptr-semdb", "ptr-storage", "ptr-verifier", "ptr-security"]},
    {"id": "foundation", "name": "Foundation", "slot": 1,
     "note": "Types, configuration, wire schemas.",
     "members": ["ptr-types", "ptr-config", "ptr-protocol"]},
]

AREAS = {
    "area_a.json": "Authority and persistence",
    "area_b.json": "Runtime and serving",
    "area_c.json": "Pods and wires",
    "area_d.json": "Learning and memory",
    "area_e.json": "Platform and model",
}


def main():
    structure = json.load(open(os.path.join(HERE, "structure.json")))
    packages = structure["packages"]
    by_name = {p["name"]: p for p in packages}
    layer_of = {}
    for layer in LAYERS:
        for member in layer["members"]:
            layer_of[member] = layer["id"]
    missing = [p["name"] for p in packages if p["name"] not in layer_of]
    assert not missing, "unassigned packages: %s" % missing

    flows = []
    problems = []
    for file_name, area in AREAS.items():
        path = os.path.join(HERE, "descriptions", file_name)
        if not os.path.exists(path):
            problems.append("missing %s" % file_name)
            continue
        document = json.load(open(path))
        for name, info in document.get("crates", {}).items():
            package = by_name.get(name)
            if not package:
                problems.append("unknown crate %s in %s" % (name, file_name))
                continue
            package["role"] = info.get("role", "")
            package["summary"] = info.get("summary", "")
            package["maturity"] = info.get("maturity", "")
            for module in package["modules"]:
                described = info.get("modules", {}).get(module["path"])
                if described:
                    module["purpose"] = described.get("purpose", "")
                    module["key_types"] = described.get("key_types", [])
                else:
                    problems.append("no description for %s/%s" % (name, module["path"]))
        for flow in document.get("flows", []):
            lane_ids = {lane["id"] for lane in flow["lanes"]}
            for index, step in enumerate(flow["steps"]):
                for end in ("from", "to"):
                    if step[end] not in lane_ids:
                        problems.append("flow %s step %d: unknown lane %s" % (flow["id"], index + 1, step[end]))
            for lane in flow["lanes"]:
                if lane.get("crate") and lane["crate"] not in by_name:
                    problems.append("flow %s lane %s: unknown crate %s" % (flow["id"], lane["id"], lane["crate"]))
                    lane["crate"] = None
            flow["area"] = area
            flows.append(flow)

    for package in packages:
        package["layer"] = layer_of[package["name"]]
        package.setdefault("role", package.get("description", ""))
        package.setdefault("summary", "")
        package.setdefault("maturity", "")
        for module in package["modules"]:
            module.setdefault("purpose", "")
            if not module["purpose"] and module.get("doc"):
                first = module["doc"].split(". ")[0].strip()
                module["purpose"] = (first[:157] + "...") if len(first) > 160 else first
            module.setdefault("key_types", [])
            module["items"] = [{"kind": i["kind"], "name": i["name"]} for i in module["items"]]
    layers_out = [{k: v for k, v in layer.items() if k != "members"} for layer in LAYERS]

    try:
        commit = subprocess.check_output(["git", "-C", REPOSITORY, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    data = {
        "meta": {"commit": commit, "generated": datetime.date.today().isoformat()},
        "layers": layers_out, "packages": packages, "flows": flows,
    }
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    template = open(os.path.join(HERE, "atlas_template.html")).read()
    assert "/*__DATA__*/" in template
    open(OUTPUT_PATH, "w").write(template.replace("/*__DATA__*/", payload))
    print("wrote %s (%d bytes); %d packages, %d flows" % (OUTPUT_PATH, os.path.getsize(OUTPUT_PATH), len(packages), len(flows)))
    for problem in problems:
        print("PROBLEM:", problem)


if __name__ == "__main__":
    main()
