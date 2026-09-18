from __future__ import annotations
import argparse, datetime as dt, json, urllib.parse, uuid
from pathlib import Path

def purl(name: str, version: str) -> str:
    return f"pkg:cargo/{urllib.parse.quote(name, safe='')}@{urllib.parse.quote(version, safe='')}"

def build(metadata: dict) -> dict:
    packages = metadata.get("packages", [])
    ref_by_id = {p["id"]: purl(p["name"], p["version"]) for p in packages}
    components = []
    for pkg in sorted(packages, key=lambda p: (p["name"], p["version"], p["id"])):
        component = {
            "type": "library",
            "bom-ref": ref_by_id[pkg["id"]],
            "name": pkg["name"],
            "version": pkg["version"],
            "purl": ref_by_id[pkg["id"]],
        }
        if pkg.get("description"):
            component["description"] = pkg["description"]
        if pkg.get("license"):
            component["licenses"] = [{"expression": pkg["license"]}]
        if pkg.get("repository"):
            component["externalReferences"] = [
                {"type": "vcs", "url": pkg["repository"]}
            ]
        components.append(component)

    nodes = (metadata.get("resolve") or {}).get("nodes") or []
    dependencies = []
    for node in sorted(nodes, key=lambda n: n["id"]):
        source_ref = ref_by_id.get(node["id"])
        if source_ref is None:
            continue
        depends_on = sorted(
            ref_by_id[dep]
            for dep in node.get("dependencies", [])
            if dep in ref_by_id
        )
        dependencies.append({"ref": source_ref, "dependsOn": depends_on})

    workspace_members = [
        ref_by_id[item]
        for item in metadata.get("workspace_members", [])
        if item in ref_by_id
    ]
    root_name = "PTR"
    root_version = "0.1.0"
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, 'https://github.com/xrey167/PTR')}",
        "version": 1,
        "metadata": {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "component": {
                "type": "application",
                "name": root_name,
                "version": root_version,
                "externalReferences": [
                    {"type": "vcs", "url": "https://github.com/xrey167/PTR"}
                ],
                "properties": [
                    {"name": "ptr:workspace-member", "value": member}
                    for member in workspace_members
                ],
            },
        },
        "components": components,
        "dependencies": dependencies,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("metadata", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args=ap.parse_args()
    metadata=json.loads(args.metadata.read_text(encoding="utf-8"))
    args.out.write_text(json.dumps(build(metadata),indent=2)+"\n",encoding="utf-8")
    print(args.out)

if __name__=="__main__":
    main()
