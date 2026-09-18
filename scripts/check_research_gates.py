from __future__ import annotations

import tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))

def is_placeholder(value: str) -> bool:
    value=value.strip().lower()
    return not value or "must-be-pinned" in value or value in {"unconfigured","none"}

def main() -> int:
    errors=[]
    experiments={}
    registry=load(ROOT/"experiments/registry.toml")
    for item in registry.get("experiment",[]):
        manifest=load(ROOT/"experiments"/item["path"]/"experiment.toml")
        experiments[item["id"]]=manifest
        status=manifest.get("status")
        if status=="completed":
            results=ROOT/"experiments"/item["path"]/manifest.get("results_dir","results")
            for artifact in manifest.get("required_artifacts",[]):
                if not (results/artifact).exists():
                    errors.append(f'{item["id"]}: completed experiment missing {artifact}')

    plain=load(ROOT/"research/baselines/plain_model/config.toml")
    for exp_id in ["M001","M002","M003","M004","M005"]:
        if experiments.get(exp_id,{}).get("status") in {"running","completed"}:
            model=plain.get("model",{})
            if is_placeholder(str(model.get("backend",""))) or is_placeholder(str(model.get("model",""))):
                errors.append(f"{exp_id}: matched plain-model baseline is not pinned")

    if experiments.get("E002",{}).get("status") in {"running","completed"}:
        rag=load(ROOT/"research/baselines/strong_rag/config.toml")
        required=[
            rag.get("dense",{}).get("revision",""),
            rag.get("reranker",{}).get("revision",""),
            rag.get("answer",{}).get("revision",""),
        ]
        if any(is_placeholder(str(value)) for value in required):
            errors.append("E002: strong RAG model/generator revisions are not pinned")
        if str(rag.get("status","")).startswith("blocked-"):
            errors.append("E002: strong RAG baseline is still blocked")

    if errors:
        print("\n".join("ERROR: "+error for error in errors))
        return 1
    print("OK: research execution gates satisfied")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
