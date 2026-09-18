from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- PTR:STATUS:BEGIN -->"
END = "<!-- PTR:STATUS:END -->"

def load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))

def registries():
    exp_raw = load_toml(ROOT / "experiments/registry.toml")
    eval_raw = load_toml(ROOT / "evaluations/registry.toml")
    exps = {x["id"]: x for x in exp_raw.get("experiment", [])}
    evals = {x["id"]: x for x in eval_raw.get("component", [])}
    return exps, evals

def bullet(items):
    return "\n".join(f"- {x}" for x in items) if items else "- None recorded."

def render_section(meta: dict, exps: dict, evals: dict) -> str:
    exp_lines = []
    for exp_id in meta.get("experiments", []):
        e = exps[exp_id]
        exp_lines.append(
            f'- [{exp_id}](../../experiments/{e["path"]}/README.md) — `{e["status"]}`'
        )
    eval_lines = []
    for eval_id in meta.get("evaluations", []):
        e = evals[eval_id]
        eval_lines.append(
            f'- [{eval_id}](../../evaluations/components/{eval_id}/README.md) — `{e["status"]}`'
        )
    decision_lines = [
        f"- [{d}](../../research/decisions/{d})" for d in meta.get("decisions", [])
    ]
    return f"""{{BEGIN}}
## Current implementation status

> **Generated section.** Source of truth: [`component.toml`](component.toml). Run `python3 scripts/update_component_docs.py --write` after editing metadata. Do not hand-edit inside this block.

**Maturity:** `{meta["maturity"]}`  
**Last reviewed:** {meta["last_reviewed"]}

### Implemented now

{bullet(meta.get("implemented", []))}

### Missing for the target architecture

{bullet(meta.get("missing", []))}

### Next milestones

{bullet(meta.get("next", []))}

### Linked experiments

{bullet(exp_lines)}

### Technology evaluations

{bullet(eval_lines)}

### Decision records

{bullet(decision_lines)}

### Current automated checks

{bullet(meta.get("checks", []))}

{{END}}
"""

def update_readme(path: Path, section: str) -> str:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if pattern.search(text):
        return pattern.sub(section.strip(), text)
    marker = "## Position in PTR"
    if marker not in text:
        raise RuntimeError(f"{path.relative_to(ROOT)} has no insertion marker")
    return text.replace(marker, section.strip() + "\n\n" + marker, 1)

def dashboard(metas: list[dict], exps: dict, evals: dict) -> str:
    rows = []
    maturity_counts: dict[str, int] = {}
    for m in metas:
        maturity_counts[m["maturity"]] = maturity_counts.get(m["maturity"], 0) + 1
        exp_status = ", ".join(
            f'{eid}:{exps[eid]["status"]}' for eid in m.get("experiments", [])
        ) or "—"
        eval_status = ", ".join(
            f'{eid}:{evals[eid]["status"]}' for eid in m.get("evaluations", [])
        ) or "—"
        rows.append(
            f'| [{m["id"]}](../../crates/{m["id"]}/README.md) | '
            f'`{m["maturity"]}` | {len(m.get("implemented", []))} | '
            f'{len(m.get("missing", []))} | {exp_status} | {eval_status} |'
        )
    counts = ", ".join(f"`{k}`: {v}" for k, v in sorted(maturity_counts.items()))
    return f"""# PTR Component Implementation Status

> Generated from every `crates/*/component.toml`. Do not hand-edit.  
> Refresh with `python3 scripts/update_component_docs.py --write`.

**Component count:** {len(metas)}  
**Maturity distribution:** {counts}

| Component | Maturity | Implemented items | Missing items | Experiments | Evaluations |
|---|---:|---:|---:|---|---|
{chr(10).join(rows)}

## Meaning of maturity labels

- `foundation` — small stable domain core already used broadly.
- `prototype` — executable behavior exists, but major target semantics are still missing.
- `scaffold` — contracts/data structures exist; core production implementation is not present.
- `research-scaffold` — architecture hypothesis is represented, but the trainable system is not implemented/proven.

The labels describe implementation maturity, not scientific novelty or production readiness.
"""

def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    exps, evals = registries()
    metas = []
    errors = []
    for meta_path in sorted((ROOT / "crates").glob("*/component.toml")):
        meta = load_toml(meta_path)
        crate = meta_path.parent.name
        if meta.get("id") != crate:
            errors.append(f"{meta_path.relative_to(ROOT)} id must equal {crate}")
            continue
        for eid in meta.get("experiments", []):
            if eid not in exps:
                errors.append(f"{crate}: unknown experiment {eid}")
        for eid in meta.get("evaluations", []):
            if eid not in evals:
                errors.append(f"{crate}: unknown evaluation {eid}")
        for decision in meta.get("decisions", []):
            if not (ROOT / "research/decisions" / decision).exists():
                errors.append(f"{crate}: missing decision {decision}")
        metas.append(meta)

    if errors:
        print("\n".join("ERROR: " + e for e in errors))
        return 1

    stale = []
    for meta in metas:
        readme = ROOT / "crates" / meta["id"] / "README.md"
        expected = update_readme(readme, render_section(meta, exps, evals))
        current = readme.read_text(encoding="utf-8")
        if current != expected:
            if args.write:
                readme.write_text(expected, encoding="utf-8")
            else:
                stale.append(str(readme.relative_to(ROOT)))

    status_path = ROOT / "docs/components/STATUS.md"
    expected_status = dashboard(metas, exps, evals)
    current_status = status_path.read_text(encoding="utf-8") if status_path.exists() else ""
    if current_status != expected_status:
        if args.write:
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(expected_status, encoding="utf-8")
        else:
            stale.append(str(status_path.relative_to(ROOT)))

    if stale:
        print("Generated component documentation is stale:")
        for p in stale:
            print(" -", p)
        print("Run: python3 scripts/update_component_docs.py --write")
        return 1

    action = "updated" if args.write else "verified"
    print(f"OK: {action} {len(metas)} component documentation records")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
