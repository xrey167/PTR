from __future__ import annotations

import argparse
import re
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

def code_metrics(crate_dir: Path) -> dict[str, int]:
    source_files = (
        sorted((crate_dir / "src").rglob("*.rs"))
        if (crate_dir / "src").exists()
        else []
    )
    test_files = (
        sorted((crate_dir / "tests").rglob("*.rs"))
        if (crate_dir / "tests").exists()
        else []
    )
    source_loc = 0
    test_markers = 0
    for path in source_files:
        text = path.read_text(encoding="utf-8")
        source_loc += sum(1 for line in text.splitlines() if line.strip())
        test_markers += text.count("#[test]")
    for path in test_files:
        text = path.read_text(encoding="utf-8")
        test_markers += text.count("#[test]")
    return {
        "files": len(source_files),
        "loc": source_loc,
        "test_files": len(test_files),
        "tests": test_markers,
    }

def render_section(meta: dict, exps: dict, evals: dict, metrics: dict[str, int]) -> str:
    exp_lines = []
    for exp_id in meta.get("experiments", []):
        e = exps[exp_id]
        exp_lines.append(
            f'[{exp_id}](../../experiments/{e["path"]}/README.md) — `{e["status"]}`'
        )
    eval_lines = []
    for eval_id in meta.get("evaluations", []):
        e = evals[eval_id]
        eval_lines.append(
            f'[{eval_id}](../../evaluations/components/{eval_id}/README.md) — `{e["status"]}`'
        )
    decision_lines = [
        f"[{d}](../../research/decisions/{d})" for d in meta.get("decisions", [])
    ]
    return f"""{BEGIN}
## Current implementation status

> **Generated section.** Source of truth: [`component.toml`](component.toml) plus code-derived metrics from `src/`. Run `python3 scripts/update_component_docs.py --write` after editing implementation metadata. Do not hand-edit inside this block.

**Maturity:** `{meta["maturity"]}`  
**Last reviewed:** {meta["last_reviewed"]}  
**Code footprint:** {metrics["files"]} Rust source files · {metrics["loc"]} nonblank source lines · {metrics["test_files"]} integration-test files · {metrics["tests"]} `#[test]` markers

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

{END}
"""

def update_readme(path: Path, section: str) -> str:
    text = path.read_text(encoding="utf-8")
    marker_pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    legacy_pattern = re.compile(r"\{BEGIN\}.*?\{END\}", re.S)
    if marker_pattern.search(text):
        return marker_pattern.sub(section.strip(), text)
    if legacy_pattern.search(text):
        return legacy_pattern.sub(section.strip(), text)
    marker = "## Position in PTR"
    if marker not in text:
        raise RuntimeError(f"{path.relative_to(ROOT)} has no insertion marker")
    return text.replace(marker, section.strip() + "\n\n" + marker, 1)

def dashboard(metas: list[dict], exps: dict, evals: dict) -> str:
    rows = []
    maturity_counts: dict[str, int] = {}
    total_loc = total_files = total_test_files = total_tests = 0
    for m in metas:
        maturity_counts[m["maturity"]] = maturity_counts.get(m["maturity"], 0) + 1
        metrics = code_metrics(ROOT / "crates" / m["id"])
        total_files += metrics["files"]
        total_loc += metrics["loc"]
        total_test_files += metrics["test_files"]
        total_tests += metrics["tests"]
        exp_status = ", ".join(
            f'{eid}:{exps[eid]["status"]}' for eid in m.get("experiments", [])
        ) or "—"
        eval_status = ", ".join(
            f'{eid}:{evals[eid]["status"]}' for eid in m.get("evaluations", [])
        ) or "—"
        rows.append(
            f'| [{m["id"]}](../../crates/{m["id"]}/README.md) | '
            f'`{m["maturity"]}` | {metrics["files"]} | {metrics["loc"]} | {metrics["test_files"]} | {metrics["tests"]} | '
            f'{len(m.get("implemented", []))} | {len(m.get("missing", []))} | '
            f'{exp_status} | {eval_status} |'
        )
    counts = ", ".join(f"`{k}`: {v}" for k, v in sorted(maturity_counts.items()))
    return f"""# PTR Component Implementation Status

> Generated from every `crates/*/component.toml` plus code-derived metrics. Do not hand-edit.  
> Refresh with `python3 scripts/update_component_docs.py --write`.

**Component count:** {len(metas)}  
**Maturity distribution:** {counts}  
**Rust footprint:** {total_files} source files · {total_loc} nonblank source lines · {total_test_files} integration-test files · {total_tests} `#[test]` markers

| Component | Maturity | Rust files | LOC | Test files | Tests | Implemented items | Missing items | Experiments | Evaluations |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
{chr(10).join(rows)}

## Meaning of maturity labels

- `foundation` — small stable domain core already used broadly.
- `prototype` — executable behavior exists, but major target semantics are still missing.
- `scaffold` — contracts/data structures exist; core production implementation is not present.
- `research-scaffold` — architecture hypothesis is represented, but the trainable system is not implemented/proven.

The labels describe implementation maturity, not scientific novelty or production readiness.

## Freshness policy

Changes under `crates/<component>/src/` or that crate's `Cargo.toml` must update the matching `component.toml` in the same change. CI enforces this metadata coupling, then verifies that generated README/status output is synchronized.
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
        crate_dir = ROOT / "crates" / meta["id"]
        readme = crate_dir / "README.md"
        expected = update_readme(
            readme,
            render_section(meta, exps, evals, code_metrics(crate_dir)),
        )
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