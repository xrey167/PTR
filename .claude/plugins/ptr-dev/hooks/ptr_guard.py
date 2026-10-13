#!/usr/bin/env python3
"""PTR guard hook for Claude Code (PreToolUse deny rules, PostToolUse reminders).

Usage: ptr_guard.py pre|post, hook JSON on stdin. Never fails the tool call:
any error allows silently (exit 0) with a one-line note on stderr.
"""
from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path, PurePosixPath

FROZEN_STATUSES = {"prepared", "running", "completed", "failed", "superseded"}


def find_repo_root(start: str) -> Path | None:
    p = Path(start)
    for cand in (p, *p.parents):
        if (cand / "Cargo.toml").is_file() and (cand / "CLAUDE.md").is_file():
            return cand
    return None


def norm_slashes(s: str) -> str:
    return s.replace("\\", "/")


def relative_to_repo(raw: str, root: Path) -> str | None:
    """Repo-relative posix path, or None when the path is outside the repo."""
    p = norm_slashes(raw)
    r = norm_slashes(str(root))
    # Windows drive letters: C:/x, /c/x (Git Bash) and /mnt/c/x compare case-insensitively.
    def key(s: str) -> str:
        m = re.match(r"^(?:/mnt/|/)?([A-Za-z])(?::)?/(.*)$", s) if re.match(
            r"^[A-Za-z]:/|^/mnt/[A-Za-z]/|^/[A-Za-z]/", s) else None
        if m:
            return f"{m.group(1).lower()}:/{m.group(2)}"
        return s
    pk, rk = key(p), key(r)
    is_win = re.match(r"^[a-z]:/", pk) is not None
    if not (pk.startswith("/") or is_win):
        pk = rk.rstrip("/") + "/" + pk  # relative to root
    parts: list[str] = []
    for seg in pk.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts:
                parts.pop()
            continue
        parts.append(seg)
    rparts = [s for s in rk.split("/") if s not in ("", ".")]
    cmp = (lambda xs: [x.lower() for x in xs]) if is_win else (lambda xs: xs)
    if cmp(parts[: len(rparts)]) != cmp(rparts):
        return None
    return "/".join(parts[len(rparts):])


def deny(reason: str) -> dict:
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                   "permissionDecision": "deny",
                                   "permissionDecisionReason": reason}}


def frozen_experiment_dirs(root: Path) -> list[tuple[str, str]]:
    """(id, repo-relative dir) of experiments the research gate freezes."""
    try:
        prereg = tomllib.loads((root / "experiments/preregistration.toml").read_text("utf-8"))
        registry = tomllib.loads((root / "experiments/registry.toml").read_text("utf-8"))
    except Exception:
        return []
    listed = prereg.get("experiment")
    if not isinstance(listed, dict):
        return []
    out = []
    for item in registry.get("experiment", []):
        if (isinstance(item, dict) and item.get("id") in listed
                and item.get("status") in FROZEN_STATUSES and isinstance(item.get("path"), str)):
            out.append((item["id"], PurePosixPath("experiments", item["path"]).as_posix()))
    return out


def new_contents(tool_input: dict) -> list[str]:
    out = []
    for k in ("content", "new_string", "new_source"):
        if isinstance(tool_input.get(k), str):
            out.append(tool_input[k])
    for e in tool_input.get("edits") or []:
        if isinstance(e, dict) and isinstance(e.get("new_string"), str):
            out.append(e["new_string"])
    return out


def pre(rel: str, root: Path, tool_input: dict) -> dict | None:
    if rel == "vendor" or rel.startswith("vendor/"):
        how = []
        if (root / "docs/VENDOR_PATCH_POLICY.md").exists():
            how.append("see docs/VENDOR_PATCH_POLICY.md")
        if (root / "scripts/refresh_vendor_upstream.py").exists():
            how.append("refresh via scripts/refresh_vendor_upstream.py")
        return deny("vendor/** holds patched upstream crates and is integrity-checked "
                    "(scripts/check_vendor_integrity.py); do not edit it casually. "
                    + "; ".join(how) + ". Prefer fixing the problem in crates/ instead.")
    if rel.startswith("training/runs/"):
        return deny("training/runs/** holds gitignored run outputs; never hand-edit them. "
                    "Re-run the training/experiment to regenerate, or write elsewhere.")
    for exp_id, d in frozen_experiment_dirs(root):
        if rel == d or rel.startswith(d + "/"):
            return deny(f"Experiment {exp_id} ({d}) is frozen by experiments/preregistration.toml "
                        "(scripts/check_research_gates.py); its records must stay as they were. "
                        "Create a successor with scripts/new_experiment.py instead of patching it.")
    if rel == ".cargo/config.toml":
        if any(re.search(r"^\s*\[env\]", c, re.M) for c in new_contents(tool_input)):
            return deny(".cargo/config.toml must not contain an [env] table "
                        "(CLAUDE.md; scripts/check_research_gates.py rejects it). Put "
                        "machine-specific settings (e.g. CUDA) in the user-level Cargo config.")
    return None


def post(rel: str) -> dict | None:
    msgs = []
    m = re.match(r"^crates/([^/]+)/(src/.*|Cargo\.toml)$", rel)
    if m:
        c = m.group(1)
        msgs.append(f"crates/{c} changed: update crates/{c}/component.toml in the same change, "
                    "then run `make docs` (checked by scripts/check_component_metadata.py).")
    if rel.startswith("experiments/"):
        msgs.append("experiments/ changed: run `python3 scripts/run_experiment.py validate` and "
                    "`python3 scripts/check_research_gates.py` before finishing.")
    if not msgs:
        return None
    return {"hookSpecificOutput": {"hookEventName": "PostToolUse",
                                   "additionalContext": " ".join(msgs)}}


def run(mode: str, payload: dict) -> dict | None:
    ti = payload.get("tool_input")
    if not isinstance(ti, dict):
        return None
    raw = ti.get("file_path") or ti.get("notebook_path")
    if not isinstance(raw, str) or not raw:
        return None
    import os
    start = os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd()
    root = find_repo_root(start)
    if root is None:
        return None
    rel = relative_to_repo(raw, root)
    if rel is None or rel == "":
        return None
    return pre(rel, root, ti) if mode == "pre" else post(rel) if mode == "post" else None


def main() -> int:
    try:
        mode = sys.argv[1] if len(sys.argv) > 1 else ""
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, dict):
            raise ValueError("stdin JSON is not an object")
        result = run(mode, payload)
        if result is not None:
            sys.stdout.write(json.dumps(result))
    except Exception as exc:  # never break the tool call
        print(f"ptr_guard: ignored error: {exc!r}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
