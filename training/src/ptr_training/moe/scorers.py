"""Task scorers for M011 (pure Python, no torch).

* GSM8K: exact-match on the final numeric answer.
* HumanEval: pass@1 (greedy, one sample) by executing completion + tests in a subprocess with a
  timeout, a scrubbed environment, resource limits and no network (see `sandbox_info`).
* BFCL `simple`: AST check of one function call against the possible-answer file. This is a
  simplified re-implementation of the BFCL AST checker for the single-call `simple` category
  (name equality, required parameters present, every value among the allowed ones, no extra
  parameters); it is not the official evaluator and the exec check does not apply to `simple`.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from fractions import Fraction

# ---------------------------------------------------------------------------- GSM8K

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def _to_fraction(token: str) -> Fraction | None:
    try:
        return Fraction(token.replace(",", "").replace("$", ""))
    except (ValueError, ZeroDivisionError):
        return None


def gsm8k_reference(answer: str) -> str:
    """The reference final answer: the text after the last `####` in a GSM8K `answer` field."""
    if "####" not in answer:
        raise ValueError("GSM8K answer without '####' marker")
    return answer.rsplit("####", 1)[1].strip().replace(",", "")


def gsm8k_extract(text: str) -> Fraction | None:
    """Final numeric answer of a generation: the number after `####` if present, else the last
    number in the text."""
    region = text.rsplit("####", 1)[1] if "####" in text else text
    found = _NUMBER.findall(region)
    if not found:
        return None
    return _to_fraction(found[0] if "####" in text else found[-1])


def score_gsm8k(generation: str, reference_final: str) -> bool:
    got = gsm8k_extract(generation)
    want = _to_fraction(reference_final)
    return got is not None and want is not None and got == want


# ---------------------------------------------------------------------------- HumanEval

STOP_SEQUENCES = ("\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n@")


def truncate_completion(text: str) -> str:
    """Cut a HumanEval completion at the standard stop sequences."""
    cut = len(text)
    for stop in STOP_SEQUENCES:
        i = text.find(stop)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


_PREAMBLE = """\
import resource as _r, socket as _s
_r.setrlimit(_r.RLIMIT_AS, ({mem}, {mem}))
_r.setrlimit(_r.RLIMIT_CPU, ({cpu}, {cpu}))
_r.setrlimit(_r.RLIMIT_FSIZE, (1 << 20, 1 << 20))
def _blocked(*a, **k):
    raise OSError("network disabled in the M011 sandbox")
_s.socket = _blocked
_s.create_connection = _blocked
_s.getaddrinfo = _blocked
"""


def sandbox_info() -> dict:
    """How network isolation is enforced here: kernel net namespace via `unshare` if usable,
    else a socket block inside the child (best effort, not a security boundary)."""
    return {"network": "unshare-net" if _unshare_usable() else "python-socket-block-only"}


_UNSHARE: bool | None = None


def _unshare_usable() -> bool:
    global _UNSHARE
    if _UNSHARE is None:
        exe = shutil.which("unshare")
        _UNSHARE = False
        if exe:
            try:
                _UNSHARE = (
                    subprocess.run(
                        [exe, "--net", "--map-root-user", "true"],
                        capture_output=True,
                        timeout=5,
                    ).returncode
                    == 0
                )
            except (OSError, subprocess.TimeoutExpired):
                _UNSHARE = False
    return _UNSHARE


def run_humaneval(
    prompt: str,
    completion: str,
    test: str,
    entry_point: str,
    timeout: float = 10.0,
    memory_bytes: int = 2 * 1024**3,
) -> bool:
    """pass@1 for one sample: True iff `prompt + completion` plus the tests run cleanly."""
    program = (
        _PREAMBLE.format(mem=memory_bytes, cpu=max(1, int(timeout)))
        + "\n"
        + prompt
        + truncate_completion(completion)
        + "\n\n"
        + test
        + f"\n\ncheck({entry_point})\n"
    )
    with tempfile.TemporaryDirectory(prefix="m011-he-") as tmp:
        path = os.path.join(tmp, "candidate.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(program)
        cmd = [sys.executable, "-I", path]
        if _unshare_usable():
            cmd = [shutil.which("unshare"), "--net", "--map-root-user", *cmd]
        proc = subprocess.Popen(
            cmd,
            cwd=tmp,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            return proc.wait(timeout=timeout) == 0
        except subprocess.TimeoutExpired:
            return False
        finally:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                proc.wait()


# ---------------------------------------------------------------------------- BFCL (simple)


def _dotted(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def parse_call(text: str) -> tuple[str, dict] | None:
    """Parse `name(a=1, b='x')` (optionally inside a one-element list) -> (name, kwargs)."""
    try:
        node = ast.parse(text.strip().split("\n")[0].strip(), mode="eval").body
    except SyntaxError:
        return None
    if isinstance(node, ast.List) and len(node.elts) == 1:
        node = node.elts[0]
    if not isinstance(node, ast.Call) or node.args:
        return None
    name = _dotted(node.func)
    if name is None:
        return None
    kwargs = {}
    for kw in node.keywords:
        if kw.arg is None:
            return None
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            return None
    return name, kwargs


def _norm(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _norm(v) for k, v in sorted(value.items())}
    return value


def bfcl_target(ground_truth: list[dict]) -> str:
    """Canonical call text for a possible-answer entry: first allowed value of each parameter,
    optional parameters (allowed list contains "") omitted."""
    (name, params), = ground_truth[0].items()
    parts = []
    for key, allowed in params.items():
        if "" in allowed:
            continue
        parts.append(f"{key}={allowed[0]!r}")
    return f"{name}({', '.join(parts)})"


def score_bfcl(generation: str, ground_truth: list[dict]) -> bool:
    parsed = parse_call(generation)
    if parsed is None or len(ground_truth) != 1:
        return False
    name, kwargs = parsed
    (want_name, params), = ground_truth[0].items()
    if name != want_name:
        return False
    if set(kwargs) - set(params):
        return False
    for key, allowed in params.items():
        if key not in kwargs:
            if "" in allowed:
                continue
            return False
        if not any(_norm(kwargs[key]) == _norm(a) for a in allowed if a != ""):
            return False
    return True


def score(domain: str, generation: str, reference: dict) -> bool:
    if domain == "math":
        return score_gsm8k(generation, reference["final"])
    if domain == "code":
        return run_humaneval(
            reference["prompt"], generation, reference["test"], reference["entry_point"]
        )
    if domain == "tool":
        return score_bfcl(generation, reference["ground_truth"])
    raise ValueError(f"unknown domain {domain!r}")


__all__ = [
    "bfcl_target",
    "gsm8k_extract",
    "gsm8k_reference",
    "parse_call",
    "run_humaneval",
    "sandbox_info",
    "score",
    "score_bfcl",
    "score_gsm8k",
    "truncate_completion",
]
