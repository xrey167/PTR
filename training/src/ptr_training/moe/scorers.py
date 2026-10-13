"""Task scorers for M011 (pure Python, no torch).

* GSM8K: exact-match on the final numeric answer.
* HumanEval: pass@1 (greedy, one sample) by executing completion + tests in a subprocess with a
  timeout, a scrubbed environment and resource limits, inside a bubblewrap sandbox: the host
  file system read-only, the home directories hidden, a private /tmp and no network (see
  `sandbox_info`). Without bubblewrap the code is refused unless the caller opts into the
  unsandboxed fallback with `PTR_M011_ALLOW_UNSANDBOXED=1`, which `sandbox_info` records.
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

# A numeric token is a maximal run of digits, commas and points (with an optional sign). It is
# read only when it is a well-formed number: commas only as thousands separators (1,234,567),
# at most one decimal point. One trailing `.` or `,` is sentence punctuation and is dropped.
# A malformed token such as `1,2` or `1,,2` is not a number, so it can never match a reference.
_TOKEN = re.compile(r"-?[\d.,]*\d[\d.,]*")
_WELL_FORMED = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|-?\.\d+")


def _number_token(token: str) -> str | None:
    if token[-1] in ".,":
        token = token[:-1]
    return token if _WELL_FORMED.fullmatch(token) else None


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
    found = _TOKEN.findall(region)
    if not found:
        return None
    token = _number_token(found[0] if "####" in text else found[-1])
    return None if token is None else _to_fraction(token)


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


UNSANDBOXED_ENV = "PTR_M011_ALLOW_UNSANDBOXED"


class SandboxUnavailable(RuntimeError):
    """Generated code would run with the user's file system access and nobody opted into that."""


def sandbox_info() -> dict:
    """How generated code is isolated here. `filesystem` is `bwrap-readonly` when bubblewrap
    runs it (host read-only, home hidden, private /tmp, no network), `unsandboxed-opt-in` when
    the caller set `PTR_M011_ALLOW_UNSANDBOXED=1` without bubblewrap, else `refused`."""
    if _bwrap_usable():
        return {"filesystem": "bwrap-readonly", "network": "bwrap-unshare-net"}
    if os.environ.get(UNSANDBOXED_ENV) == "1":
        return {
            "filesystem": "unsandboxed-opt-in",
            "network": "unshare-net" if _unshare_usable() else "python-socket-block-only",
        }
    return {"filesystem": "refused", "network": "refused"}


_BWRAP: bool | None = None


def _bwrap_usable() -> bool:
    global _BWRAP
    if _BWRAP is None:
        exe = shutil.which("bwrap")
        _BWRAP = False
        if exe:
            try:
                with tempfile.TemporaryDirectory(prefix="m011-probe-") as tmp:
                    probe = sandbox_command(exe, tmp, [sys.executable, "-I", "-c", "pass"])
                    _BWRAP = subprocess.run(probe, capture_output=True, timeout=10).returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                _BWRAP = False
    return _BWRAP


def sandbox_command(bwrap: str, workdir: str, argv: list[str]) -> list[str]:
    """`argv` under bubblewrap: the host file system read-only, the home directories replaced by
    empty tmpfs (the interpreter's own prefix is bound back read-only, since a venv often lives
    under a home directory), a private /tmp with only `workdir` writable, new namespaces (no
    network), the environment cleared and the sandbox killed with its parent."""
    cmd = [bwrap, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
    for hidden in sorted({os.path.expanduser("~"), "/root", "/home"}):
        if os.path.isdir(hidden):
            cmd += ["--tmpfs", hidden]
    interpreter = {
        os.path.realpath(sys.prefix),
        os.path.realpath(sys.base_prefix),
        os.path.dirname(os.path.realpath(sys.executable)),
        os.path.dirname(os.path.abspath(sys.executable)),
    }
    for path in sorted(interpreter):
        if os.path.isdir(path):
            cmd += ["--ro-bind", path, path]
    cmd += [
        "--bind", workdir, workdir, "--chdir", workdir,
        "--unshare-all", "--die-with-parent", "--new-session",
        "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
    ]
    return cmd + ["--", *argv]


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
    """pass@1 for one sample: True iff `prompt + completion` plus the tests run cleanly.

    Raises `SandboxUnavailable` when bubblewrap is missing and the unsandboxed fallback was not
    opted into: model-generated code is never run with the user's file system by default."""
    mode = sandbox_info()["filesystem"]
    if mode == "refused":
        raise SandboxUnavailable(
            "HumanEval scoring runs model-generated code; install bubblewrap (apt install "
            f"bubblewrap) or set {UNSANDBOXED_ENV}=1 to accept running it as this user"
        )
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
        if mode == "bwrap-readonly":
            cmd = sandbox_command(shutil.which("bwrap"), tmp, cmd)
        elif _unshare_usable():
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
        # `**x` and a repeated keyword are invalid calls (the latter is a SyntaxError in Python).
        if kw.arg is None or kw.arg in kwargs:
            return None
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            return None
    return name, kwargs


class _Unmatchable:
    """Stands in for a value that must not compare equal to anything (not even itself)."""

    def __eq__(self, other):
        return False

    __hash__ = None


def _norm(value):
    # Numbers stay as they are: Python compares int with int and int with float exactly, so
    # 1 == 1.0 still holds while 2**53 + 1 no longer equals 2**53 through a float round trip.
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    if isinstance(value, dict):
        # Keys of any type are compared as text, so no sort over incomparable keys is needed;
        # two keys that collapse to the same text (1 and '1') make the value unmatchable.
        normed = {str(k): _norm(v) for k, v in value.items()}
        return normed if len(normed) == len(value) else _Unmatchable()
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
    "sandbox_command",
    "sandbox_info",
    "SandboxUnavailable",
    "UNSANDBOXED_ENV",
    "score",
    "score_bfcl",
    "score_gsm8k",
    "truncate_completion",
]
