#!/usr/bin/env python3
"""`make vcw-demo`: drive a real ptrd through the verifier-closed workspace path.

Starts the already built `ptrd` with `config/vcw-demo.toml` (VCW path on, simulated experts, real
verifiers, data directory in a temp dir, free loopback port), sends three requests, reads the
trace and the ledger events of each one back from the daemon (`GET /v1/vcw/traces/{id}`), prints
them, stops the daemon with SIGTERM and exits 0 only if every expected property held.

  1. arithmetic, repaired: the first candidate fails the real arithmetic verifier, a routed
     repair round (the router samples between two eligible repairers of different skill) fixes
     it, the verifier accepts it, the runtime admits it.
  2. JSON object, repaired: the first candidate lacks a required field (real JSON Schema check).
  3. code, exhausted by inability: the first program fails the real code tests and no eligible
     repairer can fix that finding (simulated skill 0), so the request runs out of budget. It is
     NOT best-effort accepted. The reference is correct; the experts are unable.

Expert stages: (a) a small open-weight model on CPU (qwen-cpu, optional), (b) small self-trained
specialists (not built), (c) programmatic specialists with a controlled error rate (the default
here; see docs/architecture/39-vcw-moe.md, "Demo"). With the environment variable PTR_VCW_QWEN_DIR
pointing at a local checkpoint directory and a ptrd built with `--features qwen-cpu` (the Makefile
does that), stage (a) answers instead; repair and exhaustion are then not scripted, only the safety
properties are asserted. Python standard library only.
"""

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "vcw-demo.toml"
QWEN_ENV = "PTR_VCW_QWEN_DIR"
QWEN_FILES = {
    "weights_sha256": "model.safetensors",
    "config_sha256": "config.json",
    "tokenizer_sha256": "tokenizer.json",
    "tokenizer_config_sha256": "tokenizer_config.json",
}

FAILURES = []


def check(condition, message):
    """Record an expected property; print one line per property."""
    print(f"  [{'ok' if condition else 'FAIL'}] {message}")
    if not condition:
        FAILURES.append(message)
    return condition


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def http(method, base, path, body=None):
    """Return (status, parsed JSON). 4xx/5xx bodies are returned, not raised."""
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        base + path,
        data=data,
        method=method,
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            return error.code, json.loads(raw)
        except ValueError:
            return error.code, {"raw": raw.decode(errors="replace")}


def wait_ready(base, daemon, seconds=300):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if daemon.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(base + "/health", timeout=1):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.1)
    return False


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_config(directory, qwen_dir):
    """The demo config, or its qwen-cpu variant (digests computed from the local files)."""
    if qwen_dir is None:
        return CONFIG
    text = CONFIG.read_text()
    text = text.replace('backend = "simulated"', 'backend = "qwen-cpu"')
    # The model backend has prompts for arithmetic and json_object only.
    text = text.replace('kinds = ["arithmetic", "json_object", "code"]',
                        'kinds = ["arithmetic", "json_object"]')
    fields = {"model_dir": str(qwen_dir), "max_new_tokens": "16"}
    for field, name in QWEN_FILES.items():
        fields[field] = sha256_file(qwen_dir / name)
    for field, value in fields.items():
        replacement = value if field == "max_new_tokens" else json.dumps(value)
        text, count = re.subn(rf"(?m)^{field} = .*$", f"{field} = {replacement}", text)
        if count != 1:
            raise SystemExit(f"vcw-demo: cannot set {field} in the demo config")
    out = directory / "vcw-demo-qwen.toml"
    out.write_text(text)
    return out


def codes_of(candidate):
    return candidate.get("hard_codes", []) if candidate else []


def print_trace(report):
    trace = report["trace"]
    budget, spent = trace["budget"], trace["spent"]
    print(f"  backend={report['backend']} outcome={report['outcome']} "
          f"trace_digest={trace['digest'][:16]}...")
    for step in trace["steps"]:
        authors = ", ".join(f"{a['pod']}@{a['generation']}" for a in step["authors"]) or "-"
        print(f"  round {step['round']}  origin={step['origin']}  authors=[{authors}]")
        for route in step["routes"]:
            support = ", ".join(f"{s['pod']}:{s['width']}" for s in route["support"])
            print(f"      routed {route['class']} -> {route['pod']}@{route['generation']} "
                  f"propensity={route['propensity']:.4f}  support=[{support}]")
        if step["unroutable"]:
            print(f"      unroutable codes: {step['unroutable']}")
        candidate = step["candidate"]
        if candidate:
            print(f"      candidate {candidate['digest'][:12]}  verifier={candidate['status']} "
                  f"codes={candidate['hard_codes']}  class={json.dumps(candidate['class'])}  "
                  f"workspace={candidate['workspace']}")
        else:
            print("      no candidate")
        after = step["budget_after"]
        print(f"      decision={step['decision']}  budget_after: rounds={after['rounds']} "
              f"verifier_calls={after['verifier_calls']} pod_calls={after['pod_calls']}")
    print(f"  budget: used rounds {spent['rounds']}/{budget['rounds']}, "
          f"verifier calls {spent['verifier_calls']}/{budget['verifier_calls']}, "
          f"pod calls {spent['pod_calls']}")
    ledger = report["ledger"]
    print(f"  ledger: {ledger['count']} event(s) from index {ledger['first_index']}")
    for event in ledger["events"]:
        print(f"      #{event['index']}  {event['kind']}  ({event['bytes']} bytes, "
              f"digest {event['digest']})")


def run_request(base, request_id, task, note=""):
    """Send one task and read its report back. Returns (status, response, report)."""
    what = f"{task['kind']} {task.get('expression', task.get('instruction', ''))}".rstrip()
    print(f"\n=== {request_id}: {what}  {note}".rstrip())
    status, response = http("POST", base, "/v1/vcw/tasks", {"id": request_id, "task": task})
    shown = {k: v for k, v in response.items() if k != "trace_digest"}
    print(f"  POST /v1/vcw/tasks -> HTTP {status} {json.dumps(shown)}")
    got, report = http("GET", base, f"/v1/vcw/traces/{request_id}")
    if not check(got == 200, f"GET /v1/vcw/traces/{request_id} -> 200 (got {got})"):
        raise SystemExit("vcw-demo: cannot read the trace back; stopping")
    print_trace(report)
    check(report["trace"]["digest"] == response.get("trace_digest"),
          "the trace read back has the digest the response reported")
    return status, response, report


def kinds_of(report):
    return [event["kind"] for event in report["ledger"]["events"]]


# The repairers eligible for each finding code, as configured in config/vcw-demo.toml.
ELIGIBLE = {
    "arith.value": {"vcw-value-repair", "vcw-generalist-repair"},
    "json.required": {"vcw-schema-repair", "vcw-generalist-repair"},
    "code.test_failed": {"vcw-code-repair", "vcw-code-repair-b"},
}
RESOLUTION = float(1 << 53)


def repair_routes(report, code):
    return [
        route
        for step in report["trace"]["steps"]
        for route in step["routes"]
        if route["class"].startswith("Repair") and code in route["class"]
    ]


def expect_routing(report, code, label):
    """The repair of `code` went to an eligible Pod, with a propensity that is the logged
    support entry of the chosen Pod (not asserted: which Pod won)."""
    routes = repair_routes(report, code)
    check(bool(routes), f"{label}: {code} was routed to a repair Pod")
    eligible = ELIGIBLE[code]
    check(all(r["pod"] in eligible for r in routes),
          f"{label}: every repair of {code} went to an eligible Pod {sorted(eligible)}")
    check(all({e["pod"] for e in r["support"]} == eligible for r in routes),
          f"{label}: the logged support of each decision is exactly the eligible Pods")
    ok = True
    for route in routes:
        width = {e["pod"]: e["width"] for e in route["support"]}.get(route["pod"])
        ok = ok and width is not None and 0 < route["propensity"] < 1 \
            and abs(route["propensity"] - width / RESOLUTION) < 1e-9
    check(ok, f"{label}: propensity is in (0, 1) and equals the logged support entry of the "
              "chosen Pod")


def expect_ledger(response, report, label):
    """The ledger range of the request is non-empty and covers the request (its raw text is the
    first event), so a negative check on its events can not pass vacuously."""
    ledger = report["ledger"]
    events = ledger["events"]
    indexes = [e["index"] for e in events]
    check(ledger["count"] >= 1 and len(events) == ledger["count"],
          f"{label}: the ledger range is non-empty and fully listed ({ledger['count']} event(s))")
    check(indexes == list(range(ledger["first_index"], ledger["first_index"] + len(events))),
          f"{label}: the listed events are contiguous from the first index")
    check(bool(events) and events[0]["kind"] == "SemanticDeltaCommitted",
          f"{label}: the range starts with the committed raw text of the request")
    committed = response.get("committed")
    if committed is not None:
        check(committed["first_index"] == ledger["first_index"]
              and committed["count"] == ledger["count"],
              f"{label}: the range equals the one the response reported")


def expect_repaired(status, response, report, code, label):
    steps = report["trace"]["steps"]
    check(status == 200 and response.get("outcome") == "verified",
          f"{label}: outcome verified (HTTP 200)")
    first = steps[0]["candidate"]
    check(first is not None and first["status"] == "Fail" and code in first["hard_codes"],
          f"{label}: the first candidate FAILED the real verifier with {code}")
    check(first is not None and "repair" in first["class"],
          f"{label}: the failure was classified as repair")
    expect_routing(report, code, label)
    last = steps[-1]
    check(last["decision"] == "Accepted" and last["candidate"] is not None
          and last["candidate"]["status"] == "Pass",
          f"{label}: the last candidate PASSED the verifier and was accepted")
    check(all(s["candidate"] is not None and s["candidate"]["status"] != "Pass"
              for s in steps[:-1]),
          f"{label}: accepted only after a Pass (no earlier candidate passed)")
    check(last["origin"] == "repaired", f"{label}: the accepted candidate came from a repair round")
    check(response["rounds"] >= 1, f"{label}: at least one repair round was spent")
    expect_ledger(response, report, label)
    check(any(k.startswith("PodOutput") for k in kinds_of(report)),
          f"{label}: the ledger holds the runtime's admission of the verified answer")


def expect_exhausted(status, response, report, code):
    steps = report["trace"]["steps"]
    label = "code"
    check(status == 422 and response.get("outcome") == "exhausted", "code: outcome exhausted (HTTP 422)")
    reason = response.get("reason")
    spent, budget = report["trace"]["spent"], report["trace"]["budget"]
    dimension = {"VerifierCalls": ("verifier_calls", "verifier calls"),
                 "Rounds": ("rounds", "rounds")}.get(reason)
    check(dimension is not None and spent[dimension[0]] >= budget[dimension[0]],
          f"code: terminal reason {reason}: {dimension[1] if dimension else '?'} budget "
          f"used up ({spent[dimension[0]] if dimension else '?'}/"
          f"{budget[dimension[0]] if dimension else '?'})")
    check("answer" not in response, "code: the response carries NO answer (no best-effort accept)")
    check(all(s["candidate"] is None or s["candidate"]["status"] != "Pass" for s in steps),
          "code: no candidate ever passed the real verifier")
    check(all(code in s["candidate"]["hard_codes"] for s in steps if s["candidate"]),
          f"code: every candidate still failed with {code} (the repairers were unable)")
    expect_routing(report, code, label)
    expect_ledger(response, report, label)
    check(not any(k.startswith("PodOutput") for k in kinds_of(report)),
          "code: the ledger holds no answer admission for this request")


def drive(base, qwen):
    schema = {
        "type": "object",
        "required": ["name", "age"],
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    }
    arithmetic = {"kind": "arithmetic", "expression": "6*7"}
    person = {"kind": "json_object", "instruction": "a person with a name and an age",
              "schema": schema}
    square = {"kind": "code", "instruction": "square the argument", "entry": "f",
              "tests": [{"id": "t3", "args": [3], "expected": 9},
                        {"id": "t4", "args": [4], "expected": 16}]}
    if not qwen:
        # The simulated experts draw their clean candidates from `reference`; it never reaches
        # a verifier and all three references are correct.
        arithmetic["reference"] = "42"
        person["reference"] = {"name": "Ada", "age": 36}
        square["reference"] = "fn f(a0) { return a0 * a0; }"

    s1 = run_request(base, "demo-arith-repaired", arithmetic,
                     "" if qwen else "(first candidate carries a wrong value; expect a routed repair)")
    s2 = run_request(base, "demo-json-repaired", person,
                     "" if qwen else "(first candidate lacks a required field; expect a routed repair)")
    if qwen:
        print("\n=== expected properties")
        for status, response, _ in (s1, s2):
            answered = response.get("outcome") == "verified"
            check(answered == ("answer" in response) and answered == (status == 200),
                  f"{response['id']}: an answer is returned if and only if the verifier accepted")
        return
    s3 = run_request(base, "demo-code-exhausted", square,
                     "(no eligible repairer can fix code.test_failed; expect Exhausted)")
    print("\n=== expected properties")
    expect_repaired(*s1, "arith.value", "arithmetic")
    check(s1[1].get("answer") == "42", "arithmetic: the accepted answer is 42")
    expect_repaired(*s2, "json.required", "json")
    check(json.loads(s2[1]["answer"]) == {"name": "Ada", "age": 36},
          "json: the repaired object is the accepted answer")
    expect_exhausted(*s3, "code.test_failed")
    status, _ = http("POST", base, "/v1/vcw/tasks", {"id": "demo-code-exhausted", "task": square})
    check(status == 409, "re-using a request id is refused (HTTP 409)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bin", help="path to the built ptrd (default: <target dir>/debug/ptrd)")
    args = parser.parse_args()
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    binary = Path(args.bin) if args.bin else target / "debug" / "ptrd"
    if not binary.is_file():
        print(f"vcw-demo: {binary} not found; run `make vcw-demo` (it builds ptrd first)")
        return 2
    qwen_env = os.environ.get(QWEN_ENV)
    qwen_dir = Path(qwen_env) if qwen_env else None
    if qwen_dir is not None and not all((qwen_dir / n).is_file() for n in QWEN_FILES.values()):
        print(f"vcw-demo: {QWEN_ENV} must hold {', '.join(QWEN_FILES.values())}")
        return 2

    with tempfile.TemporaryDirectory(prefix="ptr-vcw-demo-") as tmp:
        tmp = Path(tmp)
        data_dir = tmp / "data"
        data_dir.mkdir(mode=0o700)
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        config = write_config(tmp, qwen_dir)
        log_path = tmp / "ptrd.log"
        print(f"vcw-demo: experts = {'qwen-cpu, stage (a) (' + QWEN_ENV + ')' if qwen_dir else 'simulated, stage (c)'}")
        print(f"vcw-demo: starting {binary.name} on {base}, data dir {data_dir}")
        with open(log_path, "wb") as log:
            daemon = subprocess.Popen(
                [str(binary), "--config", str(config), "--bind", f"127.0.0.1:{port}",
                 "--data-dir", str(data_dir)],
                stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                env={k: v for k, v in os.environ.items() if not k.startswith("PTR_")},
            )
        try:
            if not wait_ready(base, daemon):
                print("vcw-demo: ptrd did not become ready; its log:")
                print(log_path.read_text(errors="replace"))
                return 1
            try:
                drive(base, qwen_dir is not None)
            except SystemExit as stop:
                FAILURES.append(str(stop))
        finally:
            clean = False
            if daemon.poll() is None:
                daemon.send_signal(signal.SIGTERM)
                try:
                    clean = daemon.wait(timeout=20) == 0
                except subprocess.TimeoutExpired:
                    daemon.kill()
                    daemon.wait()
            print("\n=== shutdown")
            check(clean, "ptrd exited 0 after SIGTERM")
            log_text = log_path.read_text(errors="replace")
            check("ptrd stopped cleanly" in log_text, "ptrd reported a clean stop")
            ledger = data_dir / "ledger.log"
            check(ledger.is_file() and ledger.stat().st_size > 0,
                  f"the durable ledger file exists ({ledger.stat().st_size if ledger.is_file() else 0} bytes)")

    print()
    if FAILURES:
        print(f"vcw-demo: FAILED ({len(FAILURES)} expectation(s) not met)")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("vcw-demo: OK - every expected property held")
    return 0


if __name__ == "__main__":
    sys.exit(main())
