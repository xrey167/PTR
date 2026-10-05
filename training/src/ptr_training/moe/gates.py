"""PTR governance gates via the optional `ptr_py` binding.

The binding is built by a separate package. When it is not importable every gate reports
`status: "skipped"` with the reason; nothing is ever simulated. Binding surface used
(bindings/ptr-py/src/governance.rs):

* `aimer_score(list[float])` on each expert's flattened (gate_up + down) weights, and
  `rank_experts_for_pruning(list[(id, list[float])])` per layer;
* `subspace_overlap(a, b)` / `chance_overlap(a, b)` on `dim x k` row-major matrices whose
  column spaces are compared: here the `[hidden, intermediate]` change of each expert's
  `down_proj` between two snapshots (its output subspace in R^hidden).

A binding that is present but fails to import is reported as `status: "error"`, not skipped.
If a call raises (`PtrLineageError`, signature mismatch) the failure is recorded verbatim
as `status: "error"` and the whole report becomes "error"; it is never papered over.
The aggregate status is "ok" only when every layer ran; "partial" (with counts) when some
layers were skipped, "skipped" (with a reason) when all were.
"""

from __future__ import annotations

import importlib
import math
import random
from importlib import metadata

SKIP_REASON = "binding not installed: gates skipped"
MAX_FLAT_ELEMENTS = 1_000_000  # per vector handed to the binding; larger is skipped, not truncated


_NEEDED = ("aimer_score", "rank_experts_for_pruning", "subspace_overlap", "chance_overlap")


def binding_state() -> tuple[str, object]:
    """("installed", module) | ("absent", None) | ("broken", error text).

    Absent = `ptr_py` itself (or a submodule of it) cannot be found. Any other failure while
    importing it (a missing shared library, a dependency import error, OSError, ...) means a
    binding is present but unusable: that is an error, not a skip.
    """
    try:
        return "installed", importlib.import_module("ptr_py")
    except ModuleNotFoundError as error:
        name = error.name or ""
        if name == "ptr_py" or name.startswith("ptr_py."):
            return "absent", None
        return "broken", f"{type(error).__name__}: {error}"
    except Exception as error:  # noqa: BLE001 - recorded verbatim in the gate report
        return "broken", f"{type(error).__name__}: {error}"


def load_binding():
    state, value = binding_state()
    return value if state == "installed" else None


def binding_problem() -> dict | None:
    """None when gates can run, else the gate report to emit (skipped, or error if broken)."""
    state, value = binding_state()
    if state == "absent":
        return {"status": "skipped", "reason": SKIP_REASON}
    if state == "broken":
        return {"status": "error", "error": f"ptr_py import failed: {value}"}
    missing = [n for n in _NEEDED if not hasattr(value, n)]
    if missing:
        return {"status": "skipped", "reason": f"binding lacks {missing}: gates skipped"}
    return None


def _flat(tensor) -> list[float]:
    return tensor.detach().float().reshape(-1).cpu().tolist()


def expert_weights(model) -> list[dict]:
    """Per layer: {"gate_up": [E tensors], "down": [E tensors]} from the fused expert params."""
    layers = []
    base = getattr(model, "base_model", model)
    inner = getattr(base, "model", base)
    core = getattr(inner, "model", inner)
    for layer in core.layers:
        experts = layer.mlp.experts
        layers.append({"gate_up": experts.gate_up_proj, "down": experts.down_proj})
    return layers


def flatten_experts(model, max_elements: int | None = None) -> list:
    """[layer][expert] -> {"flat": gate_up+down weights, "down": down_proj rows [H][I]}.

    The element count is checked on the tensors first; a layer whose per-expert vector would
    exceed `max_elements` (default `MAX_FLAT_ELEMENTS`) is returned as `None` and never
    converted to Python lists.
    """
    limit = MAX_FLAT_ELEMENTS if max_elements is None else max_elements
    out = []
    for layer in expert_weights(model):
        n = layer["gate_up"].shape[0]
        if layer["gate_up"][0].numel() + layer["down"][0].numel() > limit:
            out.append(None)
            continue
        out.append(
            [
                {
                    "flat": _flat(layer["gate_up"][e]) + _flat(layer["down"][e]),
                    "down": layer["down"][e].detach().float().cpu().tolist(),
                }
                for e in range(n)
            ]
        )
    return out


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b, strict=True)]


def expert_deltas(before: list, after: list) -> list[list[dict]]:
    """Per expert change `after - before` in the same {"flat", "down"} layout."""
    return [
        None
        if la is None or lb is None
        else [
            {
                "flat": _sub(ea["flat"], eb["flat"]),
                "down": [_sub(ra, rb) for ra, rb in zip(ea["down"], eb["down"], strict=True)],
            }
            for ea, eb in zip(la, lb, strict=True)
        ]
        for la, lb in zip(after, before, strict=True)
    ]


INFORMATIVE_CHANCE_LIMIT = 0.99  # chance overlap at/above this cannot discriminate anything
_POWER_ITERATIONS = 60


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    bt = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def _transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*a)]


def _orthonormalise(cols: list[list[float]]) -> list[list[float]]:
    """Modified Gram-Schmidt over a list of column vectors; drops (numerically) null columns."""
    out: list[list[float]] = []
    for v in cols:
        w = list(v)
        for q in out:
            dot = sum(x * y for x, y in zip(w, q))
            w = [x - dot * y for x, y in zip(w, q)]
        norm = math.sqrt(sum(x * x for x in w))
        if norm > 1e-12:
            out.append([x / norm for x in w])
    return out


def top_left_subspace(matrix: list[list[float]], rank: int) -> list[list[float]]:
    """Orthonormal `dim x r` basis (row-major) of the top-`rank` left singular subspace.

    Deterministic orthogonal (subspace) iteration on `A A^T`; pure Python, so the gate module
    needs neither numpy nor torch. r is clipped to min(dim, columns). A zero matrix yields an
    empty basis (the caller records that as not informative).
    """
    dim, cols = len(matrix), len(matrix[0])
    r = min(rank, dim, cols)
    rng = random.Random(0)
    q = _orthonormalise([[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(r)])
    at = _transpose(matrix)
    for _ in range(_POWER_ITERATIONS):
        if not q:
            break
        z = _matmul(matrix, _matmul(at, _transpose(q)))  # dim x r  = A A^T Q
        q = _orthonormalise(_transpose(z))
    return _transpose(q) if q else []


def _value(call: dict):
    return call["value"] if call.get("status") == "ok" else None


def _overlap_pair(binding, da, db, subspace_rank: int | None) -> dict:
    """Overlap of two expert deltas' column spaces, with an `informative` verdict.

    With `subspace_rank` the deltas are first reduced to their top-r left singular subspaces;
    comparing full-rank subspaces of the ambient space is always ~1.0 (and chance is 1.0), which
    says nothing. `informative` is false when r >= ambient dim or chance_overlap >= 0.99.
    """
    ambient = len(da)
    if subspace_rank is not None:
        ua, ub = top_left_subspace(da, subspace_rank), top_left_subspace(db, subspace_rank)
        rank = min(subspace_rank, ambient)
        if not ua or not ub:
            return {
                "subspace_rank": subspace_rank, "ambient_dim": ambient, "informative": False,
                "subspace_overlap": {"status": "skipped", "reason": "zero expert delta"},
                "chance_overlap": {"status": "skipped", "reason": "zero expert delta"},
            }
    else:
        ua, ub, rank = da, db, None
    overlap = _call(binding.subspace_overlap, ua, ub)
    chance = _call(binding.chance_overlap, ua, ub)
    chance_value = _value(chance)
    informative = (
        chance_value is not None
        and chance_value < INFORMATIVE_CHANCE_LIMIT
        and (subspace_rank is None or subspace_rank < ambient)
        and _value(overlap) is not None
    )
    return {
        "subspace_rank": rank,
        "ambient_dim": ambient,
        "informative": informative,
        "subspace_overlap": overlap,
        "chance_overlap": chance,
    }


def binding_version(binding) -> str:
    for dist in ("ptr_py", "ptr-py"):
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    return str(getattr(binding, "__version__", "unknown"))


def _call(fn, *args):
    try:
        return {"status": "ok", "value": fn(*args)}
    except Exception as error:  # noqa: BLE001 - recorded verbatim, never swallowed silently
        return {"status": "error", "error": f"{type(error).__name__}: {error}"}


def select_pairs(all_pairs: list, cap: int) -> list:
    """All pairs if they fit under `cap`, else a deterministic stride over all of them (so the
    sample spreads across the expert range instead of covering only the first expert)."""
    total = len(all_pairs)
    if total <= cap:
        return list(all_pairs)
    return [all_pairs[i * total // cap] for i in range(cap)]


def _upgrade_for_errors(report: dict) -> dict:
    """Any embedded binding-call error makes the whole report `error` (precedence over
    ok / partial / skipped)."""
    errors = 0

    def walk(x):
        nonlocal errors
        if isinstance(x, dict):
            if x.get("status") == "error":
                errors += 1
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(report.get("layers", []))
    if errors:
        report["status"] = "error"
        report["failed_calls"] = errors
    return report


def run_gates(
    weights_before: list,
    weights_after: list,
    max_pairs_per_layer: int = 6,
    subspace_rank: int | None = None,
) -> dict:
    """weights_*: output of `flatten_experts`. Returns a JSON-ready gate report.

    Status precedence: `error` (any embedded call failed, or a broken binding) > `skipped`
    (nothing ran) > `partial` (some layers skipped) > `ok`. Gates are reporting-only: no value
    here is compared with a threshold and nothing in the pipeline reacts to them.

    `subspace_rank` (config `moe.gates.subspace_rank`) truncates each expert delta to its
    top-r left singular subspace before the overlap calls; None compares the full column
    spaces. Every pair carries an `informative` flag, and the report an aggregate one.
    """
    problem = binding_problem()
    if problem is not None:
        return problem
    binding = load_binding()
    deltas = expert_deltas(weights_before, weights_after)
    layers = []
    for li, (w_layer, d_layer) in enumerate(zip(weights_after, deltas)):
        if w_layer is None or max(len(w["flat"]) for w in w_layer) > MAX_FLAT_ELEMENTS:
            layers.append({"layer": li, "status": "skipped", "reason": "expert too large for gate input"})
            continue
        aimer = [_call(binding.aimer_score, w["flat"]) for w in w_layer]
        ranking = _call(
            binding.rank_experts_for_pruning, [(str(e), w["flat"]) for e, w in enumerate(w_layer)]
        )
        n = len(d_layer)
        all_pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
        chosen = select_pairs(all_pairs, max_pairs_per_layer)
        pairs = [
            {
                "experts": [a, b],
                **_overlap_pair(binding, d_layer[a]["down"], d_layer[b]["down"], subspace_rank),
            }
            for a, b in chosen
        ]
        layers.append(
            {
                "layer": li,
                "aimer_score": aimer,
                "prune_ranking": ranking,
                "delta_pairs": pairs,
                "pairs_total": len(all_pairs),
                "pairs_evaluated": len(pairs),
                "interference_informative": bool(pairs) and all(p["informative"] for p in pairs),
            }
        )
    skipped = sum(1 for layer in layers if layer.get("status") == "skipped")
    report = {
        "status": "ok",
        "binding_version": binding_version(binding),
        "subspace_rank": subspace_rank,
        "layers": layers,
    }
    run_layers = [layer for layer in layers if "delta_pairs" in layer]
    report["interference_informative"] = bool(run_layers) and all(
        layer["interference_informative"] for layer in run_layers
    )
    if layers and skipped == len(layers):
        report["status"] = "skipped"
        report["reason"] = f"all {skipped} layers skipped (expert too large for gate input)"
    elif skipped:
        report["status"] = "partial"
        report["layers_skipped"] = skipped
        report["layers_run"] = len(layers) - skipped
    return _upgrade_for_errors(report)
