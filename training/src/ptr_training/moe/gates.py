"""PTR governance gates via the optional `ptr_py` binding.

The binding is built by a separate package. When it is not importable every gate reports
`status: "skipped"` with the reason; nothing is ever simulated. Binding surface used
(bindings/ptr-py/src/governance.rs):

* `aimer_score(list[float])` on each expert's flattened (gate_up + down) weights, and
  `rank_experts_for_pruning(list[(id, list[float])])` per layer;
* `subspace_overlap(a, b)` / `chance_overlap(a, b)` on `dim x k` row-major matrices whose
  column spaces are compared: here the `[hidden, intermediate]` change of each expert's
  `down_proj` between two snapshots (its output subspace in R^hidden).

If a call raises (`PtrLineageError`, signature mismatch) the failure is recorded verbatim
as `status: "error"` and the whole report becomes "error"; it is never papered over.
The aggregate status is "ok" only when every layer ran; "partial" (with counts) when some
layers were skipped, "skipped" (with a reason) when all were.
"""

from __future__ import annotations

import importlib

SKIP_REASON = "binding not installed: gates skipped"
MAX_FLAT_ELEMENTS = 1_000_000  # per vector handed to the binding; larger is skipped, not truncated


def load_binding():
    try:
        return importlib.import_module("ptr_py")
    except ImportError:
        return None


_NEEDED = ("aimer_score", "rank_experts_for_pruning", "subspace_overlap", "chance_overlap")


def binding_skip_reason() -> str | None:
    """None when the binding is importable and complete, else the reason gates are skipped."""
    binding = load_binding()
    if binding is None:
        return SKIP_REASON
    missing = [n for n in _NEEDED if not hasattr(binding, n)]
    if missing:
        return f"binding lacks {missing}: gates skipped"
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


def _call(fn, *args):
    try:
        return {"status": "ok", "value": fn(*args)}
    except Exception as error:  # noqa: BLE001 - recorded verbatim, never swallowed silently
        return {"status": "error", "error": f"{type(error).__name__}: {error}"}


def run_gates(weights_before: list, weights_after: list, max_pairs_per_layer: int = 6) -> dict:
    """weights_*: output of `flatten_experts`. Returns a JSON-ready gate report."""
    reason = binding_skip_reason()
    if reason is not None:
        return {"status": "skipped", "reason": reason}
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
        pairs = []
        n = len(d_layer)
        for a in range(n):
            for b in range(a + 1, n):
                if len(pairs) >= max_pairs_per_layer:
                    break
                da, db = d_layer[a]["down"], d_layer[b]["down"]
                pairs.append(
                    {
                        "experts": [a, b],
                        "subspace_overlap": _call(binding.subspace_overlap, da, db),
                        "chance_overlap": _call(binding.chance_overlap, da, db),
                    }
                )
        layers.append(
            {"layer": li, "aimer_score": aimer, "prune_ranking": ranking, "delta_pairs": pairs}
        )
    skipped = sum(1 for layer in layers if layer.get("status") == "skipped")
    report = {
        "status": "ok",
        "binding_version": getattr(binding, "__version__", "unknown"),
        "layers": layers,
    }
    if layers and skipped == len(layers):
        report["status"] = "skipped"
        report["reason"] = f"all {skipped} layers skipped (expert too large for gate input)"
    elif skipped:
        report["status"] = "partial"
        report["layers_skipped"] = skipped
        report["layers_run"] = len(layers) - skipped
    return report
