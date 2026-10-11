"""The plateau training protocol shared by the stage-A harnesses (needs torch; import it after the environment check).

Every arm is trained identically: Adam, gradient clipping, an evaluation on a held-out in-distribution validation set every
`eval_every` steps, the best-validation weights restored, and a stop when the validation accuracy has not improved by
`min_delta_permille` for `patience` evaluations, or at the step cap (then the arm is UNCONVERGED, which the aggregator turns into a
guard). The comparison is of plateau quality, not of optimisation speed."""

from __future__ import annotations

import math

import torch

from common import permille


def train_to_plateau(net, optimiser, step_loss, validate, cfg: dict, max_steps: int | None = None):
    """Train `net`. `step_loss(step)` draws a batch, observes whatever the harness audits (lengths, hashes) and returns the loss
    tensor; `validate(net)` returns (correct, total) on the held-out validation set. Returns (net with the best weights restored,
    info, last loss, all losses finite)."""
    cap = cfg["max_steps"] if max_steps is None else max_steps
    curve, counts, window, last_loss, finite = [], [], [], 0.0, True
    best, best_state, since, converged, step = -1, None, 0, False, 0
    while step < cap:
        step += 1
        loss = step_loss(step)
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
        optimiser.step()
        last_loss = float(loss)
        window.append(last_loss)
        if not math.isfinite(last_loss):
            finite = False
        if step % cfg["eval_every"] == 0:
            c, t = validate(net)
            curve.append(permille(c, t))
            counts.append((c, t))
            if permille(c, t) > best + cfg["min_delta_permille"]:
                best, since = permille(c, t), 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                since += 1
            if since >= cfg["patience"]:
                converged = True
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    tail = window[-cfg["eval_every"] * cfg["loss_window_evals"] :]
    info = {
        "curve": curve,
        "counts": counts,
        "converged": converged,
        "steps": step,
        "cap": cap,
        "best_val": best,
        "loss_window_micro": int(1e6 * sum(tail) / len(tail)) if finite and tail else -1,
    }
    return net, info, last_loss, finite


def audit(net, info: dict, validate, cfg: dict) -> tuple[int, int]:
    """Audits of the training loop, independent of it: (checkpoint mismatch, convergence mismatch), each 0 or 1. The restored weights
    must reproduce the best validation accuracy, and the convergence flag and the step count must agree with an independent
    recomputation of the stopping rule from the recorded curve."""
    from common import plateau_reached

    c, t = validate(net)
    checkpoint = int(permille(c, t) != info["best_val"])
    stopped = plateau_reached(info["curve"], cfg["patience"], cfg["min_delta_permille"])
    expected_steps = len(info["curve"]) * cfg["eval_every"] if stopped else info["cap"]
    convergence = int(stopped != info["converged"] or info["steps"] != expected_steps)
    return checkpoint, convergence
