# ADR-0021 — O2: A Hybrid Plain-Model Baseline

## Status
Accepted by the project owner on 2026-10-06. This records owner decision O2 of `docs/RECOMMENDATIONS_20260924.md` (Step 6), which asked which backbone serves as the plain-model baseline and as the continued-pretraining backbone (ADR-0015). It is a decision record only: no experiment was run, frozen or declared running by it.

## Context
O2 gated Step 1 of the recommendations: `scripts/check_research_gates.py` refuses to let M001 to M005 be declared running until a plain-model baseline is pinned. The recommendation was option (a), one small pretrained open model, pinned once, used for both the baseline and continued pretraining. The repository already pins a different baseline: `research/baselines/plain_model/config.toml` names `ptr-burn-a0@matched-cross-attention-v1`, a plain token-to-six-latent cross-attention model with no typed slots, no latent recurrence, no router and no external memory, whose parameter and FLOP counts match A0 to within 1 percent (`compute_gate.status = "matched"`), and the Burn harness already trains it in the same process as an A0 arm (the `paired` phase). A pretrained LM baseline needs GPU training that the project does not run yet (O1 defers how A0 grows).

## Decision
The project owner decided on 2026-10-06, option (c), hybrid:

1. For the A0-scale tier of M001 to M004, the baseline is the already pinned A0-matched plain model (`research/baselines/plain_model/config.toml`, `ptr-burn-a0@matched-cross-attention-v1`).
2. A small pretrained open LM baseline is deferred to the GPU phase, as a follow-up to O1. Until it exists, all M00x claims are restricted to A0/synthetic scale.
3. M005 (epistemic calibration) has no matching plain arm at the A0 tier. It is out of scope for the A0 tier until the LM baseline exists.

## Consequences
- The A0-scale comparison in M001 to M004 can name its baseline today. Their manifests stay `planned`: they still lack an entrypoint and a clean freeze commit, and the archived A0 records remain INCONCLUSIVE (see each README). The README text of M001 to M005 points here instead of waiting on an owner decision.
- A result under this baseline supports at most a statement about A0 on the synthetic task. It is not evidence that PTR mechanisms beat a pretrained language model, and no document may say so.
- M005 makes no claim at the A0 tier.
- The pretrained LM baseline, its backbone, revision and pin, are a separate later decision. Making it must not reinterpret A0-scale results.

### What this does NOT unblock
- **E002**: the original recommendation said O2 unblocks E002. It does not: E002 needs the pretrained backbone, which this decision defers.
- **R003** (`experiments/runtime/R003-coding-domain-adaptation/`): its baseline is the unmodified pretrained base model, so it still waits on a pinned open backbone.
- **ADR-0015** (continued pretraining for a coding Pod): stays Proposed. No backbone was chosen for continued pretraining, and O3 still recommends pausing it.
- Any declaration of M001 to M005 as running, which still needs an entrypoint, a frozen preregistration and the evidence gates in `scripts/check_research_gates.py`.
