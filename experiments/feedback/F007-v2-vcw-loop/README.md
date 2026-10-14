# F007-v2-vcw-loop

## Hypothesis
Whether the verifier-closed loop with finding-routed repair (VcwLoop with finding router, learned router promoted through the off-policy gate, workspace, self-model, L3 through the ledger) verifies more requests at a matched verifier-call budget than a trained softmax gate over the same Pods and than best-of-N with the same verifier, on seeded SIMULATED expert Pods with declared per-(Pod, finding code) skill tables over arithmetic, JSON-schema and code tasks judged by the real ptr-verifier verifiers. Mechanism check: a gain is a consequence of the declared tables, a homogeneous-skill world must tie, and nothing is claimed about any model. The routing claim rests on the contrasts to the gate and the shuffled map, not on best-of-N, which has no repair Pod. Successor of F007 (issue #141): F007 is completed and frozen at f007-prereg and its records are not patched. The code under test changed after it: a repaired candidate's authors now keep its proposer and earlier repairers, so the loop and admit_verified_answer refuse a proposer revoked between rounds; the trace digest domain is v2; the payload digests of the simulated and F007 tasks and PodOutput::digest are versioned and length-prefixed; the simulated trainer's collateral capability is length-prefixed; vcw_host install replaces an existing Pod only for a reactivation; the attention schema's generation table evicts the least recently seen Pod. Same hypothesis, metrics, baselines, margins and falsification as F007; trace digests are not comparable with F007's.

## Primary metrics
false_accepts, unverified_returns, double_verifications, budget_violations, routing_outside_admissible, verdict_mismatches, verdict_overrides, verifier_call_mismatches, calls_after_acceptance, l3_unadmitted, loop_errors, product_panics, exhausted_as_success, control_shuffled, control_null, control_baseline, control_workspace, control_ctx, control_training, control_grid_edge, control_oracle, control_bon, control_l3, control_first_draft, coverage_unmet, worlds

## Baseline
learned softmax gate over Pods (task features only); best-of-N with the same verifier at matched verifier calls

## Falsification
Per seed: an integrity counter above zero (accepted candidate failing independent re-verification, budget overrun, routing outside the admissible set, a verifier verdict not followed, an L3 commit without admission, a loop error or panic) is FALSIFIED; a failed control or unmet coverage rule (shuffled map not clearly worse, null world not tied, gate or best-of-N or learned router not trained and promoted, a tuning grid edge, oracle below the full loop, best-of-N not above single shot, ablations not wired) is INVALID and takes precedence. Across the 5 seeds, on the per-seed paired difference of verified-success rate in world hetero with one-sided 95% t bounds: COMPLETED iff the lower bound of full minus each of the gate and best-of-N exceeds the margin of criteria.toml (3 points); FALSIFIED iff the upper bound against either is below it; NOT_SHOWN otherwise; the harness must agree with power.py within 2 points. Secondary hypotheses (workspace in world ctx, L3 on requests carrying the unrepairable code) are reported and never gate the verdict; the self-model is untested.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.

## Relation to F007
F007 (`experiments/feedback/F007-vcw-loop`) is completed and frozen (tag `f007-prereg`); its records
are not patched, and its `results/STALE.toml` names this successor. This experiment is only
registered (`planned`): the harness is still `ptr-bench vcw-loop`, whose records name F007, and the
preregistration, criteria, mutation plan, pilots and entrypoint are to be prepared before any
freeze. What changed under test (issue #141): a repaired candidate's authors keep its proposer and
earlier repairers, so a proposer revoked between rounds is refused by the loop and by
`admit_verified_answer`; the trace digest domain is `ptr.runtime.vcw.trace.v2`; the payload
digests of the simulated and F007 tasks are domain-tagged and length-prefixed; `PodOutput::digest`
is version 2 (counted, domain-tagged); the simulated trainer's `sim:collateral:` capability is
length-prefixed; `vcw_host` install replaces an existing Pod only for a reactivation; the attention
schema's generation table evicts the least recently seen Pod; a compacted snapshot carries the scope
registry. Trace digests are therefore not comparable with F007's.
