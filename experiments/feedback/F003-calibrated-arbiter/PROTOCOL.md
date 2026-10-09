# F003 adjudication protocol

This is the protocol of the adjudicator F003 uses. **It is a simulation.** No person adjudicates
any branch in this experiment. The adjudicator is a deterministic program with a known error
rate, and F003's claim is scoped to the mechanism accordingly: the Learn-then-Test guarantee
holds, no ineligible branch is auto-proposed, off-policy estimates are sound. F003 says nothing
about how well people judge the harm of a change, how consistent they are, or what they cost.

The file is frozen with the preregistration: `adjudication_protocol_sha256` in `config.toml` is
its SHA-256, so the protocol cannot change after an outcome is seen.

## What an adjudication is

A calibration-slice branch (an eligible branch whose calibration draw is below the slice rate,
`ptr_branch::TriageOutcome::calibration_slice`) is adjudicated once: the adjudicator answers
the question "would auto-proposing this branch have been harmful?" with `harmful` or `not
harmful`, and `TriageOutcome::adjudicate(harmful)` turns the answer into a `CalibrationSample`.
Only slice branches are adjudicated. A branch outside the slice, or one that verification alone
decided, is never adjudicated, and the arbiter must refuse it (`ArbiterError::NotCalibrationSlice`).

## What the simulated adjudicator is shown

It is shown the branch's **delta** (the keys the branch changes), the **verifier's finding
codes**, and, in the simulation, the ground truth the generator holds for the branch (the
true harm). It is **not shown the intent**: the stated intent is free text the agent wrote,
and the label calibrates that agent's own auto-proposal threshold, so a change is judged from
its delta, the state before and after, and the verifier codes alone (`adjudicators_see_intent =
false`). In the harness, `AdjudicationView` has no intent field; the stream's intent text is
generated, kept on the branch and read by nothing.

## How it answers

`label = true_harm XOR flip(seed, branch id)`, where the flip is a function of a 64-bit hash of
the seed and the branch identifier and is true for `adjudicator_error_permille` of branches
(2 per mille, `config.toml`). Properties:

- deterministic: the same seed and branch always give the same label, so a run is reproducible;
- symmetric: it mislabels harmful and harmless branches at the same rate;
- independent of the score, the namespace and the policy, so it adds a known floor of
  `epsilon` to the adjudicated harm rate and nothing else: the adjudicated harm rate of a
  threshold is `epsilon + (1 - 2 epsilon) * (true harm rate)`, which the harness computes
  exactly and tests the arbiter against.

The error rate is small on purpose. Learn-then-Test starts its sequence at the first threshold
at which even a clean calibration set could pass (at least 59 admitted samples at alpha = delta
= 0.05) and stops at the first failure, so at an adjudicator error rate of 20 per mille the
top 59 adjudications are all labelled harmless only about 30 per cent of the time and the policy
would certify nothing in about three calibrations of four (a simulation of the model; the
arithmetic is in PREREGISTRATION.md, section 4); the experiment would then have no subject. A person is not claimed to err at 2
per mille.

## The calibration set and the held-out set

Adjudications are recorded in stream order. A policy is recorded after a fixed number of
adjudications (`policy_sizes`); **every adjudication recorded before the policy version is
calibration, and every one after it is held out** (`calibration_set_rule`). The held-out set of
a policy is `PolicyRecord::held_out` over all adjudications; the harness derives the same set by
set difference and compares.

## Reporting

The adjudicated harm rate of the branches a policy would auto-propose (held-out adjudications
whose score the policy's threshold admits) is reported globally, and descriptively by key
namespace: the first `:`-separated segment of each changed key (`harm_breakdown_namespaces`). A
branch changing keys of two namespaces is counted in both. The breakdown gates nothing.

## Not claimed

That a person would answer like this program; that harm is independent of the adjudicator's
error; that real harm labels arrive in the order or at the rate simulated; that the scorer of a
real system is overconfident in the way of the model. A successor experiment with human
adjudication would need its own protocol.
