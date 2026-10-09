# F006-verifier-driven-expert-lifecycle

## Hypothesis
[GPU; extends M013] Do L3 structural operators driven by verifier statistics (Birth when a finding class has no expert that clears it; Merge when two experts clear the same findings with overlapping subspaces; Prune by AIMER score and usage; Freeze and Reactivate) keep or improve verified success at equal compute and equal active parameters compared with a static expert set? Every operator is a proposal, gated by the ForgettingGate, interference measurement and the verifier, and committed only as a ledger event.

## Primary metrics
verified success per finding class; average forgetting on the public suite; ForgettingGate and interference pass rate of proposals; training FLOPs including discarded candidates; active parameters at every stage; expert count over time; paired differences over seeds

## Baseline
static expert set at equal training compute and active parameters; the M013 heuristic baseline (threshold rules on load and overlap); verifier-statistics policy with the statistics shuffled across finding classes (negative control)

## Falsification
Falsified if the verifier-statistics-driven lifecycle does not beat BOTH the static set and the M013 heuristic baseline on cumulative verified success per unit of compute by a pinned margin with a one-sided 95% paired bound over at least 5 seeds, if its forgetting exceeds the ForgettingGate bound, or if any structural change is committed without passing the gates (hard counter gate_bypasses above 0). Shares its infrastructure and pins with M013, which tests the lifecycle without verifier statistics.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
