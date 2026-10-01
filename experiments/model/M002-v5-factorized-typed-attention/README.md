# M002-v5 — factorized typed pair attention

M002-v5 is a planned confirmatory component study. It directly compares the
same repaired A0 architecture with `FactorizedV2` versus `Off`; the control
executes the same pair-attention operations and zeros only the resulting bias.
It therefore fixes M002-v4's attribution failure without rewriting any v4
record.

The non-claimable pilot uses seeds 7 and 13 on `evidence-interventional` and the
separate `claim-temporal-development` fold. It evaluates rank 8/16, bias limit
1/2, and metadata dropout 0/0.1 at the pre-pilot learning rate `0.005`. The
selection rule in `PLAN.toml` is fixed, but its selected architecture values
remain deliberately unpinned while the experiment is `planned`. M002-v5 cannot become `prepared` until the pilot
selection, dataset lock, criteria, code and entrypoint are committed and their
digests are recorded in `config.toml` and `experiment.toml`.

From a clean development commit, run the resumable pilot into its versioned
archive and then apply the frozen selector to the complete raw stdout set:

```sh
python scripts/run_m002_v5_pilot.py --output-dir experiments/model/M002-v5-factorized-typed-attention/pilot/raw
python scripts/select_m002_v5_pilot.py experiments/model/M002-v5-factorized-typed-attention/pilot/raw/*.stdout \
  --output experiments/model/M002-v5-factorized-typed-attention/pilot-selection.json
```

`pilot-selection.json` is development evidence only. It records the fixed
rule's choice for the freeze and can never contribute a confirmatory metric.
Before the study may become `prepared`, the research gate recomputes it from
all 16 committed stdout/sidecar/stderr triples, checks their source commit and
dataset lock, and requires the frozen rank, bias limit and dropout to match
exactly.

The confirmatory unit is a seed, not a fold. Each of the five declared seeds
runs both arms over the three confirmatory folds in one process. Fold metrics
are averaged within a seed before the paired 95% Student-t interval is formed.
`scripts/aggregate_m002_v5.py` implements the conjunctive gates, including
accuracy, NLL, ECE15, abstention, matched parameters/FLOPs and p95 latency.
After the frozen five-seed run it must write the immutable decision to
`results/m002-v5-decision.json`; this exact path is the only artifact the M009
research gate accepts.

A positive claim is legal only when every gate passes. Every missing or failed
gate is `INCONCLUSIVE/NO-GO`. The only permitted positive wording is:

> Factorized Typed Pair Attention improves on operator_routing_v2 compositional
> routing accuracy at matched compute without measurable calibration
> degradation.

M009 remains locked until such a commit-bound PASS exists. The planned state in
this directory is not evidence and does not authorize Learned Backend use.
