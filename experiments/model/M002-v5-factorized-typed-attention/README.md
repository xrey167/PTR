# M002-v5 — factorized typed pair attention

M002-v5 is a prepared confirmatory component study. It directly compares the
same repaired A0 architecture with `FactorizedV2` versus `Off`; the control
executes the same pair-attention operations and zeros only the resulting bias.
It therefore fixes M002-v4's attribution failure without rewriting any v4
record.

The non-claimable pilot uses seeds 7 and 13 on `evidence-interventional` and the
separate `claim-temporal-development` fold. It evaluates rank 8/16, bias limit
1/2, and metadata dropout 0/0.1 at the pre-pilot learning rate `0.005`. The
selection rule in `PLAN.toml` selected rank 8, bias limit 1, and metadata
dropout 0. The complete archived pilot binds this choice, the dataset lock,
criteria, code and entrypoint through digests in `config.toml` and
`experiment.toml`. Pilot values remain development-only evidence.

From a clean **detached** development commit, run the resumable pilot into its
versioned archive and then apply the frozen selector to the complete raw set:

```sh
python scripts/run_m002_v5_pilot.py --output-dir experiments/model/M002-v5-factorized-typed-attention/pilot/raw
python scripts/select_m002_v5_pilot.py experiments/model/M002-v5-factorized-typed-attention/pilot/raw/*.stdout \
  --output experiments/model/M002-v5-factorized-typed-attention/pilot-selection.json
```

The runner creates a second, private detached worktree at the exact source
commit and runs Cargo there. Each cell binds that source tree, the resolved
Cargo/Rust toolchain, a build-environment digest, host facts and the measured
Windows hardware profile. It permits only verified complete triples already
below `pilot/raw` on a resumed invocation; partial or extra artifacts refuse
instead of being overwritten.

`pilot-selection.json` is development evidence only. It records the fixed
rule's choice for the freeze and can never contribute a confirmatory metric.
Before the study may become `prepared`, the research gate recomputes it from
all 16 committed stdout/sidecar/stderr triples, including all three artifact
digests and execution provenance, checks their source commit and dataset lock,
and requires the frozen rank, bias limit and dropout to match exactly.

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
this directory is not evidence and does not authorize Learned Backend use. A
future M009 must additionally bind its configuration and checkpoint contract
to the selected FactorizedV2 rank, bias limit, dropout and repaired router; an
unlisted M009 cannot bypass this lock.

At freeze the selector also produces the digest-bound
`factorized-v2-contract.txt` artifact. A learned backend must consume it with
`ptr_burn_a0::M002V5Contract` before constructing the model and must use
`save_bound`/`load_bound` for its checkpoints. Repeating the digest in a TOML
file alone is not accepted by the M009 gate.
