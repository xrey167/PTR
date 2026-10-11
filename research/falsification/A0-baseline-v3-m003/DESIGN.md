# A0 baseline study v3-m003: design

The successor study of **M003-v3** (latent recurrence) of the A0 baseline study v2, registered as **M003-v4**. Every threshold is in [`criteria.toml`](criteria.toml), the frozen text is [`PREREGISTRATION.md`](PREREGISTRATION.md), the decision procedure is `scripts/aggregate_a0_baseline_v3.py` (the v2 decider, imported unchanged). The reasoning behind arms, contrasts, gates, rules, record identity, interruption, metrics, compute budget and threats is that of [`../A0-baseline-v2/DESIGN.md`](../A0-baseline-v2/DESIGN.md) and is **not restated**: nothing in it changes.

## Why a successor: an interruption, not an outcome

M003-v3 stopped for a host event. Its seed 71 was killed by a container restart (run record `started`, no output, no exit code; seed 101 never started). The v2 interruption rule makes that INCONCLUSIVE-INTERRUPTED, forbids rerunning or completing the study and says that a successor with new ids, a new directory, preregistration and freeze is created by the owner if the question is still wanted. The owner wants it answered (decision of 2026-10-10). The question has never been answered: M003-v3 has no statistic and no verdict, its completed seeds (17, 29, 43) are recorded as unobserved and are neither read nor pooled here.

## What is identical and what is new

Identical to v2, value for value (a test compares `criteria.toml` with the v2 file): protocol, data and fold tokens, seeds, learning-rate grid and rule, arms and pins, contrasts and compute modes, every gate and threshold, decision rules, record-identity rule, interruption rule, the runner (`model/burn-a0`, unchanged), the baseline pin.

| | v2 (M003-v3) | v3-m003 (M003-v4) |
|---|---|---|
| ids | M003-v3, M003-v3-primary / -nonlinearity / -bundle | M003-v4, M003-v4-primary / -nonlinearity / -bundle |
| directory | `research/falsification/A0-baseline-v2/` | `research/falsification/A0-baseline-v3-m003/` |
| studies | M001-v6, M002-v11, M003-v3 | M003-v4 only (the other two are decided in v2, not repeated; M004 stays deferred) |
| decider | `aggregate_a0_baseline_v2.py` | `aggregate_a0_baseline_v3.py`, the v2 file imported and pinned, rebinding only the criteria name, default path, loader (adds the v2 pin) and reported script digest |
| sweep | `M003-v3-sweep` | none of its own: reuses `M003-v3-sweep` and its selection |

### Why the decider needs a new file

The v2 decider hard-codes the criteria name `A0-baseline-v2`, so it refuses a criteria file of another name, and its digest is frozen in the prepared M001-v6, M002-v11 and M003-v3, so it cannot be edited. Keeping the name `A0-baseline-v2` would label M003-v4's outputs as the old study. The v3 file is therefore a thin layer (no statistics; it also publishes run.json in the gate form: preregistration digests and seed_records) over the unchanged v2 file; the experiment pins both (`decision_script`, `decision_base`), the v2 file also through `decision.parent_sha256`, and the v1 decider and core through the criteria and the config as in v2.

### Why the sweep is reused

M003-v4 depends on a sweep only through `lr_file` (read by the runner, checked by the decider with `--lr-file`) and the name `sweep_experiment`. `M003-v3-sweep` completed with exit 0 and its selection is committed in `../A0-baseline-v2/lr_M003-v3.tsv` (commits `b65b24f3`, `bc0f75f6`, before any M003-v3 confirmatory seed). The sweep is validation-only, uses seed 17 with salted initialisation, and depends only on inputs that the two criteria files share. A new sweep would recompute the same computation on the same validation data (the same selection is expected; a float-level difference could only change a near-tie, which is noise and not information) at the cost of about 24 minutes and one more exposure to a host interruption, and it could only differ from the existing selection by noise, not by information. The reuse is made checkable: `lr_M003-v4.tsv` is a byte copy of the v2 file, `[sweep_reuse]` in `criteria.toml` names the sweep and the file, and the research gate refuses a frozen M003-v4 whose lr file differs from the original or whose `sweep_experiment` is not that sweep, or whose sweep is not frozen. If the owner prefers a fresh sweep, `M003-v4-sweep` would have to be registered, run and its selection frozen before the freeze of M003-v4; that needs a different generator path and is not done here.

## Seeds

17, 29, 43, 71 and 101 as in v2. They are new runs of a new experiment under a new id; nothing recorded by M003-v3 is read. The runs are deterministic on CPU, so seeds 17, 29 and 43 are expected to recompute bit for bit what M003-v3 computed for them; that is harmless for the decision because M003-v3's rows were never read (D1), and it is stated here so the equal numbers are not mistaken for independent replication of those three seeds.

## Budget

M003 only: about 1.6 min per fold and seed of training, about 24 minutes of evaluation (5 seeds x 3 folds x 5 arms), plus a cold release build per seed (about 3.6 min on a loaded host): about 45 to 60 minutes serial. A host interruption makes the study INCONCLUSIVE-INTERRUPTED again (D-procedure in `../A0-baseline-v2/DEVIATIONS.md`); run it on an idle host.

## Artifacts and procedure

1. The learning-rate file is `lr_M003-v4.tsv` (frozen, copy of the v2 selection).
2. Evaluate M003-v4 for the five seeds (`--phase baseline-v1`), one hardened run record per seed, each committed before the next seed.
3. Decide: `python3 scripts/aggregate_a0_baseline_v3.py --study M003-v4 --lr-file research/falsification/A0-baseline-v3-m003/lr_M003-v4.tsv --criteria-sha256 <frozen digest> --paired-ci <results/paired-ci-95.json> --decision <results/DECISION.toml> <five run records>` (exit 0 with a decision, 2 with a refusal).
4. Report every verdict, nulls and INCONCLUSIVE ones included.
