# A0 baseline v3-m003 (M003-v4): results

Successor of the interrupted M003-v3 (owner decision 2026-10-10; `../A0-baseline-v2/DEVIATIONS.md` D3). Frozen at `d8da6c1f` (tag `a0-baseline-v3-m003-freeze`), five seeds 17/29/43/71/101, each run once, all exit 0; decided by `scripts/aggregate_a0_baseline_v3.py` (the pinned v2 decider under the v3 criteria name). Seeds 17, 29 and 43 are expected to be bit-identical recomputations of M003-v3's unread runs (stated before the freeze); M003-v3's rows were never read.

**Outcome: INCONCLUSIVE, label NO-GO-CALIBRATION, `go = false`, no quality claim.**

| Contrast | Role | Isolating | OOD accuracy delta (mean, 95% CI) | Failing gates | Verdict |
|---|---|---|---|---|---|
| full vs latent-0 (recurrence k=2 vs none, equal params) | primary | yes | +0.039 [+0.015, +0.063] | ood_nll_each_fold, ood_ece_each_fold, coverage | INCONCLUSIVE / NO-GO-CALIBRATION |
| full vs latent-linear (nonlinearity) | secondary | yes | +0.011 [-0.006, +0.027] | ood_accuracy, positive_accuracy_each_fold, ood_nll_each_fold, ood_ece_each_fold, coverage | INCONCLUSIVE |
| full vs plain (bundle) | bundle | no | +0.251 [+0.223, +0.279] | coverage | INCONCLUSIVE |

Reading: the tied latent recurrence raises OOD routing accuracy over no recurrence at equal parameters (the interval excludes zero), but calibration gets worse on at least one fold (raw NLL and ECE gates fail) and the treatment's coverage gate fails, so by the preregistered rule there is no quality claim. The nonlinearity contrast shows no resolved effect. The bundle contrast against `plain` is non-isolating and is not mechanism evidence. Scope: A0 on the synthetic operator-routing v2 benchmark only; no language-model claim. The learning-rate selection (reused from M003-v3-sweep) puts the recurrent arms at the upper grid edge, unbracketed (limit, unchanged).
