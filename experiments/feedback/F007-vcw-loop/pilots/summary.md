# F007 pilots (seeds [1007, 1008, 1009]), not evidence

- seed 1007: scale pinned, hard_failures 0, counters not zero: none, coverage {'min_kind_share_permille': 253, 'min_hard_events_per_code': 1020, 'repairs_applied': 2582, 'exhausted_runs': 167, 'births': 1}, elapsed 159 s
- seed 1008: scale pinned, hard_failures 0, counters not zero: none, coverage {'min_kind_share_permille': 248, 'min_hard_events_per_code': 1219, 'repairs_applied': 2538, 'exhausted_runs': 173, 'births': 1}, elapsed 167 s
- seed 1009: scale pinned, hard_failures 0, counters not zero: none, coverage {'min_kind_share_permille': 232, 'min_hard_events_per_code': 1260, 'repairs_applied': 2520, 'exhausted_runs': 161, 'births': 1}, elapsed 156 s

# F007 vcw-loop: INVALID

Seeds: 1007, 1008, 1009.

- seed 1007: valid; 0 coverage rules unmet, 0 falsifiers tripped, 0 controls failed
- seed 1008: valid; 0 coverage rules unmet, 0 falsifiers tripped, 0 controls failed
- seed 1009: valid; 0 coverage rules unmet, 0 falsifiers tripped, 0 controls failed
- problem: controls failed over the seeds: ['null_tie_full_vs_shuffled']

Margin: 3.0 points. Differences are in points of verified-success rate per request, paired within seed, mean over seeds, one-sided 95% t bounds.

| comparison | mean | lower | upper | reading |
|---|---|---|---|---|
| full - gate | +7.50 | +6.70 | +8.30 | above the margin |
| full - bon | +6.60 | +5.60 | +7.60 | above the margin |
| no_l3 - gate (robustness) | +7.38 | +6.74 | +8.01 | above the margin |
| no_l3 - bon (robustness) | +6.48 | +5.71 | +7.24 | above the margin |
| full - no_workspace | +0.00 | +0.00 | +0.00 | non-discriminating in the primary world (see the secondary hypotheses) |
| full - no_finding | +15.97 | +15.29 | +16.66 | helps |
| full - no_selfmodel | +0.00 | +0.00 | +0.00 | non-discriminating in the primary world (see the secondary hypotheses) |
| full - no_l3 | +0.12 | -0.18 | +0.43 | non-discriminating in the primary world (see the secondary hypotheses) |
| no_l3 - no_finding | +15.85 | +15.46 | +16.24 | helps |
| oracle - full (upper reference) | +0.20 | -0.14 | +0.54 | descriptive |

Secondary hypotheses (preregistered, margins derived from the planning model, never gate the verdict):

- workspace: +2.33 [+1.71, +2.94] against margin +0.90: SUPPORTED (a harness-defined reader of the broadcast: an implementation check, not the value of the workspace)
- l3: +2.96 [-5.68, +11.59] against margin +5.40: NOT_SHOWN (low power, about 0.2 at the modelled gain: a NOT_SHOWN or REFUTED here is not evidence against L3)
- self_model: UNTESTED (no discriminating condition in this design)

Controls:

- shuffled_loses: +14.86 [+13.59, +16.12] passes
- gate_is_trained: +8.47 [+7.89, +9.06] passes
- bon_beats_single_shot: +35.76 [+35.06, +36.46] passes
- oracle_is_an_upper_reference: +0.20 [-0.14, +0.54] passes
- null_tie_no_l3_vs_gate: -0.20 [-1.32, +0.92] passes
- null_tie_no_l3_vs_no_finding: -0.20 [-1.32, +0.92] passes
- null_tie_no_l3_vs_shuffled: +0.41 [-0.68, +1.50] passes
- null_tie_full_vs_gate: -0.25 [-1.47, +0.97] passes
- null_tie_full_vs_no_finding: -0.25 [-1.47, +0.97] passes
- null_tie_full_vs_shuffled: +0.36 [-0.88, +1.59] FAILS
- workspace_withheld_changes_nothing: discordant requests 0 passes
- one_harness_for_every_seed: passes (1 distinct source digests)
- first_draft_is_shared: 0 (arm, request) pairs whose first draft differed from the full arm's, L3 population 387 passes
- harness_agrees_with_planning_model: passes ({"full": {"measured": 0.95825, "predicted": 0.962, "passes": true}, "gate": {"measured": 0.88325, "predicted": 0.883, "passes": true}, "bon": {"measured": 0.8922500000000001, "predicted": 0.891, "passes": true}})
- no_arm_at_ceiling_or_floor: passes ({"full": 0.95825, "gate": 0.88325, "bon": 0.8922500000000001})

Cost per request in the primary world (mean over seeds):

| arm | success | verifier calls | Pod calls | microseconds |
|---|---|---|---|---|
| full | 0.958 | 1.618 | 1.639 | 716 |
| gate | 0.883 | 1.670 | 1.690 | 313 |
| bon | 0.892 | 1.685 | 1.685 | 237 |
| no_workspace | 0.958 | 1.618 | 1.639 | 697 |
| no_finding | 0.799 | 1.774 | 1.799 | 353 |
| no_selfmodel | 0.958 | 1.618 | 1.639 | 711 |
| no_l3 | 0.957 | 1.619 | 1.640 | 701 |
| shuffled | 0.810 | 1.763 | 1.789 | 742 |

Descriptive world mild (never gates the verdict):

- full_vs_gate: +5.14 [+4.04, +6.24]
- full_vs_bon: +3.68 [+3.55, +3.82]
- full_vs_no_workspace: +0.00 [+0.00, +0.00]
- full_vs_no_finding: +10.94 [+10.03, +11.85]
- full_vs_no_selfmodel: +0.03 [-0.05, +0.10]
- full_vs_no_l3: +0.13 [-0.12, +0.39]
- full_vs_shuffled: +9.24 [+8.70, +9.78]
- full_vs_oracle: -0.38 [-0.92, +0.17]

Descriptive world ctx (never gates the verdict):

- full_vs_gate: +2.47 [+1.30, +3.64]
- full_vs_bon: +8.91 [+8.22, +9.59]
- full_vs_no_workspace: +2.33 [+1.71, +2.94]
- full_vs_no_finding: +5.51 [+4.74, +6.27]
- full_vs_no_selfmodel: +0.00 [+0.00, +0.00]
- full_vs_no_l3: +0.22 [-0.14, +0.57]
- full_vs_shuffled: +4.44 [+3.72, +5.16]
- full_vs_oracle: -0.18 [-0.56, +0.21]

Null world, descriptive (what a repair Pod is worth where nothing is specialised):

- full_vs_bon: +10.38 [+8.87, +11.89]
- gate_vs_bon: +10.63 [+10.04, +11.22]

Limitations: the expert Pods are seeded simulations with declared skill tables; whether finding routing helps is decided by those tables (the assumption under test), and the null world shows only that the harness adds no advantage of its own where the tables are equal; a trained gate over task features can capture at most the kind-level skill; in the declared worlds the codes of one kind are served best by different Pods, so a task-feature gate is handicapped by construction, which is the structure finding routing exploits; no Pod reads the workspace broadcast outside the descriptive world that declares it, so the workspace ablation is an exact tie by construction (a wiring check), not a measurement of the workspace's value; the self-model acts only through bounded hint boosts; the ablation measures that path alone; L3 is Birth only, on one code that no initial Pod can repair; the baselines have no L3, and the effect of L3 is shown by the no-L3 arm; the learned arms are trained once per run seed on pilot-domain traffic with no online learning during evaluation; the reported intervals are over seeds and so include the variance of training; five seeds give a one-sided t interval with four degrees of freedom, which is wide; a small true gain is NOT_SHOWN, not absent; the workspace, the self-model and L3 are not discriminated by the primary world (the tie of the workspace is by construction, the self-model has no discriminating condition in this design and is untested, L3 acts on one rare code); the workspace and L3 have preregistered secondary hypotheses with margins derived from the planning model, the workspace secondary tests a harness-defined reader of the broadcast and not the workspace's value, and the L3 secondary has low power; best-of-N lacks any repair Pod, so a gain over it bundles having a repair Pod with routing it; the routing claim rests on the contrasts to the gate and to the shuffled map; success is verified success under the real verifiers; a task whose verifier is blind to a defect would count as success (the independent oracle counts these as false accepts, and none was expected).


(The verdict line above applies the confirmatory rule to pilot data for illustration only.)
