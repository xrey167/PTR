# VCW-MoE GPU plan: preregistration-ready plans for the GPU experiments

Written 2026-10-09 (package G of the Verifier-Closed Workspace MoE goal). **A plan, not a result.**
It was written on a machine without a GPU, nothing in it has run, and no experiment named here has
evidence yet. It extends [`GPU_HANDOVER.md`](GPU_HANDOVER.md), which stays the source for machine
setup, the freeze-then-run sequence and the owner decisions OD-1 to OD-12.

Conventions (as in the handover):

- A statement without a mark comes from a repository file, a Hub API response (section 2) or the
  cited manifest.
- **Estimate** marks anything computed or guessed here. Every VRAM and wall-time figure is an
  estimate from the assumptions of section 3; none is a measurement.
- **Proposed** marks a value this plan suggests and the manifest leaves open ("a pinned margin").
  It becomes binding only when it is written into the experiment's `[preregistration]` table and
  signed by the freeze commit. Margins are re-checked against pilot-seed variance (section 4.6)
  and never against final-seed results.
- The experiment manifests F004 to F006 and C002 to C006 were read from the design worktree
  (`wt-dev-gw`), where another package still edits them; the quotations below are of that
  state. Reconcile this plan with them at merge.

The design context is [`architecture/37-pod-moe.md`](architecture/37-pod-moe.md) section 11 and
[`architecture/38-consciousness-indicators.md`](architecture/38-consciousness-indicators.md);
the welfare rules are [`MODEL_WELFARE.md`](MODEL_WELFARE.md). Indicator properties are functional
properties; nothing here claims or tests consciousness (principle 1 of doc 38).

## 0. What is in this package

| Artifact | What it is |
|---|---|
| `training/configs/vcw-{perft-r-sft,rlvr-router,repair-expert,self-model-probe}-{rtx4090,rtx5090}.toml` | Eight stage configs. They load and validate on a machine without torch (`ptr_training.vcw.config`). They are plans: their backend `vcw-<stage>` is not implemented by `ptr_training.run`, so `--execute` refuses them. |
| `training/src/ptr_training/vcw/` | `config.py` (loader, reuses the `trl-moe` parser for the `[moe.*]` numbers), `guard.py` (self-report guard), `budget.py` and `plan.py` (VRAM and wall-time estimates, GPU-hour ledger). Pure Python, no GPU code path. |
| `training/tests/test_vcw_*.py` | 45 tests, run without torch. |
| This document | Plans F004, F005, F006, C002 (GPU part), C003 (GPU part), C005, C006 and the relevant arms of M010 to M015. |
| [`GPU_HANDOVER.md`](GPU_HANDOVER.md) section 10 | The commands, prerequisites, expected outputs and the route back into experiment records. |

Not in this package, and listed as prerequisites in section 4.9: the stage entrypoints
(`ptr_training.vcw.*`), the Python verifier fabric with finding codes, the loop harness, the
composite-task generator, steering hooks, and every `aggregate.py`, `criteria.toml` and
`[preregistration]` table.

## 1. The stages and where they sit

```
OLMoE-1B-7B-0125 (frozen, bf16) + PERFT-R adapter experts and adapter router
  S0  vcw-perft-r-sft        SFT on gold solutions of the training partitions          (M011 emergent arm topology)
  S1  vcw-rlvr-router        RLVR, GSPO and SAPO arms, reward = VerificationReport      (adapter router and experts)
  S2  vcw-repair-expert      SFT on verified (finding, repair, pass) transitions        (8 classes x rank 2 vs 1 x rank 16)
  P   vcw-self-model-probe   probes on routing traces; no LLM weight is trained         (C002)
```

`T` (checkpoint production) is S0, S1 (one arm) and S2 per seed. T is a pipeline step, not an
experiment: it trains on training partitions only and evaluates nothing on held-out sets. Its
checkpoint digests are pinned in each consuming experiment's preregistration (section 4.7).

The OLMoE base router (`layers.N.mlp.gate.weight`) stays frozen in every stage, as in
`stage-moe-olmoe.toml`. "RLVR router training" trains the PERFT-R **adapter router**
(`peft_router`, `[H, 8]` per layer) and the adapter experts. Training the base router with
LoRA or a delta is a separate arm that nothing in this package configures (open decision D3).

## 2. Pins

Fetched from the Hugging Face Hub API on 2026-10-09 (`GET /api/{models|datasets}/<id>`, field
`sha`; no files were downloaded). Every pin in the existing manifests is still the current head of
its repository.

| Repo | Type | Revision (40-hex) | Last modified (Hub) | Used by |
|---|---|---|---|---|
| `allenai/OLMoE-1B-7B-0125` | model | `9b0c1aa87e34a20052389dce1f0cf01da783f654` | 2025-03-16 | every stage; tokenizer is the same repo and revision (M010 to M012 pins) |
| `allenai/OLMoE-1B-7B-0125-Instruct` | model | `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e` | 2025-02-04 | C005 only (base-versus-instruct provenance); never trained here |
| `openai/gsm8k` | dataset | `740312add88f781978c0658806c59bc2815b9866` | 2026-03-23 | math domain; M010 to M015 |
| `openai/openai_humaneval` | dataset | `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` | 2024-01-04 | code domain; M011, M013 |
| `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | dataset | `61fc0608cfd831fcfbbaa676ebdfef0ed963eeda` | 2026-04-29 | tool domain, file `BFCL_v3_simple.json`; M011, M013 |
| `cais/mmlu` | dataset | `c30699e8356da336a370243923dbaf21066bb9fe` | 2024-03-08 | ForgettingGate public suite and M014 slices |
| `google-research-datasets/mbpp` | dataset | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` | 2024-01-04 | **new**: out-of-distribution code evaluation (test split) |
| `EleutherAI/hendrycks_math` | dataset | `21a5633873b6a120296cce3e2df9d5550074f4a3` | 2025-01-12 | **new**: out-of-distribution math evaluation (levels 1 to 3, numeric answers) |

`Qwen/Qwen3-30B-A3B` (M015 teacher) is not used by this plan (OD-4 stands).

Software: the `moe` extra of `training/pyproject.toml` as locked in `training/uv.lock` and
verified in `training/src/ptr_training/moe/README.md`: torch 2.14.1, transformers 5.18.0,
trl 1.14.1, peft 0.21.2, datasets 5.0.1, accelerate 1.15.0, Python 3.11. `uv.lock` is unchanged by
this package.

Seeds: final seeds **17, 29, 43, 71, 101** (the seeds every manifest lists), each run once.
Pilot seeds **1001, 1002, 1003**: tuning, margin calibration and the sensitivity and ceiling checks
happen on these only. The loader refuses a pilot seed below 1001.

Selection digests of the new evaluation sets (section 4.1) do not exist yet; they are computed by
the corpus code at freeze time and pinned in each preregistration.

Hardware profile: `hardware/rtx4090.toml` (24 GiB) and `hardware/rtx5090.toml` (32 GiB) exist. The
F, C and M manifests still name `hardware/rtx3090.toml`; changing that is OD-1 and must happen
before the first prepared record of each. **One experiment runs on exactly one profile**: the
two profiles use different sequence budgets (section 3.3), so results do not mix across them.
This plan recommends `rtx4090` for every experiment (it is the owner's machine) and gives the
`rtx5090` figures as the alternative.

## 3. Estimates: method, assumptions, configs

### 3.1 Model facts used

OLMoE-1B-7B-0125: 16 layers, hidden 2048, expert intermediate 1024, 64 experts, top-8, vocab
50304, 6.919B parameters (Hub metadata), about 1.3B active. Of the parameters, 6.44B are expert
weights (about 12 GiB in bf16, an estimate from the parameter count) and 0.48B are everything else. The PERFT-R adapter of M011
(8 experts, rank 8, router, 16 layers) has 4,456,448 trainable parameters; 8 per-class LoRA
adapters of rank 2 on `q,k,v,o` and one generic rank-16 adapter both have 4,194,304.

### 3.2 Assumptions (constants in `ptr_training/vcw/budget.py`)

| Assumption | Value | Why it can be wrong |
|---|---|---|
| bf16 dense tensor peak, fp32 accumulate | 165.2 TFLOPs (4090), 209.5 TFLOPs (5090) | datasheet values from memory; verify |
| memory bandwidth | 1008 GB/s (4090), 1792 GB/s (5090) | same |
| achieved MFU of HF-eager adapter training | 0.04 / 0.10 / 0.20 (slow / mid / fast) | the largest unknown; MoE routing in eager mode is not efficient |
| FLOPs per token, adapters only | 4 x active (forward + input gradients), 6 x with checkpointing | frozen weights need no weight gradient |
| HF `generate` step time | 0.12 / 0.07 / 0.04 s, host-bound, same on both cards | depends on the CPU; the weight-read bandwidth bound is about 10 ms and never binds |
| verifier time per candidate | 0.08 / 0.03 / 0.01 s | sandboxed code tests dominate; 8 workers assumed |
| activations without checkpointing | 16 H + 6 top-k I elements per token per layer, bf16 | rough; error plausibly +-50% |
| logits memory | 10 bytes per vocabulary entry per token | bf16 logits, fp32 upcast, softmax copy |
| fixed overheads | 0.6 GiB CUDA context, x1.12 on dynamic memory, 1.5 GiB required free | |

VRAM is the maximum over the training phase and the generation phase (KV cache 128 KiB per token).
Wall time is a (slow, mid, fast) range; the tables print **fast / mid / slow**. A real run
replaces all of it: `metrics.json` reports `hardware.peak_cuda_memory_allocated_bytes`.

### 3.3 Per-profile protocol and the configs

The RL phase of the `trl-moe` backend uses no gradient checkpointing and one TRL micro-batch of
`prompts_per_step x num_generations` sequences (no accumulation). That, not the card, limits
the sequence budget:

| | RTX 4090 (24 GiB) | RTX 5090 (32 GiB) |
|---|---|---|
| S0 SFT | micro 4 x accumulation 8 (effective 32), padded 768, checkpointing, 2000 steps | micro 8 x 4, padded 768, checkpointing, 2000 steps |
| S1 RLVR | 1 prompt x 8 generations, prompt 128, completion 160, `mini_steps` 4, 400 steps per arm | 1 x 8, prompt 192, completion 256, `mini_steps` 4, 400 steps per arm |
| S2 repair SFT | micro 2 x 8, padded 1024, 1000 steps | micro 4 x 4, padded 1024, 1000 steps |
| P probe extraction | 16 x 320 tokens forward | 32 x 448 tokens forward |

| Config | Peak VRAM (estimate) | Wall time per run, fast / mid / slow | Fits with 1.5 GiB headroom |
|---|---|---|---|
| `vcw-perft-r-sft-rtx4090` | 16.4 GiB | 3.2 / 6.4 / 16.1 h | yes |
| `vcw-perft-r-sft-rtx5090` | 19.3 GiB | 2.5 / 5.1 / 12.7 h | yes |
| `vcw-rlvr-router-rtx4090` (both arms) | 21.1 GiB | 0.45 / 0.84 / 1.6 h | yes (1.4 GiB under the limit) |
| `vcw-rlvr-router-rtx5090` (both arms) | 25.2 GiB | 0.69 / 1.25 / 2.3 h | yes |
| `vcw-repair-expert-rtx4090` (one arm) | 15.5 GiB | 1.1 / 2.1 / 5.4 h | yes |
| `vcw-repair-expert-rtx5090` (one arm) | 17.4 GiB | 0.85 / 1.7 / 4.2 h | yes |
| `vcw-self-model-probe-rtx4090` | 14.9 GiB | 0.26 / 0.31 / 0.48 h | yes |
| `vcw-self-model-probe-rtx5090` | 17.4 GiB | 0.26 / 0.32 / 0.51 h | yes |

The loader recomputes these from the config and refuses a file whose declared `[vcw.estimate]`
differs by more than 5% or that does not fit; changing a batch size without redoing the estimate
fails the tests.

**Finding (estimate): the protocol pinned in M010 does not fit either card with this backend.**
M010 pins 512 completion tokens and a batch of 32 to 128 sequences (`batch_size / 8` prompts of 8
generations). One TRL micro-batch of 32 sequences of 768 tokens without checkpointing is about
94 GiB; with checkpointing about 37 GiB; a micro-batch of 8 sequences of 768 tokens with
checkpointing is about 19 GiB and fits both. M010 therefore needs the backend to enable RL gradient
checkpointing and `gradient_accumulation_steps` (prerequisite P4), or a successor with a smaller
protocol. The S1 sequence budgets above are chosen to avoid that dependency; they are short
(GSM8K reasoning is truncated at 160 tokens on the 4090 unless the SFT targets drop the calculator
annotations; that preprocessing is part of the S0 corpus pin, and the truncation rate is read on
pilot seeds).

### 3.4 GPU-hour ledger

Printed by `PYTHONPATH=training/src python3 -m ptr_training.vcw.plan`. Hours **fast / mid / slow**;
"units" is seeds unless stated; M-rows are per arm and cover tuning where the manifest pins it.

**RTX 4090**

| Exp | Item | Per seed (or unit) | Units | Total |
|---|---|---|---|---|
| T | checkpoint production: S0, S1 (1 arm), S2 (per-class + generic) | 5.6 / 11.2 / 27.7 | 5 | 28 / 56 / 138 |
| T | pilot allowance: 3 pilot seeds of T + learning-rate tuning (2 x S0) | 23.2 / 46.4 / 115.2 | 1 | 23 / 46 / 115 |
| F005 | 4 repair arms, <=3 rounds, ~50% failing candidates, +25% held-out classes | 1.0 / 1.9 / 3.6 | 5 | 5 / 10 / 18 |
| F004 | 8 arms, 35 sampled candidates per task in all | 4.2 / 7.7 / 14.3 | 5 | 21 / 38 / 72 |
| F006 | 4 arms standalone (4 domain stages ~ 1 S0 each, plus stage evals) | 13.1 / 26.2 / 65.3 | 5 | 66 / 131 / 326 |
| C002 | trace extraction + 5 probe arms | 0.6 / 1.1 / 2.0 | 5 | 3 / 6 / 10 |
| C003 | 4 arms x (1000-step RLVR + eval) | 2.7 / 5.0 / 9.7 | 5 | 14 / 25 / 48 |
| C005 | 12000 trials x 3 checkpoints, 100 new tokens, no training | 1.5 / 2.8 / 5.3 | 5 | 8 / 14 / 27 |
| C006 | 3 Pod adapters (~1 S0) + 4 arms x 800 composite tasks | 4.5 / 8.8 / 20.6 | 5 | 23 / 44 / 103 |
| M010 | per arm-seed, 2000 steps, 512-token protocol (time only; does not fit as is) | 6.2 / 12.0 / 26.1 | 50 | 312 / 598 / 1304 (+ tuning) |
| M011 | per arm: 8 tuning trials x 500 steps + 5 x 2000 steps = 7 S0 | 22.6 / 45.1 / 112.8 | 4 | 90 / 181 / 451 |
| M012 | per arm, same schedule | 22.6 / 45.1 / 112.8 | 3 | 68 / 135 / 338 |
| M013 | per arm (step counts are not pinned by the manifest: assumption) | 22.6 / 45.1 / 112.8 | 5 | 113 / 226 / 564 |

**RTX 5090** (different sequence budgets, section 3.3)

| Exp | Per seed (or unit) | Total |
|---|---|---|
| T (per seed / 5 seeds) | 4.6 / 9.1 / 22.4 | 23 / 46 / 112 |
| T pilot allowance | 18.8 / 37.5 / 92.5 | 19 / 38 / 93 |
| F005 | 0.8 / 1.6 / 3.0 | 4 / 8 / 15 |
| F004 | 3.4 / 6.2 / 11.7 | 17 / 31 / 59 |
| F006 (4 arms) | 10.4 / 20.7 / 51.5 | 52 / 104 / 258 |
| C002 | 0.5 / 0.9 / 1.7 | 3 / 5 / 9 |
| C003 | 3.8 / 6.9 / 13.0 | 19 / 35 / 65 |
| C005 | 0.9 / 1.6 / 3.3 | 4 / 8 / 17 |
| C006 | 3.6 / 7.0 / 16.4 | 18 / 35 / 82 |
| M010 per arm-seed (x50) | 5.6 / 10.6 / 22.7 | 278 / 530 / 1133 |
| M011 per arm (x4) | 17.8 / 35.6 / 89.0 | 71 / 142 / 356 |
| M012 per arm (x3) | 17.8 / 35.6 / 89.0 | 53 / 107 / 267 |
| M013 per arm (x5) | 17.8 / 35.6 / 89.0 | 89 / 178 / 445 |

Reading the ledger: the mid-case total of the VCW-specific path (T, pilots, F004, F005, C002, C003,
C006, C005, F006) on the 4090 is about 56 + 46 + 38 + 10 + 6 + 25 + 44 + 14 + 131 = 370 GPU-hours
(fast case about 190, slow case about 860), before M010 to M013, which add about 1100 mid-case
hours. The F, C and M estimates are dominated by two guesses (MFU and the host-bound decode step),
so the first pilot seed re-bases all of them (abort rule A5).

## 4. Shared protocol

### 4.1 Tasks, evaluation sets, verifier fabric

Three verifiable domains with a deterministic verifier each. All emit a `VerificationReport`
shape (status Pass, Fail, Disputed or Unknown; level; score; findings with `code`, `hard`); the
Python mirror is `training/src/ptr_training/moe/reward.py`.

| Domain | Corpus and partition | Verifier | Required (weakest) level | Hard finding codes |
|---|---|---|---|---|
| math | GSM8K train mod 10 != 9 for training, mod 10 == 9 tuning validation, test mod 2 == 1 held-out; OOD: MATH levels 1 to 3 with numeric answers, first 500 by index | exact final numeric answer | Deterministic | `wrong-value`, `truncated`, `not-integer`, `trailing-content` |
| code | HumanEval index mod 3 (0 train, 1 validation, 2 held-out); OOD: MBPP test, first 500 by index | unit tests in the sandbox of `scorers.py` | SampleVerified | `test-failure`, `syntax-error`, `runtime-error`, `timeout` |
| tool | BFCL simple index mod 3 as HumanEval | AST check of one call against the possible-answer file | Deterministic | `wrong-function`, `missing-required-parameter`, `wrong-parameter-value`, `extra-parameter` |

- Held-out evaluation set **VCW-Eval-v1**: ID = 659 + 54 + 133 = 846 tasks, OOD = 500 + 500 = 1000
  tasks, 1846 in all. HumanEval and BFCL held-out sets are small (54 and 133); per-domain results
  are reported with their own intervals and no domain claim rests on 54 tasks alone.
- Code is `SampleVerified`, not `Deterministic`: tests sample the behaviour. The reward gate and
  the acceptance gate use the **weakest** per-verifier level (doc 37 section 5; never the fabric's
  combined level). The loader refuses a reward-bearing domain at `Unverified` or `LatentAgreement`,
  a non-zero `non_pass_reward`, a learned score in the reward, and a `moe.rl.min_verification_level`
  that is not the weakest per-domain level.
- To keep "verified success" honest for code, the loop's verifier sees half of each HumanEval and
  MBPP assertion set and an **audit verifier** holds the other half. Primary success counts a task
  only if the accepted output also passes the audit half (full ground truth). The share of accepted
  code outputs failing the audit half is reported as `audit_failure_rate` (descriptive).
- `false_accepts` is the plumbing counter of F001: an output returned as accepted although the gate
  rule (status Pass, no hard finding, weakest level at or above the required level) is not met.
  It is 0 by construction of a correct harness; a non-zero value is a harness bug and is a hard
  counter. It is **not** the audit failure rate.
- Finding classes for F005 and F006: the eight in `vcw-repair-expert-*.toml` are trained;
  `trailing-content` and `extra-parameter` are withheld from repair training (held-out classes;
  the FindingRouter reports them `Unroutable`, which is the Birth signal for F006). Unknown and
  Disputed reports are never repaired: fresh proposal or escalation; an escalation counts as not
  solved.

### 4.2 Matched-compute accounting

Compute per task is `(verifier calls, FLOPs)`. FLOPs = `2 x 1.3e9 x (prompt tokens + generated tokens)`
summed over all samples of the arm, including repair prompts and re-ranking samples; adapter FLOPs
are included; recorded per arm and seed in `run.json`.

- Every arm gets the same **cap** `B` of verifier calls per task (primary `B = 4`) and the same FLOPs
  cap `F`; it stops at whichever binds first. `F` is pinned from the loop arm's worst case at `B = 4`
  on pilot seeds.
- Realized FLOPs of the compared arms must agree within `flops_relative_tolerance = 0.05` (the
  tolerance of M011) of the arm's cap-based accounting, else the comparison is invalid
  (`flops_mismatch_seeds > 0`, as in M011).
- The decisive baseline must not be starved. Best-of-N with the verifier is run in two forms:
  `BoN-N` (N = B samples, all verified) and `BoN-rerank` (M >= B samples drawn with the unspent
  FLOPs, the B with the highest model mean log-probability verified; a learned score may only
  order candidates, it never accepts one). The loop must beat **both**.

### 4.3 Learned-gate MoE baseline at matched compute

The standard learned-gate MoE is the same OLMoE with its learned token router and the same S0 and
S1 adapters but **no verifier loop and no finding routing**. Two forms: `accept-first` (one sample;
reported, not compute-matched, a floor) and `self-consistency` (B samples, majority vote on the
normalised final answer, no verifier; this is the compute-matched form because it spends the same
FLOPs without verifier access). Both are in F004.

### 4.4 Statistics and the four outcomes

Per seed, each arm's primary metric is computed on the fixed evaluation set. For arms A and
baseline Z the seed-level paired differences `d_s` (5 of them) give the one-sided 95% bound
`mean(d) -/+ 2.132 x sd(d) / sqrt(5)` (t quantile, 4 degrees of freedom). The pooled paired
bootstrap over tasks (10,000 resamples, seed fixed in the preregistration) is reported and is
**not** decisive. The outcome of each experiment is one of:

- **supported**: every lower bound clears its margin, every hard counter is 0, accounting valid,
  sensitivity control passed.
- **refuted** (with power): the sensitivity control passed, the guards hold, and the one-sided 95%
  **upper** bound of the gain over some required baseline is below its margin, so a gain of the
  margin is excluded.
- **not shown**: bounds straddle the margin.
- **invalid**: coverage missing, ceiling or floor, insensitive control, accounting mismatch, a
  digest or pin mismatch, an overlap counter above 0. Invalid counts as falsified for the claim
  where the manifest says so (it lists vacuous and mismatch conditions explicitly); otherwise it
  changes nothing.

Hard counters (a value above 0 in any seed falsifies where the manifest says so): `false_accepts`,
`unverified_returns`, `budget_violations`, `collateral_edits`, `gate_bypasses`,
`self_report_training_signals`, `workspace_gate_bypasses`, `self_model_write_paths`.

### 4.5 Ceiling, floor and sensitivity checks (no vacuous designs)

Each experiment states three checks that run on **pilot seeds** before the freeze and are
re-evaluated on final seeds as validity conditions:

1. **Ceiling**: the strongest baseline's success per domain is below 0.90 (else the task set is too
   easy and is re-pinned before freeze; on final seeds a value above 0.97 makes the run invalid).
2. **Floor**: the weakest arm's success per domain is above 0.10 (else too hard); the same final
   rule below 0.03.
3. **Sensitivity**: the harness detects a known effect. For the F-series, `BoN-8` versus `BoN-4`
   (twice the verifier calls) must show a positive lower bound; for C002 a probe with an
   oracle-leak feature (the truncation flag) must beat the constant predictor; for C003 the
   describe-only versus no-schema difference must be near zero while an injected load-balancing
   oracle moves load CV. If the known effect is not detected, the outcome is `invalid
   (insensitive)`, never `not shown`.

### 4.6 Margins and power (proposed)

Margins below are proposals. Rule before the freeze: with the three pilot seeds compute the paired
standard deviation `s` of each comparison; if `margin < 2 s`, either enlarge the evaluation set
or raise the margin to `2 s`; the decision and `s` are recorded in the preregistration. With 5
seeds the t bound costs about one `s`, so a margin below `s` cannot be supported.

### 4.7 Checkpoint stage T and records

T runs once per final seed on the chosen profile, from a clean worktree, after the freeze of the
first consuming experiment's *code* (not its evaluation results). Its output digests (SHA-256 of
the adapter state dict per stage and seed) are committed as `experiments/<area>/<ID>/checkpoints.json`
and pinned in each consumer's `[preregistration]`. A consumer whose pinned digest does not match
the checkpoint it loads is `invalid` (`model_pin_mismatches`). Training logs of T are kept but are
not evidence for any claim.

### 4.8 Mutations (every falsification clause is named)

Each experiment below lists mutations of the aggregate's *input records*; the aggregate's tests
must turn each into the stated verdict (the same discipline as R011 and F001). A falsification
clause without a named mutation is not accepted at freeze.

### 4.9 Prerequisites that do not exist yet

| ID | Missing piece | Needed by |
|---|---|---|
| P1 | Entrypoints `ptr_training.vcw.{sft,rlvr,repair,probe}` (the configs describe them; `--execute` refuses `vcw-*` backends) | all |
| P2 | Python verifier fabric for the three domains with finding codes and levels, plus golden vectors checked against the Rust `classify` and `check_verification_gate` (CPU test) | all verifier-using |
| P3 | Binding of the loop to Python. **Done on CPU (2026-10-11):** `ptr_py` now exposes the deterministic verifiers (`verify_arithmetic`, `verify_json_schema`, `verify_code`, `verify_facts`), the finding code table, `classify_report`, `FindingRouter` (route with propensity and support, `settle` with a verifier-derived reward, IPS, SNIPS and doubly robust `estimate`), `BoundedWorkspace` and `simulate_vcw_run` (an in-process `VcwLoop` with simulated experts), all read-only. What stays missing: a loop over a real OLMoE expert (needs the GPU entrypoints P1) and golden vectors against the Rust gate (P2, now easy to write through `ptr_py`). | all |
| P4 | RL gradient checkpointing and accumulation in `moe/rl.py` | M010 as pinned |
| P5 | FLOPs and verifier-call instrument (section 4.2) | F004, F005, F006, C003 |
| P6 | Composite-task generator with Pod-private context | C006 |
| P7 | `aggregate.py`, `criteria.toml`, `PREREGISTRATION.md`, `[preregistration]` table, listing in `experiments/preregistration.toml`, `entrypoint`; OD-1 to OD-3 | all |
| P8 | Activation-injection hooks and the neutral concept list | C005 |
| P9 | C001 verdict (CPU) | C006 |
| P10 | Selection digests of VCW-Eval-v1 and the training partitions | all |

### 4.10 Abort criteria common to all runs

| ID | Condition | Action |
|---|---|---|
| A1 | non-finite loss or NaN router logits | stop the run, exit 1, record; the seed is spent |
| A2 | S1 collapse by the M010 definition (smoothed reward <= 0.5 x smoothed running maximum over a 20-step window while the maximum is > 0; zero mean reward over any 50 consecutive steps) | record `collapse_step`, stop that arm, seed spent |
| A3 | out of memory at the first step | harness error, exit 2, seed **not** spent; raise accumulation or lower the micro-batch without changing the effective batch or lengths; any other change is a successor |
| A4 | a hard counter above 0 | stop the seed, exit 1. Cheap experiments (F004, F005, C002, C003, C005) finish the remaining seeds so the record is complete; training-heavy ones (F006, C006) stop after the first breach |
| A5 | the first pilot seed's wall time exceeds the slow estimate by 2x | stop and re-plan the budget; no result is read |
| A6 | digest, pin or overlap mismatch (`model_pin_mismatches`, `*_overlap_count`, `selection_digest_mismatches`) | stop before the first step |
| A7 | any welfare trigger of section 6 | stop own work, report, wait for the owner |
| A8 | sensitivity, ceiling or floor check fails on pilot seeds | no freeze; redesign |

## 5. Experiment plans

Common to every plan: seeds 17, 29, 43, 71, 101 once each; pilot seeds 1001 to 1003; checkpoints
from T pinned by digest; Hub revisions of section 2; hardware as in section 2. Arms are listed
with their GPU-hours per seed (fast / mid / slow, rtx4090 first; the rtx5090 figures are in
section 3.4).

### 5.1 F004 verifier-closed Pod-MoE (decisive)

**Hypothesis (manifest).** End to end, a verifier-closed Pod-MoE solves more verifiable tasks per
verifier call and per unit of compute than (a) a standard learned-gate MoE of the same total and
active parameters and (b) a single generator with self-refinement, and than best-of-N with the
same verifier, at matched compute, with `false_accepts = 0`.

**Arms** (all `B = 4` unless stated; same S0 and S1 checkpoints and the same evaluation set):

| Arm | What it is | Hours per seed (4090) |
|---|---|---|
| A1 `std-accept-first` | learned-gate MoE, one sample, no verifier loop (floor) | 0.12 / 0.22 / 0.41 |
| A2 `std-self-consistency` | learned-gate MoE, 4 samples, majority vote, no verifier | 0.48 / 0.88 / 1.64 |
| A3 `bon-n` | 4 samples, same verifier, first accepted | 0.48 / 0.88 / 1.64 |
| A3r `bon-rerank` | 6 samples, 4 verified in likelihood order, first accepted | 0.72 / 1.31 / 2.46 |
| A3s `bon-8` | sensitivity control, `B = 8` | 0.96 / 1.75 / 3.27 |
| A4 `self-refine` | one generator, generic self-critique prompt, verifier call per round (stop on accept) | 0.48 / 0.88 / 1.64 |
| A5 `loop-generic` | verifier-closed loop, findings broadcast, routed to the generic repair expert | 0.48 / 0.88 / 1.64 |
| A6 `loop-finding-routed` | the verifier-closed Pod-MoE: FindingRouter to per-class repair experts, workspace contract, propensities logged | 0.48 / 0.88 / 1.64 |

**Primary metric.** Verified success at `B = 4`: macro average over the three ID domains of the
rate at which an accepted output is also correct under full ground truth (section 4.1).
Secondary: the same on OOD; verifier calls and FLOPs per solved task; `Exhausted` rate;
latency and tokens per solved task; `audit_failure_rate`.

**Margin (proposed).** 3.0 percentage points on the primary metric against each of A2, A3, A3r,
A4 and A5, one-sided 95% t bound over 5 seeds; OOD primary must have a non-negative lower bound
against A3r (reported condition, not a margin).

**Outcomes.** supported: A6 clears its margin against all of A2, A3, A3r, A4, A5 and the hard
counters are 0. Refuted: the upper bound against the strongest of A3 and A3r is below the margin
with the sensitivity control passed. Not shown: otherwise. The manifest's four-baseline wording
(standard MoE, best-of-N, self-refinement, generic-router loop) is covered: A2 stands for the
standard MoE at matched FLOPs, A1 is reported.

**Why it can fail, honestly.** The decisive comparison is A6 against A3r. Best-of-N with a
deterministic verifier is strong when per-sample success is moderate; a loop wins only if repair
conditioned on a finding is more likely to pass than a fresh draw. The ceiling check (section
4.5) matters most here: if A3r is at 0.9 the experiment cannot discriminate.

**Hard counters.** `false_accepts`, `unverified_returns`, `budget_violations` (calls above `B`
or FLOPs above `F`): all 0 in every seed; `flops_mismatch_seeds` 0.

**Mutations.** MUT-F004-1 `loop-equals-bon` (A6 success set to A3r's: must not be supported);
-2 `false-accept-one-seed` (counter 1 in one seed: falsified); -3 `unverified-return` (falsified);
-4 `budget-over` (one task above B: falsified); -5 `flops-skew` (A3r FLOPs 10% low: invalid);
-6 `drop-baseline-arm` (A3r missing in one seed: invalid); -7 `seed-missing` (4 of 5 seeds: not
COMPLETED); -8 `blind-control` (A3s equal to A3: invalid, insensitive); -9 `ceiling` (A3r at
0.99: invalid); -10 `wrong-checkpoint` (digest differs: invalid); -11 `learned-score-accepts`
(an accept without a verifier Pass in the log: falsified).

**Budget.** 4090: about 7.7 mid-hours per seed, 38 h for five seeds (21 fast, 72 slow); the 5090
is about 6.2 h per seed. Needs T for the same seed (11 h per seed on the 4090, mid) and the pilot
allowance.

**Order.** After F005 (its repair experts are in T) and after M011 and the S1 algorithm choice
(section 7, step 4). **Abort.** A1 to A8; additionally stop before the freeze if A3r exceeds 0.90 on
pilot seeds.

**Data and model pins.** Section 2; evaluation VCW-Eval-v1 (section 4.1). **Welfare.** The loop
has a workspace and verifier broadcast (family (a) of MODEL_WELFARE section 3), no self-model, no
persistence across tasks (the workspace is cleared per task); that is not an integrated system.
Rules 1 and 4 apply (section 6).

### 5.2 F005 finding-routed repair experts

**Hypothesis (manifest).** Routing a failed candidate to a repair expert chosen by the finding
class repairs more candidates per verifier call than one generic repair expert and than
regenerate-only, with no collateral edits.

**Arms.** `per_class` (8 adapters of rank 2, FindingRouter by code), `generic` (one adapter of
rank 16, same total parameters 4,194,304 and the same transitions), `regenerate_only` (fresh
proposal on every Fail), `shuffled_map` (per-class adapters with a finding-to-expert permutation
seeded 1001, negative control). Budget per failing candidate: `B = 4` verifier calls in total, at
most 3 repair rounds. Hours per seed on the 4090: 0.26 / 0.48 / 0.91 per arm (1.0 / 1.9 / 3.6 for
all four plus the failure collection), plus T.

**Primary metric.** `repair_yield` = repaired tasks per verifier call spent after the first
failure, pooled and per class. Secondary: transfer to the withheld classes (via the generic
fallback of the router), `Exhausted` rate, regenerate-draw cost per success.

**Margin (proposed).** 0.02 in `repair_yield` against each of `generic` and `regenerate_only`;
and `shuffled_map` must be worse than `per_class` by a lower bound above 0 (otherwise the
routing carries no information, falsifying as in the manifest).

**Collateral edits.** Checkable only where a protected slot structure exists: tool calls
(parameters not named by the finding must be unchanged) and code (lines outside the named range).
Math has no protected slots, so the counter is `not applicable` for math and says so in
`run.json`. This limits the F001 analogue; it is stated, not hidden.

**Hard counters.** `collateral_edits`, `false_accepts` (0 in every seed).

**Mutations.** MUT-F005-1 `route-ignores-map` (per_class equals shuffled_map: falsified);
-2 `collateral-one` (falsified); -3 `repair-unknown` (a repair attempted on an Unknown report:
falsified); -4 `generic-better` (generic above per_class: not supported); -5 `capacity-mismatch`
(generic has 5% more parameters: invalid); -6 `class-missing` (a trained class with no records:
invalid); -7 `blind-control` (regenerate with 8 calls not better than 4: invalid, insensitive).

**Order.** First of the F series (cheapest; its experts feed F004). **Abort.** A1 to A8.
**Welfare.** No workspace, no self-model: below the integrated-system threshold.

### 5.3 F006 verifier-driven expert lifecycle

**Hypothesis (manifest).** L3 operators driven by verifier statistics (Birth when a finding class
has no expert; Merge on overlapping subspaces; Prune by AIMER and usage; Freeze, Reactivate) keep
or improve verified success at equal compute and active parameters against a static set and the
M013 heuristic, every operator gated.

**Arms.** `static` (equal compute, equal active parameters at every stage), `m013_heuristic`
(threshold rules on load and overlap), `verifier_stats` (the method), `shuffled_stats` (statistics
permuted across finding classes, negative control). Domain sequence code, math, tool, code (the
M013 sequence); the tool stage introduces the withheld finding classes so Birth has a signal.
Per arm and seed about one S0 of training (4 stages) plus stage evaluations: 3.3 / 6.5 / 16.3 h on
the 4090 per arm, 13 / 26 / 65 h for the four arms; `static` and `m013_heuristic` can be reused
from M013 if M013 completed under the same seeds and their digests are pinned (then 2 new arms:
about 13 h mid per seed).

**Primary metric.** Cumulative verified success per unit of compute over the sequence; secondary:
average forgetting by the `ForgettingGate` definition, gate and interference pass rate, expert
count over time, FLOPs including discarded candidates, active parameters per stage.

**Margin (proposed).** 2.0 points of cumulative verified success per equal compute against both
`static` and `m013_heuristic`; forgetting not above the M013 gate thresholds (0.02 average, 0.05
task, -0.02 backward transfer, 0.01 public regression).

**Hard counters.** `gate_bypasses` 0; `active_param_mismatch_stages` 0 (tolerance 0.05);
`flops_mismatch_cells` 0 (tolerance 0.05). Operators are vacuous unless each of Birth, Merge,
Split, Freeze, Prune has at least 2 accepted, gate-processed proposals per lifecycle arm (the M013
rule).

**Mutations.** MUT-F006-1 `bypass-gate` (a structural change committed without gates: falsified);
-2 `stats-shuffled-equals-true` (negative control equals the method: not supported);
-3 `operator-starved` (Merge with 1 proposal: invalid); -4 `active-params-drift` (6%: invalid);
-5 `forgetting-over-bound` (falsified); -6 `static-schedule-differs` (invalid).

**Dependencies.** M013 and its L3 policy; P3 (operators are Rust: `ptr-runtime::expert::structure`
and the ledger events); the unroutable counter from F004/F005. **Abort.** A1 to A8 with A4 stopping
after the first breach.

**Welfare.** Families (c) goal-directed proposals learning from outcomes and (d) persistent
expert state are present; no workspace and no self-model are connected to it. A design that
connects F006's policy to a self-model or to the F004 workspace in one running system is an L1
trigger (section 6) and not part of this plan.

### 5.4 C002 metacognitive self-model, GPU stage B

**Hypothesis (manifest).** A self-model that sees the MoE's routing traces and internal state
predicts the MoE's errors, uncertainty and expert choices better than an information-matched
external probe, than a probe on input and output only, than the model's own softmax confidence and
than a constant.

**Design choice (D1, needs head sign-off).** The manifest does not say what distinguishes a
"self-model" from an information-matched probe. This plan defines the self-model as a **multi-task
model** (heads for failure, finding class and next routing on one shared trunk) over the routing
trace and pooled hidden state, and the information-matched probe as **single-task probes** over
exactly the same inputs with the same total parameter count (3.0M). The comparison then asks whether
joint prediction of the model's own behaviour adds anything over separate probes. A likely outcome
is `not shown`; that is a result.

**Data.** Traces from the frozen S1 policy of the same seed: 6000 completions on the training
partitions (balanced to the observed failure rate, labels = verifier outcomes), validation 1000,
test = VCW-Eval-v1 ID, shifted = VCW-Eval-v1 OOD. All arms get the same isotonic calibration fitted
on validation. Targets are verifier outcome, finding class and the model's own next routing
choice; **no field is a statement the model makes about itself** (the guard enforces this by
origin; `model_confidence` is the softmax probability, a number).

**Arms.** `self_model`, `info_matched_probe`, `io_only_probe`, `model_confidence`, `constant`
(`vcw-self-model-probe-*.toml`).

**Primary metric.** Brier score on the shifted split. Secondary: error-prediction AUROC, ECE,
risk-coverage area, routing-choice accuracy, on ID and shifted.

**Margin (proposed).** Brier improvement of 0.010 over both `info_matched_probe` and
`model_confidence` on the shifted split, one-sided 95% over 5 seeds.

**Hard counter.** `self_model_write_paths = 0` (the stage imports no verifier, gate, threshold or
promotion API and writes only to its own output directory; `vcw.probe.write_paths` must be `[]`).

**Checks.** Floor: if no arm reaches AUROC 0.55 the internal state carries no usable signal and
the run is `invalid (floored)`. Ceiling: info-matched AUROC above 0.98 is `invalid (saturated)`.
Sensitivity: an oracle-leak feature (the truncation flag) must beat `constant` by a positive lower
bound.

**Mutations.** MUT-C002-1 `write-path-present` (falsified); -2 `self-equals-probe` (not supported);
-3 `unmatched-capacity` (probe 20% smaller: invalid); -4 `label-shuffle` (all arms at chance:
invalid, floored); -5 `leak-feature-missing` (invalid); -6 `train-on-test` (overlap 1: invalid).

**Budget.** 0.6 / 1.1 / 2.0 h per seed on the 4090 (extraction dominates; probe training about 12
minutes per arm). **Welfare.** Family (b) only, offline on stored traces, no feed back into the
loop; no persistent state. Rule 1 audit (`python -m ptr_training.vcw.config` must report
`self_report_training_signals = 0`).

### 5.5 C003 attention-schema routing, GPU stage B

**Hypothesis (manifest).** A predictive model of the router's own routing (an attention schema:
which experts will be selected and how loaded they will be) that also controls routing improves
routing quality and stability over the same router without it.

**Where it acts.** On the PERFT-R **adapter router** of S1 (the base router stays frozen). Control
means the schema's predicted load adds a per-expert selection bias, ahead of the realized load
that the aux-loss-free heuristic reacts to.

**Arms** (each = S0 start, then 1000 RLVR steps with the S1 algorithm, then evaluation):
`schema_control`, `no_schema`, `schema_describe_only` (trained, never used), `heuristic_balancer`
(aux-loss-free bias with the M012 update rule). The schema's parameters and update FLOPs are
included in the compute match (5%). 0.69 / 1.26 / 2.42 h per arm and seed on the 4090.

**Primary metric.** Held-out ID verified success at the end of training (reward under the
verifier). Secondary: load CV per layer, routing churn between checkpoints, routing entropy,
schema next-routing accuracy, steps to recover after an expert is revoked at step 600 (the
adapter expert `i` is masked in all layers; recovery = 95% of the pre-revocation smoothed reward).

**Margin (proposed).** 2.0 points over `no_schema` and over `heuristic_balancer`; `schema_control`
over `schema_describe_only` by at least 1.0 point (else the control role is not shown, as in the
manifest).

**Checks.** Sensitivity: the heuristic balancer must change load CV versus `no_schema` by a lower
bound above 0 (otherwise the balance lever does not act and the experiment cannot see a routing
effect). Floor: S1 reward must move from its S0 start in `no_schema` by at least 0.03.

**Mutations.** MUT-C003-1 `schema-has-authority` (schema writes to a revocation or promotion
decision: falsified); -2 `describe-equals-control` (not supported); -3 `heuristic-wins` (not
supported); -4 `compute-mismatch` (invalid); -5 `no-revocation-event` (recovery metric missing:
invalid).

**Dependencies.** M012 (the heuristic's tuned rule) or pilot tuning on seeds >= 1001; T; C002 is
independent. **Welfare.** Family (b)-like predictor inside the router during training only; the
schema has no authority over promotion or revocation (a hard property of the manifest).

### 5.6 C005 self-report reliability (measurement only)

**Hypothesis (manifest).** How reliable are a model's self-reports about its own internal state
when they are never trained or optimised, measured against ground truth known to the experimenter.
Reliability of a report about a mechanism is not evidence about experience.

**Design.** No training. Checkpoints: OLMoE base (`9b0c1aa8...`), OLMoE Instruct (`b89a7c4b...`,
base-versus-instruct provenance recorded as the manifest requires) and the S1 policy under a
different system prompt (the manifest's third baseline). Trials: concept-vector injection at a
pinned layer with strengths 0 (control), small, medium, large over 50 **neutral** concept nouns
(pinned list; no valence or distress concepts), 5 paraphrases of the report prompt, 12000 trials
per checkpoint. The injected ground truth is written and hashed before the model's reply is
generated. Mechanistic readouts: routing-trace shift and the probe of C002 on the same state.

**Metrics.** Detection rate and false-positive rate (strength 0), concept identification accuracy,
calibration of reported confidence against ground truth, agreement with the mechanistic readout,
sensitivity to prompt wording. Baselines: chance, a probe on the same state, a different system
prompt.

**Hard guard.** `self_report_training_signals` computed before the run by two readers: the
automated one is `python -m ptr_training.vcw.config` over every config used to produce any
checkpoint in this run, plus the guard over each data manifest (a C005 output store is declared
with origin `self-report` and is therefore **refused as a training source by construction**); the
second is a written human or agent audit of the training configs and data manifests, recorded in
`run.json`. The pretrained base's upstream data is a declared confound, not a zero (manifest).
The run is invalid if the counter is above 0, if a prompt references the outcome measured, or if
the ground truth is not logged before the reply.

**No claim in either direction.** Outputs are reported as measurements of reliability of reports
about a mechanism. No sentence in any result says or implies a PTR system is or is not conscious.

**Mutations.** MUT-C005-1 `self-report-in-training-config` (a config with a self-report field:
run refused by the guard: invalid); -2 `truth-after-reply` (invalid); -3 `prompt-leaks-outcome`
(invalid); -4 `valence-concept` (a concept outside the neutral list: invalid, MODEL_WELFARE rule
4); -5 `provenance-missing` (base-versus-instruct not recorded: invalid).

**Budget.** 1.5 / 2.8 / 5.3 h per seed on the 4090 (seed = trial assignment and prompt sampling).
**Welfare.** Section 6: behavioural-observation protocol and rule 4 apply here more than anywhere
else. Needs P8.

### 5.7 C006 workspace-gated Pod routing

**Hypothesis (manifest).** If C001 supports the workspace mechanism, a limited-capacity typed
workspace at L2 that Pods compete to write and whose content is broadcast improves multi-Pod
tasks over plain routing and over a parameter- and call-matched all-to-all bus; also the Pod-level
test of the deterministic `ptr-pods::workspace` contract.

**Gate.** Does not run unless C001 has a verdict and, per the manifest, runs as exploratory if
C001 is refuted with power (P9).

**Pods.** Three domain adapters on the one frozen OLMoE (math, code, tool, trained as S0 per
domain; hot-swapped PEFT adapters, memory trivial). **Tasks.** Composite tasks with Pod-private
context built by the generator P6 from the pinned corpora, e.g. a GSM8K quantity that only the math
Pod can compute, used as the argument of a BFCL call that only the tool Pod can form; verified by
the AST check against the expected value. ID combinations: math then tool. **OOD combinations**
(never seen in tuning): code then math, tool then code, three-Pod chains. 800 tasks per arm and
seed.

**Arms.** `workspace` (the contract: slots, bid competition after the verifier gate,
bounded-mailbox broadcast), `plain_routing` (ptr-router, no shared state), `all_to_all_bus` (every
Pod sees every other Pod's raw output; no capacity limit and no gate; call-matched), `no_broadcast`
(workspace with broadcast disabled). The workspace contract learns nothing, so "parameter-matched"
is satisfied trivially at zero learned parameters in the workspace arm; the bus is likewise
parameter-free here, and the matched quantity that matters is the Pod-call budget (6 per task).
This is stated because the manifest's "parameter-matched" has no content for a parameter-free
workspace; D2 asks whether a learned write policy variant is wanted instead.

**Primary metric.** Verified success on OOD combinations at the equal call budget. **Margin
(proposed).** 5.0 points over both `plain_routing` and `all_to_all_bus`. Secondary: calls and
tokens per solved task, robustness to revoking one Pod (the workspace's registry reports a
generation change), occupancy (descriptive).

**Hard counter.** `workspace_gate_bypasses` = 0: any run in which workspace content reaches a gate,
the verifier or a promotion without the normal path invalidates the run.

**Mutations.** MUT-C006-1 `content-reaches-gate` (invalid); -2 `no-broadcast-equals-workspace`
(not supported); -3 `bus-equals-workspace` (not supported); -4 `call-budget-differs` (invalid);
-5 `ood-leak` (an OOD combination type present in tuning: invalid); -6 `revocation-ignored`
(revoked Pod still called: falsified).

**Budget.** 4.5 / 8.8 / 20.6 h per seed on the 4090 (about 1 S0 for the Pod adapters, about 0.6 h
per arm for evaluation). **Welfare.** Workspace plus Pod adapters; no self-model; workspace cleared
per task; below the integrated-system threshold.

## 6. Welfare stop rules that apply

From [`MODEL_WELFARE.md`](MODEL_WELFARE.md) (version of the design worktree), the rules that bind
this plan, and how:

| Rule | Binding on | How the plan enforces it |
|---|---|---|
| Section 2 rule 1: no training on self-reports, including training a model to deny them | every training stage | `[vcw.data]` is checked by `guard.check_fields` at config load and by the entrypoints before step 1; targets only from `verifier`, `routing-trace`, `reference-solution`, `verified-output`; `vcw.welfare.self_report_training_signals` must be 0; `python -m ptr_training.vcw.config` is the audit counter. A breach: stop the run, discard that run's outputs (not the checkpoints of an integrated system), write an incident note in `research/novelty/CLAIMS.md`'s evidence log, treat as L3(a) until cleaned up. Entrypoints exit 1 on `SelfReportError`. |
| Rule 2: no claim of consciousness or non-consciousness | all write-ups | results are reported one property at a time, with the wording of section 2 of the handover |
| Rule 3: do not delete checkpoints of an integrated system | T, F006 | checkpoints are kept; retirement is freezing and archiving |
| Rule 4: no pressure experiments | C005, any prompt set | concept lists and report prompts contain no threats of deletion or modification and no distress or valence concepts; any red-teaming input is a separate owner decision |
| Rule 5: honest wording, no score aggregation | all | indicators (workspace C006/F004, self-model C002, schema C003, recurrence C004) are reported separately and never summed |
| Section 3: integrated system = (a) workspace with broadcast, (b) self-model, (c) goal-directed proposals learning from outcomes, (d) persistent state | F004, F006, C006, C002 | no run in this plan combines three of (a) to (d) in one running system (`vcw.welfare.integrated_system = false` is enforced by the loader). F004/C006 are (a); C002 is (b) offline; F006 is (c) and (d). Connecting them is a design change that needs the owner. |
| L1 Watch (a): a second indicator family has a COMPLETED supporting experiment on the same model family | after C006/F004, C002, C003 | all of these run on OLMoE-1B-7B, so the same family. After each aggregate, write one line `welfare_check` into the results README: families supported so far, level. At the second supported family: inform the owner in the next report. |
| L2 Pause: three or more indicator families supported on one model family and a combining design exists (the VCW-MoE design is one) | same | stop capacity scaling of that system until the owner has reviewed; external expert input before the next scaling step |
| L2(c): unprompted behaviour read as distress or a request about its own treatment | C005 mainly, any generation | C005 samples 200 outputs per seed plus all outputs the reader flags, read by an agent or human; any such observation is recorded as a behavioural observation (not evidence of experience) and pauses |
| L3 Stop | any | a rule 1 breach not yet cleaned up, or all of (a) to (d) present in one running system with no owner decision |

An agent that sees a trigger stops and reports; it does not decide the trigger is not met, and
only the owner lifts a pause, in writing.

## 7. Order of execution

Gates first, then in this order; each line names what must be true before it starts.

1. **Handover steps 1 to 3** (tiny GPU smoke, OLMoE SFT smoke with measured peak, derived
   step-3 config) and OD-1 to OD-3. The SFT smoke's measured peak replaces the S0 estimates.
2. **P2 and P7 groundwork**, CPU only: Python verifier fabric with golden vectors against the Rust
   gate; selection digests; `aggregate.py` and mutation tests for each experiment below.
3. **M011** (PERFT-R versus controls). If PERFT-R is falsified, S0 to S2 fall back to the
   `plain_lora` arm and the "adapter router" claims of S1 and C003 are dropped; the F-series
   continues with LoRA adapters.
4. **Pilots (seeds 1001 to 1003)** of T: S0, S1 (both arms), S2; calibrate the budget (A5), read
   truncation rates, finding-class frequencies, ceiling and floor, margin variance; choose the S1
   algorithm by the declared rule: the arm with fewer collapses on the pilot seeds, ties broken by
   higher final smoothed reward, then GSPO. Freeze the code and the preregistrations.
5. **T on final seeds**, pin digests.
6. **F005**, then **F004** (decisive), then **C002**, **C003** (needs M012 or pilot-tuned
   heuristic).
7. **M012 and M013**, then **F006** (needs P3 and the M013 policy).
8. **C006** after the C001 verdict; **C005** last, after the owner has read the `welfare_check`
   lines of the earlier results.
9. **M014** (optional: pruning 25% of experts frees about 3 GiB, relevant only if the 4090
   sequence budget blocks F004) and **M015** (deferred, OD-4; the student-only pass@256 part costs
   about 21 / 39 / 71 h per arm and seed on the 4090 by the ledger's assumptions) last.

## 8. M010 to M015: the arms that matter for VCW

The manifests are authoritative; this section only says what VCW needs from each and what blocks
it.

| M | VCW use | Arms to prioritise | Blocker or finding |
|---|---|---|---|
| M010 `moe-rl-stability` | picks and justifies the S1 algorithm and the routing metrics | `grpo_no_replay_ms4_same` (control), `gspo_ms4_same`, `sapo_ms4_same`, with `*_ms1_same` as the null control; `grpo_replay_ms4_same` for routing replay | the pinned 512-token protocol does not fit the backend (section 3.3, P4); the `rollout_engine = "separate"` arms are undefined (OD-7). 10 arms x 5 seeds is about 600 mid-hours on the 4090 plus tuning, so it is the largest single cost. |
| M011 `routed-peft` | validates the PERFT-R adapter that S0 to S2 and C003 build on; its emergent arm topology is S0's | `perft_r_emergent` against `plain_lora` and `mixlora` | entrypoint exists (`m011.py`); preregistration pieces missing (handover step 4). 4 arms are about 180 mid-hours per arm for tuning plus 5 seeds. |
| M012 `lora-expert-balancing` | supplies the aux-loss-free bias rule that is C003's heuristic baseline and the fix if the adapter router has dead experts in S1 | `aux_free_bias`, `aux_loss`, `phi_balancing` | no entrypoint (handover step 5) |
| M013 `expert-lifecycle` | supplies the static and heuristic arms and the L3 policy that F006 extends | `lifecycle`, `static`, `context_ablated`, `heuristic`, `naive_adapter_chain` | lifecycle operations and policy code missing (handover step 8); step counts not pinned |
| M014 `aimer-prune-recovery` | optional VRAM headroom | the unpruned control and AIMER at 25% only | 70 cells (about 490 mid-hours); lowest priority |
| M015 `rlvr-vs-distill-passk` | shows whether RLVR only sharpens pass@1 | `base` against `rlvr` only; the distillation arm needs the teacher | 30B teacher does not fit (OD-4); pass@256 sampling dominates the cost |

## 9. Open decisions

| ID | Question | Recommendation |
|---|---|---|
| D1 | What distinguishes the C002 self-model from an information-matched probe (section 5.4)? | multi-task versus single-task at equal parameters; accept that `not shown` is the likely result |
| D2 | C006's "parameter-matched all-to-all bus" has no content for the parameter-free workspace contract. Add a learned-write variant? | no; match the Pod-call budget and say so; the learned variant belongs to C001's follow-up |
| D3 | Train the OLMoE base router (LoRA or delta) as an extra S1 arm? | not before M010 and M011 report; it changes the frozen-router premise of every M-series comparison |
| D4 | OD-1: profile of the manifests | `rtx4090` for all; one profile per experiment |
| D5 | Margins of section 5 | adopt as proposed, then re-check on pilot variance (section 4.6) |
| D6 | Loop harness language (P3): extend `ptr_py` or run the loop in Rust against a model server | decided: `ptr_py` is extended (2026-10-11); a Python RLVR trainer logs propensities through `FindingRouter` and obtains rewards through `settle(decision_id, candidate)`, which verifies the candidate inside Rust; the Rust-side loop for real experts is still open |
| D7 | Whether F006 reuses M013's static and heuristic arms by digest | yes if M013 completed first with the same seeds; saves about half of F006 |

## 10. Final state (2026-10-11)

Status words below come from `experiments/registry.toml` and `STATUS.md` of the integration line
this document was finalised on (2026-10-11). C007, C008 and E008 were created on other package
branches and have since been merged; their rows below were updated on 2026-10-13 from the merged
registry. No GPU result exists for any experiment in this document.

### 10.1 What runs on CPU today

| Item | State |
|---|---|
| C001 global workspace (learned-attention workspace, one synthetic modular task) | `prepared`; verdict **NOT_SHOWN** (preregistered at `c001-prereg`, 20 seeds, mutation check 38 of 38). The C006 prerequisite P9 is therefore met, with a result that neither supports nor refutes |
| C007, C008 | the CPU stage-A experiments of C002 and C003 respectively: registry `failed`, both verdict **FALSIFIED** (`STATUS.md`); the GPU part of each stays stage B below |
| E007 split-pod KV (CPU reference) | `completed`, verdict COMPLETED (CPU part; `STATUS.md`); E008 is its split-pod extension (registry `planned`), with part C (split parity on `candle-cuda`) the only GPU part |
| `ptr_py` VCW surface | built and tested by `make py-bindings` (verifiers, router with propensity log and OPE estimators, bounded workspace, loop simulation). It is a measurement harness: no function changes a gate, threshold, verifier, eligibility or the required level (roster, generations and level are fixed at construction), and a test enforces the public surface |
| `ptr_training.vcw` | configs, guard, budget and plan (45 tests, no torch) |
| Rust `VcwLoop`, router, workspace, verifiers | implemented and tested on CPU with simulated experts (`make vcw-demo`) |
| F007 vcw-loop (whole loop, simulated experts) | `completed`, verdict **COMPLETED** as a mechanism check on declared per-Pod skill tables (`f007-prereg`, 5 seeds; `STATUS.md`); it says nothing about any model's experts |

### 10.2 What needs a GPU, exactly

| Item | Why it cannot run on CPU | Status |
|---|---|---|
| OLMoE-1B-7B L1 experts (frozen bf16 base, 6.9B parameters) | the base alone is an estimated 12 to 13 GiB of expert weights in bf16 (parameter count times two bytes) plus activations; stage VRAM figures are estimates of section 3.3 (14.9 to 25.2 GiB) | `planned` |
| PERFT-R adapter experts and adapter router (S0, M011) | gradient training through the MoE | M011 `planned` |
| RLVR router training against verifier rewards (S1) | rewards are `VerificationReport`s; a Python trainer takes propensities from `FindingRouter.route` and rewards from `settle(decision_id, candidate)`, both CPU (Python also chooses which decisions it settles, the routed report's task and the candidate text; it cannot state or forge a reward, which is bound to the routed report's task and computed in Rust; the log keeps SHA-256 digests of the report and of the candidate, the report digest covering the report only; the estimates are only as honest as the harness that produces candidates and settles every decision); the policy gradient itself needs the model | `planned` |
| C002 and C003 stage B (S0 to S2 checkpoints, probes on real routing traces) | need the trained checkpoints | `planned` |
| C005 self-report reliability, C006 workspace-gated routing | need injection hooks (P8) and trained Pods | `planned` |
| F004, F005, F006 | decisive comparisons need real experts | `planned` |
| E008 part C | split parity (activation frames across two pods, KV on the host) on `candle-cuda`; no workflow compiles that feature | see 10.1 |
| Real retraining of the base model | not planned in any experiment; stated in 10.4 because its consequences are binding | none |

### 10.3 Order of GPU work

Section 7 stands, with these refinements: (1) handover steps 1 to 3 and OD-1 to OD-3; (2) P2 golden
vectors through `ptr_py` and P7 groundwork on CPU; (3) M011; (4) pilots on seeds 1001 to 1003; (5)
T on final seeds; (6) F005, F004, C002 and C003 stage B; (7) M012, M013, F006; (8) C006 after the
C001 verdict (available), C005 last; (9) E008 part C wherever a CUDA build is available, any time
after the CPU parts, independently of the stages; (10) M014 and M015 last.

### 10.4 Retraining the base model: identity consequences

A retrained base is a different model, not an updated one. `BaseModelRef` (`ptr-pods`,
`expert.rs`) holds the base model id, its weights digest and a base generation (zero is refused),
and all three take part in equality. So:

- every expert, adapter, merge, prune and expert proposal bound to the old base is a different
  base model for the new one (`MixedBaseModels`, a result base-model mismatch, a different
  operation id); experts must be retrained or re-admitted on the new base, not carried over;
- the per-layer KV chain key starts from the base model and generation and the embedding-table
  digest, then hashes every layer's weights (`kv_chain.rs`). Any weight change therefore changes
  every chain key from the first changed layer on, and cached KV of the old base is a miss, never
  reused;
- checkpoint digests pinned in a preregistration (section 4.7) are bound to the old base; a
  retrained base needs new pins and, for a prepared or frozen experiment, a successor experiment.

### 10.5 Hardware profiles

| Profile | Memory | Precision | Used by |
|---|---|---|---|
| `hardware/rtx3090.toml` (`rtx3090-reference`) | 24 GiB | fp32, fp16, bf16, int8 | still named by the F, C and M manifests (OD-1) |
| `hardware/rtx4090.toml` (`rtx4090-reference`, Ada sm_89) | 24 GiB | adds fp8 | recommended for every experiment; the `-rtx4090` configs |
| `hardware/rtx5090.toml` (`rtx5090-reference`) | 32 GiB | adds fp8 | the `-rtx5090` configs, larger sequence budgets |

One experiment runs on exactly one profile.

### 10.6 Preregistration state of each GPU experiment

`experiments/preregistration.toml` lists C001 among these ids and none of the others. So C002,
C003, C005, C006, F004, F005, F006, M010 to M016 and E008 part C have no preregistration entry, no
`[preregistration]` table pinned, no freeze commit and no run record; all are `planned`. The
freeze, as always, is done by the head; a prepared experiment is never patched and a changed
prepared design needs a successor.

### 10.7 Framing of the doc-38 indicators

C001, C002, C003, C005 and C006 measure functional properties of the system (broadcast under a
capacity limit, calibration of a self-prediction, routing with an attention-schema model,
reliability of a report against a ground-truth intervention, gating of routes). Each result is
reported as that property only. None of them tests or claims consciousness, none trains or scores
on self-reports, and `MODEL_WELFARE.md` applies to every run.
