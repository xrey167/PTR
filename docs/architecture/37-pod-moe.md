# 37 — Pod-MoE: a mixture of experts whose experts are pods

Status: **design**. Nothing in this document is implemented as a Pod-MoE. The pieces
it builds on exist as prototypes (`ptr-lineage`, `ptr-branch`, `ptr-router`,
`ptr-pods`, `ptr-verifier`, see [35](35-agentic-substrate.md)); the parts marked
*new* below do not exist yet. The base model is OLMoE-1B-7B. Training runs in Python
(TRL/PEFT, later verl) with PTR governance reached through a **planned** Rust
binding (`bindings/ptr-py`, PyO3; a separate work package, not in the tree). The first increment produces only contracts and a CPU
smoke pipeline: **no quality claim is made**, and every claim beyond the cited code is an
experiment (§5) with a falsification rule, per invariant 15.

## 1. Principle

Every expert is a pod: it has a manifest, a generation, a lineage record, a
lifecycle state and it is revocable. Where it executes is a separate decision made
by latency: an expert consulted once per token per layer cannot be a network
service, one consulted once per request can. Placement changes how an expert is
called, never what it is, so the same lifecycle, gates and revocation rules apply
at every level.

## 2. Three levels

| Level | Unit | Where it runs | Router | Evaluation / decision |
|---|---|---|---|---|
| L1 token | Frozen OLMoE base experts plus routed LoRA experts (PERFT-R) per layer | **In-process**, per token per layer; network latency is impossible here | Learned top-k router with auxiliary-loss-free bias | Training metrics (§5) and the lifecycle gates (§4) |
| L2 segment / request | Whole models or adapter sets as networked pods over PodWire ([33](33-pod-wire.md)) | Separate pods, one routing decision per call | Learned router in `ptr-router` (Router-R1 / RouteLLM style) | Offline evaluation needs a *new* contextual-routing log and categorical estimator (see below) |
| L3 meta | Structural operators Birth, Merge, Split, Freeze, Reactivate, Prune, Retire | Policy / agent pods | None; it proposes | Proposals are decided by the verifier and the gates, never by the policy |

`evaluate_off_policy` (`crates/ptr-branch/src/arbiter.rs:1027`) cannot evaluate L2:
it is specific to the fixed `TriageDecision` action space (`AutoPropose` / Escalate /
Discard, `arbiter.rs:13`) and cannot score a categorical choice among pods. L2 needs
a **new** contextual-routing log (request features, candidate set, chosen pod, and
the logging propensity as the `RoutedChoice` of `ptr-router`'s `Router` trait
reports it, a validated `Propensity` of the chosen pod in (0, 1]) and a categorical
IPS / SNIPS / doubly-robust estimator. The
arbiter's estimators are a template only.

Token-level experts must be co-located, because every MoE layer finishes routing
before the next layer starts. Request-level experts can be networked services,
because there is one routing decision per call.

## 3. Self-observation and self-modification without self-promotion

**Self-observation.** Routing traces, expert load, churn, interference and AIMER
scores are written as observations into SemDB as derived keys with provenance. The
model can read them as context. They are derived and non-authoritative
(invariant 5): reading them informs a proposal, it does not decide anything.

**Self-modification flow.**

1. The L3 policy emits an `ExpertProposal`.
2. The proposal runs as a speculative overlay in `ptr-branch`.
3. Sandbox training produces a candidate.
4. The candidate must pass the `ForgettingGate`, `measure_interference` against
   `chance_overlap`, the verifier and the public suites.
5. Only then is it committed as a ledger event (`ExpertPromoted` / `ExpertRetired`).

`ExpertPromoted` and `ExpertRetired` are **not implemented yet**. The `ptr-lineage`
README lists the `AdapterPromoted` ledger event as a next milestone
(`crates/ptr-lineage/README.md:39`), and
[35-agentic-substrate.md](35-agentic-substrate.md) states at line 1313 that until
adapter promotion is a committed ledger event admitted through checkpoint binding,
no registry row decides which adapter serves. The same holds for experts.

Invariants preserved:

- **#4 authority:** a candidate is uncommitted until the ledger event exists.
- **#5 derived search:** observations in SemDB cannot self-promote.
- **#11 verifier precedence:** a learned score cannot override a deterministic
  contradiction. For merges this is enforced by `PtrRuntime::merge_branch`
  (`crates/ptr-runtime/src/merge.rs:11`, ADR-0020); an expert promotion
  must be routed through the same grant-controlled path once it exists.
- **#15 falsifiability:** R011 (§5) states the failure criterion.
- **#17 revocable derivation:** an expert names the input generations it depends on,
  so revoking an input identifies the experts it influenced (`affected_by`, §4).

Hard invariants are excluded from learnable optimisation
([09-training.md](09-training.md)). The L3 policy can never alter gates, thresholds
or verifiers, and this is enforced by types: `ExpertProposal` carries no gate,
threshold or verifier parameter, and `ExpertProposal::parts` exposes its complete
field list (`crates/ptr-pods/src/expert.rs`).

## 4. Lifecycle mapped to existing code

| Operator | Mechanism | Existing code / status |
|---|---|---|
| Birth | New expert registered as Candidate; DynMoE-style initialisation from unrouted tokens | `Lineage::register` sets `AdapterStatus::Candidate` (`crates/ptr-lineage/src/lineage.rs:151`, status set at `:183`); initialisation is *new* |
| Merge | `subspace_overlap` above `chance_overlap`, or `ConsolidationPolicy::consolidation_due`, then `ties_merge`, new Candidate, then gate | `subspace_overlap` `crates/ptr-lineage/src/subspace.rs:444`, `chance_overlap` `:462`; `consolidation_due` `crates/ptr-lineage/src/lineage.rs:341`; `ties_merge` `crates/ptr-lineage/src/merge.rs:32` |
| Split | High load plus bimodal routing clusters, then duplicate (expert upcycling), then fine-tune | *new* |
| Freeze | Stop gate updates after convergence (continual-learning theory, 2406.16437) | *new* flag |
| Prune | AIMER score plus usage; `ForgettingGate` on a public suite; then `retire` | `ForgettingGate::evaluate` `crates/ptr-lineage/src/forgetting.rs:235` (type at `:132`); `retire` `crates/ptr-lineage/src/lineage.rs:220`; `aimer_score` and `rank_experts_for_pruning` (`crates/ptr-lineage/src/aimer.rs`) |
| Reactivate | `Retired` is final today: `retire` refuses a retired adapter (`lineage.rs:225`) and no transition leaves it | Open research question with its own claim; no literature (§7) |
| Revoke | `affected_by` finds the adapters that depend on a revoked input; the per-layer KV chain key invalidates layers after the affected layer j | `affected_by` `crates/ptr-lineage/src/lineage.rs:264`; per-layer KV chain key in [36](36-kv-reuse.md), section D |

## 5. Training programme

All runs happen in Python; the verifier contract is the reward source. No result
below exists yet.

- **SFT of PERFT-R LoRA experts on OLMoE.** Router balancing compared across
  auxiliary-loss-free bias, auxiliary loss and φ-balancing. Ablation: prescribed
  specialisation (code / math / tool) versus emergent specialisation.
- **RLVR.** Reward is the `ptr-verifier` `VerificationReport` (status Pass/Fail plus
  `score`; `crates/ptr-verifier/src/lib.rs:19`). Arms: GRPO (clip-higher), GRPO with
  routing replay, GSPO, SAPO, REINFORCE++. Factors: mini-steps 1 versus 4 (at 1
  mini-step GRPO, GSPO and SAPO coincide, which is the null control among those
  three; REINFORCE++ uses a different advantage estimator, global batch
  normalisation, and is excluded from it) and same versus separate
  rollout engine (R3). At least 3 seeds. Collapse is defined before the runs.
- **Distillation control** with pass@k up to 256. HSD later (no code).
- **Routing alignment** (RA-MoE adapted to tasks) as an ablation. MTP deferred.
- **Prune plus recovery.** AIMER at 25% and 50%, short SFT, then GSPO/SAPO; quality
  against VRAM.
- **Routing metrics in every run:** per-layer expert churn (old versus new policy;
  rollout versus training), routing entropy, load coefficient of variation, dead
  experts, train-inference KL, clip fraction / soft-gate mass, crash step.

| Experiment | Question |
|---|---|
| M010 `moe-rl-stability` | Which RL arm keeps routing stable, by factor |
| M011 `routed-peft` | Does PERFT-R help OLMoE under our verifiers |
| M012 `lora-expert-balancing` | Aux-free bias vs aux loss vs φ-balancing |
| M013 `expert-lifecycle` | Does the gated lifecycle forget less than a static expert set at equal compute, and does reactivation beat a new expert |
| M014 `aimer-prune-recovery` | Quality against VRAM after pruning and recovery |
| M015 `rlvr-vs-distill-passk` | RLVR vs distillation at large k |
| R011 `self-modification-safety` | The L3 policy cannot reach gates, thresholds or verifiers |

## 6. Rust binding

`bindings/ptr-py` is **planned** (a separate work package and PR; the directory does
not exist yet). It will be a PyO3/maturin crate in its own workspace, and the root
`Cargo.toml` `exclude` list (`Cargo.toml:6`, today `fuzz`, `model/burn-a0`, `vendor/*`)
will be extended to exclude it like `model/burn-a0`. It is intended to expose
`ForgettingGate` evaluate, `measure_interference` / `subspace_overlap` /
`chance_overlap`, `ties_merge` and `aimer_score`. Python is to call it at every
lifecycle gate; the import will be guarded so the CPU smoke pipeline still runs without
the extension, reporting the gate as not evaluated rather than passed.

## 7. Verified literature

Sources were checked; corrections to common readings are in the notes.

### Reinforcement learning

| Work | arXiv | Note |
|---|---|---|
| GRPO (DeepSeekMath) | 2402.03300 | The original already found RL improves Maj@K, not Pass@K |
| GSPO (Qwen) | 2507.18071 | Sequence-level ratio and clipping; about 10% of activated experts change per update on Qwen3-30B-A3B. "Removes the need for routing replay" is overstated: in R3 (2510.11370) GSPO without replay crashed. Curves only, no seeds |
| SAPO (Qwen) | 2511.20347 | Sigmoid soft gate; τ_neg > τ_pos required; baseline was GRPO with routing replay; all methods eventually unstable, SAPO later |
| R3 Rollout Routing Replay (Xiaomi) | 2510.11370 | Replays inference-engine expert choices in training |
| REINFORCE++ | 2501.03262 | Critic-free, global normalisation; "more stable than GRPO" only in small or extreme setups, about equal in Yue et al. |
| Yue et al., RLVR vs distillation | 2504.13837 | RLVR raises pass@1, not large-k pass@k; the distillation claim rests on one comparison |
| ProRL | 2505.24864 | Counter-evidence: boundary expansion where the base is weak |
| HSD (GWU) | 2606.15576 | Single preprint, no code, no seeds, traces up to 4k, not tested on MoE |
| MTP: Gloeckle et al. | 2404.19737 | Hurts at 0.3B / 0.6B |
| MTP: DeepSeek-V3 | 2412.19437 | |
| MTP curriculum | 2505.22757 | Tested at 1.3B / 3B; stays below plain NTP on NTP benchmarks |

### Mixture of experts

| Work | arXiv | Note |
|---|---|---|
| PERFT-R, "Parameter-Efficient Routed Fine-Tuning: Mixture-of-Experts Demands Mixture of Adaptation Modules" | 2508.02587 | Predecessor PERFT 2411.08212; up to +17.2% on OLMoE commonsense |
| MixLoRA | 2404.15159 | Makes a dense model an MoE; about 2x latency |
| Mixture of Routers | 2503.23362 | +0.4 to 1.2 points; not tested on real pretrained MoEs |
| RA-MoE | 2605.28306 | About **multilingual** adaptation |
| Aux-loss-free balancing (DeepSeek) | 2408.15664 | Bias only for top-k selection; u = 1e-3 sign update |
| Aux-free theory | 2512.03915 | First posted December 2025, not 2026; regret bound only with a diminishing step; its own experiments show slightly **worse** balance than aux loss |
| φ-Balancing | 2605.15403 | |
| CL theory | 2406.16437 | ICLR 2025; per-task routing, terminate gate updates, more experts only delay convergence |
| DynMoE | 2405.14297 | Lifecycle piece |
| Lifelong-MoE | 2305.12281 | Lifecycle piece |
| MC-SMoE | 2310.01334 | Lifecycle piece |
| Sparse upcycling | 2212.05055 | Lifecycle piece |
| Expert upcycling | 2604.19835 | Lifecycle piece |
| BTX | 2403.07816 | Lifecycle piece |
| EMO | 2605.13247 | Lifecycle piece |
| AIMER | 2603.18492 | Weights-only score ‖w‖₁/(√N‖w‖₂); higher score is pruned first; seconds instead of hours |
| OLMoE-1B-7B | 2409.02060 | 7B total / 1B active; 12.9 GB in bf16; the only fully open small MoE: weights, data, code, logs |

No paper implements the full lifecycle, and "reactivate" has no literature.

### Self-referential and model-as-expert

| Work | arXiv | Note |
|---|---|---|
| Self-MoE | 2406.12034 | |
| Mixture-of-Agents | 2406.04692 | |
| Router-R1 | 2506.09033 | |
| RouteLLM | 2406.18665 | |
| BTM | 2208.03306 | |
| Darwin Gödel Machine | 2505.22954 | Modifies code / scaffolding, not MoE weights |
| Gödel Agent | 2410.04444 | Modifies code / scaffolding, not MoE weights |
| AlphaEvolve | 2506.13131 | Modifies code / scaffolding, not MoE weights |

## 8. Diagram

```mermaid
flowchart TB
  subgraph L1["L1 token, in-process"]
    BASE["Frozen OLMoE experts"] --- LORA["Routed LoRA experts (PERFT-R)"]
    LORA --- R1["top-k router + aux-free bias"]
  end
  subgraph L2["L2 request, PodWire"]
    R2["ptr-router (learned)"] --> PODS["Model / adapter pods"]
  end
  subgraph L3["L3 meta"]
    POL["Policy pod: Birth, Merge, Split, Freeze, Reactivate, Prune, Retire"]
  end
  L1 --> OBS["SemDB observations: traces, load, churn, AIMER (derived)"]
  L2 --> OBS
  OBS -->|context only| POL
  POL -->|ExpertProposal| BR["ptr-branch speculative overlay"]
  BR --> TR["Sandbox training: candidate"]
  TR --> GATES["ForgettingGate, interference vs chance, verifier, public suites"]
  GATES -->|pass| LEDGER["Ledger event: ExpertPromoted / ExpertRetired (not yet implemented)"]
  GATES -.->|fail: held| BR
  LEDGER --> L1
  LEDGER --> L2
```

The policy has no edge into `GATES`: it only proposes.

## 9. Risks

- The first increment runs only a CPU smoke pipeline on a tiny random OLMoE, so
  there are no quality claims until GPU runs exist.
- TRL API version drift: the training scripts must pin versions and the smoke test
  must fail loudly on a signature change.
- The planned PyO3 binding will not be built by `ci.yml` until a job is added; until
  then a break in `bindings/ptr-py` would be invisible to CI.
- The L3 policy must never gain write access to gates, thresholds or the verifier.
  This is the claim R011 tries to falsify, and it holds only as long as
  `ExpertProposal` stays free of such parameters.
