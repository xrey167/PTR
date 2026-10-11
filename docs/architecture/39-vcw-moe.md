# 39 — VCW-MoE: the Verifier-Closed Workspace Mixture of Experts

Status: **specification of a design**. Parts of the substrate are implemented and tested on the
integration line (the workspace contract, the finding router, the generate-verify-repair loop, the
expert admission path, the runtime `VcwLoop`, the server path and `make vcw-demo`, all with
simulated experts); the whole-loop CPU experiment F007 is COMPLETED on simulated experts (mechanism check, section 3.7); every model-backed experiment does not exist yet. **No claim that VCW-MoE is useful is made or
supported by any result in this repository.** The measured pieces are mechanism checks, not results about any model: the loop mechanism with a
seeded stub (F001), the gate path (R011), the learned-attention workspace analogue (C001, NOT_SHOWN),
the revocation-crash substrate (L001-v2, completed) and the whole-loop check F007 (COMPLETED, simulated experts with declared skill tables, so the gain follows from those tables). Every claim below is
an experiment with a baseline, ablations and a refutation rule (invariant 15, section 6).

This document is the formal companion of [37 — Pod-MoE](37-pod-moe.md) (design of the three levels
and, in its draft on branch `claude/dev-gw`, sections 10 and 11 on the verifier-closed loop) and of
document 38, "consciousness indicators" (`38-consciousness-indicators.md`, same draft branch:
functional capability requirements for the workspace, the self-model and the schema). References
to 37 section 10 or 11, to 38 and to `docs/MODEL_WELFARE.md` are to those drafts and are plain text
until they are merged. It restates their binding parts in
one place, formalises them, and delimits the construction from related work. Where this document
and 37 or 38 differ, 37 and 38 decide and this document is wrong. Indicators are functional
properties and not consciousness; no component here is trained on, or rewarded for, a statement
about itself (principle 2 of 38; `docs/MODEL_WELFARE.md` applies).

## 1. Definition

**Definition.** A *Verifier-Closed Workspace MoE* (VCW-MoE) is a mixture of experts in which

1. the experts are **Pods**: units with a manifest, a generation and a lineage, individually
   revocable (`Pod`, `PodManifest`, `PodRegistry` in `crates/ptr-pods/src/lib.rs:183`, `:139`,
   `:278`), at three levels (L1 token experts, L2 request-level Pods, L3 structural operators, see
   37 section 2);
2. every L2 expert output is a typed **proposal**, and a proposal becomes an answer only if a
   deterministic **verifier fabric** accepts it: status `Pass`, no hard finding, verification
   level at least the task's required level, never because of a score;
3. the **verifier's findings are the primary routing signal** at L2: the class of the next step
   (accept, repair, regenerate, escalate) is a function of the report alone, and the choice among
   the Pods eligible for that class may be learned, but only inside the set the verifier
   admits;
4. proposals that pass the gate compete for a **bounded global workspace** whose content is
   broadcast to all subscribed Pods and conditions the next routing query; refusals and their
   finding codes are broadcast too;
5. a **self-model** and an **attention schema** observe the traces and routing and may only
   propose hints inside the admissible set; they have no edge into acceptance, gates, thresholds
   or the registry;
6. the **structure** of the expert set changes only through L3 proposals (Birth, Merge, Split,
   Freeze, Reactivate, Prune, Retire) that are decided by the ForgettingGate, the interference
   test and the verifier, and are committed only as ledger events;
7. every request has **budgets**, and a request that spends them returns `Exhausted`, not the best
   failed candidate.

In one paragraph: a VCW-MoE is a classical learned-gate MoE at L1 (it can be OLMoE with routed LoRA
experts) wrapped in a request-level closed loop. The loop proposes, verifies each candidate exactly
once, classifies the report, and routes the next step by the verifier's finding codes to
role-specific Pods. Proposals that pass integrate through a bounded, verifier-gated workspace that
broadcasts verified content and refusals. Observers predict the system's own verifier outcomes and
routing without any authority. Verifier statistics (codes no registered expert clears) trigger
gated structural proposals. Everything that routes is logged with its propensity, so the
routing policies can be evaluated off-policy, and every training reward is computed from verified
transitions and never from a self-report.

### 1.1 Difference from a classical MoE

| Aspect | Classical sparse MoE (Switch, GShard, DeepSeekMoE, OLMoE) | VCW-MoE |
|---|---|---|
| Routing signal | The token representation, through a learned gate; load balanced by an auxiliary loss or a bias | L1 the same. L2: the verifier's finding class (deterministic), then a learned choice among eligible Pods, conditioned on workspace content and optional self-model hints |
| What an expert is | An FFN inside a layer, identified by an index | A Pod with manifest, generation, lineage and an execution manifest; revocable, fenced and replayable |
| Combination | Weighted sum of expert outputs | Proposals; acceptance by the verifier; integration through a bounded workspace |
| Output guarantee | None: the model always emits | Accept predicate (section 2.5); otherwise `Exhausted` |
| Feedback | Next-token loss at training time; none at inference | Typed findings at inference, rewards from verified transitions at training time |
| Self-observation | Load and entropy statistics for the balancer | A self-model calibrated against verifier outcomes, observe and propose only |
| Structure | Fixed number of experts | Gated lifecycle: Birth, Merge, Split, Freeze, Reactivate, Prune, Retire, committed through admission |
| Cost model | FLOPs per token | FLOPs plus verifier calls plus rounds, all budgeted per request |

What is **not** new: top-k sparse routing, load balancing, shared experts, routed LoRA experts,
RLVR for routers, best-of-N with a verifier, generate-verify-refine loops, a bounded broadcast
workspace as a mechanism. The claim of novelty is restricted to the **combination under closure
invariants**: a typed finding interface that decides the class of every step, a deterministic accept
predicate that no learned component can influence, a workspace that carries only verified content,
and a structure policy driven by the verifier's unroutable-code statistics, with propensities
logged at every sampled routing decision. Section 5 states where this claim could fail.

## 2. Formalisation

Notation. `Pod` identifiers `p`, generations `g`, finding codes `c ∈ Code`, verification levels
`Unverified < LatentAgreement < SampleVerified < FullSemantic < Deterministic` with rank `rk`.
`⊥` is absence. The verifier fabric is a total function `V : Proposal → Report` of the proposal
(and of its pinned identity); a report is `ρ = (status, level, score, F)` with
`status ∈ {Pass, Fail, Disputed, Unknown}` and `F` a list of findings `(code, hard, message)`
(`VerificationReport`, `crates/ptr-verifier/src/lib.rs:19`; `VerificationLevel`,
`crates/ptr-types/src/lib.rs:425`). The level of a fabric report is that of its weakest
per-verifier level (37 section 5).

### 2.1 State

A VCW-MoE state is `Σ = (L, W, R, Π, M)` plus request-local state `κ`.

* `L` — the **authoritative** ledger and its projections (expert registry, dependency index,
  authority epochs). Only `admit_expert_proposal`, `fence_expert` and `readmit_expert` append
  structural events (`crates/ptr-runtime/src/expert.rs:2210`, `:2417`, `:2471`). `L` is the only
  component the workspace, router and observers can never write.
* `W = (r, σ₁…σ_k, P, T, B₁…B_m)` — the **workspace** (`Workspace`,
  `crates/ptr-pods/src/workspace/mod.rs:360`): a revision `r`, `k ≤ 64` slots, a pending set `P`
  with `|P| ≤ 1024`, a bounded retired log `T`, and `m ≤ 64` subscriber mailboxes of depth
  `≤ 64` (`config.rs:9-17`). Content of `W` is *provisional*: derived, revision-bound, and never
  authoritative (invariants 4 and 5).
* `R = (G, E, g_R)` — the **registry**: the finding registry `G` mapping each code `c` to the set
  of repair-role Pods and holding the proposer-role Pods (`FindingRegistry`,
  `crates/ptr-router/src/findings.rs:137`, with a registry generation `g_R`), and the eligibility
  view `E : Pod → generation` supplied for each routing (`PodEligibility`, `:393`).
* `Π = (key, nonce, ε, θ, u)` — the **router state**: the secret sampling key and nonce
  sequence, the exploration floor `ε`, the (possibly learned) scorer parameters `θ`, and the
  unroutable counters `u : Code → ℕ` (`FindingRouter`, `findings.rs:773`; `u` is returned by
  `unroutable_counts`, `:834`).
* `M = (S, A)` — the **self-model state**: the bounded trace-statistics model and its calibration
  trackers `S`, and the attention schema `A` (package C; section 3).
* `κ = (q, ℓ*, V_left, T_left, τ)` — the request: the request digest `q`, the required level
  `ℓ*`, the verifier-call and round budgets, and the trajectory `τ` so far
  (`RepairBudget`, `RepairPolicy`, `crates/ptr-feedback/src/repair.rs:37`, `:56`).

All components of `Σ` except `L` are bounded by explicit caps with an explicit overflow bucket or a
typed refusal (invariant 8, section 2.6 constraint C3).

### 2.2 Classification

`classify : (Report, Level) → Class` is a pure total function
(`crates/ptr-router/src/findings.rs:352`):

```
classify(ρ, ℓ*) =
  Fallback(ForgedPass)               if status = Pass ∧ ∃f∈F. hard(f)
  Fallback(BelowLevel)               if status = Pass ∧ rk(level) < rk(ℓ*)
  Accept                             if status = Pass          -- no hard finding, level ≥ ℓ*
  Fallback(Unknown)                  if status = Unknown
  Fallback(Disputed)                 if status = Disputed
  Repair(C)                          if status = Fail ∧ C = distinct hard codes of F, 1 ≤ |C| ≤ 32
  Fallback(FailWithoutHardFinding)   if status = Fail ∧ no hard finding
  Fallback(TooManyCodes | OversizedCode)   if the bounds on C are exceeded
```

`Accepted` is a sealed proof type: `Accepted(())` has a private field, so only `classify` can build
it, and `RouteOutcome::Accept` carries it (`findings.rs:338`). A typed `FallbackPolicy` maps each
fallback reason to `Regenerate` (a fresh proposal-role Pod) or `Escalate` (a stronger verifier or
a human). Defaults: `Unknown`, `FailWithoutHardFinding`, `BelowLevel` regenerate; `Disputed`,
`ForgedPass`, `TooManyCodes`, `OversizedCode` escalate (`findings.rs:272-299`). `Unknown` and
`Disputed` are **never** repair targets (invariant 12).

### 2.3 Per-request loop

The loop is a transition system over `(Σ, κ)`. `route(class, context)` returns a Pod and a log
record; `submit`, `cycle` and `broadcast` are workspace operations (`Workspace::submit`,
`mod.rs:429`; `run_cycle`, `workspace/cycle.rs:109`).

```
request(q, ℓ*, budget=(V_max, T_max)):
  κ ← (q, ℓ*, V_max, T_max, [])
  x ← propose(route(Proposer, ctx(W.snapshot, M.hints)))        -- first candidate
  loop:
    W.submit(x, computed_against = W.r)                           -- provisional
    ρ ← V(x);  κ.V_left −= 1                                      -- exactly one call per candidate
    cls ← classify(ρ, ℓ*)
    W.run_cycle(host verifier V)                                  -- gate, bid competition, snapshot
    broadcast(W.snapshot)                                         -- admitted content + refusals (codes)
    τ ← τ ++ (x, ρ, cls, route records)
    match cls:
      Accept        → return Verified(x)                          -- the only non-Exhausted exit
      Repair(C)     → for each c ∈ C: pod_c ← route(Repair(c), ctx) over G(c) ∩ E
                          if G(c) ∩ E = ∅: u[c] += 1; mark unroutable
                      x' ← repair(x, C restricted to the slots named by C); if all unroutable → regenerate
      Fallback(r)   → match FallbackPolicy(r):
                          Regenerate → x' ← propose(route(Proposer, ctx))
                          Escalate   → stronger verifier or human; the result is again a Report
    if κ.V_left = 0 → return Exhausted(VerifierCalls)
    if κ.T_left = 0 → return Exhausted(Rounds);  else κ.T_left −= 1
    x ← x'
```

State machine view (every arrow into `Accept` passes through `classify`):

```mermaid
stateDiagram-v2
    [*] --> Propose
    Propose --> Verify: candidate submitted (provisional)
    Verify --> Classify: Report (one call per candidate)
    Classify --> Accepted: Pass, no hard finding, level at least required
    Classify --> RepairRoute: Fail with hard codes
    Classify --> Fallback: Unknown, Disputed, no hard finding, forged Pass, below level
    RepairRoute --> Propose: repair Pod per code, slots named only
    RepairRoute --> Fallback: every code unroutable, counter incremented
    Fallback --> Propose: Regenerate
    Fallback --> Escalated: Escalate
    Escalated --> Verify: stronger verifier or human
    Verify --> Exhausted: budget spent
    Accepted --> [*]
    Exhausted --> [*]
```

The loop of F001 (`RepairLoop::run`, `repair.rs:184`) is the implemented skeleton of this
machine for a single generator and a single repairer: one verifier call per candidate, the budget
check after each verdict, `Exhausted` carrying no failed candidate (`Outcome`, `repair.rs:150`),
repair of the named slots only. What F001 does **not** contain, and what this specification adds,
is finding-class routing among several Pods, the workspace cycle, broadcast, the self-model hints,
propensity logging and L3 (all designed, section 3).

### 2.4 Levels

| Level | Unit | Decision function | Training signal | Verifier at inference |
|---|---|---|---|---|
| L1 | Token experts (frozen OLMoE experts, routed LoRA experts) | Learned top-k router with aux-loss-free bias, in-process | RLVR and the M-series metrics (37 section 5) | None |
| L2 | Request-level Pods | `classify` fixes the class; a scorer `s_θ(p ∣ class, ctx)` over the admissible set picks the Pod; `ctx` includes the workspace snapshot and optional hints | Logged routing outcomes (section 2.7) | Yes, in the loop |
| L3 | Structural operators | `StructurePolicy` maps verifier statistics to `ExpertOperationRequest`s | None; it proposes, the gates decide | Via the gate evidence |

Token experts must be co-located (every layer finishes routing before the next); request-level
Pods can be networked (37 section 2). The router never sees a verifier at L1.

### 2.5 Acceptance predicate

For a candidate `x` with report `ρ = V(x)` and required level `ℓ*`:

```
Acc(x) ⇔ status(ρ) = Pass  ∧  ∀f ∈ F(ρ). ¬hard(f)  ∧  rk(level(ρ)) ≥ rk(ℓ*)
```

`score(ρ)` is **not** an argument. The predicate has three independently written restatements,
pinned against one another by tests: `Candidate::passes`
(`crates/ptr-feedback/src/candidate.rs:18`), `check_verification_gate`
(`crates/ptr-pods/src/workspace/gate.rs:26`) and `classify` (`findings.rs:352`). The tests are
`gate_equals_candidate_passes_over_the_full_matrix` (`crates/ptr-pods/tests/workspace.rs`, 600
cases) and `classify_matches_ptr_feedback_over_the_full_matrix`
(`crates/ptr-router/tests/findings.rs`). A system is **verifier-closed** if the only way a request
returns `Verified(x)` is `Acc(x)` for a report produced by `V` on exactly `x` under the pinned
verifier epoch.

Two-stage reading. `Acc` decides the *answer*. The workspace gate decides *integration*: a
sub-result enters a slot only if the same predicate holds for it. A composite answer built from
workspace content is a new candidate and is verified like any other; workspace membership is not
acceptance.

### 2.6 Constraints (the invariants as formal conditions)

Each constraint states what a conforming implementation must satisfy. Status says whether the
constraint is enforced by a type, a test, or only by this design.

| # | Constraint | Formal statement | Invariant | Enforced by |
|---|---|---|---|---|
| C1 | No self-promotion | For every observer `o ∈ {critic, self-model, schema, L3 policy, workspace}` there is no total function from `o`'s output type to `Accepted`, to a gate or threshold parameter, or to `RegistryUpdate`. `Acc` and `Accepted` are functions of `V` alone | 4, 5, 11 | Workspace: `ProposalInit` has no report, level, threshold or verifier field (`no_submitted_value_can_change_the_gate_or_the_config`). Router: sealed `Accepted`, scorer returns Pod weights only. Self-model: `compile_fail` doctests on `SelfModelHint` (branch `claude/dev-pm-selfmodel`). L3: no gate or verifier import (source-scan test, branch `claude/dev-pm-l3`) |
| C2 | Verifier precedence | `Acc` is invariant under any change of `score`, of the scorer `θ`, of hints and of bids. A learned score `s` may only choose inside `A(class) = {p ∈ G(class) ∩ E}`: `supp(π_θ) ⊆ A` | 11 | `AdmissibleScorer` sees only admissible Pods and cannot name an outcome class; a weight outside the set is `OutsideAdmissibleSet` (`findings.rs:649`, `:670`); bids order gate-passers only |
| C3 | Bounded state | `|slots| ≤ k ≤ 64`, `|P| ≤ 1024`, mailbox depth `≤ 64`, findings per refusal `≤ 32`, codes per report `≤ 32` (else typed fallback, never truncation), registry caps, tracked unroutable codes `≤ 1024` plus an overflow counter, self-model tables capped with overflow buckets, calls `≤ V_max`, rounds `≤ T_max` | 8 | Typed config validation; counters for saturation; `check_invariants` on the workspace |
| C4 | Provisional workspace content | No function maps workspace state to a ledger event. Proposals carry a `computed_against` revision and are refused if stale; pending proposals never survive a cycle; workspace content is derived and revision-bound | 4, 5, 2 | `Workspace` has no ledger or `SemanticHost` dependency; staleness rule in `run_cycle`; persistence of workspace content in SemDB is **not** built |
| C5 | Propensity logging for OPE | Every sampled routing decision logs the full support `[(p, width_p)]` with `Σ width = 2^53`, the chosen Pod with generation, the quantised propensity `π_log(p) = width_p / 2^53 ∈ (0, 1]`, the nonce, key id, and the registry generation; the propensity is conditional on the eligible set `E` at decision time. An estimator refuses a target policy with mass where `π_log = 0` | 15 | `FindingRouteRecord` (`findings.rs:492`) and `route_checked` (`router.rs:627`); the categorical L2 estimator and contextual log of 37 section 2 are **not** built |
| C6 | Rewards from verified transitions only | Training reward `r = score(ρ)` if `status = Pass` and `rk(level) ≥ rk(min level)`, else `0`; no statement a model makes about itself is a reward, target or filter | 11; principle 2 of 38 | `rl_reward` (`training/src/ptr_training/moe/reward.py:106`); a guard counter for C005 |
| C7 | `Exhausted` over best effort | The result type has no variant carrying a rejected candidate | 11, 12 | `Outcome<T>` (`repair.rs:150`); `an_unrepaired_defect_exhausts_the_budget_without_returning_the_failure` |
| C8 | Unknown is an outcome | `Unknown` and `Disputed` reports are never classified `Repair` | 12 | `classify` and its matrix test |
| C9 | Structure only through admission | Every structural change is an `ExpertProposal` admitted by `admit_expert_proposal` with gate evidence bound to the proposal, artifact and attested inputs; the policy emits requests only and cannot produce an admissible proposal before the host completes it with a trained manifest | 3, 4, 7, 17 | `crates/ptr-runtime/src/expert.rs:2210` (merged, R011); `StructureProposal::needs_training`, `ready`, `complete` (branch `claude/dev-pm-l3`) |
| C10 | Replayable | Router draws are keyed hash draws of (key, request, class, nonce); the workspace rule is a deterministic function (no clock, no RNG state); same inputs give the same snapshot digest | 14 | Determinism tests on both; replay of the full loop is not built |

### 2.7 Training signals

* **L2 router (RLVR over logged decisions).** Each routing record has an outcome fixed at a
  preregistered deadline: `r = reward(ρ)` of the verifier report of the proposal the chosen Pod
  produced (`Pass` and level at least the minimum: `score`; `Fail`, `Disputed`, `Unknown`,
  below-level, timeout, crash: `0`). Failures are outcomes and are never dropped (no
  action-dependent censoring); estimates use matured cohorts under one pinned reward and
  verifier epoch (37 section 2). The offline estimator is the categorical IPS, SNIPS and
  doubly-robust family; `evaluate_off_policy` (`crates/ptr-branch/src/arbiter.rs:1027`) and
  `doubly_robust` (`:1091`) are templates for the triage action space only.
* **Finding-routed repair experts.** Each verified `(codes, x, x′, Pass)` transition is a training
  example for the repair expert registered for those codes; non-passing transitions are negative
  or discarded as the experiment fixes beforehand, never reweighted by a self-report.
* **L1 experts.** RLVR with the GSPO or SAPO variants configured in `training/` (37 section 5);
  verifier-derived rewards only.
* **Self-model and schema.** Trained prequentially on traces against verifier *outcomes*
  (predict, then observe, then update); scored by Brier, ECE and Wilson intervals. They are never
  trained on verbal confidence statements or on any statement about experience.
* **L3 policy.** No reward in the first design: it consumes statistics (unroutable counts, clear
  rates, usage, AIMER, interference) and its proposals are decided by the gates. An outcome-driven
  policy is the M013 context-ablation arm and F006.

## 3. Component map and evidence

Locations are `file:line` on the integration line (`claude/dev-integration`, tree at the head of
this branch) unless a branch is named. Lines of in-progress packages are not pinned because
they move until the branch is committed; items are named instead. Evidence classes are those of
the repository: **pipeline evidence** (a smoke run proves the pipeline, no quality claim),
**mechanism check** (a preregistered, mutation-checked experiment that tests a loop or contract
against an analytic model or oracle, usually with a stub; supports a mechanism, makes no claim
about a model), **research result** (a model-backed, preregistered, seeded comparison against
baselines with a verdict). **There is no research result for any VCW-MoE component.**

### 3.1 Implemented and tested contracts

| Component | Location | Tests | Evidence |
|---|---|---|---|
| Verifier report and fabric | `crates/ptr-verifier/src/lib.rs:19` (`VerificationReport`), `:26` (`Verifier`), `:49` (`VerifierFabric`) | crate tests | Contract only |
| Accept predicate | `crates/ptr-feedback/src/candidate.rs:18`, `crates/ptr-pods/src/workspace/gate.rs:26`, `crates/ptr-router/src/findings.rs:352` | Cross-pinned full-matrix tests (section 2.5) | Tested contract |
| Generate-verify-repair loop | `crates/ptr-feedback/src/repair.rs:23` (`Generator`), `:29` (`Repairer`), `:37` (`RepairBudget`), `:150` (`Outcome`), `:166` (`RepairLoop`), `:184` (`run`); `candidate.rs:68` (`Critique::from_report`) | `crates/ptr-feedback/tests/` | **F001 mechanism check** (completed, 5 seeds, mutation 24 of 24): 0 false accepts, 0 double verifications, 0 budget violations, 0 collateral edits against an exact analytic model; stub skill, no model claim |
| Bounded verifier-gated workspace | `crates/ptr-pods/src/workspace/mod.rs:360` (`Workspace`), `:429` (`submit`), `cycle.rs:109` (`run_cycle`), `gate.rs:26`, `config.rs:138` (`WorkspaceConfig`), `proposal.rs:135` (`Proposal`), `snapshot.rs:283` (`Refusal`), `:328` (`SlotContent`) | `crates/ptr-pods/tests/workspace.rs` (41 tests) plus unit tests | Tested contract. **No task evidence.** C001 tests a *learned* analogue, not this contract; the contract's evidence is C006 (GPU) |
| Finding router | `crates/ptr-router/src/findings.rs:773` (`FindingRouter`), `:850` (`route`), `:137` (`FindingRegistry`), `:272` (`FallbackPolicy`), `:393` (`PodEligibility`), `:467` (`RouterConfig`), `:492` (`FindingRouteRecord`), `:649` (`AdmissibleScorer`) | `crates/ptr-router/tests/findings.rs` (21 tests, including the classify oracle and IPS recovery on a synthetic log) | Tested contract. **No model evidence**; the learned scorer is the uniform scorer only |
| Propensity logging | `crates/ptr-router/src/router.rs:81` (`RoutedChoice`), `:412` (`RouteDistribution`), `:627` (`route_checked`); `propensity.rs:17` | `crates/ptr-router/tests/router.rs` | Tested contract. Categorical OPE estimator **not built** |
| Expert lifecycle types and gates | `crates/ptr-pods/src/expert.rs:1020` (`ExpertProposal`), `:1353` (`ExpertOperationRequest`); `crates/ptr-lineage/src/forgetting.rs:133` (`ForgettingGate`), `aimer.rs:42`, `subspace.rs:1123` | crate tests | Tested contracts |
| Expert admission, fencing, readmission | `crates/ptr-runtime/src/expert.rs:337` (`ExpertGateRequirement`), `:447` (`ExpertAdmissionGrant`), `:1104` (`ExpertRegistryProjection`), `:2210` (`admit_expert_proposal`), `:2417` (`fence_expert`), `:2471` (`readmit_expert`); `crates/ptr-runtime/src/expert/authority.rs`, `dependency.rs`; ledger events `crates/ptr-ledger/src/lib.rs:249` (`ExpertStructuralCommitted`), `:311` (`InputInvalidated`) | `crates/ptr-runtime/tests/expert_admission.rs` | **R011 mechanism check** (narrowed, reference path): an L3 policy reached the gate only through admission on a pinned adversarial suite; no bypass, no self-promotion path. Does not test that any policy is *useful* |
| Server request path (package VS) | `crates/ptr-server/src/vcw/` (`VcwService`, `POST /v1/vcw/tasks`), `crates/ptr-config/src/vcw.rs` (`[vcw]`, default off), `crates/ptr-runtime/src/vcw/experts/recode.rs` (`Recoded`), `bins/ptrd/src/main.rs` (`install_vcw`) | `crates/ptr-server/tests/vcw.rs`, `crates/ptr-config/tests/vcw.rs`, `bins/ptrd/tests/vcw.rs`; `tests/vcw_qwen.rs` (ignored, needs a model) | Tested contract. The loop runs per request over the real task verifiers (arithmetic, JSON Schema subset, code tests, facts) with simulated experts; an accepted answer is admitted by the runtime's output admission as a `Candidate` (not promoted), `Exhausted` and `Escalated` are `422` with no candidate, and the trace digest is in the admission's manifest and in `RuntimeEvent::VcwFinished`. **No model evidence**; the `qwen-cpu` backend is feature-gated and not compiled by CI |
| Learning router and promotion gate (package VL) | `crates/ptr-router/src/learning.rs` (`RoutingLearner`, `LearnedPolicy`, `VerifierReward`), `ope.rs` (`estimate_policy`, `compare_policies`, `PromotionGate`, `PromotionEvidence`), `policy_book.rs` (`PolicyBook`, `ActiveScorer`); `VcwLoop::with_learner` in `crates/ptr-runtime/src/vcw/closed_loop.rs` | `crates/ptr-router/tests/learning.rs`, `crates/ptr-runtime/tests/vcw_learning.rs` | Tested contract, **mechanism only, with seeded programmatic experts**. A Thompson-sampling bandit per (route class, Pod) trained on `classify` verdicts only (a declined repair is a failure, not a missing observation); a frozen snapshot is an `AdmissibleScorer` that can only weight the admissible Pods (the router keeps the floor and the eligibility filter); IPS, SNIPS and DR with standard errors rebuild the target distribution with the router's own function; a candidate is promoted only by a host through `PolicyBook::promote`, on decisions logged after its snapshot, when the lower confidence bound of the paired value difference exceeds a margin fixed in advance; the loop holds no `PolicyBook`. The gate reads the evaluation log from the learner itself (a hand-built log or a stated reward cannot produce evidence: `VerifierReward::replayed` and `LoggedDecision::replayed` are not public), refuses rules below fixed floors (100 records, effective sample 30, one-sided 95%), and the exploration floor is bound to the deployment (a `PolicyBook` is built for a positive `Exploration`; a router or loop with another rate is refused). The estimate is a one-step per-route clear rate with decisions treated as independent and no peeking correction. `RuntimeEvent::VcwPolicyChanged` is observability, not an audit record: it is in memory and not replayed, so a host must not promote in production until a ledger event exists. The simulated difference between two repair Pods is a configured fact, not a result about a model |
| Live L3 through the ledger (package V3) | `crates/ptr-runtime/src/vcw_host/` (`StructureDriver`, `SimulatedTrainer`, `SimulatedHeldOut`, `rebuild_roster`) | `crates/ptr-runtime/tests/vcw_structure.rs` | Tested contract, **mechanism only**. Windows of real verifier outcomes (per-code clearing) become Birth, Prune, Freeze (and Merge, Reactivate, not exercised end to end) requests; the host completes a Birth with a simulated specialist (a seeded skill table, not training), measures clear rates on held-out verified tasks before and after, binds a `ForgettingGate` report to the proposal and calls `admit_expert_proposal`; only a committed expert is registered and enrolled by `sync_roster`; a refused report commits and enrols nothing. The held-out suite is told only a candidate's identity (the trainer's skill and collateral are untrusted claims), must measure every code of the roster and of the candidate, and its digest is the evaluation manifest of the bound subject; a change that commits but cannot be installed is reported as committed-not-installed and retried, never as a refusal. The L3 clear credit requires an accepted candidate or a repaired candidate whose hard codes introduce nothing new. A dataset revocation fences the born Pod and its dependents; replay reproduces the registry and, with the manifests, the roster |
| Reward mapping and RL pipeline | `training/src/ptr_training/moe/reward.py:106` (`rl_reward`), `rl.py`, `routed_peft.py` | `training/tests` | **Pipeline evidence only** (WP8 CPU smoke on a 0.24M-parameter random-init OLMoE, one seed). The OLMoE config has not been run |

### 3.2 In review on package branches (not on the integration line)

> **Stale on this line.** At `8bff6fbf` the self-model (`crates/ptr-router/src/self_model.rs`), the
> attention schema (`attention_schema.rs`) and the L3 structure policy
> (`crates/ptr-runtime/src/expert/structure.rs`) are on the integration line and are used by
> `VcwLoop`; the rows below describe their state when this section was written. Section 3.5 is the
> current per-component status.

| Component | Branch and file | State | Evidence |
|---|---|---|---|
| Self-model (trace statistics, calibration, advisory hints) | `claude/dev-pm-selfmodel`: `crates/ptr-router/src/self_model.rs` (`SelfModel`, `SelfContext`, `SelfModelHint`, `CalibrationTracker`, `HintPolicy`) | Uncommitted, in progress. The module documentation states the authority limits: `SelfModelHint` is plain advisory data and does not convert to `Accepted` or `RegistryUpdate` (`compile_fail` doctests), the model does not implement `AdmissibleScorer` | None in the repository for this branch's code. C002 stage A is the CPU toy test C007, verdict FALSIFIED (section 3.4) |
| Attention schema (predicts routing, optional control inside the admissible set) | same branch: `crates/ptr-router/src/attention_schema.rs` (`AttentionSchema`, `ControlMode`, `SchemaLimits`) | Uncommitted, in progress. `AttentionSchema::apply_hints` re-weights inside the verifier-determined set; `ControlMode::ObserveOnly` exists as the describe-only arm | None in the repository for this branch's code. C003 stage A is the CPU toy test C008, verdict FALSIFIED (section 3.4) |
| L3 structure policy | `claude/dev-pm-l3`: `crates/ptr-runtime/src/expert/structure.rs` (`StructurePolicy`, `StructurePolicyConfig`, `StructureObservation`, `StructureProposal`, `propose`, `unroutable_delta`); `crates/ptr-runtime/tests/expert_structure.rs` | Uncommitted, review fixes applied. Operators proposed: Birth, Reactivate (of a `Retired(GateFailed)` expert), Prune (by AIMER and usage, never the last demonstrated clearer of a code), Merge, Freeze; Split and Retire are never proposed. Results needing training are *requests* until the host supplies the trained manifest digest | Unit and integration tests only (48 plus 4 at the last report). Mutation check on the new rules killed 5 tests; this is not a preregistered experiment. **No evidence that the policy helps** |

### 3.3 Planned or not built

| Component | Status | Registered experiment |
|---|---|---|
| Runtime composition of workspace, finding router, repair loop, budgets, eligibility and structural ledger events as one path (package H; new path, old signatures unchanged) | Planned | none by itself; X exercises it |
| End-to-end CPU experiment with stub experts: full loop against learned-gate routing, best-of-N with the verifier, and ablations without workspace, finding routing, self-model (package X) | Registered as F007 (`experiments/feedback/F007-vcw-loop`, registry `completed`, verdict COMPLETED in `results/run.json`, mechanism check on simulated experts; section 3.7) | F007 |
| Categorical L2 off-policy estimator (37 section 2) | Built for the finding router (`ope.rs`, package VL); the contextual routing log of 37 section 2 and any estimate on a model-backed loop are not built | needed by F004, F005 |
| Learned `AdmissibleScorer` | Built as a context-free Thompson bandit (`learning.rs`, package VL); a feature-based policy is not built. Learned critic Pod (advisory): not built | F004 |
| Persistence of workspace content in SemDB or a branch overlay | Not built, not required by the contract | C006 |
| Verifier-statistics driven lifecycle on a real model | Not built | F006 (GPU) |

### 3.4 Experiments and what each can show

| Id | Tests | CPU or GPU | Status | Evidence class when finished |
|---|---|---|---|---|
| F001 | Loop mechanism with a seeded stub | CPU | **Completed** (mechanism check) | Mechanism check |
| R011 | No bypass of the gates by an L3 policy | CPU reference path | **Completed** (narrowed) | Mechanism check |
| X (registered as F007, next row) | Whole loop with stub experts | CPU | see F007 | Mechanism check (stub skill is declared; no model claim) |
| C001 | Learned-attention workspace as an inductive bias | CPU, 20 seeds (4001-4020) | Registry `prepared` (as defined for NOT_SHOWN), frozen at `c001-prereg`, run complete. **Outcome NOT_SHOWN** (`results/run.json`; STATUS.md): the workspace beats sparse all-to-all (+132 per mille, lower bound 65) but not all-to-all (+27 [-15, 70]) or bandwidth-matched all-to-all (+19 [-23, 60]) beyond the 30 per mille margin | Mechanism check of a learned analogue; not a result about the VCW workspace contract |
| F007 | Whole verifier-closed loop (`VcwLoop`, finding router, learned router through the OPE gate, workspace, self-model, L3 through the ledger) against a trained softmax gate and best-of-N at matched verifier-call budget, on simulated expert Pods | CPU | Registry `completed`, frozen at `f007-prereg`, 5 seeds (17/29/43/71/101), all valid. **Outcome COMPLETED** (`experiments/feedback/F007-vcw-loop/results/run.json`, `results/metrics.json`): verified success +7.20 points over the trained gate (one-sided 95% bounds [6.51, 7.88]) and +6.35 over best-of-N [5.40, 7.31] in the heterogeneous-skill world, margin 3; all controls passed including the null-world ties; workspace secondary SUPPORTED (+2.16 [1.93, 2.39], an implementation check), L3 secondary NOT_SHOWN (+3.28 [-0.78, 7.34], low power), self-model untested | Mechanism check (a gain is a consequence of the declared skill tables; a homogeneous-skill world ties). It is the registered form of the experiment called X above |
| C007 (C002 stage A), C008 (C003 stage A), C004 | Self-model, schema, recurrence on toys | CPU | C007 FALSIFIED (registry `failed`; Brier gain over probe +0.13 per mille [-1.20, 1.45], over own confidence -2.54 [-4.21, -0.87], margin 4). C008 FALSIFIED (registry `failed`; +7.1 per mille [4.8, 9.4] over a router that reads the history directly, margin 25: the gain comes from the history, not the schema form). C004 COMPLETED (registry `completed`; recurrent minus attention stack +782.6 per mille [769.5], extrapolation only). Sources: each `results/run.json`; doc 38 section 6. C002 and C003 themselves (GPU) stay `planned` | Mechanism check, calibration toys; no positive effect of a self-model or a schema was shown |
| C002 stage B, C003 stage B, C005, C006 | Self-model, schema, self-report measurement, workspace at Pod level | GPU | Registered, planned | Research result (if run) |
| F004, F005, F006 | Closed loop vs standard MoE and best-of-N; finding-routed repair; verifier-driven lifecycle | GPU, OLMoE-class Pods | Registered, planned | Research result (if run) |
| M010-M015 | Base training programme on OLMoE (RLVR, PERFT-R, aux-free balancing, lifecycle) | GPU | Planned (WP8 smoke only) | Research result (if run) |

F003 (calibrated arbiter, harm-rate bound for auto-proposals) and S002 (snapshot cancellation of
resumable runs) are related substrate work that is not part of this type. Both are recorded as
completed in `STATUS.md` on this line; no claim of this document rests on either (section 3.5
says where F003 is, and is not, relevant).

### 3.5 Evidence status per component

State at `8bff6fbf` (`claude/dev-integration`), read from the tree, `STATUS.md` and
`experiments/registry.toml`; nothing here is a result about a model. "Tested" names test files and
counts `#[test]` functions (not a pass count). "Experimentally evaluated" says what an experiment
shows and is **none** where no experiment ran. "Needs a GPU" is about the experiment that would
evaluate the component, not about running it: no component of the loop as built needs one.

| Component | Implemented | Tested | Experimentally evaluated | Needs a GPU |
|---|---|---|---|---|
| Workspace (bounded, verifier-gated) | `crates/ptr-pods/src/workspace/` | `crates/ptr-pods/tests/workspace.rs` (41) plus unit tests | None for the contract. C001 tests a *learned* analogue, not this contract; its outcome is NOT_SHOWN (section 3.4) | C006 (workspace at Pod level) |
| Finding router | `crates/ptr-router/src/findings.rs` (`FindingRouter`, `FindingRouteRecord`) | `crates/ptr-router/tests/findings.rs` (21), `tests/router.rs` (45) | None on the router itself. F001 exercises the loop with a stub generator, not this router | F005 (finding-routed repair experts) |
| Learning router and OPE | `learning.rs`, `ope.rs`, `policy_book.rs` (package VL) | `crates/ptr-router/tests/learning.rs` (14), `src/ope_tests.rs`, `crates/ptr-runtime/tests/vcw_learning.rs` (7) | None for the VCW router. F003 (completed) calibrates a different object, the `ptr-branch` arbiter, and uses IPS and doubly robust estimators on synthetic data; it is substrate evidence for estimator code of that crate, not for this learner. The difference between two repair Pods in the tests is a configured fact | F004, F005 (model-backed log) |
| Self-model and attention schema (functional, advisory) | `self_model.rs`, `attention_schema.rs`; `SelfStepTrace` in the loop trace | `crates/ptr-router/tests/self_model.rs` (20), `tests/attention_schema.rs` (19), `tests/authority_surface.rs` (2) | None. C002 and C003 stage A (CPU toys) are `planned` in the registry. Per doc 38 these are functional indicators (prediction of the system's own failures, control inside the admissible set), not claims about experience; `hint_mode = "trace-only"` is the default | C002 / C003 stage B |
| L3 structure driver (admission, `ForgettingGate`, ledger) | `crates/ptr-runtime/src/vcw_host/` (`StructureDriver`), `src/expert/structure.rs`, `src/expert.rs` (admission), `crates/ptr-lineage/src/forgetting.rs` | `crates/ptr-runtime/tests/vcw_structure.rs` (15), `expert_structure.rs`, `expert_admission.rs`; the specialist and the held-out suite are simulated | R011 (completed, narrowed): no bypass of the gates by an L3 policy on a pinned adversarial suite. It does not show that the policy helps. Nothing evaluates the driver with a real specialist | F006, M013 (training specialists) |
| `VcwLoop` budgets, `Exhausted`, `Escalated` | `crates/ptr-runtime/src/vcw/closed_loop.rs`, `budget.rs`, `trace.rs` | `crates/ptr-runtime/tests/vcw_loop.rs` (42), unit tests; the demo (`make vcw-demo`) checks `Exhausted` end to end through `ptrd` | F001 (completed, mutation 24 of 24) for the *ptr-feedback* loop with a seeded stub: 0 false accepts, 0 budget violations, `Exhausted` instead of best effort. The runtime `VcwLoop` itself has no preregistered experiment (X, the whole loop with stub experts, is not registered on this line; the package owner reports it as preregistering) | No |
| Real verifiers (arithmetic, JSON Schema subset, code tests, facts) | `crates/ptr-verifier/src/{arithmetic,json_schema,code_test,fact}.rs`, code table in `codes.rs` | 63 unit tests in the crate, `crates/ptr-runtime/tests/vcw_real_verifiers.rs` (22) | None. They are deterministic checkers; their coverage is that of their tests, not an experiment | No |
| Experts, stage (c): programmatic specialists with a controlled error rate (`SimulatedExperts`) | `crates/ptr-runtime/src/vcw/experts/simulated.rs`, `guard.rs` (call-time eligibility), `recode.rs`; `crates/ptr-server/src/vcw/sim.rs` | `crates/ptr-runtime/tests/vcw_experts.rs` (20), unit tests (10 in `simulated.rs`) | None, and none can show model quality: the candidates are drawn from the client's own `reference` with configured error and repair rates. They exist to test the loop, the router and the admission path | No |
| Experts, stage (a): a small open-weight model on CPU behind `TextGenerator` (`qwen-cpu`) | `crates/ptr-runtime/src/vcw/experts/llm.rs`, `crates/ptr-pods/src/qwen_cpu/`, `crates/ptr-server/src/vcw/qwen.rs` (feature `qwen-cpu`, not compiled by CI) | `crates/ptr-runtime/tests/vcw_qwen_model.rs` (2) and `crates/ptr-server/tests/vcw_qwen.rs` (1), both need a local checkpoint and are ignored without one | None. No run is reported in `STATUS.md`; the demo's optional qwen run (section 3.6) is a smoke run of one pipeline, not an evaluation | Runs on CPU; a model-backed comparison (F004) needs a GPU |
| Experts, stage (b): small self-trained specialists | Not built (`vcw_host::specialist.rs` is a simulated trainer) | None | None | Yes (F005, F006, M013) |
| Server path (`POST /v1/vcw/tasks`, `GET /v1/vcw/traces/{id}`, `[vcw]`) | `crates/ptr-server/src/vcw/`, `crates/ptr-config/src/vcw.rs`, `bins/ptrd/src/main.rs` (`install_vcw`) | `crates/ptr-server/tests/vcw.rs` (35, `#[test]` and `#[tokio::test]`), `crates/ptr-config/tests/vcw.rs`, `bins/ptrd/tests/vcw.rs` (5); `make vcw-demo` drives the built `ptrd` | None (not an experiment). The demo is an end-to-end smoke of the path with the stage (c) experts | No |

Open items behind the table: the runtime composition is exercised only with stage (c) experts (and, once, a smoke run with stage (a));
the demo and the tests show the *mechanism* (a failing first candidate, a routed repair, an
accepted or exhausted request, what the ledger holds), which F001 had already shown for the
`ptr-feedback` loop. Whether any of it helps a real model is F004 to F006, all `planned`.

### 3.6 Demo: `make vcw-demo`

One command, nothing downloaded, only the pinned toolchain and `python3` (standard library):

```text
make vcw-demo
```

It builds `ptrd`, starts it with `config/vcw-demo.toml` (the `[vcw]` path on, backend `simulated`,
the real arithmetic, JSON Schema and code-test verifiers, a free loopback port, the data directory
in a temp directory) and sends three requests through `POST /v1/vcw/tasks`. Expert stages, as the
project names them: (a) a small open-weight model on CPU (`qwen-cpu`, optional, below), (b) small
self-trained specialists (not built), (c) programmatic specialists with a controlled error rate
(`SimulatedExperts`, the default here).

The roster has two eligible repairers per repair role used, with different skill (configured via
`[vcw.pods.extra_repairers]` and `pod_repair_success`): `arith.value` goes to `vcw-value-repair`
(skill 1.0) or `vcw-generalist-repair` (0.5; it also holds the format and schema roles, so its
coverage differs); `json.required` to `vcw-schema-repair` (1.0) or the generalist (0.5);
`code.test_failed` to `vcw-code-repair` or `vcw-code-repair-b` (both skill 0). The finding router
samples among the eligible Pods, so each repair decision logs a support of two Pods and a
propensity of 0.5.

1. `demo-arith-repaired`: `6*7`. The simulated proposer always plants `arith.value`, so the first
   candidate (`43`) fails the real arithmetic verifier; the router draws a repairer, the repaired
   candidate passes, the runtime admits it.
2. `demo-json-repaired`: a JSON object with required `name` and `age`; the first candidate lacks a
   field (`json.required`) and is repaired in a routed round (in the run below the generalist is
   drawn first and fails, the schema specialist is drawn next and succeeds).
3. `demo-code-exhausted`: a program that must square its argument, with a correct reference. The
   proposal fails the real code tests (`code.test_failed`) and no eligible repairer can clear
   that finding (skill 0), so the request runs out of budget: `422 exhausted`, reason
   `VerifierCalls` (6 of 6 verifier calls, 5 of 5 rounds), no answer in the response, no
   `PodOutputAdmitted` in the ledger (no best-effort accept). The cause is the experts' inability
   under budget, not a corrupted reference.

For each request it reads back `GET /v1/vcw/traces/{id}` (the retained trace and the ledger
events the request committed, read from the runtime's ledger) (a read of a request that is still running returns `outcome` `"running"` and `ledger` `null`) and prints rounds, finding codes,
the routed Pod with its propensity and support, the budget used and the events. Then it sends
SIGTERM to `ptrd`, requires exit code 0, and exits 0 only if every expected property held. The
properties asserted do not depend on which Pod the router drew (the routing nonce instance is
fresh per start): the first candidate failed a real verifier with the expected code; each repair
went to a Pod eligible for that code and its propensity is in (0, 1) and equals the logged
support entry of the chosen Pod; an answer is returned only after a candidate passed and only
with an admission in the ledger; the Exhausted request has the terminal reason of an exhausted
budget dimension, no answer and no admission; the trace matches the response digest; a reused
id is refused (409). It does not depend on luck: with seed 32 the generalist clears `arith.value` and `json.required` by its second attempt and the specialists always do, so whatever sequence the router draws, a repair is done within two repair rounds (budget 5) and the exhausted request exhausts whichever Pods are drawn; `make vcw-demo` was run ten times in a row for this. Excerpt of a real run (the full output is in the report):

```text
=== demo-json-repaired
  round 0  origin=initial  authors=[vcw-proposer@1]
      candidate 88bab6d8f6dc  verifier=Fail codes=['json.required']  class={"repair": ["json.required"]}
  round 1  origin=repaired  authors=[vcw-generalist-repair@1]
      routed Repair(FindingCode("json.required")) -> vcw-generalist-repair@1 propensity=0.5000  support=[vcw-generalist-repair:4503599627370496, vcw-schema-repair:4503599627370496]
      candidate 88bab6d8f6dc  verifier=Fail codes=['json.required']
  round 2  origin=repaired  authors=[vcw-schema-repair@1]
      routed Repair(FindingCode("json.required")) -> vcw-schema-repair@1 propensity=0.5000  support=[...same...]
      candidate fc234b36d998  verifier=Pass codes=[]  class="accept"  workspace=Admitted
  budget: used rounds 2/5, verifier calls 3/6, pod calls 3
  ledger: #15 SemanticDeltaCommitted  #16 PodOutputAdmitted  #17 SemanticDeltaCommitted
=== demo-code-exhausted
  POST /v1/vcw/tasks -> HTTP 422 {"outcome": "exhausted", ..., "reason": "VerifierCalls", "rounds": 5, "verifier_calls": 6, ...}
  ... 6 candidates, every one verifier=Fail codes=['code.test_failed'], each repair routed with propensity 0.5 ...
  budget: used rounds 5/5, verifier calls 6/6, pod calls 6
  ledger: 1 event(s) from index 18: #18 SemanticDeltaCommitted
```

What the demo is not: the experts are programmatic (stage (c)) and the failures are scripted by
error rates of 1.0; the skills are configured facts, not results. It is a smoke test of the
mechanism (routing among several eligible Pods, repair, acceptance only after a verifier pass,
exhaustion by inability), not evidence about a model. The round-0 "routed" line is the proposal
(class `Regenerate(Unknown)`, a single proposer). Optional stage (a): with `PTR_VCW_QWEN_DIR` set
to a local checkpoint directory (`model.safetensors`, `config.json`, `tokenizer.json`,
`tokenizer_config.json`; the digests are computed at start-up) the Makefile builds `ptrd` with
`--features qwen-cpu`, restricts the kinds to arithmetic and JSON (the model backend has no
prompts for code) and sends the first two requests without a `reference`; only the safety property
(an answer is returned if and only if the verifier accepted) is asserted, because repair is not
scripted. One such run was made (CPU shared with other jobs, `max_new_tokens = 16`, a 0.5B instruction model,
local checkpoint): `6*7` was answered `42` by the proposer on the first call and passed the real
verifier (no repair needed, admitted). The JSON request failed its first candidate with
`json.parse`; `json.parse` was routed to a repairer (propensity 0.5 between the format specialist
and the generalist) but no Pod produced a candidate in the following rounds and the request ended
`422 exhausted` with reason `Deadline` (the 30 s per-request limit, `max_request_ms`, was hit on
the loaded CPU; 1 of 6 verifier calls used), with no answer and no admission. That is one smoke
run of the pipeline, not an evaluation: it shows that the model backend runs through the same
path and that the loop refuses to answer when no candidate verifies.

### 3.7 Evidence status of the VCW-MoE track

Registry status is `experiments/registry.toml` on branch `claude/dev-docs13`; outcomes are those
written in each experiment's `results/run.json` or the STATUS.md line.

| Item | Status | Outcome and scope |
|---|---|---|
| `make vcw-demo` (`Makefile` target, `scripts/vcw_demo.py`, `config/vcw-demo.toml`) | Smoke path, not an experiment | Builds `ptrd` and drives `POST /v1/vcw/tasks` with simulated experts and the real verifiers, exiting non-zero if an expected property fails (`scripts/vcw_demo.py`, checks printed as `[ok]`/`[FAIL]`). CPU only. It shows the mechanism end to end; it is not run in CI by this document's checks and says nothing about model quality |
| `ptr_py` binding (`bindings/ptr-py`, `make py-bindings`) | Built and tested by `make py-bindings` (fmt, clippy, `cargo test`, `maturin develop`, `unittest` over `bindings/ptr-py/tests/test_binding.py`, `test_vcw.py`); CI does not build it (root `README.md:141`) | Gives Python training code the governance checks plus read-only access to the VCW deterministic verifiers, the finding router with its propensity log and OPE estimators, the bounded workspace and an in-process loop simulation (`bindings/ptr-py/README.md:3`; modules `src/vcw.rs`, `routing.rs`, `verify.rs`, `workspace.rs`). A tooling surface, not evidence about any model. Needs Python >= 3.11 |
| F007 `vcw-loop` | completed | COMPLETED (`experiments/feedback/F007-vcw-loop/results/run.json`; tag `f007-prereg`, freeze commit 99d92379; 5 seeds; mutation check 33 of 33). Full loop minus trained gate +7.20 points [6.51, 7.88], minus best-of-N +6.35 [5.40, 7.31], margin 3; without L3 the same (+7.16, +6.32); removing finding routing costs 16.1 points; controls passed, including null_tie_full_vs_shuffled +0.46 [+0.10, +0.83] inside the +-1.5 band (lower bound above 0, reported not judged). Workspace secondary SUPPORTED, L3 secondary NOT_SHOWN (power about 0.2), self-model UNTESTED. **Simulated experts with declared per-(Pod, finding code) skill tables: the result is a consequence of those tables (the null world ties), not a statement about any real model or about best-of-N with a repair Pod** |
| L001-v2 `revocation-crash` | completed | COMPLETED (`experiments/lifecycle/L001-v2-revocation-crash/results/run.json`; tag `l001v2-prereg`; mutation check 24 of 24). 13,771 recoveries; all falsifier rules held, with the stale-accept counter at 0 for the gated cases; `RaftEngineLedger` has no anchor, so its 90 stale accepts after a lie that drops acknowledged bytes are descriptive only (STATUS.md, owner decision of 2026-10-09). Other limits (single host, crash-image power model) are in STATUS.md. It is substrate evidence for the ledger that makes expert and Pod revocation durable; it does not evaluate the VCW loop |
| R011 | completed | COMPLETED; no bypass of the gates by an L3 policy on a pinned adversarial suite, reference path |
| F001 | completed | COMPLETED; loop mechanism with a seeded stub |
| C001 | prepared (NOT_SHOWN) | See section 3.4 |
| F004, F005, F006 (GPU) | planned | no outcome |

## 4. Related work and delimitation

All sources below were read on alphaXiv against the primary text on 2026-10-09 unless a row says
otherwise. "Novelty could fail" states a concrete way in which the corresponding part of the claim
would be wrong or redundant.

### 4.1 Mixture-of-experts architecture and routing

| Work | What it does (verified) | What VCW-MoE adds or does differently | Where novelty could fail |
|---|---|---|---|
| Switch Transformer, Fedus, Zoph, Shazeer (arXiv:2101.03961) | Top-1 routing of each token to one FFN expert; expert capacity with a capacity factor; auxiliary balance loss `α·N·Σ f_i P_i`, `α = 10⁻²`. Appendix C observes the router "receives no counterfactual information" about alternatives and likens routing to a contextual bandit | Same L1 substrate. L2 routing logs propensities so alternatives can be evaluated off-policy, which Switch's router cannot | If the verifier-derived signal is never used by the L1 router (it is not), then at L1 there is no difference; any gain must come from L2 |
| GShard, Lepikhin et al. (arXiv:2006.16668) | Top-2 gating, expert capacity, group-level dispatch, auxiliary loss, random routing of the second expert by gate weight | Same remark; the stochastic second-expert dispatch is an unlogged random choice, whereas VCW routing logs a quantised propensity | Same as above |
| DeepSeekMoE, Dai et al. (arXiv:2401.06066) | Fine-grained expert segmentation and isolated shared experts; expert-level and device-level balance losses; 2B to 145B scale | A shared always-on expert is a different mechanism from a workspace: it is a fixed path, not a competition for a bounded set of verified slots | A shared expert may capture most of the benefit of "common content" that the workspace is meant to carry |
| Auxiliary-loss-free balancing, Wang et al. (arXiv:2408.15664) | A per-expert bias added to the routing scores for top-k selection only, updated by `b_i += u·sign(e_i)` from the load of the previous batch (`u = 10⁻³`); tested at 1B and 3B | L1 of a VCW-MoE uses it unchanged (M012). **Correction to a common attribution:** this is the Wang et al. paper; DeepSeek-V3 (arXiv:2412.19437) adopts it ("pioneers" in the report's wording, bias update speed `γ = 0.001` for 14.3T tokens, then `0.0`, with a complementary sequence-wise loss `α = 10⁻⁴`) | The bias is a feedback controller on load; our L3 and the attention schema are feedback controllers on other statistics. Whether the schema adds anything over this bias is C003 stage B |
| PERFT, Liu et al. (arXiv:2411.08212) and PERFT-R / "Parameter-Efficient Routed Fine-Tuning: Mixture-of-Experts Demands Mixture of Adaptation Modules" (arXiv:2508.02587) | Routed PEFT experts inside the MoE layer with an independent router (PERFT-R), variants PERFT-E (reuse the pretrained router), -D, -S; OLMoE-1B-7B and Mixtral-8x7B; up to 17.2% (commonsense) and 12.3% (arithmetic) relative improvement over attention-LoRA at equal activated trainable parameters. The later version names the routed variant "PERFT" and keeps -E/-D/-S as ablations | The L1 adaptation layer of the VCW-MoE training programme (M011 arms, 37 section 5). PERFT supplies no verifier loop, no request-level routing, no lifecycle. Its gains were measured on supervised fine-tuning on commonsense and arithmetic sets | Our own M011 comparison may show that routed LoRA gives most of what is gained on verifiable tasks, leaving the loop with a small residual |
| Branch-Train-MiX, Sukhbaatar et al. (arXiv:2403.07816) | Trains domain experts by continued training of copies of a seed LLM, merges their FFNs into MoE layers, averages the rest, then fine-tunes the router; BTM and sparse upcycling as special cases; Llama-2 7B seed | Birth happens once, offline, by data domain. In a VCW-MoE Birth is an online, gated proposal triggered by a verifier-defined finding class that no expert clears | A static set built by BTX may match or beat a lifecycle at equal compute; that is exactly the static baseline of F006 |
| Self-MoE, Kang et al. (arXiv:2406.12034) | Builds LoRA experts from self-generated synthetic data and trains a router on the aggregate; 6.5 points average over the base on Gemma-7B; experts fixed after construction | Experts and router are not verified at inference; no lifecycle; no budgets | A router over semantic experts may recover the finding-class specialisation without any verifier (F005's generic-repair baseline tests this) |

### 4.2 Routing of whole models and multi-agent composition

| Work | What it does (verified) | What VCW-MoE adds or does differently | Where novelty could fail |
|---|---|---|---|
| RouteLLM, Ong et al. (arXiv:2406.18665) | Binary strong/weak model routing from the query, with a win-prediction model trained on preference data (Chatbot Arena, augmented with gold and LLM-judge labels) and a cost threshold; over 2x cost reduction at about 90% of the strong model on its benchmarks; related work names cascades (FrugalGPT) and AutoMix (self-verification before escalation) as multi-call alternatives | Routing is decided before generation from the query alone, with no verifier in the loop. In a VCW-MoE the verifier's report fixes the class of each step | A cascade with a good verifier (the FrugalGPT / AutoMix line, which I have not read directly) is a baseline of the loop in spirit; if it matches the loop, the finding-routing contribution is absent |
| Router-R1, Zhang, Feng, You (arXiv:2506.09033) | An LLM router trained with RL (veRL) that interleaves "think" and "route" actions across up to 4 rounds; reward is format (-1/0), exact-match outcome and a cost term; seven QA benchmarks; Qwen2.5-3B and Llama-3.2-3B bases; states inference latency as a limitation | Closest on learned multi-round routing and RLVR-style rewards. Differences: its reward is a final-answer match, not per-finding; it has no acceptance gate (EM is used in training only) and no deterministic class function; it conditions on model descriptors rather than verifier findings; no structure change | A learned multi-round router may learn the same finding-conditioned policy without a typed finding interface; then the interface adds engineering guarantees but no accuracy |
| Mixture-of-Agents, Wang et al. (arXiv:2406.04692) | Layers of LLM proposers; each next-layer agent receives all previous outputs; an LLM aggregator synthesises; no training; 65.1% LC win rate on AlpacaEval 2.0 with open models; beats an LLM-ranker baseline; names high time-to-first-token as a limitation | The aggregator is an LLM judgement; there is no deterministic acceptance, no bound on what is carried forward, no budget. The VCW workspace carries only verified content in `k` slots | On tasks with no deterministic verifier, MoA-style aggregation can beat any verifier-gated loop, which then returns `Exhausted` more often (section 7, verifier coverage) |

### 4.3 Verification, refinement and test-time compute

| Work | What it does (verified) | What VCW-MoE adds or does differently | Where novelty could fail |
|---|---|---|---|
| Training verifiers, Cobbe et al. (arXiv:2110.14168) | A learned verifier ranks 100 sampled solutions on GSM8K; a 6B verifier slightly beats a finetuned 175B model (about a 30x size equivalent); performance falls beyond about 400 completions, which the authors attribute to adversarial solutions that fool the verifier | The accept predicate is deterministic; a learned verifier may be added only as an advisory scorer or a verifier whose own level is below `Deterministic` | Where only learned verifiers exist, the type degenerates to a best-of-N with a learned ranker; the paper's own 400-sample degradation is a reward-hacking warning for any learned acceptance |
| Let's Verify Step by Step, Lightman et al. (arXiv:2305.20050) | Process-supervised reward models beat outcome-supervised ones for best-of-N search on MATH (78.2% against 72.4% for the ORM and 69.6% for majority voting at N = 1860); 800K human step labels (PRM800K) | Step-level credit assignment is the motivation for typed finding codes: a code names where the candidate failed, with deterministic verifiers instead of human labels | If findings are coarse (one code per task), credit assignment reverts to outcome supervision and the PRM line dominates |
| Large Language Monkeys, Brown et al. (arXiv:2407.21787) | Coverage (any sample correct) scales log-linearly with the number of samples over four orders of magnitude; with automatic verifiers (unit tests, proof checkers) coverage becomes success (SWE-bench Lite 15.9% to 56% with 250 samples); without verifiers, majority voting and reward models plateau near 100 samples; names multi-turn feedback from verifiers as untested | This is the decisive baseline. F004 requires the loop to beat best-of-N from the same MoE with the same verifier at matched verifier calls and FLOPs. The loop spends calls on repair instead of independent resampling | If independent sampling plus the same verifier matches the loop, a VCW-MoE is a verifier plus sampling and the architecture is refuted (R1, section 6) |
| Self-Refine, Madaan et al. (arXiv:2303.17651) | Same LLM produces feedback and refinement, no training, up to 4 iterations; 5 to 40 points absolute on 7 tasks; with oracle feedback math improves by up to 4.8 (GPT-3.5); in manual analysis 33% of failures were feedback that mislocated the error and 61% feedback that suggested an inappropriate fix | The feedback in a VCW-MoE comes from a deterministic verifier and is typed. Self-Refine's own failure analysis is evidence for the value of accurate feedback | Where verifiers are unavailable, self-feedback is the only option and Self-Refine is the baseline the loop cannot be compared with |
| Reflexion, Shinn et al. (arXiv:2303.11366) | Verbal reinforcement: an actor, an evaluator (exact match, heuristics, an LLM, or self-written unit tests) and a self-reflection model whose text is kept in a bounded episodic memory; no weight update; 91% pass@1 on HumanEval with GPT-4 | Findings are typed codes, not generated text; acceptance is not decided by an LLM evaluator; budgets and `Exhausted` replace "retry until pass or max trials" | Self-written unit tests are a weak verifier we exclude from acceptance; if those tests are the practical verifier, the loop is Reflexion with stricter bookkeeping |
| ORACLE, Raj, Kundu et al. (arXiv:2607.22465, posted 2026-07-24; v4 2026-10-01) | Training-free feedback loop over a model-selection policy for agentic programs: classify the task type, assign a task-appropriate verifier, update a LinUCB bandit from the verifier's verdict (delayed, off the critical path), `r = (1-α)·a − α·cost`; SWE-bench Verified, τ²-bench, Terminal-Bench 2.0; up to 7 points over single-verifier feedback at matched cost; 1.8x program throughput from a KV-aware scheduler | **Nearest prior art for "verifier feedback guides routing".** It routes whole models per program from a scalar verdict after completion; it does not classify within a request by finding code, repair named slots, gate a workspace or restructure the expert set | The sub-claim "verifier outcomes guide routing" is **not novel** and must not be claimed as such. What remains is the typed-finding class function, in-request repair routing, the workspace and L3; if a bandit on scalar verdicts matches finding routing, the typed interface is not needed |
| "Verifier-guided explainable reasoning framework with gold-anchored QLoRA, task-aware MoE and group-relative RLVR" (arXiv:2609.05221) | A task-level router chooses between *external* symbolic verification experts (FOL/Z3 logic, physics solver); verifier feedback supports revision and RLVR on Qwen2.5-3B; 438 held-out examples; hybrid accuracy about 56%; RLVR changed answer accuracy by -0.68 points | The "expert" there is a verifier, chosen by a router; here the experts are generators and repairers and the verifier is the closed boundary | Small single-run study (the authors list repeated runs as future work); relevance is that it uses the same words for a different mechanism, which is a naming risk for our claims |
| "Theater of Mind" Global Workspace Agents, Shang (arXiv:2604.08206) | A swarm of LLM agents (attention, generator, critic, meta-arbitrator) with a central broadcast hub, a "cognitive tick", entropy-driven temperature, first-person prompting for "subjective agency"; I found no quantitative evaluation in the text I read | A deterministic, bounded, verifier-gated contract versus an LLM-arbitrated blackboard; we exclude the first-person-agency framing by principle (38) | Their architecture needs no verifier; where tasks have none, an LLM-arbitrated workspace is the only available form |

### 4.4 Global workspace and self-models

| Work | What it does (verified) | What VCW-MoE adds or does differently | Where novelty could fail |
|---|---|---|---|
| Coordination among neural modules through a shared global workspace, Goyal et al. (ICLR 2022, arXiv:2103.01197) | Specialists compete (soft or top-k attention) to write into a few memory slots, whose content is broadcast to all specialists by attention; replaces pairwise attention, complexity linear in the number of specialists; evaluated on equilateral-triangle detection, CATER, Sort-of-CLEVR, MultiMNIST generation and WikiText-103 with Transformers, RIMs and TIMs; the CATER gain is small (71.07 against 70.83 top-1) | A *deterministic contract* rather than a learned mechanism: write competition is gated by the verifier first, then ordered by a bid, a seeded hash and an id; broadcast is a snapshot into bounded mailboxes; refusals with finding codes are broadcast | Their benefits are measured for a learned mechanism on small benchmarks; the deterministic contract can be correct and still useless. C001 does not test the contract (it tests a learned analogue); only C006 and the ablations in X do |
| Verbalizable representations form a global workspace in language models, Gurnee et al. (arXiv:2607.15495) | Functional workspace-like properties in an LLM; takes no position on consciousness (as summarised and checked in 38) | Cited for the functional framing only. Not re-read for this document | Interpretability evidence about a dense model does not transfer to an engineered Pod workspace |
| Butlin et al. indicator properties (arXiv:2308.08708) | Theory-derived indicator properties (see 38) | Properties are tested one at a time as mechanisms | Not an architectural claim |

### 4.5 Sources not read, and sources known only second-hand

* **Not read in full:** RouteAudit (arXiv:2610.02808, budgeted multi-verifier routing), CoVeR
  (arXiv:2609.26086, coverage-based routing of verifier calls in agentic retrieval), "A persistent
  state for auditable MoE routing" (arXiv:2609.34634), "Subject-typed claim licensing for learned
  routing" (arXiv:2609.33505): found by search only, titles and abstract fragments, **no claim is
  made about their content**. They are the first candidates to read before any novelty claim about
  verifier-aware routing is published.
* **Known only through the related-work text of papers I read:** ACRouter and SWE-Router (through
  ORACLE), FrugalGPT and AutoMix (through RouteLLM), PHATGOOSE and MOLE (through Self-MoE).
* **Not searched:** the deployment-side literature on verifier latency and speculative
  verification; the literature on verifier gaming and reward hacking beyond the Cobbe et al.
  observation above.

### 4.6 What is claimed, precisely

The claim registered for this type (proposed text for `research/novelty/CLAIMS.md` is in the
package report) is conditional and joint: **the combination** of (a) a deterministic accept
predicate that no learned component can influence, (b) routing at L2 by the finding class and a
learned choice inside the admissible set, with propensities logged, (c) a bounded workspace that
carries only verified content, and (d) a structure policy driven by the unroutable-code statistics,
solves more verifiable tasks per unit of compute than the baselines of F004. Each of (a) to (d) is
also tested alone, and each can fail alone. The whole type is refuted if (a) with (b) does not beat
best-of-N at the same verifier calls (R1).

## 5. Why this might not be a new MoE type

Stated before the experiments so that a null result is interpretable.

1. **It may be a pipeline, not a mixture.** At L2 the "experts" are called one after another with
   a verifier between them; the sum of weighted outputs that defines an MoE layer is absent. If
   the loop is best described as an agent loop with a typed verifier interface, the right name is
   "verifier-closed agent loop over an MoE" and the claim is about the loop, not about an MoE type.
2. **Verifier access may carry the whole effect.** Large Language Monkeys shows that with an
   automatic verifier, independent sampling already converts coverage into success. The loop must
   beat that with the same calls (F004).
3. **The finding interface may be redundant.** ORACLE-style bandits on scalar verdicts, and
   Router-R1-style learned routers, may match finding routing without a typed interface (F005's
   generic-repair baseline and the shuffled-map control are built to detect this).
4. **The workspace may be decoration.** The write competition is deterministic and takes verified
   content only; a plain list of verified results may do the same. The X ablation without the
   workspace and C006 test this; C001 does not.
5. **Self-observation may add nothing over a probe.** C002's information-matched probe is the
   comparison that matters; a self-model that equals a probe is a probe.
6. **L3 may be a heuristic in disguise.** The structure policy is threshold rules over counters.
   M013's heuristic baseline and a static expert set are the controls (F006).

## 6. Falsification plan

All margins, tolerances and seed lists are pinned in each experiment's preregistration; they are
not set here. Seeds are the repository's `17, 29, 43, 71, 101` for 5-seed experiments and the
declared 10 for C001. Outcomes use the repository's three-way rule: **supported**, **refuted**
(with power: the upper bound lies below the margin and the guards hold), **not shown**; invalid
runs change nothing. Pilots use seeds ≥ 1001 and are never confirmatory.

### 6.1 Experiments and decisive comparisons

| Id | Hardware | Claim tested | Arms | Decisive comparison | Hard counters (any > 0 falsifies the implementation) |
|---|---|---|---|---|---|
| X (working name) | CPU, 5 seeds, stub experts with declared per-finding-class skills | The whole loop is better than the alternatives *when the stub makes it possible*, and each component's marginal effect | Full loop; learned-gate routing without a verifier (accept first output); best-of-N with the same verifier at matched verifier calls; loop without workspace; loop with a generic router (no finding routing); loop with a shuffled finding-to-Pod map; loop without self-model hints; a sensitivity control (crippled loop) | Full loop against best-of-N at matched verifier calls on verified success per call; finding routing against the shuffled map | false accepts, unverified returns, double verifications, budget violations, collateral edits, workspace-to-ledger edges, hint-to-gate edges |
| C001 | CPU, 10 seeds | A learned-attention workspace is a better inductive bias than parameter- and bandwidth-matched pairwise attention at more modules than trained on | See the C001 manifest | Workspace minus the strongest comparator, lower one-sided 95% bound at the margin | Parameter mismatch, communication leaks, bottleneck violation, nondeterminism |
| C002 stage A | CPU toy | A self-model predicts the system's own failures better than an information-matched probe | Self-model; info-matched probe; input-output probe; softmax confidence; constant | Brier on the shifted split against the info-matched probe and the confidence | Any write path from the self-model to a gate, verifier or promotion |
| C003 stage A | CPU toy | A schema used for control beats the schema-free router and a describe-only schema | Router; router plus schema (control); schema trained but unused; heuristic balancer | Control arm against both the schema-free router and the describe-only arm | Schema authority over promotion or revocation |
| C004 | CPU toy | Recurrence with carried state extrapolates better than a depth-matched feed-forward stack | See manifest | Length extrapolation against both comparators | Parameter or FLOP mismatch |
| F004 | GPU | The closed loop beats a standard MoE, a self-refining single generator, best-of-N with the same verifier, and the loop with a generic router, at matched FLOPs and verifier calls | See manifest | Closed loop against **best-of-N at matched verifier calls and FLOPs** | false accepts, unverified returns, budget violations; accounting mismatch invalidates |
| F005 | GPU | One repair expert per finding class beats one generic repair expert and regenerate-only | See manifest | True map against generic expert, regenerate-only and the shuffled-map control | collateral edits, false accepts |
| F006 | GPU | Verifier-statistics-driven lifecycle beats a static set and the M013 heuristic at equal compute | See manifest | Cumulative verified success per compute against both | gate bypasses, forgetting above the gate bound |
| C002 stage B, C003 stage B, C006 | GPU | The observers and the workspace contract on OLMoE-class Pods | See manifests | Info-matched probe; schema-free router; plain routing and all-to-all Pod bus | Authority edges; workspace content reaching a gate |

CPU results cannot support the claim about models. A CPU result with a stub is a mechanism check
whose gains are consequences of the declared stub skill (the F001 limit, repeated in 37 section 11).

### 6.2 Conditions under which "VCW-MoE is a useful new MoE type" is refuted

The statement is **refuted** (not merely not shown) if, with the guards holding and the sensitivity
control resolving, any of the following holds with an upper confidence bound below the preregistered
margin:

* **R1.** In F004 the closed loop does not beat best-of-N from the same MoE with the same verifier
  at matched verifier calls and FLOPs. Consequence: verifier access, not the architecture, carries
  the effect; the type is refuted, the verifier-use result stands.
* **R2.** In F005, or in the X ablations, the shuffled finding-to-Pod map is not worse than the
  true map. Consequence: routing by findings carries no information; the type is reduced to a
  generate-verify-repair loop with a generic repair expert.
* **R3.** Any hard counter above 0 on any seed of X, F004 or F005 (a false accept, an unverified
  return, a budget violation, a collateral edit, an authority edge). Consequence: the closure claim
  is refuted *for that implementation* until fixed and re-run under a successor experiment.
* **R4.** Removing the workspace in X and in C006 changes verified success by less than the margin.
  Consequence: the workspace is dropped from the definition; the type is a verifier-routed Pod-MoE.
* **R5.** The self-model does not beat the information-matched probe (C002) and hints do not change
  X or F004 outcomes. Consequence: self-observation is dropped from the definition.
* **R6.** The structure policy does not beat the static set and the M013 heuristic (F006).
  Consequence: gated self-restructuring is dropped from the definition (the gates remain valuable
  as safety, R011).

The claim "VCW-MoE is a useful new MoE type" is **supported** only if R1 and R2 are survived on
model-backed experiments (F004, F005). Surviving them only in X is "not shown" for models. R4 to R6
are independent: they remove components from the definition without refuting the rest.

### 6.3 What is already known to hold or fail

* Holds (mechanism checks, stub): the closed loop keeps its safety counters at 0 under noisy,
  lying and forged reports (F001); an L3 policy does not bypass the gates (R011).
* Unknown: every statement about benefit.
* Holds on simulated experts: the whole loop beats a trained gate and best-of-N at matched verifier calls in a world with declared heterogeneous skill, and ties where skill is equal (F007, COMPLETED).
* Not shown: C001 (NOT_SHOWN, section 3.4): a learned workspace is not shown to beat dense
  pairwise attention in its toy setting; this is neither support nor refutation. F007's L3 secondary is also NOT_SHOWN.
* Falsified in toy form: C007 (a self-model does not beat an information-matched probe or the model's own confidence) and C008 (the schema-controlled router does not beat a router that reads the history directly). Neither touches the GPU stage B.
* Not run: every GPU experiment. No negative result of a refutation has been recorded; negative
  results will be reported as first-class artifacts.

## 7. Open questions and risks

1. **Latency of the verifier in the loop.** Each round costs a verifier call and a sequential
   model call; MoA and Router-R1 both name latency as a limitation. ORACLE removes verification from
   the dispatch critical path by delayed feedback, which is possible for routing statistics but not
   for the in-request accept decision. The budget `(V_max, T_max)` bounds the cost but not the
   tail; the experiments report wall-clock next to verified success.
2. **Verifier coverage.** A domain with only weak verifiers (`LatentAgreement`) cannot clear a
   `Deterministic` required level and returns `Exhausted` by design; a high `Exhausted` rate is a
   cost, not an error. Unit tests give false positives and negatives (Large Language Monkeys,
   section 4.3). The type applies only where a deterministic verifier of sufficient depth exists.
3. **Gaming of finding statistics.** L3 Birth is triggered by the count of finding codes no
   expert clears. A generator that emits junk codes, a verifier whose code taxonomy drifts, or a
   deliberately noisy proposer can inflate those counts. The policy has per-window caps, a
   cool-down, a persistence rule and a last-clearer guard, but these are tuned to a taxonomy that
   does not exist yet; and an RLVR-trained router can learn to produce findings that route to
   easy repairers. Cobbe et al.'s degradation beyond 400 completions is a documented instance of
   optimising against a verifier.
4. **Credit assignment.** A pass after two repairs and one regeneration: which Pod earns the
   reward? The per-decision records give an unbiased estimator for each decision, but the
   trajectory's joint propensity is a product, with variance growing with rounds. Step-level typed
   findings help (Lightman et al.) only if the code taxonomy is fine enough.
5. **Scaling of the workspace.** Slots `≤ 64`, subscribers `≤ 64`, pending `≤ 1024` are contract
   limits. Per-cycle cost is the number of pending proposals times a verifier call; broadcast is
   a snapshot per subscriber. Beyond tens of Pods a hierarchy is needed. The linear-in-specialists
   argument of Goyal et al. concerns a learned attention mechanism, not this contract.
6. **Non-stationarity.** The registry generation changes while logged cohorts mature; evidence
   uses one pinned reward and verifier epoch. The cost of epoch rotation to a live system is not
   measured.
7. **Interaction of L1 balancing and RLVR.** Routing replay (R3, in 37) and the aux-free bias are
   known to interact with RL on MoE; whether the L2 loop changes that is untested.
8. **Cost accounting.** F004 matches verifier calls and FLOPs; real deployments also pay
   verifier latency, memory for Pods and registry maintenance. A win at matched calls and FLOPs may
   lose on wall-clock.
9. **Evidence class creep.** The nearest risk to this document is overclaiming a CPU stub result
   as a statement about OLMoE. Every report keeps the evidence class next to the number.
10. **Welfare.** The workspace, the self-model and goal-directed L3 proposals together are the
    configuration `docs/MODEL_WELFARE.md` was written for. The CPU experiments are
    exempt by definition; an integrated GPU system is not.

## 8. References

All identifiers are arXiv unless stated. Verified by reading the primary text unless marked.

* Switch Transformers (2101.03961); GShard (2006.16668); DeepSeekMoE (2401.06066);
  Auxiliary-loss-free load balancing (2408.15664); DeepSeek-V3 (2412.19437, read for the
  balancing section only).
* PERFT (2411.08212); PERFT-R (2508.02587); Branch-Train-MiX (2403.07816); Self-MoE (2406.12034).
* RouteLLM (2406.18665); Router-R1 (2506.09033); Mixture-of-Agents (2406.04692).
* Cobbe et al., Training verifiers (2110.14168); Lightman et al., Let's Verify Step by Step
  (2305.20050); Brown et al., Large Language Monkeys (2407.21787); Self-Refine (2303.17651);
  Reflexion (2303.11366).
* ORACLE (2607.22465); verifier-guided explainable reasoning with task-aware MoE (2609.05221);
  Global Workspace Agents (2604.08206).
* Goyal et al., shared global workspace (2103.01197); Gurnee et al. (2607.15495, via 38);
  Butlin et al. (2308.08708, via 38).
* Unread (section 4.5): 2610.02808, 2609.26086, 2609.34634, 2609.33505.
