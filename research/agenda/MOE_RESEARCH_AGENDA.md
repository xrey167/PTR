# MoE research agenda: from scaffolding to three falsifiable theses

Status: **proposal**, written 2026-10-13 on branch `claude/ws-research`. Nothing in this
document is evidence. It registers no experiment and changes no claim; every edit to
`research/novelty/CLAIMS.md`, `research/novelty/PRIOR_ART_MATRIX.md`, `experiments/registry.toml`
or an ADR is *proposed* here (section 7) and is the head's or owner's to make.

Reading order: section 1 (where the evidence stands), section 2 (what the 2024–2026 literature
already covers), section 3 (the three theses), section 4 (work packages), section 5 (risks),
section 6 (what is deliberately dropped), section 7 (proposed edits).

Conventions used below. *Evidence class* is one of: **mechanism check** (a contract or harness
behaves as specified on the CPU reference path or with simulated experts; says nothing about a
model), **pipeline proof** (a pipeline runs end to end and produces finite numbers; no
comparison), **research result** (a preregistered comparison with baseline and decision rule;
the scope is stated next to it), **NO-GO** (the study ran and its own decision rule says no
claim may be made, or the claim failed), and **none** (planned or prepared, no outcome). Literature
status is one of **done** (a published paper does the thing, at least in some setting),
**partially done** (a published paper does a close variant; the open part is named), **open**
(not found in the searches listed in section 2.3; that is not proof of absence).

---

## 1. Inventory: MoE-relevant claims, mechanisms and experiments

### 1.1 Claims (from `research/novelty/CLAIMS.md`)

| Claim | Content (short) | Tested by | Evidence class today |
|---|---|---|---|
| 1 | Raw + typed latent workspace (typed slots) | M001 series, A0-ablations-v2 | **NO-GO** (M001-v4 `INCONCLUSIVE/NO-GO`, `experiments/model/M001-v4-semantic-slots/DECISION.toml`; M001-v6 `INCONCLUSIVE`, `experiments/model/M001-v6-semantic-slots/results/run.json`) |
| 2 | Epistemic slot types + typed attention | M002 series, M005 | **NO-GO** (M002-v4 `INCONCLUSIVE/NO-GO`, typed attention not isolated; M002-v11 `INCONCLUSIVE`; A0-ablations-v2 necessity falsified in the toy, `research/falsification/A0-ablations-v2/`) |
| 3 | Operator routing replaces token-heavy reasoning | M004, M003 | **NO-GO** (M003-v4 `INCONCLUSIVE`, label `NO-GO-CALIBRATION`, `experiments/model/M003-v4-latent-recurrence/results/run.json`); M004 none |
| 11 | Gated expert lifecycle (Birth, Merge, Split, Freeze, Prune) forgets less than a static MoE at equal compute | M013, R011 | R011 **mechanism check** (no-bypass half only); M013 **none** |
| 12 | Reactivating a retired expert beats training a new one | M013 | **none** |
| 13 | Routing and interference context improves L3 structural proposals | M013, R011, R013 | R011/R013 **mechanism check** (no self-promotion only); improvement **none** |
| 14 | Never-stale prefix/KV reuse under adapter, expert and input revocation, incl. deterministic MoE reuse | R005 (stale), R006, R007, R009, R010, R012, E006, E007 | **mechanism check** on the CPU reference path (tiny seeded models); E006 (engines) **none** |
| 15 | Learned shared workspace beats pairwise attention (C001) | C001 | **research result: NOT_SHOWN** (CPU toy, `experiments/consciousness/C001-global-workspace/results/run.json`) |
| 16 | Self-model Pod predicts the MoE's own errors | C007 (stage A), C002 | **research result: FALSIFIED** in the CPU toy (C007); C002 **none** |
| 17 | Attention schema of the router improves routing | C008 (stage A), C003 | **research result: FALSIFIED** in the CPU toy (C008: the gain comes from the history, not the schema form); C003 **none** |
| 21 | Combination: verifier-closed acceptance + finding-routed Pods + workspace + self-observation + gated L3 beats learned-gate MoE and best-of-N | F004; F001 | F001 **mechanism check** (seeded stub); F004 **none** |
| 22 | Repair experts routed by finding class beat one generic repairer and regenerate-only | F005 | **none** |
| 23 | Verifier-statistics-driven Birth/Merge beat a static set | F006 | **none** |
| 24 | VCW-MoE as a useful MoE type | F004–F006; F007 | F007 **mechanism check** (simulated experts; gain follows from declared skill tables, `experiments/feedback/F007-vcw-loop/results/run.json`) |
| 25 | Whole loop with stub experts beats best-of-N and a trained gate | F007 | **mechanism check** (COMPLETED; L3 secondary NOT_SHOWN, power about 0.2–0.31) |

Claims 4–10 and 18–20 are not MoE claims and are left out, except where a mechanism below
depends on them.

### 1.2 Mechanisms in code (what exists, what it has been shown to do)

| Mechanism | Location | What is shown | Class |
|---|---|---|---|
| Verifier finding table (hard/soft codes, repair roles: format, value, schema, code, resource, fact) | `crates/ptr-verifier/src/codes.rs` (`CODE_TABLE`) | unit tests | mechanism check |
| Deterministic verifiers (arithmetic, JSON-schema subset, tiny sandboxed code language, facts) | `crates/ptr-verifier/src/{arithmetic,json_schema,code_test,fact}.rs` | unit tests, `crates/ptr-runtime/tests/vcw_real_verifiers.rs` | mechanism check |
| Finding router (class decided by the report; learned choice only inside the admissible set; quantised propensities; unroutable counters) | `crates/ptr-router/src/findings.rs` | `crates/ptr-router/tests/findings.rs` | mechanism check |
| Categorical OPE (IPS, SNIPS, DR) with positivity refusal and a promotion gate | `crates/ptr-router/src/ope.rs`, `learning.rs`, `policy_book.rs` | `src/ope_tests.rs`, `tests/learning.rs` | mechanism check (no model log) |
| Self-model, attention schema (advisory) | `crates/ptr-router/src/self_model.rs`, `attention_schema.rs` | tests; C007/C008 falsified the toy forms | mechanism check / NO-GO for the toy claims |
| Bounded verifier-gated workspace | `crates/ptr-pods/src/workspace/` | `crates/ptr-pods/tests/workspace.rs`; F007 secondary "supported" as an implementation check | mechanism check |
| Expert lifecycle types, two-stage proposal identity, state table | `crates/ptr-pods/src/expert.rs` | `crates/ptr-pods/tests/expert.rs` | mechanism check |
| Expert admission, dependency index, compound `InputInvalidated`, authority epochs, Ed25519 attestation | `crates/ptr-runtime/src/expert.rs`, `expert/dependency.rs`, `expert/authority.rs`, `expert/attestation.rs` | R011, R013 (COMPLETED, reference path) | mechanism check |
| Closed loop with budgets and `Exhausted`; L3 structure step; simulated and CPU-model experts | `crates/ptr-runtime/src/vcw/` (`closed_loop.rs`, `l3.rs`, `experts/`), `crates/ptr-runtime/src/vcw_host/` | `tests/vcw_loop.rs`, F007, `make vcw-demo` | mechanism check; demo is a pipeline proof |
| ForgettingGate, subspace interference, TIES merge, AIMER, replay | `crates/ptr-lineage/src/{forgetting,subspace,merge,aimer,replay}.rs` | unit tests; no experiment | mechanism check |
| Per-layer KV chain key; tiny dropless top-k MoE with expert masking | `crates/ptr-pods/src/kv_chain.rs`, `crates/ptr-pods/src/qwen_moe.rs` | R009, R010 | mechanism check |
| Python binding (verifiers, finding router, OPE, workspace, loop simulation, `ties_merge`, `aimer_score`, ForgettingGate) | `bindings/ptr-py/src/` | `make py-bindings` tests; not built by CI | pipeline proof |
| `trl-moe` backend: OLMoE build, LoRA, SFT, GRPO-family RL (GSPO, SAPO), verifier-derived reward, routing metrics (churn, entropy, load CV, dead experts), PERFT-R and MixLoRA arms, M011 entrypoint | `training/src/ptr_training/moe/` (`routed_peft.py`, `m011.py`, `rl.py`, `reward.py`, `routing_metrics.py`, `gates.py`) | WP8 CPU smoke on a 0.24M random-init OLMoE, one seed (`training/README.md:34`) | pipeline proof |

### 1.3 Experiments that touch MoE (from `experiments/registry.toml` and results)

| Id | Question | Status / outcome | Class |
|---|---|---|---|
| A0 studies M001-v4, M002-v4 | Typed slots / typed attention in the A0 toy | `INCONCLUSIVE/NO-GO` (accuracy up, ECE15 and NLL worse; M002-v4 does not isolate the factor) | NO-GO |
| M001-v6, M002-v11, M003-v4 | Successor A0 studies | `INCONCLUSIVE` (M003-v4 `NO-GO-CALIBRATION`); results marked stale by repository provenance (`results/STALE.toml`) | NO-GO |
| A0-ablations-v2 | Toy ablations | mixed, toy only | research result (toy) |
| M010 | MoE RL stability (GRPO/GSPO/SAPO, routing replay) | planned | none |
| M011 | PERFT-R vs single LoRA vs MixLoRA on OLMoE | planned; entrypoint `training/src/ptr_training/moe/m011.py` exists | none |
| M012 | Aux-loss-free bias vs aux loss vs φ-balancing | planned | none |
| M013 | Gated lifecycle vs static set; Reactivate vs new expert; context ablation | planned | none |
| M014 | AIMER prune + recovery | planned | none |
| M015 | RLVR vs distillation pass@k | planned; teacher does not fit (docs/GPU_HANDOVER.md step 9) | none |
| M016 | Approximate chunk reuse on a small dense model | planned | none |
| C001 | Learned workspace vs pairwise attention | NOT_SHOWN (20 seeds) | research result (toy) |
| C002, C003 (stage B), C005, C006 | Self-model, schema, self-report, Pod-level workspace | planned | none |
| C004 | Recurrence extrapolation (counter toy) | COMPLETED, close to a known answer | research result (toy) |
| C007, C008 | Stage-A self-model and schema toys | FALSIFIED | research result (toy, negative) |
| F001 | Generate-verify-repair loop, seeded stub | COMPLETED | mechanism check |
| F004, F005, F006 | Closed loop vs MoE and best-of-N; finding-routed repair; verifier-driven lifecycle | planned (GPU) | none |
| F007 | Whole loop, simulated experts | COMPLETED (+7.20 over trained gate, +6.35 over best-of-N at matched calls; null world ties) | mechanism check |
| R005 | Exact prefix reuse | COMPLETED, stale | mechanism check (not current) |
| R006, R007, R009, R010, R012 | Positional reuse, span forget, LoRA KV, MoE expert KV revocation, approximate taint | COMPLETED (CPU reference) | mechanism check |
| R011, R013 | L3 cannot bypass gates; deferred authority mechanisms | COMPLETED (narrowed) | mechanism check |
| R008, R014, E006, E008 | Lossy KV tier, remaining authority mechanisms, engine prefix reuse, split-Pod extensions | planned | none |
| E007 | Split-Pod activation transport | COMPLETED (plumbing, not math) | mechanism check |
| WP8 | `trl-moe` CPU smoke | ran once | pipeline proof |

### 1.4 Summary of the inventory

There is **no research result about any mixture-of-experts model** in the repository. Every
completed MoE-adjacent item is a mechanism check (governance, revocation, the closed loop with
stubs) or a pipeline proof. The model-side studies that did run (A0) are NO-GO. Every GPU
experiment is planned, and the GPU bring-up (docs/GPU_HANDOVER.md steps 1–3) has not run. The
substrate is unusually strong for a research project at this stage (authority, revocation,
propensity logging, deterministic verifiers); the open question is which *scientific* claim that
substrate makes testable that the field does not already have.

---

## 2. Literature check (2024–2026)

All rows marked "read" were checked against the paper text on alphaXiv on 2026-10-13 for this
document. Rows marked "repo" were read by an earlier PTR document (doc 37, doc 39 or the prior-art
matrix) and were not re-read here. "Cited only" means known through another paper's related work.

### 2.1 Per candidate idea

| Idea | Status | Closest prior work (arXiv id) | What is open |
|---|---|---|---|
| **Verifier-conditioned routing** (route the next step by the verifier's outcome) | **done** | ORACLE 2607.22465 (repo, re-found): LinUCB model selection from verifier verdicts; RouteAudit 2610.02808 (read): budgeted routing *between verifiers*, with a contract for matched route contrasts; CoVeR 2609.26086 (read): gating *verifier calls* in agentic retrieval; Router-R1 2506.09033 (repo) | Nothing at the level of "verdicts guide routing". RouteAudit's matched verifier-set control is a methodological requirement PTR should adopt (section 3). |
| **Finding-driven repair routing** (route a failed candidate by failure class) | **partially done** | CodeRescue 2607.19338 (read): a fine-tuned router reads verdict class + stderr and chooses among reflect / replan / escalate; beats fixed actions and a binary cascade; CRC budget control. Self-Refine 2303.17651, Reflexion 2303.11366, Large Language Monkeys 2407.21787 (repo) | Routing to *specialised repair experts per failure class*, trained on finding-conditioned transitions, against (a) one repairer that sees the same findings in context and (b) best-of-N at matched verifier calls and FLOPs. CodeRescue's actions are generic strategies, not specialised experts, and it does not compare against resampling at matched verifier calls. |
| **Governed expert lifecycle** (birth / merge / split / prune / reactivate with gates) | **partially done** | DynMoE 2405.14297 (read): adds an expert when tokens activate none, removes unused experts; BAR 2604.18473 (read): add or upgrade domain experts independently, router retrained; FlexOlmo 2507.07024 (read): experts added or removed at inference; Lifelong-MoE 2305.12281, MC-SMoE 2310.01334, sparse upcycling 2212.05055, expert upcycling 2604.19835, BTX 2403.07816 (repo); AIMER 2603.18492 (repo); nanoFMT 2604.00812 (read): market-driven birth/death with dormant experts that "reactivate" 9–11x faster in a 2-layer toy, single runs | Gates as *pass/fail predicates with ledger commit* are an engineering contribution, not a scientific one. **Reactivate is no longer unpublished** (nanoFMT, toy, no seeds). **Birth from an unroutable signal has a token-level precedent** (DynMoE). The open scientific question is narrower: what lifecycle operations do to guarantees (section 3, thesis C). |
| **Experts as revocable units with provenance** | **partially done** | FlexOlmo 2507.07024 (read): each expert trained on one data owner's set with frozen shared layers; removing the expert is stated to remove that data's influence; extraction test 0.7% with the expert vs 1.6% for the dense expert. Policy-Masked Private Experts 2608.06690 (read): trusted authorization masks a private expert pool before top-k; 0 unauthorized private executions; policy-tagged KV caches. BAR 2604.18473 (read): independently trained experts, **diverged shared layers averaged across experts** at merge. Gradient routing 2410.04332 and modular information-flow control (Tiwari et al.) cited only (via 2608.06690). Exact unlearning by sharding: SISA 1912.03817 (repo) | Whether the *removal* guarantee survives lifecycle operations that mix parameters across experts (shared-layer averaging as in BAR, router fine-tuning on mixed data, merging, upcycling a new expert from an old one). Not found measured anywhere. |
| **Routing on typed or semantic slots** | **open** (not found) | Nearest: SA-MoE persistent routing state 2609.34634 (read; a learned router-only state with an exact write ledger, no types); attention-aware routing 2609.20974 (title and abstract only) | Open in the literature, but PTR's own A0 evidence is NO-GO (section 1). Not proposed as a thesis now (section 6). |
| **KV reuse across experts / adapters** | **done** (for adapters and engine tiers), **partially done** (for expert revocation) | KV provenance contract 2609.38706 (read): adapter identity, weight version, precision, KV format and tenant salt bound into shared-tier keys in vLLM and SGLang; CacheReforge 2609.30884 (read): per-layer adapter-version anchors, restart from the first affected layer, bounded recomputation; LRAgent 2602.01053, ForkKV 2604.06370 (titles and abstracts); Activated LoRA 2504.12397 (repo); cross-adapter aLoRA reuse 2512.17910 cited only | PTR's per-layer chain key invalidating only layers above a revoked expert (R010) is a small step beyond CacheReforge's adapter anchors. Claim 14's engine half (E006) overlaps 2609.38706 heavily. Not a thesis. |
| **Counterfactual evaluation of MoE routers** (related idea surfaced by the search) | **partially done** | Counterfactual routing analysis 2605.07260 (read): re-executes 32 sampled equal-compute routes per token, finds the trained top-k router uninformative on low-confidence tokens across four open MoEs including OLMoE; a router-only update shifts pass@K. Cross-entropy-guided routing 2609.37751 (read): supervises routing with token loss and states that this "supplies no counterfactual expert-error target". Slate off-policy evaluation 1605.04812 (read): pseudoinverse estimator under an additive-reward assumption | **Estimating a candidate router's value from logged stochastic routing, without re-execution** (off-policy evaluation at the token level) is not found. This becomes thesis B. |

### 2.2 Corrections this check forces on existing PTR text

1. `PRIOR_ART_MATRIX.md`, Pod-MoE section, says "Reactivate is unpublished" and doc 37 §7 says
   "'reactivate' has no literature". nanoFMT (2604.00812) reports dormant-expert recovery in a toy.
   The statement must be narrowed (section 7).
2. The four "found but not read" papers of doc 39 §4.5 are now read: none of them routes *experts*
   by verifier findings. RouteAudit and SCOPE-Routing (2609.33505, read: subject-typed claim
   licensing) are methodological; both argue that a composite system's gain must not be credited to
   a component without a matched control. That is the same discipline as F004's best-of-N rule.
3. CodeRescue (2607.19338) narrows claim 22: learned post-failure routing over recovery actions is
   published. What remains is the expert-specialisation axis.
4. DynMoE (2405.14297) is the token-level precedent for "Birth when something is unroutable"
   (claims 23, 24(d)).
5. The engine half of claim 14 (E006) overlaps the KV provenance contract (2609.38706).

### 2.3 Search coverage and limits

Searches run (alphaXiv discovery, 2026-10-13): verifier-conditioned / repair routing; expert
lifecycle in continual learning; unlearning by expert removal (two phrasings); router training
from verifier outcomes; cross-adapter KV reuse; expert reactivation; failure-driven expert
expansion; off-policy evaluation of routers; stochastic routing for counterfactual signals;
error-type-specialised program repair; exact unlearning by sharding or merging. Each returned 8–19
candidates; the ones named above were opened. **Not verified:** Kim and Kang (2025),
exploration-driven RL for expert routing (cited by 2605.07260; not opened); "Certain head,
uncertain tail" 2602.02443 (cited only); 2606.10338 (routing-aware expert calibration for MoE
unlearning; PDF fetch failed, abstract only: it targets unlearning inside an MoE but its method was
not read). Absence of a paper in these searches is not proof that the idea is unpublished; every
thesis below repeats the search before preregistration (work package X1).

---

## 3. Three theses

Selection rule: (a) not found published, (b) decidable on a CPU harness first and then on two
RTX 3090 cards (24 GB each, PCIe, `hardware/dual-rtx3090.toml`), (c) if it holds, it changes a
design decision people make when building or evaluating MoEs. Each thesis reuses existing PTR
substrate, so the governance work already done becomes an instrument rather than the claim.

Common rules for all three (from `AGENTS.md` §3): pilots on seeds ≥ 1001 only; confirmatory seeds
`17, 29, 43, 71, 101`, each run once; thresholds below are proposals to be frozen at
preregistration and never loosened afterwards; decision order validity → falsifier → controls;
exit code 1 is a product finding, 2 a harness failure; COMPLETED only with the whole mutation check
killed. On the desktop, frozen GPU seeds run one at a time on card 0 unless the manifest declares
`PTR_CUDA_DEVICE` (OD-3); card 1 is for pilots and tuning. "FLOPs" means the analytic forward
(and, where trained, backward) FLOPs of the active parameters, counted by the harness, as in M011.

### Thesis A — Failure-mode experts

**Hypothesis.** For repair after a deterministic verifier rejects a candidate, experts
*specialised by failure class* and *routed by the verifier's finding code* repair more tasks per
verifier call than (i) independent resampling with the same verifier at matched verifier calls and
FLOPs, (ii) a single repair expert with the same total trainable parameters that receives the same
finding codes in its input, and (iii) the same parameter budget split into experts by *task domain*.
If true, the axis along which repair experts should be split is the failure mode, not the input
domain, and routing by failure carries information that conditioning on it does not.

**Why this is the decisive form of claims 21, 22 and 24.** It isolates the one ingredient of
VCW-MoE whose value is not already published (section 2.1) and tests it against the two baselines
that would make it redundant: resampling (Large Language Monkeys) and conditioning (one model that
reads the finding).

**Strongest baseline.** A1, the *conditioned generic repairer*: one LoRA on the frozen base with
rank equal to the sum of the specialists' ranks (trainable parameters within 1%), trained on the
union of all repair transitions, whose prompt contains the full verifier report (codes, pointers,
messages). It has every bit of information the routed system has. Second strongest: A0, best-of-N
from the base generator with the same verifier, N set so that verifier calls equal the routed
loop's budget and generation FLOPs are within 5%.

**Arms.**

| Arm | Description |
|---|---|
| A0 | Best-of-N resampling, same generator and verifier, matched verifier calls and generation FLOPs |
| A1 | Conditioned generic repairer (strongest baseline) |
| A2 | Domain experts: the same parameter budget split by task family (arithmetic, JSON, code), routed by task family, report in the prompt |
| A3 | **Failure-mode experts**: one LoRA per repair role of `CODE_TABLE` used by the task mix (format, value, schema, code, resource; the fact role is out of scope), routed by `FindingRouter` |
| A4 | A3 with a fixed random permutation of the code-to-expert map (R2 control) |
| A5 | Scalar-verdict bandit over A3's experts (LinUCB on pass/fail only; ORACLE-style) |
| A6 | Learned recovery router over A3's experts that reads the raw report text but not the code table (CodeRescue-style) |
| S+ | Positive sensitivity control: the F007 simulated world with declared heterogeneous skill; must show A3 ahead |
| S0 | Null sensitivity control: the verifier's codes deterministically relabelled per task so they carry no information about the defect; A3 must tie A1 |

**Ablations.** Remove the code from A3's prompt (routing only, no conditioning); collapse roles
(format + schema merged); vary the specialist rank at fixed total parameters (2 levels); repair
budget 2 vs 4 verifier calls.

**Metrics and pinned thresholds (proposed).**

| Metric | Rule |
|---|---|
| Primary: verified success within a budget of 4 verifier calls per task (accept predicate as in `Candidate::passes`: status Pass, no hard finding, weakest level at or above the required one) | A3 − max(A0, A1, A2): one-sided 95% lower bound ≥ +3.0 points in every confirmatory seed (1,000 held-out tasks per seed; pilot checks power ≥ 0.8 at a true effect of +5) |
| R2 control | A3 − A4 lower bound ≥ +3.0 points in every seed |
| Accounting validity (INVALID if violated) | generation FLOPs of every arm within ±5% of A3; verifier calls per task exactly the budget; trainable parameters A1–A3 within ±1%; identical task ids and order across arms (digest recorded) |
| Hard counters (implementation refuted if > 0) | false accepts, unverified returns, budget violations, collateral edits (JSON: every field not named by a finding pointer byte-identical; arithmetic: n/a; code: functions not named by a finding unchanged) |
| Sensitivity | S+ shows A3 − A1 ≥ +3.0; S0 shows \|A3 − A1\| ≤ 1.5 |
| Secondary (reported, not decisive) | A3 − A5, A3 − A6, wall-clock per accepted task, repairs per verifier call |

**Kill criteria.**
- **K-A1**: A3 does not beat A1 by the margin → routing by failure class adds nothing over
  conditioning on the same findings. Thesis A is dropped; claim 22 is refuted in this setting and
  claim 24(b) loses its only non-published part.
- **K-A2**: A0 is not beaten → R1 of doc 39 §6.2 triggers: verifier access, not the architecture,
  carries the effect.
- **K-A3**: A2 ≥ A3 − 1.0 → the failure-mode axis is not better than the domain axis; the "axis"
  half of the thesis is dropped even if K-A1 passes.
- **K-A4**: A4 is not worse → finding routing carries no information (R2).
- If A6 ≥ A3 − 1.0 while K-A1 passes: the *typed* code table is unnecessary; the claim narrows to
  "routing by failure" and drops "typed".

**Compute estimate (unmeasured; to be replaced by the step-2 memory measurement of
docs/GPU_HANDOVER.md).** CPU stage: harness, S+ and S0 through the `ptr_py` loop simulation
(`bindings/ptr-py/src/vcw.rs`), a few CPU-hours. GPU stage on the pinned OLMoE base (about 1B
active / 7B total, bf16 about 12.9 GB per doc 37): repair-transition collection about 5 GPU-h; 9
LoRA trainings (A1, 3× A2, 5× A3) × 5 seeds at about 1–1.5 GPU-h each ≈ 45–70 GPU-h; evaluation
of 7 arms × 1,000 tasks × 4 calls × 5 seeds ≈ 15–25 GPU-h. Total ≈ 70–100 GPU-h, i.e. 2–3 days
on two cards with pilots on card 1.

**Reuses.** `crates/ptr-verifier` (codes, verifiers) through `ptr_py.verify_*`;
`crates/ptr-router/src/findings.rs` through `ptr_py` routing (`bindings/ptr-py/src/routing.rs`),
so the Rust router is the router under test and its `FindingRouteRecord` propensities are logged;
`training/src/ptr_training/moe/` (`model.py` LoRA builder, `m011.py` training loop and tuning
protocol, `tasks.py`, `reward.py`); F007's harness design, matched-call accounting and mutation
plan (`experiments/feedback/F007-vcw-loop/`); F005's manifest as the record to supersede.

**Minimal decisive experiment.** JSON-schema and arithmetic tasks only (deterministic
verifiers, collateral-edit check is exact), three roles (format, value, schema), arms A0, A1, A3,
A4, S0, 5 seeds. The two comparisons A3 vs A1 (matched parameters) and A3 vs A0 (matched calls
and FLOPs) decide the thesis; everything else is secondary.

### Thesis B — Off-policy-auditable routing

**Hypothesis.** A sparse MoE router can be made *off-policy evaluable* at negligible quality cost:
if each MoE layer's top-k set is perturbed by a small, exactly logged exploration (one slot
swapped with probability ε to a uniformly drawn non-selected expert, per token and layer), then a
slate pseudoinverse or doubly robust estimator computed from ordinary traffic ranks candidate
routers in the same order as running each candidate on-policy. If true, router changes (balancing
schemes, post-training router updates, pruning-induced reroutes) can be evaluated and gated from
logs, the way recommender systems evaluate rankers, instead of by retrain-and-benchmark; and
"routing quality" becomes a measurable quantity rather than something inferred from load and
entropy statistics.

**Why it is plausible.** An MoE layer's output is a gated *sum* of expert outputs, which is the
additive structure the slate pseudoinverse estimator needs (1605.04812, Assumption 1) at the
level of the layer output; the reward (log-probability of the realized token) is non-linear in
that output, so the assumption holds only approximately. How approximately is the empirical
question. 2605.07260 shows that better equal-compute routes exist on hard tokens, so there is
signal to rank.

**Strongest baseline.** The status quo for judging routers without re-running them: (a) routing
statistics (load CV, routing entropy, router margin, churn; already computed by
`training/src/ptr_training/moe/routing_metrics.py`), and (b) the direct method, a learned reward
model of per-token route value. For the estimator itself, set-level IPS/SNIPS (known to fail on
large slates) is the naive comparator.

**Candidate routers (targets).** At least 8, spanning a known value range: the trained router;
the trained router with aux-loss-free bias updates at two step sizes (M012's arms); a router-only
counterfactual update of the final layer (the EPO procedure of 2605.07260); two perturbed routers
of known worse value (logit noise at two scales); a token-error-supervised router (2609.37751's
headless variant, if it can be reproduced at this scale); and the uniform-within-top-2k router.

**Ablations.** ε ∈ {0.02, 0.05, 0.10} (pre-declared); exploration in one layer vs all layers;
estimator: IPS, SNIPS, PI, wPI, DM, DR; reward: local per-layer teacher-forced log-probability on
verified-correct trajectories (as 2605.07260) vs sequence-level verified success.

**Metrics and pinned thresholds (proposed).**

| Metric | Rule |
|---|---|
| Primary: rank agreement between estimated and on-policy value over the candidate set | Kendall τ_b ≥ 0.80 in every confirmatory seed |
| Calibration of the estimate | for every candidate, \|V̂ − V\| ≤ 0.25 × (max V − min V) over the candidate set |
| Exploration cost (co-primary) | at the chosen ε: held-out loss increase ≤ 1% relative and verified pass@1 on the pinned GSM8K slice drop ≤ 0.5 points vs deterministic routing |
| Superiority over the status quo | τ_b(OPE) − max(τ_b(routing statistics), τ_b(DM)) ≥ 0.15 |
| Validity | candidates outside the logging support are refused and counted (the `ope.rs` positivity rule), never estimated; a refused candidate does not count as a rank error but more than 2 refusals out of 8 invalidates the run |
| Sensitivity | synthetic world with an exactly additive reward: τ_b ≥ 0.95; placebo with shuffled propensities: τ_b ≤ 0.30 |

**Kill criteria.**
- **K-B1**: τ_b < 0.80 in any seed at the largest ε that satisfies the cost bound → token-level OPE
  of routers is not practical; thesis dropped.
- **K-B2**: the smallest ε that reaches τ_b ≥ 0.80 violates the cost bound → it works only at a
  price nobody would pay; thesis dropped as an engineering proposal, kept as a measurement result.
- **K-B3**: routing statistics or the direct method alone come within 0.05 of OPE's τ_b → logging
  propensities is unnecessary; thesis dropped.

**Compute estimate.** CPU stage: a tiny MoE (4 layers, 16 experts, top-2, built with
`model.build_tiny_model` and trained on a synthetic verifiable task) – training about 1–2 CPU-h
per seed; the on-policy ground truth for 8 deterministic candidates is 8 forward passes over the
evaluation set, so the whole CPU stage is about 10 CPU-h for 5 seeds. GPU stage on the pinned
OLMoE base (64 experts, top-8): logging pass over about 2M tokens with exploration ≈ 1–2 GPU-h per
seed; on-policy ground truth for 8 candidates ≈ 8 forward passes ≈ 1–2 GPU-h per seed;
generation for the cost bound ≈ 2 GPU-h per seed. Total ≈ 20–30 GPU-h. This is the cheapest of the
three theses.

**Reuses.** `crates/ptr-router/src/ope.rs` (estimator discipline, positivity refusal,
`PromotionGate`) as the reference to port the slate estimator into later;
`crates/ptr-router/src/sampling.rs` and `propensity.rs` (exact quantised propensities);
`training/src/ptr_training/moe/routing_metrics.py` (status-quo statistics);
`training/src/ptr_training/moe/routed_peft.py` (`route_topk` with an `allowed` mask, the hook
point for exploration over routed LoRA experts; the base model's native router needs a forward
hook, work package B1); M012's balancing arms as candidates; F003's matured-cohort and
positivity rules as the logging contract.

**Minimal decisive experiment.** CPU tiny MoE, exploration in the final MoE layer only, ε = 0.05,
8 candidates, wPI and DR against on-policy truth, 5 seeds, with the additive-reward and
shuffled-propensity controls. If it fails on CPU (where the model is small and the support is
dense), the thesis is dead; if it passes, the GPU stage on OLMoE's final layer decides.

### Thesis C — Lineage-closed revocation of experts

**Hypothesis.** Modular MoE "opt-out by removing the expert" (FlexOlmo, BAR) is exact only while
no other parameter that remains in the model has seen the removed data. Common lifecycle
operations break that silently: averaging diverged shared layers across experts (BAR's merge
step), fine-tuning the router on mixed data, merging experts, and upcycling a new expert from an
old one. After such operations, removing the expert leaves measurable influence of the revoked
data in the surviving model. Revocation over the *transitive closure* of a typed dependency graph
(the R011/R013 mechanism: fence and retrain every derived component, keep planned-succession
edges distinct from live dependencies) restores indistinguishability from a model trained without
the data, at a cost well below full retraining and with better forget quality than approximate
unlearning. If true, modular-MoE practice has to treat routers, shared layers and merges as part of
the data-provenance graph; "remove the expert" is not a deletion mechanism on its own.

**Strongest baseline.** Two. (1) For the remedy: approximate unlearning applied to the
ungoverned model at equal compute, NPO with a retain loss (2404.05868, read). (2) For the
problem: FlexOlmo-style removal with frozen shared layers and per-expert router embeddings, which
should *not* leak and serves as the control that the measurement is not detecting noise.

**Arms.** For each lifecycle schedule s ∈ {frozen-shared removal (control), shared-layer
averaging, router fine-tune on mixed data, TIES merge of the revoked expert into a descendant,
upcycled descendant, reactivated descendant}:
- U_s: ungoverned removal (drop the revoked expert only);
- L_s: lineage-closed revocation (fence the closure; retrain fenced members from non-revoked
  inputs, reusing planned-succession edges);
- N_s: NPO+RT on the ungoverned model at the compute L_s used;
- G_s: gold reference, the same schedule retrained without the revoked shard (two gold seeds, to
  measure the gold's own seed band).

**Metrics and pinned thresholds (proposed).** Leakage is measured with planted canaries per shard
(32 per shard, format with a randomness space of 10^6) using exposure (1802.08232, read) and a
loss-threshold membership-inference AUC on held-out shard examples.

| Metric | Rule |
|---|---|
| Problem existence (H-C1) | for at least two of the four mixing schedules, exposure(U_s) − exposure(G_s) ≥ 2.0 bits (one-sided 95% lower bound) and MIA AUC(U_s) ≥ 0.60, in every seed |
| Control | frozen-shared removal: exposure(U) − exposure(G) ≤ 0.5 bits |
| Remedy (H-C2) | exposure(L_s) − exposure(G_s) ≤ 0.5 bits and MIA AUC(L_s) ≤ 0.55 for every schedule, every seed; retained-shard utility of L_s within 1.0 point of G_s |
| Cost | compute of L_s ≤ 50% of G_s (full retrain without the shard) |
| Comparison with approximate unlearning | at equal compute, N_s has exposure ≥ L_s + 1.0 bits or utility ≤ L_s − 1.0 point |
| Validity | gold seeds' exposure band reported; a schedule where the two gold seeds differ by more than the 0.5-bit tolerance makes the tolerance invalid for that schedule (reported, not judged) |

**Kill criteria.**
- **K-C1**: H-C1 fails (no mixing schedule leaks measurably) → removing the expert is already a
  sufficient deletion mechanism in practice; the lineage machinery has no measured privacy value
  beyond FlexOlmo-style design; thesis dropped and reported as a negative result.
- **K-C2**: closure cost ≥ full retrain (the closure cascades to everything, for example because the
  router and shared layers depend on every shard) → lineage closure is correct but useless; thesis
  dropped.
- **K-C3**: N_s matches L_s on exposure and utility at equal or lower compute → approximate
  unlearning suffices.

**Compute estimate.** CPU stage: tiny MoE, 4 shards, one revoked shard per seed, 6 schedules × (U,
L, N, 2 G) ≈ 30 trainings per seed at about 0.3–0.5 CPU-h each ≈ 50–75 CPU-h for 5 seeds (days on
4 cores; can run unattended). GPU stage: LoRA experts per shard on the pinned OLMoE base (the
PERFT-R arm of `routed_peft.py`), ≈ 1 GPU-h per expert training; 6 schedules × 5 seeds × about 3
GPU-h ≈ 90 GPU-h. The gold retrains dominate.

**Reuses.** R011/R013's typed dependency index and closure rule
(`crates/ptr-runtime/src/expert/dependency.rs`, compound `InputInvalidated`), as the reference
semantics the Python harness must match; `crates/ptr-lineage/src/merge.rs` (`ties_merge`, exposed
as `ptr_py.ties_merge`); `routed_peft.py` (PERFT-R / MixLoRA arms); `m011.py` training loop;
`ptr-lineage`'s ForgettingGate for the utility side; M013's domain sequence and pinned corpora as
shard sources.

**Minimal decisive experiment.** CPU tiny MoE, schedules: frozen-shared removal (control),
shared-layer averaging, router fine-tune on mixed data; arms U and G only; 5 seeds. This decides
H-C1 cheaply. Only if H-C1 holds is the remedy (L, N) built and run, first on CPU, then on GPU. A
CPU *negative* on H-C1 is weak evidence (tiny models memorise differently) and is followed by one
GPU pilot before K-C1 is declared.

### 3.4 How the three theses relate to the existing programme

| Existing item | Relation |
|---|---|
| F004 (closed loop vs everything) | Deferred until thesis A survives; its R1 comparison is A0 here |
| F005 (finding-routed repair experts) | Superseded by thesis A (adds the conditioned-repairer and domain-split baselines it lacks) |
| F006, M013 claims 11–13 (statistics-driven lifecycle, context ablation) | Deprioritised (section 6); the provenance part moves to thesis C |
| M010, M011, M012, M014 | Kept as the base programme; M011 provides the LoRA training path all three theses use, M012 provides candidate routers for thesis B |
| R011, R013, R014 | Become instruments: their closure semantics is thesis C's remedy |
| E006 | Narrowed to the PTR-specific difference (section 7); not a thesis |

---

## 4. Gap list: work packages

Sizes are estimates of new code (lines) and focused working days. All paths are proposed new
files unless marked otherwise. A change under `crates/<crate>/src/**` also updates
`crates/<crate>/component.toml` and regenerates docs (`make docs`).

### 4.0 Cross-cutting

| WP | Content | Paths | Size | Depends on |
|---|---|---|---|---|
| X0 | GPU bring-up: tiny GPU smoke, OLMoE LoRA SFT memory measurement on a 3090, derived config | existing targets `make smoke-gpu`, `make olmoe-sft-smoke`; docs/GPU_HANDOVER.md steps 1–3 | 0 LOC, 1–2 days on the desktop | owner hardware |
| X1 | Re-run the literature searches of §2.3 immediately before each preregistration; record the diff | `research/agenda/` (successor note) | 0.5 day per thesis | — |
| X2 | Build `ptr_py` in a reproducible environment the training venv can import (CI does not build it today) | existing `make py-bindings`; possibly a CI job (later, head's decision) | 1 day | — |

### 4.1 Thesis A

| WP | Content | Paths | Size | Depends on |
|---|---|---|---|---|
| A1 | Repair-transition builder: generate tasks, run the base generator, verify through `ptr_py.verify_*`, keep (task, candidate, report, codes, reference) | `training/src/ptr_training/moe/repair_data.py`, `training/tests/test_repair_data.py` | ~400 LOC, 2 days | X2 |
| A2 | Code-table parity test: Python's view of codes and roles equals `CODE_TABLE` | `training/tests/test_repair_codes.py` | ~100 LOC, 0.5 day | X2 |
| A3 | Repair-expert trainer with parameter-matched arms (A1 generic, A2 domain, A3 failure-mode) | `training/src/ptr_training/moe/repair_experts.py` + tests | ~500 LOC, 3 days | A1, M011 path |
| A4 | Loop evaluator: matched verifier-call and FLOP accounting, arms A0–A6, S0 relabelling, collateral-edit checker, routing through `ptr_py` `FindingRouter` with propensity logs | `training/src/ptr_training/moe/repair_eval.py` + tests | ~700 LOC, 4 days | A1–A3 |
| A5 | CPU harness stage with `ptr_py` loop simulation for S+ and S0 | `training/src/ptr_training/moe/repair_sim.py` | ~250 LOC, 1.5 days | A4 |
| A6 | Experiment records: CPU harness experiment and GPU successor (manifest, PREREGISTRATION, criteria, `aggregate.py`, mutation plan) | `experiments/feedback/<new id>/` via `scripts/new_experiment.py` | 2 days each | A4, A5 |

### 4.2 Thesis B

| WP | Content | Paths | Size | Depends on |
|---|---|---|---|---|
| B1 | Exploration hook: single-slot swap with probability ε per token and layer, exact propensities, per-token logging (layer, chosen set, top-k set, ε, router logits digest) | `training/src/ptr_training/moe/route_explore.py` + tests | ~300 LOC, 2 days | — |
| B2 | Estimators: set-level IPS/SNIPS, slate PI/wPI, DM, DR; positivity refusal mirroring `ope.rs` | `training/src/ptr_training/moe/route_ope.py` + tests (unit tests against the additive synthetic world) | ~450 LOC, 3 days | B1 |
| B3 | On-policy ground truth and candidate-router family (bias variants, EPO-style final-layer update, noise-perturbed routers) | `training/src/ptr_training/moe/route_candidates.py`, `route_truth.py` | ~400 LOC, 2.5 days | B1 |
| B4 | CPU tiny-MoE training script on a synthetic verifiable task (verified-correct trajectories for the local reward) | `training/src/ptr_training/moe/tiny_moe_task.py` | ~250 LOC, 1.5 days | — |
| B5 | Experiment records, CPU then GPU | `experiments/model/<new id>/` | 2 days each | B1–B4 |
| B6 | (Only if B passes on GPU) port the slate estimator into Rust next to the categorical one, so the `PromotionGate` can gate L1 router changes | `crates/ptr-router/src/slate_ope.rs`, `crates/ptr-router/component.toml` | ~500 LOC Rust, 3 days | B5 outcome |

### 4.3 Thesis C

| WP | Content | Paths | Size | Depends on |
|---|---|---|---|---|
| C1 | Canary injection per shard; exposure (rank estimate over the randomness space) and loss-MIA AUC | `training/src/ptr_training/moe/canaries.py` + tests | ~300 LOC, 2 days | — |
| C2 | Lifecycle operations on LoRA experts in Python: per-shard expert, shared-layer averaging, router fine-tune on mixed data, TIES merge (via `ptr_py.ties_merge`), upcycle, reactivate | `training/src/ptr_training/moe/lifecycle_ops.py` + tests | ~500 LOC, 3–4 days | M011 path, X2 |
| C3 | Dependency graph and closure in the harness, with a conformance test against R011/R013's semantics (typed edges, planned succession vs live dependency) | `training/src/ptr_training/moe/lineage_graph.py`; later `bindings/ptr-py/src/lineage.rs` exposing the runtime's closure so the harness calls the Rust rule directly | ~300 LOC Python, 2 days; Rust binding ~300 LOC, 2 days | C2 |
| C4 | Gold retrain-without-shard runner and the closure retrain (remedy) | `training/src/ptr_training/moe/revocation_eval.py` | ~350 LOC, 2 days | C1–C3 |
| C5 | NPO+RT baseline | `training/src/ptr_training/moe/npo.py` | ~200 LOC, 1.5 days | C4 |
| C6 | Experiment records: problem-existence (CPU), remedy (CPU), GPU successor | `experiments/lifecycle/<new id>/` | 2 days each | C1–C5 |

### 4.4 Order

1. **B1–B4 on CPU** (cheapest decisive test; can start now, no GPU, no Rust change).
2. **C1, C2 (mixing schedules only), C4 (U and G only) on CPU** — decides H-C1 before any remedy
   code is written.
3. **A1, A2** (dataset and parity; CPU; needs X2).
4. **X0** on the owner's desktop, in parallel with 1–3.
5. **A3, A4** and the GPU stage of A; then GPU stages of B and C according to the CPU outcomes.

Each experiment is preregistered and frozen before its confirmatory seeds; a CPU pass never
counts as model evidence.

---

## 5. Honest risks: why each thesis might fail or be incremental

### Thesis A
- **Most likely failure: K-A1.** A single LoRA that reads the finding code in its prompt may do
  everything three specialists do; at small adapter ranks the specialisation axis may not matter.
  This is the outcome to expect, and the reason A1 is the strongest baseline.
- **The base may produce the wrong failures.** A 1B-active MoE may fail mostly in ways the
  verifiers classify as one or two codes (for example `json.parse`), leaving too few transitions per
  role for specialists to differ. The pilot must report the per-code transition counts before the
  freeze; fewer than 2,000 transitions for any role used in A3 invalidates the design.
- **Resampling is strong on easy tasks.** On short JSON and arithmetic tasks, best-of-N may solve
  nearly everything solvable (Large Language Monkeys), leaving no headroom; the task mix must be hard
  enough that A0's success at budget 4 is below about 70% in the pilot.
- **External validity is narrow.** The verifiers cover arithmetic, a JSON-schema subset, a toy
  code language and facts. A positive result says "for deterministic, typed verifiers", not "for
  agents".
- **Incremental reading.** Even if positive, reviewers can call it "a mixture of LoRA specialists
  for self-repair, routed by error type" — close to CodeRescue plus mixture-of-LoRA work. The
  defensible contribution is the matched-baseline result about the *axis*, nothing more.

### Thesis B
- **Linearity may be badly violated.** The reward is a non-linear function of the layer output;
  the pseudoinverse estimator may be biased enough that DR's reward model does all the work, in
  which case K-B3 triggers.
- **Positivity.** Single-slot exploration only supports routers within one swap per token under the
  set-level view; the slate estimator relies on per-expert marginal support plus additivity to
  reach further. Routers that differ in many slots per token may be refused or badly estimated.
- **Cost.** Exploration in every layer may cost more than 1% loss; exploring only the final layer
  limits what can be audited.
- **Proxy competition.** If cheap routing statistics already rank routers well, logging
  propensities is unnecessary.
- **Incremental reading.** Off-policy slate evaluation is a known method (1605.04812); the
  contribution is showing that it transfers to token routing at acceptable cost, plus the measured
  answer to "how good are trained routers" without re-execution. If the answer is "it works only
  for the final layer", the contribution is small.

### Thesis C
- **No leakage to find (K-C1).** Shared-layer averaging over several experts may dilute one
  shard's influence below measurement; tiny CPU models may not memorise canaries at all.
- **Gold noise.** Two gold retrains differ by seed; if that band exceeds the 0.5-bit tolerance,
  exact indistinguishability is not measurable at this scale.
- **"Of course it leaks".** If leakage is found, some readers will call it obvious (averaging
  parameters that saw the data keeps the data). The value is in the measurement across schedules
  and in the cost of the remedy, not in surprise.
- **Closure cascades.** If the router is trained jointly on every shard, every revocation fences
  the router; the remedy then depends on cheap router retraining, which may break K-C2.
- **Determinism.** "Indistinguishable from gold" is statistical, not bit-exact, on GPU (batch- and
  kernel-dependent numerics; doc 36 §F and the batch-invariance row of the prior-art matrix); the
  claim must stay statistical on GPU.

### Programme-level risks
- **Capacity.** One owner, two consumer cards, a strict preregistration discipline: each
  confirmatory run costs days of setup. Running all three GPU stages is about 200 GPU-h plus
  setup; the order in §4.4 is chosen so that two theses can die on CPU first.
- **Evidence-class creep.** The nearest risk is presenting a CPU harness pass as a statement about
  OLMoE. Every report keeps the evidence class next to the number.
- **Tooling gaps.** `ptr_py` is not built by CI; the GPU path of `trl-moe` has never run; the
  M011 harness is unexercised on real weights. X0 and X2 are prerequisites, not formalities.

---

## 6. What is deliberately not pursued now (and why)

| Item | Reason | Can return as |
|---|---|---|
| Typed / semantic-slot routing (claims 1–3) | Open in the literature, but PTR's own A0 studies are NO-GO with calibration degradation | A later study only after an A0 study passes its gates (ADR-0022, S8) |
| Self-model and attention schema at stage B (C002, C003) | Stage-A toys FALSIFIED; no positive effect has been shown | Secondary arm of thesis A (self-model hint vs none) if A passes |
| Learned workspace and Pod-level workspace (C001, C006) | C001 NOT_SHOWN; F007's workspace effect is an implementation check | Ablation inside thesis A's loop |
| Verifier-statistics-driven Birth/Merge (F006, M013 claims 11–13) | Token-level precedent (DynMoE); F007's L3 secondary NOT_SHOWN with power about 0.2–0.3; doc 39 §5.6 "heuristic in disguise" | Thesis C supplies the provenance half; a statistics-driven Birth study only after A shows failure-mode experts help |
| Reactivate (claim 12) | Published in toy form (nanoFMT 2604.00812); not a strong open question | Schedule "reactivated descendant" inside thesis C |
| KV reuse claims beyond the mechanism checks (E006, E008) | Largely published (2609.38706, 2609.30884, aLoRA line) | Engineering, not research |

---

## 7. Proposed edits (not made; for the head or owner)

### 7.1 `research/novelty/CLAIMS.md`

- **Claim 12**: append "Dormant-expert recovery after a domain returns is reported in a 2-layer
  toy without seeds (nanoFMT, arXiv:2604.00812); the claim is narrowed to a matched comparison
  at OLMoE scale with every seed passing."
- **Claim 14**: append "The engine-cache half (E006) overlaps the KV provenance contract
  (arXiv:2609.38706: adapter identity, weight version, precision, KV format and salt bound into
  shared-tier keys) and layerwise version anchors for evolving adapters (arXiv:2609.30884). What
  remains PTR-specific is revocation checked at admission and the per-layer chain over expert
  generations."
- **Claim 22**: append "Learned post-failure routing over recovery actions from verdict and stderr
  is published (CodeRescue, arXiv:2607.19338). The claim is restricted to specialised repair
  experts per finding class against a single repairer that receives the same findings, and is
  tested as thesis A of `research/agenda/MOE_RESEARCH_AGENDA.md`."
- **Claims 23 and 24(d)**: append "Token-level precedent for Birth from an unroutable signal:
  DynMoE (arXiv:2405.14297) adds an expert when tokens activate none."
- **New claim 26 (thesis A)**, draft: "In a verifier-closed loop, repair experts specialised by
  verifier failure class and routed by finding code verify more tasks per verifier call than
  best-of-N with the same verifier at matched verifier calls and FLOPs, than one parameter-matched
  repair expert that receives the same findings, and than the same budget split by task domain.
  Unproven."
- **New claim 27 (thesis B)**, draft: "With single-slot exploration logged at every MoE routing
  decision, slate off-policy estimators rank candidate routers in the same order as on-policy
  evaluation (Kendall τ_b ≥ 0.80) at a quality cost of at most 1% held-out loss. Unproven."
- **New claim 28 (thesis C)**, draft: "Removing an expert does not remove its training data's
  influence once shared layers are averaged, the router is fine-tuned on mixed data, or the expert
  is merged or upcycled; revocation over the transitive dependency closure restores
  indistinguishability from a model trained without the data at less than half the cost of full
  retraining. Unproven."

### 7.2 `research/novelty/PRIOR_ART_MATRIX.md`

- Pod-MoE section, lifecycle row: replace "we know of no prior work covering the full lifecycle,
  and Reactivate is unpublished" with a statement that names DynMoE (2405.14297), BAR
  (2604.18473), FlexOlmo (2507.07024) and nanoFMT (2604.00812), and keeps only "no prior work
  gates the operations with committed evidence; whether that matters scientifically is thesis C".
- New rows: Policy-Masked Private Experts (2608.06690); FlexOlmo (2507.07024); BAR (2604.18473);
  counterfactual routing analysis (2605.07260); cross-entropy-guided routing (2609.37751); slate
  OPE (1605.04812); CodeRescue (2607.19338); KV provenance contract (2609.38706); CacheReforge
  (2609.30884); SA-MoE routing ledger (2609.34634).
- Verifier-closed routing section: the row "RouteAudit, CoVeR, 2609.34634, 2609.33505 — found but
  not read" becomes read, with the content of §2.1 and §2.2 above (none routes experts by
  findings; RouteAudit and SCOPE-Routing are methodological).

### 7.3 `experiments/registry.toml` (names only; ids are assigned by `scripts/new_experiment.py`)

- Thesis A: a CPU harness experiment (`failure-mode-experts-harness`) and a GPU successor
  (`failure-mode-experts`); F005 marked superseded by the GPU successor once it is registered.
- Thesis B: `router-ope-cpu` and `router-ope-olmoe`.
- Thesis C: `expert-revocation-leakage-cpu` (problem existence), `expert-revocation-closure-cpu`
  (remedy), `expert-revocation-olmoe`.
- Every new preregistered experiment is also added to `LISTED_UNFROZEN` in
  `scripts/tests/test_research_gates.py` (AGENTS.md §3).

### 7.4 ADRs

- A new ADR (next free number) recording the research focus: the three theses, the
  deprioritisations of §6, and the rule that F004 runs only after thesis A survives. No existing ADR
  needs to change; ADR-0022's S8 (A0 bridge only after an A0 study passes) is consistent with §6.

### 7.5 Architecture docs

- doc 37 §7 ("'reactivate' has no literature") and doc 39 §4.5 ("not read") are out of date for the
  reasons in §2.2; the docs-refresh workstream owns those files.
