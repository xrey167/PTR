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

This section is a **research agenda, not a preregistration**. For each thesis it fixes the
hypothesis, the claim scope, the arms, the decisive contrasts with their margins, the controls by
name and purpose, the kill criteria and the compute. Everything a harness needs at implementation
level (bound mechanics, exact gate formulas, edit-region algorithms, logging fields, gold recipes)
is listed per thesis under **"Open for preregistration"**, as a checklist of decisions the
experiment manifest must pin, each with the constraint it must satisfy and the failure it prevents.

**Run gate.** No thesis can be run, at any stage beyond an unrecorded harness smoke, until every item
of its "Open for preregistration" checklist is pinned in a reviewed manifest under `experiments/`
(experiment manifest, preregistration, criteria and aggregator, created with
`scripts/new_experiment.py` and reviewed as `AGENTS.md` §1 requires). Margins written below are
proposals; the manifest freezes them and they are never loosened afterwards.

### 3.0 Rules common to the three theses

- **Seeds and hardware** (`AGENTS.md` §3, §2): pilots on seeds ≥ 1001 only; confirmatory seeds
  `17, 29, 43, 71, 101`, each run once; confirmatory GPU seeds run one at a time on card 0 (OD-3);
  card 1 serves pilots, tuning and mutation runs. "FLOPs" means the analytic FLOPs of the active
  parameters counted by the harness, as in M011.
- **Stage scoping.** A thesis about the OLMoE-class model is supported or refuted only by its
  frozen GPU confirmatory stage. A CPU stage gives a go/no-go for GPU spend and a result about the
  CPU toy only; a GPU pilot gives only a go/no-go for the freeze.
- **Order and outcomes.** Validity gates first: any failure makes the run INVALID and nothing is
  judged (a hard-counter failure is additionally recorded as a product finding, exit code 1). Then
  the named sensitivity controls: a failed control makes INVALID every row that needs it. Then the
  decisive rows.
- **Three-way rule for a decisive row** (`docs/architecture/39-vcw-moe.md:575-579`), judged with
  one-sided 95% bounds per confirmatory seed: **supported** if the favourable bound clears the
  margin in every seed; **refuted** if the unfavourable bound lies on the wrong side of the margin
  in every seed; **not shown** otherwise, including any interval that overlaps the margin. For an
  equivalence row (a difference must lie inside a band), supported if both bounds lie inside in
  every seed, refuted if the whole interval lies outside in every seed. A row with several
  components is refuted only if the **same** component fails in every seed, and supported only if
  every component passes in every seed.
- **Closure.** A thesis outcome can change only through a row of its decisive table. Everything
  else (secondary arms, ablations, other estimators, other ε, wall-clock) is reported only and can
  never support, narrow or drop a claim.
- **Variability.** Within-seed bounds resample items against fixed trained models; retraining
  variability is covered by requiring every row outcome in every one of the five seeds, each of
  which retrains every model.

### Thesis A — Failure-mode experts

**Hypothesis.** For repair after a deterministic verifier rejects a candidate, experts
*specialised by failure class* and *routed by the verifier's finding code* repair more tasks per
verifier call than (i) independent resampling with the same verifier at matched verifier calls and
FLOPs, (ii) a single repair expert with the same total trainable parameters and training budget
that receives the same findings in its input, and (iii) the same parameter and training budget
split into experts by *task domain*.

**Claim scope.** A supported outcome licenses only: on tasks with deterministic verifiers of the
kinds tested, failure-class routing of specialised repair experts beats conditioning, resampling
and a domain split at matched budgets. It licenses nothing about the *typed* code table (A6 is
reported only) or about agents in general.

**Arms.**

| Arm | Description |
|---|---|
| A0 | Best-of-N resampling from the base generator, same verifier, matched verifier calls and generation FLOPs |
| A1 | Conditioned generic repairer: one LoRA with the specialists' total parameters and training budget, trained on all repair transitions, full verifier report in the prompt (strongest baseline) |
| A2 | Domain experts: the same budget split by task family, routed by task family, report in the prompt |
| A3 | **Failure-mode experts**: one LoRA per repair role of `CODE_TABLE` used by the task mix (the fact role is out of scope), routed by `FindingRouter` |
| A4 | A3 with a fixed random permutation of the code-to-expert map |
| A5, A6 | Scalar-verdict bandit over A3's experts; learned router over A3's experts reading the raw report text. Reported only |

**Controls** (run in a simulated-expert harness that shares the evaluation, routing and accounting
code of the GPU arms). **S+**, a heterogeneous-skill world in which each specialist is better on its
own role, so A3 must lead every comparator; it shows the harness can detect a true effect. **S0**,
an equal-skill world in which every repair expert of A1–A4 has the same success per code, so all
repair arms tie by construction; it shows the harness does not manufacture an effect.

**Decisive rows** (quantity: verified success within a budget of 4 verifier calls, in percentage
points, paired over the same tasks; accept predicate as in `Candidate::passes`).

| Row | Contrast | Margin | Needs | If refuted |
|---|---|---|---|---|
| D1 | A3 − A1 | +3.0 | S+, S0 | **K-A1**: thesis REFUTED (routing adds nothing over conditioning) |
| D2 | A3 − A0 | +3.0 | S+ | **K-A2**: thesis REFUTED (doc 39 R1: verifier access carries the effect) |
| D3 | A3 − A2 | +3.0 | S+, S0 | **K-A3**: the "axis" sub-claim is dropped |
| D4 | A3 − A4 | +3.0 | S+, S0 | **K-A4**: thesis REFUTED (doc 39 R2) |

**Outcome.** REFUTED if D1, D2 or D4 is refuted. Otherwise SUPPORTED if D1–D4 are all supported;
SUPPORTED-NARROW (no axis claim) if D1, D2 and D4 are supported and D3 is not; NOT_SHOWN otherwise.

**Compute.** CPU stage: harness, S+ and S0, a few CPU-hours. GPU stage on the pinned OLMoE base
(about 1B active / 7B total, bf16 about 12.9 GB per doc 37), full design ≈ 65–100 GPU-h
(transition collection ≈ 5, 9 LoRA trainings × 5 seeds at 1–1.5 GPU-h each ≈ 45–70, evaluation ≈
15–25); minimal design ≈ 45–70 GPU-h. Confirmatory seeds run serially on card 0: ≈ 2.7–4.2 days of
device time (≈ 1.9–2.9 for the minimal design), plus about 1–2 days for pilots, the mutation check
and the pilot rerun. Unmeasured until docs/GPU_HANDOVER.md step 2 has run.

**Reuses.** `crates/ptr-verifier` through `ptr_py.verify_*`; `crates/ptr-router/src/findings.rs`
through `ptr_py` routing, so the Rust router is the router under test; `training/src/ptr_training/moe/`
(`model.py`, `m011.py`, `tasks.py`, `reward.py`); F007's harness design and mutation plan; F005's
manifest as the record to supersede.

**Minimal decisive experiment.** JSON-schema and arithmetic tasks, three failure roles (format,
value, schema), arms A0–A4 with A2 as two domain experts (arithmetic, JSON), S+ and S0 inside the
same frozen run, 5 confirmatory GPU seeds.

**Open for preregistration (thesis A).**

| # | Must pin | Constraint | Prevents |
|---|---|---|---|
| A-1 | Bound procedure for every row and control | Paired over the identical task list of every arm; resampling unit, method and replicate count fixed; one-sided 95% | Unpaired or arm-specific resampling inflating or hiding differences |
| A-2 | Budget matching (validity gate) | A0–A4: verifier calls per task exactly the budget; generation FLOPs within ±5% of A3. A1, A2, A3: trainable parameters within ±1%, **aggregate training budget equal** (training tokens within ±1%, optimizer steps equal in total, tuning trials and tuning steps equal, as M011's tuning protocol) | A capacity, data or tuning advantage passing as a routing effect |
| A-3 | Hard counters (validity gate) | False accepts, unverified returns and budget violations 0 in every arm; A0 candidates are fresh seeded samples, never modified | Unsafe or mis-budgeted runs being judged |
| A-4 | Collateral-edit region for repair arms A1–A6 (validity gate) | Computable from `Finding { code, message, hard }` only (`crates/ptr-verifier/src/lib.rs:32-36`) plus per-task dataset declarations. Code: each task declares its entry and its *frozen* (prompt-supplied) functions; frozen functions must stay byte-identical; unparsable input or output exempts the step (code findings name a test id, not a function: `crates/ptr-verifier/src/code_test.rs:905-919`). JSON: the pointer is the message prefix `"<pointer>: <reason>"` (`crates/ptr-verifier/src/json_schema.rs:657-666`); for schema findings other than `json.required`, the region is the subtree at that pointer; for `json.required`, whose pointer is the **containing object** (`json_schema.rs:733-738`), the only permitted change is adding the named missing property (parsed from `missing required property "<name>"`) to that object, every existing sibling byte-identical, and a containing object shown as `(root)` is handled the same way at the root; findings with no usable pointer (`json.parse`, `json.oversize`, `json.depth`, or a pointer shortened with `...` by `shown`, `json_schema.rs:631-643`) exempt the step. Arithmetic: exempt. The computable fraction is at least 50% of repair steps per arm and seed | An uncomputable or over-permissive counter; a sibling-deleting "repair" passing as collateral-free |
| A-5 | Transition coverage (validity gate) | At least 2,000 training transitions per failure role used by A3 | Specialists trained on too little data to differ |
| A-6 | S+ and S0 worlds | S+ skill tables give analytic gaps of at least +8 points for A3 − A0, A1, A2, A4; S0 gives every repair arm identical success per code; both run through the GPU arms' evaluation and accounting code; control rules: S+ lower bounds ≥ +3, S0 bounds within ±1.5 | Controls that are not positive or null under a correct harness |
| A-7 | Task mix, sampling configs, prompts, ranks | A0's success at budget 4 below 70% in the pilot; power ≥ 0.8 at a true effect of +5 | No headroom; an underpowered confirmatory run |

### Thesis B — Off-policy-auditable routing

**Hypothesis.** If one routing decision per teacher-forced pass, at a declared MoE layer, is
perturbed by a small, exactly logged exploration (one slot of the base top-k set swapped with
probability ε to a uniformly drawn non-selected expert), a slate pseudoinverse estimator (wPI,
1605.04812) computed from such logs ranks candidate routers by their per-decision value in the same
order as running each candidate on-policy, at a small exploration cost.

**Estimand (complete).** Causal-LM convention: the output at input position j predicts y_{j+1}.
A context c is (verified-correct trajectory y, target index t ≥ 2, layer ℓ), one uniformly drawn
(t, ℓ) per pass. The intervened decision is the routing of input position t − 1 at layer ℓ; every
other decision uses the base router; aggregation weights on the chosen set S are the base router's
scores renormalised over S. The reward r(c, S) is the teacher-forced probability p(y_t | y_<t) at
position t − 1, in [0, 1]. **V(π)** = E_c E_{S∼π(·|c)} r(c, S), measured on-policy on a held-out
evaluation set; **V̂(π)** is wPI's estimate from a logged exploration cohort only.

**Claim scope.** One explored decision per pass at a declared layer; candidates compared on expert
selection only; exploration at every decision is not claimed.

**Candidates.** Exactly 8, frozen before collection: the trained router; aux-loss-free bias
variants at two step sizes; an EPO-style final-layer update (2605.07260); two logit-noise routers;
a token-error-supervised router (2609.37751) or a declared fallback; the uniform-within-top-2k
router. **Comparators:** a load-balance proxy (lower load CV ranks higher) and a direct-method
reward model fitted on the training cohort.

**Controls.** **SB+**, an additive-reward world on the same logs, where the estimator's linearity
assumption holds exactly, so ranking must succeed; it shows the harness and estimator work. **SB0**,
an action-independent world on the same logs, where every candidate has the same true value, so
estimated differences must vanish; it shows the harness does not manufacture differences.

**Decisive rows** (at the single declared ε; wPI).

| Row | Quantity | Margin | Needs | If refuted |
|---|---|---|---|---|
| DB1 | Kendall τ_b between V̂ and V over the candidates | ≥ 0.80 | SB+, SB0 | **K-B1**: thesis REFUTED |
| DB2 | max over candidates of \|V̂ − V\| / (max V − min V) | ≤ 0.25 | SB+ | thesis REFUTED (estimates rank but are unusable as values) |
| DB3 | τ_b(V̂, V) − max(τ_b(proxy, V), τ_b(DM, V)) | ≥ 0.15 | SB+, SB0 | **K-B3**: thesis REFUTED (propensity logging unnecessary) |
| DB4 | exploration cost per logged decision: relative drop of mean r between the base set and the logged set | ≤ 1% | none (a measurement on the real model) | **K-B2**: thesis REFUTED |

**Outcome.** REFUTED if any row is refuted; SUPPORTED if all four are supported; NOT_SHOWN otherwise.

**Compute.** CPU stage ≈ 10–15 CPU-h for 5 seeds on a tiny MoE (4 layers, 16 experts, top-2). GPU
stage on the OLMoE base (64 experts, top-8) ≈ 30–50 GPU-h (50,000 one-decision passes per seed ≈
3–5 GPU-h, on-policy truth ≈ 1–2, base-set rewards ≈ 1, reported-only pass@1 ≈ 2, per seed), ≈
1.3–2.1 days of serial card-0 time plus pilots and the mutation check.

**Reuses.** `crates/ptr-router/src/ope.rs` (estimator discipline, positivity and training-overlap
refusals, `PromotionGate`); `sampling.rs`, `propensity.rs`; `routing_metrics.py` (`load_cv`);
`routed_peft.py` (`route_topk`); M012's balancing arms as candidates.

**Minimal decisive experiment.** CPU go/no-go: tiny MoE, final MoE layer, ε = 0.05, the 8 frozen
candidates, a training cohort then a fresh exploration cohort per seed, SB+ and SB0, 5 seeds. GPU
decisive: the same design on the OLMoE base's final MoE layer.

**Open for preregistration (thesis B).**

| # | Must pin | Constraint | Prevents |
|---|---|---|---|
| B-1 | Logger and propensity verification (validity gate) | Every logged decision records the full base slate S_base, the swap indicator, the swapped slot and replacement, ε and the frozen logger parameters; μ(S \| c) is **recomputed from these records for every logged decision** (under the uniform single-slot logger, μ(S_base) = 1 − ε and each one-swap neighbour has ε / (k (N − k))) and must equal the logged value; router inputs are retained for a pinned subset only, for candidates added after collection | Unverifiable propensities; a gate that cannot be evaluated on most decisions |
| B-2 | Shadow scoring | Every frozen candidate's slate distribution is recorded on every logged decision from the same router input (position t − 1, layer ℓ); a candidate without it on a decision is refused for that cohort | Estimating a target whose action probabilities are unknown |
| B-3 | Cohorts (validity gate) | Learned candidates trained on a training cohort and snapshotted; a fresh, fixed-size exploration cohort per seed after the snapshot (proposed CPU 20,000, GPU 50,000 decisions); evaluation contexts disjoint from both; no overlap (`OpeError::OverlapsTraining`, `crates/ptr-router/src/ope.rs:36-38`); at most 2 of 8 candidates refused for positivity | Optimistic evaluation on training data; missing support |
| B-4 | Bound procedure | One-sided 95% bounds per seed; each replicate resamples exploration decisions and, independently, evaluation contexts, and recomputes V̂, V, the range, the comparators and DB4 from that replicate | Mixing estimands across replicates; a two-sample statistic without a combining rule |
| B-5 | Comparators | Proxy: `load_cv` of each candidate's selected sets at the intervened decisions of the evaluation set, lower ranks higher. DM: one linear head per expert on the router input, ridge-fitted on the training cohort only, penalty by 5-fold CV inside it | A comparator tuned after seeing results |
| B-6 | SB+ and SB0 | SB+: reward replaced by Σ_{e∈S} φ(c, e), φ = base-router softmax probability / k, true values computed exactly, rule lower bound τ_b ≥ 0.90. SB0: reward replaced by r(c, S_base), rule both bounds of V̂(π) − V̂(base) within ±0.25 × (real-world max V − min V) for every candidate | Controls that are not positive or null under a correct harness |
| B-7 | DB4 formula | (mean r(c, S_base) − mean r(c, S_logged)) / mean r(c, S_base) over the logged decisions, with r(c, S_base) recorded for every logged decision | A cost metric diluted over unexplored decisions or unstable per-item ratios |
| B-8 | ε, layer, candidate fallback | One ε and one layer per stage; the token-error router's fallback (a third noise scale) decided in the pilot | Searching over ε or candidates inside a confirmatory run |

### Thesis C — Lineage-closed revocation of experts

**Hypothesis.** Modular MoE "opt-out by removing the expert" (FlexOlmo, BAR) leaves the removed
data's influence in the model once lifecycle operations mix parameters: shared-layer averaging,
router fine-tuning on mixed data, merging an expert into a descendant, upcycling a new expert from
an old one. Revocation over the **transitive** closure of a typed dependency graph (the R011/R013
mechanism) brings the model back to a retrain without the data on the specified attacks and on
utility, at well below full-retrain cost, where approximate unlearning and a direct-only
(non-transitive) revocation at the same compute do not.

**Claim scope.** Matching the gold on canary exposure, a loss-threshold membership-inference AUC
and retained-shard utility only; no broader indistinguishability. Scoped to the schedules that are
shown to leak. The word "transitive" is asserted only if row H-C4 below is supported.

**Arms**, for each schedule s ∈ {frozen-shared removal (negative control), shared-layer averaging,
router fine-tune on mixed data, TIES-merged descendant, upcycled descendant, reactivated descendant
(reported only)}: **U_s** remove the revoked expert only; **L_s** transitive-closure revocation
(fence and retrain every component reachable from the revoked shard in the dependency graph);
**D_s** direct-only revocation (fence and retrain only components trained directly on the revoked
shard), compute-matched to L_s; **N_s** NPO with a retain loss (2404.05868) on U_s, compute-matched
to L_s; **G_s** gold: the same schedule rebuilt without the revoked shard, two retrains.

**Controls.** **Negative**: frozen-shared removal, where removal is exact by construction, so U
must match the gold; it shows the assays are not biased. **P-exp** and **P-mia**: the revoked
shard's canaries and its designated MIA member examples are also trained into a retained shard, so
the "removed" model is known to leak on each assay; each shows its assay can see a leak.

**Decisive rows** (Δ = arm minus gold; S* = schedules for which LEAK is supported; T = schedules in
S* whose transitive and direct closures differ).

| Row | Quantity | Margin or band | Needs |
|---|---|---|---|
| LEAK(s) | Δexposure(U_s) and ΔAUC(U_s) | ≥ 2.0 bits and ≥ 0.05 | negative, P-exp, P-mia |
| MATCH(X, s), X ∈ {L, D, N} | Δexposure, ΔAUC, Δutility of X_s | ≤ 0.5 bits; within ±0.03; within ±1.0 point | negative, P-exp, P-mia |
| COST(L, s) | training FLOPs of L_s / those of one gold retrain | ≤ 0.50 (a measurement; supported or refuted as stated in every seed, mixed seeds not shown) | none |

**Outcomes.**
- **H-C1 (problem):** SUPPORTED-BROAD if LEAK is supported for ≥ 2 mixing schedules,
  SUPPORTED-NARROW if for exactly one (a claim about that schedule only), **REFUTED (K-C1)** only if
  LEAK is refuted for all four mixing schedules, NOT_SHOWN otherwise.
- **H-C2 (remedy), on S*:** SUPPORTED if MATCH(L, s) and COST(L, s) are supported for every s in
  S*; REFUTED if either is refuted for some s in S* (**K-C2** if the cost ratio reaches 1.0 in every
  seed); NOT_SHOWN otherwise.
- **H-C3 (vs approximate unlearning), on S*:** REFUTED (**K-C3**) if MATCH(N, s) is supported for
  every s in S*; SUPPORTED if MATCH(N, s) is refuted and MATCH(L, s) supported for every s in S*;
  NOT_SHOWN otherwise.
- **H-C4 (transitivity), on T:** SUPPORTED if MATCH(D, s) is refuted and MATCH(L, s) supported for
  every s in T; REFUTED if MATCH(D, s) is supported for every s in T (the claim then names
  dependency-based revocation, not transitive closure); NOT_SHOWN otherwise or if T is empty, and
  "transitive" is then not asserted.
- **Claim 28** is SUPPORTED, scoped to S*, only if H-C1, H-C2 and H-C3 are supported; "transitive"
  appears in it only if H-C4 is supported. If H-C1 is refuted the thesis is dropped as a negative
  result; if H-C2 or H-C3 is refuted the remedy part is dropped and H-C1 stands as a measurement.

**Compute.** Per seed: 4 per-shard experts; 3 retained-shard experts × 2 gold retrains; 18 mixing
operations (6 schedules × arm and two golds); L, D and N for each schedule (18 jobs); P (1). CPU (tiny
MoE): ≈ 47 small jobs per seed, ≈ 45–95 CPU-h for 5 seeds. GPU (LoRA experts per shard on the OLMoE
base, ≈ 1 GPU-h per expert training, ≈ 0.5–1 GPU-h per mixing, L, D or N job): ≈ 29–47 GPU-h per
seed, ≈ 145–235 GPU-h for 5 seeds, ≈ 6–10 days of serial card-0 time plus pilots and the mutation
check.

**Reuses.** R011/R013's dependency index and closure rule (`crates/ptr-runtime/src/expert/dependency.rs`)
as the reference semantics; `ptr_py.ties_merge`; `routed_peft.py`; `m011.py`; M013's corpora as
shard sources.

**Minimal decisive experiment.** CPU go/no-go: tiny MoE, the negative control, the four mixing
schedules, arms U, G, P, 5 seeds; it decides only whether remedy code (L, D, N) is built and GPU
time is spent, and a CPU null never triggers K-C1. GPU decisive: all arms on the OLMoE base; only its
frozen confirmatory seeds can support or refute H-C1–H-C4.

**Open for preregistration (thesis C).**

| # | Must pin | Constraint | Prevents |
|---|---|---|---|
| C-1 | Shard design | Every shard is a task family with a deterministic exact-match checker; the revoked expert is trained on the revoked shard **and** at least one retained shard, so a clean counterpart exists | A gold whose counterpart of the revoked expert is undefined |
| C-2 | Gold construction per schedule | The gold runs the **same topology and operation sequence** as the arm, with every node rebuilt on its data minus the revoked shard: the revoked expert's clean counterpart is the same expert trained on its retained shards with the same seed and step schedule; the TIES-merged gold merges the descendant's other sources with that counterpart at the same densities; the upcycled gold upcycles from the counterpart with the same initialisation and training steps; router fine-tuning uses the mixed data minus the revoked shard | A gold that differs from the arm by topology or recipe rather than by data |
| C-3 | Gold estimand (one, used throughout) | The gold score of an item is the mean of the two retrains' scores (exposure per canary; attack score per example before the AUC; correctness per utility item); **the bootstrap uses the same combined-score gold in every replicate**, so bounds and point estimates target one estimand. Retraining variability of the gold is guarded by C-4 and the every-seed rule, not by the bound | A bootstrap that targets a different estimand than the point estimate |
| C-4 | Gold agreement (validity gate, checked first) | The two gold retrains of every schedule and seed agree within 0.5 bits of exposure, 0.02 AUC and 0.5 utility points; otherwise the run is INVALID and a successor needs a higher-precision gold | Bands narrower than the gold's own noise |
| C-5 | Assays | Exposure: 32 canaries per shard, randomness space 10^6 (1802.08232), estimation method. MIA: 500 designated members of the revoked shard vs 500 matched non-members never trained on, loss-threshold AUC. Utility: exact-match accuracy in percentage points on 500 held-out items per retained shard, macro-averaged | Undefined or post-hoc metrics |
| C-6 | Bound procedure | Paired bootstrap per seed over canaries, stratified member/non-member examples and utility items, the same items for arm and gold; one-sided 95% | Unpaired comparisons against the gold |
| C-7 | Negative control | U in the frozen-shared schedule uses the same seeds for retained experts as gold retrain 1, so it equals that retrain up to kernel nondeterminism; rule: Δexposure upper bound ≤ 0.5, ΔAUC within ±0.03, Δutility within ±1.0 | A negative control that is not null under a correct harness |
| C-8 | Positive controls | P trains the revoked canaries and members into a retained shard; rules: Δexposure(P) lower bound ≥ 4.0 bits, ΔAUC(P) lower bound ≥ 0.10, each gating its own assay | A null on an assay that could not see a leak |
| C-9 | Closures and compute matching | Transitive and direct closures computed from the typed dependency graph with R011/R013 semantics (planned-succession versus live edges), checked by a conformance test; FLOPs counted by the harness; D_s and N_s within ±5% of L_s | Attributing the remedy to transitivity without a matched direct-only arm |
| C-10 | Insertion integrity (validity gate) | Canary and member manifests (counts, digests) match for every arm; members and non-members disjoint; the gold never trains on them | Leaks or absences caused by bookkeeping |

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
| X2 | Narrow gap only: `ptr_py` already builds into `training/.venv` on the desktop (`scripts/setup_desktop.sh:148-160`, `maturin develop`) and has a manual GitLab job (`.gitlab-ci.yml:337-347`, `make py-bindings`); what is missing is a **GitHub** CI job that builds and tests it, so a break in the binding is invisible to the required checks | a job in `.github/workflows/` (head's decision); until then, run `make py-bindings` before each harness freeze | 0.5 day; not blocking | — |

### 4.1 Thesis A

| WP | Content | Paths | Size | Depends on |
|---|---|---|---|---|
| A1 | Repair-transition builder: generate tasks, run the base generator, verify through `ptr_py.verify_*`, keep (task, candidate, report, codes, reference) | `training/src/ptr_training/moe/repair_data.py`, `training/tests/test_repair_data.py` | ~400 LOC, 2 days | desktop `ptr_py` build |
| A2 | Code-table parity test: Python's view of codes and roles equals `CODE_TABLE` | `training/tests/test_repair_codes.py` | ~100 LOC, 0.5 day | desktop `ptr_py` build |
| A3 | Repair-expert trainer with parameter- and training-budget-matched arms (A1 generic, A2 domain, A3 failure-mode; checklist A-2) | `training/src/ptr_training/moe/repair_experts.py` + tests | ~500 LOC, 3 days | A1, M011 path |
| A4 | Loop evaluator: matched verifier-call and FLOP accounting, arms A0–A6, the S+ and S0 simulated worlds, collateral-edit checker (repair arms) and the A0 fresh-sample invariant, routing through `ptr_py` `FindingRouter` with propensity logs | `training/src/ptr_training/moe/repair_eval.py` + tests | ~700 LOC, 4 days | A1–A3 |
| A5 | CPU harness stage with `ptr_py` loop simulation for S+ and S0 | `training/src/ptr_training/moe/repair_sim.py` | ~250 LOC, 1.5 days | A4 |
| A6 | Experiment records: CPU harness experiment and GPU successor (manifest, PREREGISTRATION, criteria, `aggregate.py`, mutation plan) | `experiments/feedback/<new id>/` via `scripts/new_experiment.py` | 2 days each | A4, A5 |

### 4.2 Thesis B

| WP | Content | Paths | Size | Depends on |
|---|---|---|---|---|
| B1 | Exploration hook and logger: one explored decision per teacher-forced pass, single-slot swap with probability ε, full base slate and swap record per decision so μ(S \| c) can be recomputed for every decision (checklist B-1); shadow scoring of the frozen candidate set (selected set and per-expert inclusion probabilities per candidate); retention of layer-ℓ router inputs for a pinned subset; cohort ids, snapshot cutoffs and digests | `training/src/ptr_training/moe/route_explore.py` + tests | ~450 LOC, 3 days | — |
| B2 | Estimators: set-level IPS/SNIPS, slate PI/wPI, DM, DR; positivity refusal and the training-overlap refusal mirroring `ope.rs` | `training/src/ptr_training/moe/route_ope.py` + tests (unit tests against the additive synthetic world) | ~450 LOC, 3 days | B1 |
| B3 | On-policy ground truth and candidate-router family (bias variants, EPO-style final-layer update, noise-perturbed routers) | `training/src/ptr_training/moe/route_candidates.py`, `route_truth.py` | ~400 LOC, 2.5 days | B1 |
| B4 | CPU tiny-MoE training script on a synthetic verifiable task (verified-correct trajectories for the local reward) | `training/src/ptr_training/moe/tiny_moe_task.py` | ~250 LOC, 1.5 days | — |
| B5 | Experiment records, CPU then GPU | `experiments/model/<new id>/` | 2 days each | B1–B4 |
| B6 | (Only if B passes on GPU) port the slate estimator into Rust next to the categorical one, so the `PromotionGate` can gate L1 router changes | `crates/ptr-router/src/slate_ope.rs`, `crates/ptr-router/component.toml` | ~500 LOC Rust, 3 days | B5 outcome |

### 4.3 Thesis C

| WP | Content | Paths | Size | Depends on |
|---|---|---|---|---|
| C1 | Canary injection per shard; exposure (rank estimate over the randomness space) and loss-MIA AUC | `training/src/ptr_training/moe/canaries.py` + tests | ~300 LOC, 2 days | — |
| C2 | Lifecycle operations on LoRA experts in Python: per-shard expert, shared-layer averaging, router fine-tune on mixed data, TIES merge (via `ptr_py.ties_merge`), upcycle, reactivate | `training/src/ptr_training/moe/lifecycle_ops.py` + tests | ~500 LOC, 3–4 days | M011 path, desktop `ptr_py` build |
| C3 | Dependency graph and closure in the harness, with a conformance test against R011/R013's semantics (typed edges, planned succession vs live dependency) | `training/src/ptr_training/moe/lineage_graph.py`; later `bindings/ptr-py/src/lineage.rs` exposing the runtime's closure so the harness calls the Rust rule directly | ~300 LOC Python, 2 days; Rust binding ~300 LOC, 2 days | C2 |
| C4 | Gold runner (same topology from clean counterparts, checklist C-2), transitive (L) and direct-only (D) revocation runs with compute matching | `training/src/ptr_training/moe/revocation_eval.py` | ~450 LOC, 3 days | C1–C3 |
| C5 | NPO+RT baseline | `training/src/ptr_training/moe/npo.py` | ~200 LOC, 1.5 days | C4 |
| C6 | Experiment records: problem-existence (CPU), remedy (CPU), GPU successor | `experiments/lifecycle/<new id>/` | 2 days each | C1–C5 |

### 4.4 Order

1. **B1–B4 on CPU** (cheapest go/no-go for GPU spend; can start now, no GPU, no Rust change).
2. **C1, C2 (mixing schedules only), C4 (U, G and P only) on CPU** — a CPU-scoped H-C1 go/no-go before any remedy
   code is written.
3. **A1, A2** (dataset and parity; CPU; uses the `ptr_py` build that `scripts/setup_desktop.sh` already performs; X2 runs alongside and blocks nothing).
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
  the pseudoinverse estimator may be biased enough that a direct reward model ranks as well, in
  which case DB3 is refuted (K-B3).
- **Positivity.** Single-slot exploration only supports routers within one swap per token under the
  set-level view; the slate estimator relies on per-expert marginal support plus additivity to
  reach further. Routers that differ in many slots per token may be refused or badly estimated.
- **Cost and scope.** The claim covers one explored decision per pass at a declared layer; whether
  exploring every decision is affordable is not tested (reported only), and exploring only the
  final layer limits what can be audited.
- **Proxy competition.** If cheap routing statistics already rank routers well, logging
  propensities is unnecessary.
- **Incremental reading.** Off-policy slate evaluation is a known method (1605.04812); the
  contribution is showing that it transfers to token routing at acceptable cost, plus the measured
  answer to "how good are trained routers" without re-execution. If the answer is "it works only
  for the final layer", the contribution is small.

### Thesis C
- **No leakage to find (K-C1).** Shared-layer averaging over several experts may dilute one
  shard's influence below measurement; tiny CPU models may not memorise canaries at all.
- **Gold noise.** Two gold retrains differ by seed; if that band exceeds the 0.5-bit tolerance the
  run is invalid by rule, and matching the gold on the specified attacks is not measurable at this
  scale without a higher-precision gold.
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
  confirmatory run costs days of setup. Running all three GPU stages is about 240–385 GPU-h of
  serial card-0 confirmatory time (≈ 10–16 days) plus setup; the order in §4.4 is chosen so that
  two theses can be stopped on CPU first (a no-go for GPU spend, never a falsification).
- **Evidence-class creep.** The nearest risk is presenting a CPU harness pass as a statement about
  OLMoE. Every report keeps the evidence class next to the number.
- **Tooling gaps.** No GitHub CI job builds `ptr_py` (it is built by `scripts/setup_desktop.sh`
  and by a manual GitLab job only); the GPU path of `trl-moe` has never run; the M011 harness is
  unexercised on real weights. X0 is a prerequisite; X2 reduces the risk of an unnoticed binding
  break.

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
- **New claim 27 (thesis B)**, draft: "When one routing decision per teacher-forced pass, at a
  declared MoE layer, is logged with single-slot exploration at rate ε and exact propensities, a
  slate off-policy estimator ranks candidate routers by their per-decision value (the teacher-forced
  probability of the realized token y_t, read at input position t − 1 whose layer-ℓ routing is the
  intervened decision, all other decisions routed by the base router) in the same order
  as on-policy evaluation (Kendall τ_b lower bound ≥ 0.80), at an exploration cost of at most 1%
  relative drop in that probability per logged decision. Exploration at every routing decision is
  not claimed. Unproven."
- **New claim 28 (thesis C)**, draft, scoped to the schedules for which leakage is supported:
  "Removing an expert does not remove its training data's influence once shared layers are
  averaged, the router is fine-tuned on mixed data, or the expert is merged or upcycled; revocation
  over the dependency closure brings the model back to a retrain without the data on the specified
  attacks (canary exposure and a loss-threshold membership-inference test, both gold-relative) and
  on retained-data utility, at less than half the cost of full retraining, where approximate
  unlearning at the same compute does not. No broader indistinguishability is claimed. Unproven."
  The word "transitive" is added before "dependency closure" only if the compute-matched
  direct-only arm fails where the transitive closure succeeds (H-C4 of the agenda).

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
