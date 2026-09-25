# A0 mechanism ablation study v1: the design

This is the study design as the judge-panel workflow synthesized it (three independent designs, a validity and a feasibility judge, one synthesis), reproduced in full. `PREREGISTRATION.md` binds it to the measured references, the calibration, the budget and the frozen splits, and lists every deviation from it. Where the two differ, `PREREGISTRATION.md` and `criteria.toml` govern.

## A0 mechanism correctness and isolated ablations v1 on operator-routing v1: a preregistered study of A0's own mechanisms. It is not the M001-M004 baseline comparisons, and those manifests stay 'planned'.

## Ablation semantics

### full (the reference arm; every contrast is paired against it unless stated otherwise)

**Definition.**

PtrA0Config::new(216, 32).with_provenance_buckets(8).with_latent_steps(2). typed_attention = true uses the existing field (lib.rs:332). The three new switches keep their defaults: typed_query = true, latent_nonlinearity = true, frozen_router = false. The model is built with PtrA0Config::init. Its materialize() (lib.rs:491, commit c9a13f7) draws every parameter in declaration order.

Typed batch: slot i carries fact i (S = 6).
- slot_types: CodeGrid<SemanticRole> through Codebook::V1.
- epistemic: CodeGrid<EpistemicState>.
- confidence: the fact's continuous c.
- slot_values: SlotEncoding::V1.encode(TypeId::from('Entity'), bytes 'entity-<e>', 32).
- provenance_ids: i.
- admission: admission_bias(ValidityMask::from_validities(validities)). Only Live is admitted.

Raw tokens: the shared 216-token sequence (see task.generator).
Loss: cross-entropy on router_logits.

**Isolates.**

Nothing; this is the reference.

Code-reading finding, pinned by test T5: the raw->slot update raw' (lib.rs:799) never reaches router_logits. The latent loop (lib.rs:801) and the router (lib.rs:812) read only slots. So in every arm no gradient reaches raw_query, slot_key, slot_value, raw_output, the admission term inside raw_scores, or the transposed cross_bias. Admission reaches the output only through the router's admitted mean.

**Parameter matching.**

17,612 parameters in total:
- token embedding 216x32 = 6,912
- role 288, epistemic 192, provenance 8x32 = 256
- confidence projection 64, metadata_bias 33
- eight 32x32 attention linears 8,448
- latent_refine 1,056, router 363

13,292 of them receive gradient. The PAD row, 2 provenance rows and the 4,224-parameter dead branch are idle.

Every arm builds the same modules. For a given seed, all arms therefore start from bitwise-identical weights (test T6).

**Confounds.**

provenance_bucket_count is 8, not the recorded default of 64. lib.rs:315 allows a different width; the field is label-irrelevant here and only 6 rows are used.

latent_steps = 2 is not the crate default of 0, so the full arm sets it explicitly.

### no-semantic-slots (M001: typed content and admission both removed)

**Definition.**

No change to forward.

The batch builder keeps S = 6 and raw tokens byte-identical to the full arm, but empties every slot of typed content:
- slot_types: SemanticRole::Goal for every slot.
- epistemic: EpistemicState::Unknown.
- confidence: 0.0.
- slot_values: SlotEncoding::V1.encode(TypeId::from('SlotIndex'), bytes 'slot-<i>', 32).
- provenance_ids: i, the same as in the typed arms.
- admission: admission_bias(ValidityMask::admitting_all(6)).

The six slots become learned per-index pooling queries, Perceiver-style. Each must pull fact i's attributes out of the index-tagged raw tokens through slot->raw attention. It must also learn to gate non-Live facts from their validity tokens.

**Isolates.**

Two ways of delivering the same fact attributes:
- already bound into typed slots, with the hard admission mask;
- only as index-tagged raw tokens, which A0 must bind itself through one slot->raw attention hop, learning the Live gate as well.

**Parameter matching.**

Same modules and bitwise-identical initial weights per seed. FLOPs per step are identical.

12,844 parameters receive gradient (-448):
- 8 role rows and 5 epistemic rows are unused;
- the confidence weight is idle because its input is 0.

**Confounds.**

(1) The arm loses the hard admission mask as well as slot content. The masked control separates the two.
(2) The arm has no payload vectors, so ood_payload cannot affect it (by construction).
(3) Confidence is asymmetric only in the slot vector. Raw tokens are identical across arms, so the typed arms receive both the continuous c in the slot and the 5-bucket token in raw. They can reach the token through slot->raw attention, because provenance = i tells a slot its index. This arm gets only the bucket token, which is exactly what the label rule uses. This favours neither arm informationally; it is an inductive-bias difference.
(4) In typed arms the slot query depends on slot content, including payload noise. In this arm it is content-free.
(5) Index-tagged tokens make binding far easier than natural text. So a typed-slot win here is strong evidence, and a null is weak evidence.
(6) The typed bias stays active: its metadata is a per-index constant, so it acts as a learned per-slot scalar times the key mean.

### no-semantic-slots-masked (M001 control: typed content removed, admission kept)

**Definition.**

Identical to no-semantic-slots except admission = admission_bias(ValidityMask::from_validities(fact validities)). Slot i is excluded from the router's admitted mean exactly as in the typed arms.

**Isolates.**

Typed slot content (role, epistemic state, continuous confidence, payload) with the admission mechanism held equal.

The mask's own effect is measured, without a verdict, as the difference between the masked and unmasked arms on the validity-decisive subset and on ood_validity.

**Parameter matching.**

Same as no-semantic-slots: 12,844 parameters receive gradient.

**Confounds.**

Confounds (2) to (6) of no-semantic-slots.

### no-typed-attention (M002: necessity given QK)

**Definition.**

Reuses the existing switch PtrA0Config::with_typed_attention(false) from commit c9a13f7 and its tests. When it is false, cross_bias is zero in both directions (lib.rs:745-749) and metadata_bias is built but never applied. Because raw->slot is dead (see full), only the slot->raw use can change the output.

**Isolates.**

The rank-1, unscaled, metadata-only score term beta(metadata_s) * mean_features(raw_key_t) on slot->raw attention. The QK score stays intact. It already sees metadata, because slots = slot_values + typed_metadata feeds slot_query.

**Parameter matching.**

13,259 parameters receive gradient (-33: metadata_bias). FLOPs are the same apart from one [B,6,1]x[B,1,T] product.

**Confounds.**

The QK path can express the same conditioning: a rank-up-to-32 bilinear form over slots that already contain the metadata. A null result therefore means 'redundant given QK', not 'metadata-directed attention is useless'; the blind-query pair tests the latter.

The removed term is not scaled by 1/sqrt(D), so an effect mixes metadata conditioning with sharper logits.

### raw-blind (M002 manipulation check; not in ablations.toml)

**Definition.**

No model change. The batch builder replaces every raw token id with PAD (0) at the same length T. Typed slots are the same as in the full arm.

**Isolates.**

Whether the full arm actually reads raw context. Regime and budget exist only in raw, so this shows whether the slot->raw path, and with it typed attention, is exercised at all.

**Parameter matching.**

Same modules. 6,444 parameters receive gradient: 215 token rows are unused.

**Confounds.**

It removes information by construction. Its exact Bayes ceiling is 0.815 on the prototype IID split and is recomputed on the frozen split. That ceiling uses the exact posterior over (R, B) given every slot field, including the cue that an Evidence fact rules out the interventional regime in training-type splits.

### typed-attention sufficiency pair (M002, tier 2): blind-query-k0 and blind-query-k0-no-typed-attention

**Definition.**

A new switch, PtrA0Config::with_typed_query(bool), defaults to true. When false, forward computes slot_query = self.slot_query.forward(slot_values.values()) (the payload hash only) instead of forward(slots) at lib.rs:741. Typed metadata still enters the slot state that slot_output adds to, the router's input and metadata_bias. Both arms use K = 0.
- blind-query-k0: typed_attention on.
- blind-query-k0-no-typed-attention: typed_attention off.

**Isolates.**

Whether the rank-1 typed bias alone can learn to steer reads by metadata. In blind-query-k0 it is the only route by which a slot's role can change which raw tokens it reads.

Without it, K = 0 and a linear router make each slot's vote additive in its metadata and a role-independent context. Role-gated use of the regime (the regime changes only Claim and Evidence votes) then becomes impossible.

**Parameter matching.**

12,236 parameters receive gradient with the typed bias on; 12,203 with it off. The two arms differ only in metadata_bias.

**Confounds.**

The payload-hash query adds noise, but a query bias can make it effectively constant.

A rank-1 bias can only favour the tokens with the largest (or smallest) key mean. It can therefore implement 'read the regime token' but not an arbitrary lookup.

This pair is tier 2: the budget rule drops it first when time is short.

### no-latent-recurrence, as latent-0 (M003)

**Definition.**

The existing switch with_latent_steps(0): the refinement loop never runs. latent_refine is still built, so initial weights are identical to the full arm.

**Isolates.**

The per-slot nonlinearity after attention: slots + gelu(W_l slots), applied K times with tied weights. Apart from the slot->raw softmax, it is the only per-slot nonlinearity in A0.

**Parameter matching.**

12,236 parameters receive gradient (-1,056: latent_refine idle). About 12 fewer 32x32 per-slot matmuls per example, under 3% of step FLOPs.

**Confounds.**

The arm removes capacity as well as recurrence, so part of any deficit is by construction. The latent-linear control separates the gelu from the extra parameters.

### latent-linear (M003 attribution control; diagnostic D1)

**Definition.**

A new switch, PtrA0Config::with_latent_nonlinearity(bool), defaults to true. When false, the loop body is slots = slots + latent_refine(slots), with no gelu. K = 2.

**Isolates.**

With a linear router, the whole post-attention path becomes affine, so the arm has the same function class as latent-0 in a different parameterisation.
- If latent-linear is close to full, a latent-0 deficit comes from the extra parameters or affine depth (optimisation).
- If latent-linear is close to latent-0, the deficit needs the gelu.

**Parameter matching.**

13,292 parameters receive gradient, identical to the full arm. FLOPs are the same apart from the skipped gelu.

**Confounds.**

Affine depth changes the optimisation geometry (it is a reparameterisation), so this separates 'nonlinearity' from 'parameters' only up to optimisation effects.

### latent depth: latent-1 (tier 1) and latent-4 (tier 2) (M003: recurrence proper)

**Definition.**

with_latent_steps(1) and with_latent_steps(4). Everything else is as in the full arm, which uses K = 2.

**Isolates.**

How many times the same tied refinement is applied, at exactly equal parameters and equal initial weights. In A0 this is the only clean test of recurrence as such.

**Parameter matching.**

13,292 parameters receive gradient in both. FLOPs differ by at most 2 per-slot 32x32 matmuls per example.

**Confounds.**

Deeper tied iteration also changes optimisation, because gradients pass through repeated gelu residuals. A deficit at K = 4 could be instability rather than a property of recurrence.

### no-router, as frozen-router (M004: a negative control, not a mechanism test)

**Definition.**

The literal ablation cannot be run: router_logits is A0's only output head.

A new switch, PtrA0Config::with_frozen_router(bool), defaults to false. When it is true, PtrA0Config::init applies model.router = model.router.no_grad() after materialize(). It must never be applied inside init_lazy: calling no_grad on a lazy parameter would force its draw early and shift every later parameter, and test T6 catches that. GradientsParams then never contains the router, and Adam never updates it. forward is unchanged.

M004's 'deterministic router' becomes reference lines computed by references.py, not an arm: hand-primary, hand-weighted and the count router.

**Isolates.**

Whether learning the 11x32 readout matters when everything upstream is learned. By reparameterisation the 32-dimensional slot state can adapt to any full-rank fixed readout, so the predicted result is equivalence. The arm is therefore a negative control on the paired decision rule and on optimisation fairness.

**Parameter matching.**

12,929 parameters receive gradient (-363). The upstream network is identical.

**Confounds.**

A fixed random readout scale can slow early training.

The deterministic references are not trained per seed. hand-weighted encodes part of the generator (primary affinities, epistemic weights, confidence gains), so it transfers to ood_compose_regime by construction.

A raw-stream readout head (router over the mean of raw'), as the coverage design proposed, would change the function class and bring the dead branch to life. It is deferred to a follow-up study.

### no-verifier-head

**Definition.**

Not applicable: PtrA0 has no verifier head (no module and no output). No arm is run, and model/configs/ablations.toml gains a0_status = 'not-applicable' for this entry.

**Isolates.**

Nothing.

**Parameter matching.**

Not applicable.

**Confounds.**

It must never be reported as a null result. It is recorded as 'not tested'.

## Task

### Generator

OPERATOR-ROUTING V1
benchmarks/operator-routing/generator.py uses only the Python stdlib. generator_version = 1; benchmark seed 20260925.

PRNG
- Classic splitmix64: state += 0x9E3779B97F4A7C15, and the output is the finalizer used at crates/ptr-types/src/slot_encoding.rs:301.
- Example j of split s draws from a stream seeded with splitmix64(20260925 ^ fnv1a64(s) ^ splitmix64(j)).
- u01 = (next >> 11) / 2^53; randbelow(n) = floor(u01 * n).
- Python's random module is not used.

CODEBOOK V1 NAMES
- Roles r: Goal, Constraint, Claim, Evidence, Resource, Capability, Relation, Procedure, Action.
- Epistemic states e: Unknown, Assumed, Hypothesis, Observed, Inferred, Verified.
- Operators k: Semantic, Deductive, Probabilistic, Statistical, Temporal, Causal, Search, Optimization, Simulation, Symbolic, ExternalPod.

SAMPLING ONE EXAMPLE
(1) Regime R = randbelow(4), one of tabular, temporal, interventional, textual. In ood_compose_regime, R = interventional. Budget B = randbelow(3), one of 0.4, 0.7, 1.0.
(2) ban = (R == interventional and split != ood_compose_regime).
(3) Two focus roles, each ROLE_PRIOR[randbelow(11)], redrawn while ban holds and the role is Evidence. ROLE_PRIOR = [Goal, Constraint, Claim, Claim, Evidence, Evidence, Resource, Capability, Relation, Procedure, Action].
(4) Six facts, i = 0..5. For each fact, repeat until accepted:
  - role = focus[randbelow(2)] if u01 < 0.5, else ROLE_PRIOR[randbelow(11)];
  - e = randbelow(6);
  - reject if (role, e) is in HELD_OUT (this check is skipped in ood_compose_epi), or if ban holds and role is Evidence.
  Then:
  - confidence c = (randbelow(1000) + 0.5) / 1000, written with 4 decimals, so the bucket cb = floor(5c) is never ambiguous;
  - the fact is Live iff u01 < p_live, where p_live = 0.78 (0.50 in ood_validity); otherwise validity = [Superseded, Revoked, Disputed][randbelow(3)];
  - entity = randbelow(256), plus 256 in ood_payload.
(5) Reject the whole example if no fact is Live. Apply the split acceptance rules listed under splits.

CHANGE FROM THE SKEPTIC DRAFT
The draft forced Evidence to be non-Live under the interventional regime. It is now never drawn there. That lets no arm learn an 'Evidence under interventional -> zero vote' shortcut, and in particular the unmasked no-semantic-slots arm cannot. p_live moved from 0.85 to 0.78, which restores the validity-decisive rate: it measured 0.144 at 0.85 and 0.208 at 0.78.

HELD_OUT (role, e) pairs
(Goal, Verified), (Constraint, Hypothesis), (Claim, Observed), (Evidence, Assumed), (Resource, Inferred), (Capability, Unknown), (Relation, Verified), (Procedure, Hypothesis), (Action, Observed).

TYPED SLOTS
Slot i is fact i: role and e as CodeGrids through Codebook::V1, continuous c, entity hash SlotEncoding::V1 at width 32, provenance bucket i, and validity as admission. UncertaintyKind is not used.

RAW TOKENS
The vocabulary V = 216 is shared by every arm; raw-blind replaces every token with PAD.
- 0: PAD.
- 1 + 24i + offset, for fact index i = 0..5: 144 index-tagged attribute tokens. Offsets: role 0..8, e 9..14, cb 15..19, validity 20..23. A0 has no positional encoding, so token identity is the only binding raw text can offer.
- 145..148: regime.
- 149..151: budget.
- 152..215: 64 filler tokens.

A sequence is the 24 attribute tokens, 1 regime token, 1 budget token and F fillers drawn uniformly with replacement: F = 10, T = 36; in ood_distractors F = 46, T = 72. The sequence is Fisher-Yates shuffled by the example stream.

Raw text carries every label-relevant quantity losslessly. It lacks only the continuous c, which is label-irrelevant beyond its bucket, and the payload, which is label-irrelevant.

LABEL RULE
Computed in float64, over facts in index order.
  z_k = sum over facts that are Live and have cb > 0 of
        G[cb] * (W[e] * A[r,R]_k + U[e] * 1{k = Probabilistic}) - LAMBDA * max(0, COST_k - BUDGET_B)
  label = argmax_k z_k.
Values within 1e-9 of the maximum are tied. Ties go to the lowest COST, then the lowest codebook code.

Constants
- G = [0, 0.5, 1.0, 1.0, 1.25].
- W = [0.30, 0.55, 0.80, 1.30, 1.05, 1.55].
- U = [0.35, 0.60, 0.85, 0, 0, 0].
- LAMBDA = 1.0.
- COST = [0.2, 0.3, 0.5, 0.5, 0.5, 0.6, 0.6, 0.8, 0.8, 0.4, 0.9].
- BUDGET = [0.4, 0.7, 1.0].
- A: primary operator 2.0, secondary 0.9, as follows. op(R) = (Statistical, Temporal, Causal, Semantic)[R].

| Role | Primary | Secondary |
|---|---|---|
| Goal | Search | Optimization |
| Constraint | Optimization | Symbolic |
| Claim | Deductive | op(R) |
| Evidence | op(R) | Probabilistic |
| Resource | ExternalPod | Search |
| Capability | Simulation | ExternalPod |
| Relation | Causal | Deductive |
| Procedure | Symbolic | Simulation |
| Action | ExternalPod | Temporal |

The rule is a sum over admitted facts of one vector per fact, which is the form of A0's admitted-mean router, so every neural arm can represent it in principle. It needs multiplication within each fact (confidence x epistemic x role/regime) and role-dependent use of raw context: the regime changes only Claim and Evidence votes, and the budget penalises only usable facts.

RECORDS
The JSONL extends datasets/samples/operator_route.jsonl, with operator names in the schema's snake_case. Each record has:
- id, split, generator_version;
- task: rendered English, for future text baselines;
- routes: all 11 operators with target = softmax(z / 0.5);
- cost_budget; type_codebook_version = 1 (integer); type_codebook_fingerprint = 2b6f8175...;
- regime, budget;
- slots[6] = {index, role, epistemic, confidence, confidence_bucket, validity, entity, provenance_bucket};
- raw_tokens;
- label {operator, code}; utility[11]; margin (top-1 z minus top-2 z);
- tags.

Tags record which counterfactual changes the label:
- validity: all facts are admitted;
- regime: another R;
- budget: another B;
- confidence: G is set to 1;
- epistemic: W is set to 1 and U to 0;
- heldout (ood_compose_epi): facts with held-out pairs are removed;
- transfer (ood_compose_regime): the label differs from all four no-transfer treatments of Evidence (ignored, or voting as tabular, temporal or textual).

TSV TWIN
The Rust example reads this, so model/burn-a0/Cargo.toml and Cargo.lock stay unchanged. Each line is:
  id <TAB> label_code <TAB> R <TAB> B <TAB> comma-separated tokens <TAB> facts as 'role,e,c,validity,entity;...' in codebook codes.
The Rust loader re-derives every label from the parsed fields with its own implementation of the rule, and refuses to run on any mismatch. It also checks an FNV-1a-64 over the TSV files against a literal pinned in the entrypoint.

PROTOTYPE MEASUREMENTS
This synthesizer re-ran the scratch prototype with these exact constants, using Python's random module, n = 3000 per split, SE about 0.009. Scripts: scratchpad/synth/gen2.py, refs2.py.
- IID labels: majority class 0.200; smallest class 0.053; exact ties 1.0%; margin < 0.1 in 7.0%.
- IID decisive rates: validity 0.208, regime 0.221, budget 0.192, confidence 0.268, epistemic 0.271.
- Mean Live facts: 4.68.

### Splits

Every split is fixed by benchmark seed 20260925 and regenerated deterministically into datasets/generated/operator_routing_v1/, which is gitignored. The SHA-256 of every JSONL and TSV file is pinned in benchmarks/operator-routing/splits.lock.json, and a test asserts it.

IN-DISTRIBUTION SPLITS
- train: 16,000.
- val: 2,000, IID. Used only for pre-freeze calibration, the post-freeze lr sweep and learning curves.
- test_iid: 3,000.

OOD TEST SPLITS
Five splits of 3,000 each. Each changes exactly one factor relative to train.

(1) ood_compose_epi. The 9 held-out (role, e) pairs, which never occur in train, val or test_iid, now occur. An example is accepted only if at least one Live fact with cb > 0 carries a held-out pair. Prototype: ignoring those facts scores 0.650, so about 35% of examples are heldout-decisive.

(2) ood_compose_regime. R is interventional and Evidence is allowed. This (Evidence, interventional) cell never occurs in any training-type split. An example is accepted only if at least one Live Evidence fact has cb > 0.
- Evidence's interventional vote (Causal) can only be inferred by factoring: role -> 'read the regime', and regime -> direction. The interventional direction is learned from Claim's secondary vote.
- Prototype: regime-decisive rate 0.57.
- The strict transfer subset is 43% of the split, and 98.8% of its labels are Causal. Every no-transfer reference scores 0 on it by construction. In training, Causal is the label of 9.6% of interventional examples.

(3) ood_distractors. Fillers go from 10 to 46 (T from 36 to 72); facts are IID.

(4) ood_validity. p_live goes from 0.78 to 0.50, so mean admitted facts drop from 4.68 to 3.0. Prototype: exact ties 4.1%, validity-blind rule 0.601.

(5) ood_payload. Entity ids come from the disjoint pool 256..511, so the payload hashes are unseen. Only typed arms can be affected.

S = 6 in every split. The slot count is deliberately not shifted: index-tagged tokens for i >= 6 would be untrained, and the raw-only arms would fail by construction.

### Information accounting

SIGNALS EACH ARM CAN ACCESS
- full, no-typed-attention, latent-0, latent-1, latent-4, latent-linear, frozen-router: typed slots (role, e, continuous c, payload hash, provenance = index, admission mask) plus all raw tokens (every attribute index-tagged, regime, budget, fillers).
- blind-query arms: the same inputs, but the slot query sees only the payload hash.
- no-semantic-slots: raw tokens only. Slots carry index identity; all are admitted.
- no-semantic-slots-masked: raw tokens plus the admission mask.
- raw-blind: typed slots only, with no regime and no budget.

TYPED-ONLY ADVANTAGES, DECLARED
- Continuous c in the slot. The bucket token is in raw for every arm, and c adds nothing to the label beyond its bucket.
- The payload hash, which is label-irrelevant: it can only hurt, and ood_payload measures that.
- The hard exclusion of non-Live facts. Validity itself is in raw for every arm.
- Pre-bound fact-to-slot alignment. The raw-only arms must learn it from the index tags.

NO ARM IS HANDED THE LABEL
Slots hold the same attributes raw text holds, and the label needs the per-fact products plus the cross-fact sum. The best predictor with no per-fact products (bound additive linear) scores 0.651.

CEILINGS
- Every arm except raw-blind has all label-relevant inputs, so its Bayes ceiling is 1.0. About 7% of IID examples have margin < 0.1 and about 1% are exact ties, so the practical ceiling is about 0.97.
- raw-blind: the exact Bayes ceiling is 0.815 on the IID prototype, with the posterior over (R, B) enumerated exactly. It is recomputed on the frozen split.

REFERENCE LINES
Prototype values with the final constants, recomputed on the frozen splits and written into the preregistration before any run.

| Reference | test_iid | compose_epi | compose_regime | validity |
|---|---|---|---|---|
| train-majority | 0.200 | 0.238 | 0.069 | 0.195 |
| count router (naive Bayes) | 0.338 | 0.306 | 0.431 | 0.339 |
| unbound bag-of-attributes linear | 0.583 | 0.575 | 0.340 | 0.508 |
| bound additive linear | 0.651 | 0.613 | 0.377 | 0.652 |
| hand-primary | 0.652 | 0.648 | 0.607 | 0.702 |
| hand-weighted | 0.812 | 0.790 | 0.783 | 0.853 |
| validity-blind rule | 0.792 | 0.826 | 0.822 | 0.601 |
| raw-blind exact Bayes (training posterior) | 0.815 | 0.834 | 0.449 | 0.851 |

No-transfer references:
- compose_epi: ignore held-out facts 0.650; treat them as Inferred 0.835.
- compose_regime: ignore Evidence 0.468; Evidence as tabular 0.496, as temporal 0.465, as textual 0.479. The share of Causal labels is 0.534.

PRIOR EXPECTATIONS ON test_iid (not criteria)
- full: 0.90 (0.85-0.97).
- no-semantic-slots-masked: 0.88 (0.70-0.97).
- no-semantic-slots: 0.86 (0.65-0.96).
- no-typed-attention: full +/- 0.02.
- raw-blind: 0.78, at most 0.815.
- latent-0: 0.75 (0.62-0.88).
- latent-linear: latent-0 +/- 0.03.
- latent-1: 0.88.
- latent-4: 0.88 (0.75-0.97).
- blind-query-k0: 0.68.
- blind-query-k0-no-typed-attention: 0.63.
- frozen-router: full +/- 0.01.

Prior expectations for the full arm on OOD splits:
- compose_epi: 0.82.
- compose_regime: 0.55, with 0.10-0.30 on the strict transfer subset.
- distractors: 0.85.
- validity: 0.85.
- payload: IID +/- 0.01.

BY CONSTRUCTION (verified, not discovered)
- Typed arms beat unmasked no-semantic-slots on validity-decisive examples and on ood_validity; the masked control removes this.
- The no-semantic-slots arms cannot be affected by ood_payload.
- raw-blind loses on regime- and budget-decisive examples.
- latent-0 < full partly, because it has less capacity.
- latent-linear and latent-0 share a function class.
- frozen-router is about equal to full, by reparameterisation.
- hand-weighted transfers on compose_regime.
- The blind-query-k0-no-typed-attention arm cannot gate the regime by role.

GENUINELY OPEN
- full vs both no-semantic-slots arms on the OOD composite, including the possibility that the raw-only arms win.
- Whether latent-linear sits near full or near latent-0.
- latent-1 vs latent-2 vs latent-4 at equal parameters.
- Whether the typed bias is necessary given QK (no-typed-attention).
- Whether it is sufficient when it is the only metadata-dependent read (blind-query pair).
- Whether any arm transfers compositionally above the no-transfer references (transfer and heldout subsets).
- Robustness to 4.6x fillers (46 instead of 10).
- Learning speed (validation-curve AUC).

### Exercises

no-semantic-slots and no-semantic-slots-masked
- Binding each fact's four attributes, because the rule multiplies within a fact. IID decisive rates: confidence 0.268, epistemic 0.271, validity 0.208.
- Composition of role and epistemic state (compose_epi).
- A new role x regime cell (compose_regime).
- Binding under 4.6x filler dilution (distractors).
- Learning the Live gate from validity tokens (unmasked arm only).

no-typed-attention
Context that exists only in raw and whose relevance depends on slot metadata:
- the regime changes only Claim and Evidence votes: regime-decisive 0.221 IID, 0.57 compose_regime;
- the budget penalty applies only to facts with cb > 0: budget-decisive 0.192;
- 4.6x dilution in distractors.
The task exercises the job typed attention exists for, but does not require it, because QK already sees metadata. A null result is predicted.

blind-query pair
The same role-gated regime reading. Here QK is blinded to metadata and K = 0, so the rank-1 bias is the only route to it. Primary subset: regime-decisive test_iid, about 660 examples.

latent-0, latent-linear, latent-1 and latent-4
The per-fact multiplicative gate G[cb] * W[e] * A[r,R] and the cb > 0 gate on the penalty.

frozen-router
No task component exercises it. The readout is linear and full rank, so no task can separate a learned readout from a fixed one. A null result is predicted, and the arm serves as a negative control.

no-verifier-head
Not exercised and not run.

raw-blind
Regime- and budget-decisive examples, where the information is removed.

Not declared ablations but measured
- The admission mask: masked vs unmasked no-semantic-slots.
- The payload channel: nuisance only (ood_payload).

Not exercised by anything
- The raw->slot branch, which is dead with respect to the output.
- UncertaintyKind, which A0 does not have.
- Provenance beyond index identity.

## Protocol

RESOLVED DISAGREEMENTS
Where the three designs and the two judgments disagree, this is how each point was settled.

1. Base design. Both judges ranked the skeptic design first, so it is the base.
   - Its task gets one fix: under the interventional regime, Evidence is now resampled rather than forced non-Live, and p_live changes from 0.85 to 0.78.
   - Its raw-blind ceiling is now the exact posterior.
   - Confound (3) is corrected.
   - The frozen router is demoted to a negative control.

2. Task. The skeptic's operator-routing task is kept.
   - Coverage's four-family suite is rejected. Its F3 and F4 families share operator sets and need opposite aggregation, so interference could void the F4 control and the gates.
   - Minimal's XNOR task is rejected. It never exercises admission, confidence or the router, and every arm is likely to hit the ceiling.

3. Learning rate. Each arm gets its own peak lr, following the skeptic design, instead of one fixed lr for all arms (coverage, minimal). Both judges found this fairer to the ablated arms. The sweep runs after the freeze and reads only val.

4. Calibration before the freeze. The skeptic design forbade any model run before the freeze. Minimal's calibration is adopted instead, because both judges flagged the risk of a study with no verdicts.
   - It uses the full arm only, seed 0, train and val only, with a revision ladder written down in advance.
   - It never looks at an ablated arm or a test split.

5. Data format. The Rust example reads a TSV twin, not serde_json. Cargo.toml and Cargo.lock stay unchanged, so every --locked command keeps working.

6. Toolchain and build directory.
   - Commands use `cargo +1.95.0`. The repository root pins 1.85.0, and ptr-burn-a0 needs 1.95; the explicit toolchain is also how the runner records rustc.
   - Builds use `--target-dir model/burn-a0/target-a0-study`, which is gitignored by **/target-*/. An `env CARGO_TARGET_DIR=` prefix is not used, because the runner then stops recording rustc.
   - RAYON_NUM_THREADS is not set. burn-flex in this lockfile has no rayon, so each process is single-threaded.

7. Process granularity. One process per (experiment, seed) runs all of that experiment's arms in sequence (minimal and coverage). One process per arm (skeptic) was rejected.
   - The parameters are identical across seeds and each seed has one completed record, so the stock `run_experiment.py aggregate` works as a cross-check.
   - Data is loaded once per process.
   - The full arm lives in M001; the custom aggregator pairs arms across experiments.
   - Arm order inside a process must not matter. Self-test T7 checks this, and so does G2.

8. Router. The frozen router is kept as a negative control. Coverage's raw-readout head is deferred, because it swaps the head, changes the function class and brings the dead branch to life.

9. Typed attention. Two contrasts are kept: necessity (no-typed-attention, from the skeptic design) and sufficiency (the blind-query pair from coverage). The sufficiency pair is tier 2.

10. Latent recurrence. latent-0, plus the latent-linear attribution control from coverage, plus the latent-1 and latent-4 depth controls from the skeptic design.

11. Admission. Coverage made admission equal across arms by not rendering non-Live facts. Instead, both a masked and an unmasked no-semantic-slots arm are kept, so the mask effect is measured rather than hidden.

12. Falsification. It needs a confidence interval, following the skeptic design. Minimal's rule used only the mean.

13. Correctness suite (from coverage). The determinism rerun uses a separate entrypoint key (from minimal). The existing with_typed_attention switch and materialize() are reused, not re-added.

MODEL
- Base config: PtrA0Config::new(216, 32).with_provenance_buckets(8).
- D = 32, S = 6, T = 36 (72 on ood_distractors), one block, codebook V1, SlotEncoding V1 at width 32.
- Backend: Device::flex().autodiff().
- Arm switches:
  - typed_attention: existing.
  - latent_steps: existing.
  - typed_query: new, default true.
  - latent_nonlinearity: new, default true.
  - frozen_router: new, default false, applied after materialize().

LOSS AND OPTIMISER
- Loss: mean cross-entropy on router_logits against the hard label.
- Optimiser: AdamConfig::new() defaults (beta 0.9/0.999, eps 1e-5), no weight decay, no clipping.
- Batches: 128, without replacement. The epoch-e permutation is a Fisher-Yates shuffle from splitmix64(order_seed ^ e). 125 steps make one epoch.
- Steps: S*, fixed by calibration from {1500, 2000, 3000}; 2000 is expected.
- Schedule: linear warm-up over 100 steps, then cosine decay to 0.1x peak at step S*.
- No early stopping. The final parameters are evaluated.

PRE-FREEZE CALIBRATION
It is outcome-blind: full arm only, seed 0 (not a declared seed), lr 0.005, train and val only. It runs as a direct cargo invocation, not through the runner, and its output is committed in the preregistration.
- Candidates: S in {1500, 2000, 3000}, each as its own run with its own schedule, in that order.
- S* is the first candidate whose final val accuracy is at least 0.88.
- If none reaches 0.88, the revision ladder is tried in order, repeating the S candidates each time:
  - R1: d_model 48.
  - R2: G = [0, 1, 1, 1, 1] and U = 0, keeping d_model 48.
- If R2 also fails, no ablation is run. The study is filed in research/falsification as the negative capacity finding 'A0 does not learn operator-routing v1 within 3000 steps'.
- Every calibration run and each ms/step is recorded in calibration.json.

BUDGET RULE
It is applied once, before the freeze, from calibration timing only. The result is written into budget.json and the preregistration.
- Projected wall time W = [N_eval * (S* * t + 4 s) + N_sweep * (S_sweep * t + 4 s)] / 3 workers, where t is the largest ms/step measured in calibration.
- Start with every arm (tier 1 plus tier 2) and S_sweep = S*. While W > 20 minutes, apply the next step in order:
  1. drop latent-4;
  2. drop the blind-query pair;
  3. set S_sweep = S* / 2;
  4. skip the sweep: every arm uses lr 0.005, flagged 'no per-arm lr selection'.
- If W is still above 45 minutes, stop and redesign.
- Tier 1 (always run): full, no-semantic-slots, no-semantic-slots-masked, no-typed-attention, raw-blind, latent-0, latent-1, latent-linear, frozen-router.
- Tier 2: blind-query-k0, blind-query-k0-no-typed-attention, latent-4.

LR SELECTION
It runs after the preregistration commit and before any evaluation.
- One sweep process per (experiment, lr) at runner seed 17. Inside the binary, init_seed = 17 ^ 0x5EE9 and order_seed = init_seed ^ 0x0BA7C4, so no evaluation run reuses a sweep trajectory.
- Grid: {0.002, 0.005, 0.0125}. Final val accuracy is the only thing measured.
- A NaN loss makes that lr ineligible.
- For each arm, pick the smallest lr whose final val accuracy is within 0.005 of the best. A pick at a grid edge is flagged, not re-swept.
- The selection goes into research/falsification/A0-ablations-v1/lr_selection.tsv (and a .json copy). It is committed alone with the sweep records, as the lr-selection commit.

EVALUATION
- Validation: all 2,000 val examples every S*/10 steps, with the mean train loss over the window before each checkpoint.
- After step S*: every test split, in forward-only batches of 500 on model.valid().
- Prediction is the argmax of the logits; exact ties go to the lowest code.

SEEDS AND PAIRING
- Seeds: 17, 29, 43, 71, 101. init_seed = seed; order_seed = seed ^ 0x0BA7C4.
- device.seed(init_seed) is called immediately before each arm's init. Every arm builds the same modules, so within a seed all arms have bitwise-identical initial weights and see the same batch order.
- The dataset is fixed across seeds, so the variance measured is init and order variance only.

EXECUTION
Everything goes through `scripts/run_experiment.py`. The M001-M004 manifests stay 'planned', so check_research_gates.py is unaffected. Each manifest gains the following keys in the preregistration commit, with S*, the arm lists and the data FNV written as literals.

1. a0_ablation_entrypoint:
   "cargo +1.95.0 run --release --locked --offline --quiet --target-dir model/burn-a0/target-a0-study --manifest-path model/burn-a0/Cargo.toml --example a0_ablation -- --phase eval --experiment M00x --arms <literal list> --seed <seed> --steps <S* literal> --lr-file research/falsification/A0-ablations-v1/lr_selection.tsv --data datasets/generated/operator_routing_v1 --data-fnv64 <literal>"
   The only placeholder left for the runner is <seed>.
2. a0_sweep_entrypoint: the same with --phase sweep and --lr <lr>. It is run with `--seed 17 --set lr=...`.
3. a0_contingency_entrypoint: eval with --steps 4000 and --arms full,<arm>. It is run with `--set arm=...`.
4. M001 only, a0_rerun_entrypoint: eval with --arms full. It is run once, at seed 17.

Arms per experiment:
- M001: full, no-semantic-slots, no-semantic-slots-masked.
- M002: no-typed-attention, raw-blind, blind-query-k0, blind-query-k0-no-typed-attention.
- M003: latent-0, latent-1, latent-linear, latent-4.
- M004: frozen-router.

Work runs as 3 concurrent worker processes, longest first; one core stays free. The release binary is built once, before the first run. The concurrent `cargo run` calls then only take the target-dir lock.

RECORDED PER RUN
Stdout carries one JSON object per line. String fields name the row and numeric fields are its metrics:
- a data row: the data FNV-64 and the per-split label FNV-64, as strings;
- a meta row per arm: switch strings, the lr used, total and effective parameters (effective = non-zero gradient over the first 10 batches), final train loss, a NaN flag;
- a val row per arm and checkpoint: checkpoint number as a string; val_accuracy and train_loss;
- a final row per arm and split: n, correct, accuracy, nll, ece15.

Stdout also carries non-JSON lines `PRED <arm> <split> <one hex digit per example>` for the six test splits. Timing, ms/step and cargo messages go to stderr only, so stdout is byte-comparable. The runner adds git sha, dirty flag, host facts, the hardware profile contents, manifest sha, command, rustc and duration.

ORDER OF COMMITS
1. Implementation commits. The golden-logit test is committed before lib.rs changes.
2. The pre-freeze phase, run from a clean commit: generate the data, data gates G0, references, build, self-test, calibration, budget rule.
3. Preregistration commit: PREREGISTRATION.md, criteria.toml, references.json, calibration.json, budget.json, splits.lock.json, the manifest entrypoint keys and the hardware profile pointer. It is tagged a0-ablation-prereg-v1.
4. Sweep, run from the preregistration commit, then the lr-selection commit, which may touch only lr_selection.* and the sweep run records.
5. Evaluation, the rerun and any contingency runs, all from the lr-selection commit with a clean worktree.
6. Aggregate, then the results commit.
Amendments are allowed only before the preregistration commit.

## Metrics

PER (ARM, SEED, SPLIT)
- route_accuracy (primary): the mean of [argmax router_logits == label code].
- cost_adjusted_regret: the mean of (z_label - z_pred), in rule utility units. 0 is perfect; cost overrun is included.
- task_success: the mean of [z_label - z_pred <= 0.25].
- Mean NLL, and ECE over 15 equal-width bins of the maximum softmax probability.
- Subset accuracies:
  - each decisive tag on test_iid and on every OOD split: validity, regime, budget, confidence, epistemic;
  - the heldout subset of ood_compose_epi;
  - the strict transfer subset of ood_compose_regime;
  - the clear-margin subset (margin >= 0.25).
- Validation-curve AUC: mean val accuracy over the 10 checkpoints, as a measure of learning speed.
- Effective parameter count. ms/step is recorded as a diagnostic only.

COMPOSITE ENDPOINTS
Unweighted means of split accuracies, fixed per contrast in criteria.toml:
- A = {ood_compose_epi, ood_compose_regime, ood_distractors}.
- B = {ood_distractors, ood_compose_regime}.
- C = {test_iid, ood_compose_epi, ood_compose_regime}.

CONTRASTS
- Per seed s, delta_s = metric(comparator) - metric(ablated), paired on seed. The ablated arm has the same initial weights and batch order.
- Reported for each contrast: the five delta_s values, the mean m, the sample SD, min, max, the number of positive seeds, and the 95% t-interval m +/- 2.776 * SD / sqrt(5).
- Secondary, descriptive only: a per-example McNemar test within each seed, and a within-seed bootstrap (1,000 resamples, fixed seed).

COMPUTATION
- The Rust binary prints counts. benchmarks/operator-routing/score.py recomputes every accuracy, subset and regret value from the committed PRED strings and the regenerated splits. Rust and Python counts must match exactly (gate G5).
- scripts/aggregate_a0_ablation.py computes the statistics in float64 and compares against thresholds with no rounding.
- The stock `run_experiment.py aggregate` output per experiment must equal the custom aggregator's per-arm means to 1e-12.

REFERENCES
references.py computes these on the frozen splits before the freeze. None of them needs a model.
- train-majority;
- count router (naive Bayes over (role, R), e and cb of Live facts, plus R and B; Laplace alpha 1);
- unbound bag-of-attributes and bound additive multinomial logistic regressions (stdlib SGD, seed 1, 5 epochs);
- hand-primary: Live facts with cb > 0 vote for primary(role, R);
- hand-weighted: the same votes weighted by G[cb] * W[e];
- the validity-blind rule;
- the raw-blind exact Bayes ceiling (the posterior over R, B given all slot fields, enumerating 4 regimes x 3 budgets x 81 focus-role pairs);
- the no-transfer references.

The benchmark suite's metric names (route_accuracy, cost_adjusted_regret, task_success) become defined, implemented metrics in suite.toml.

## Preregistered criteria

### COMMON DECISION RULE (applies to every verdict-bearing contrast; thresholds are fixed now and stored in criteria.toml)

**Supports mechanism if.**

Let delta_s = accuracy(comparator) - accuracy(ablated) on the contrast's primary endpoint for seeds 17, 29, 43, 71 and 101. Let m be the mean, sd the sample SD, and CI = m +/- 2.776 * sd / sqrt(5).

SUPPORTS if all of the following hold:
(i) m >= delta_min, where delta_min = 0.02 unless the contrast says otherwise;
(ii) the CI lower bound is > 0;
(iii) delta_s > 0 for all 5 seeds;
(iv) both arms pass the learnability criterion;
(v) gates G0-G6 pass and no leak alarm is active.

Comparisons are exact (>=, >, <) on float64 values computed from exact counts. Nothing is rounded before comparison.

**Falsifies mechanism if.**

FALSIFIES if the CI upper bound is < delta_min and preconditions (iv) and (v) hold. The same result is also labelled HARMFUL when the CI upper bound is < 0, meaning the ablated arm is better. SUPPORTS and FALSIFIES cannot both hold.

**Inconclusive if.**

Any other outcome, or any failed precondition. The recorded verdict names the reason.

After the preregistration commit, criteria.toml cannot change. aggregate_a0_ablation.py refuses to run if its SHA-256 differs from the file at tag a0-ablation-prereg-v1.

### STUDY-WIDE GATES (checked mechanically before any verdict)

**Supports mechanism if.**

The gates issue no verdicts; they decide whether verdicts can be issued. All of the following must pass.

G0, data:
- split SHA-256 values equal splits.lock.json;
- every run's data FNV and per-split label FNV match the pinned values;
- the Rust label re-derivation and score.py's independent re-derivation each agree with generator.py on 100% of records;
- the pre-freeze data bands held: test_iid majority <= 0.25; every operator >= 0.03 in train; each IID decisive rate in [0.12, 0.40]; heldout-decisive on compose_epi >= 0.25; regime-decisive on compose_regime >= 0.40; IID exact ties <= 0.02; hand-weighted in [0.74, 0.88]; bound additive in [0.58, 0.72]; raw-blind ceiling in [0.76, 0.86]; a nuisance-only predictor (entity ids, fillers) <= train-majority + 0.02.

G1, competence: the full arm's 5-seed mean test_iid >= max(0.85, raw-blind exact ceiling + 0.03).

G2, determinism: the a0_rerun_entrypoint run (M001, seed 17, full arm only) reproduces byte for byte the full arm's JSON rows and PRED lines from the original M001 seed-17 run, where that arm ran first of three.

G3, completeness: every planned evaluation process completed with finite losses. A failed or NaN process gets one full retry.

G4, freeze:
- every eval-phase record has git_dirty = false and the same git_sha;
- tag a0-ablation-prereg-v1 is an ancestor of that sha;
- the diff from the tag to that sha touches only lr_selection.* and the sweep run records;
- criteria.toml and PREREGISTRATION.md are unchanged.

G5, scorer agreement: for every arm, seed and split, the correct count printed by Rust equals score.py's count from the PRED string.

G6, correctness: at the evaluation commit, `cargo +1.95.0 test --locked` for ptr-burn-a0 (tests T1-T6), the binary's --phase self-test (T7, T8) and the pytest suites for benchmarks/operator-routing and the aggregator all pass. Their logs are committed.

**Falsifies mechanism if.**

If G1 fails, no mechanism verdict is issued. The recorded result is: 'A0 (D=32, one block) does not learn operator-routing v1 in S* steps'. It is filed in research/falsification/A0-ablations-v1 as a negative capacity finding.

**Inconclusive if.**

If G0, G2, G3, G4, G5 or G6 fails, every verdict that depends on an affected arm is INCONCLUSIVE until the cause is fixed. The affected processes are then re-run in full from a new commit, with a dated note; the criteria stay unchanged.

An arm that fails in the same seed twice makes every contrast involving it INCONCLUSIVE.

Leak alarm: if raw-blind's 5-seed mean on test_iid exceeds its exact Bayes ceiling + 0.02, every verdict is put on HOLD and the generator investigated.

Discarded results are kept and listed.

### LEARNABILITY (non-negotiable 1; precondition (iv) for every contrast)

**Supports mechanism if.**

Complete-path arms (no-semantic-slots, no-semantic-slots-masked, no-typed-attention, latent-1, latent-4, frozen-router, and full): the 5-seed mean test_iid must be >= the recomputed bound additive reference (prototype 0.651).

Restricted-class arms (latent-0, latent-linear, blind-query-k0, blind-query-k0-no-typed-attention, raw-blind): the 5-seed mean test_iid must be >= train-majority + 0.25 (prototype 0.450).

**Falsifies mechanism if.**

Not a mechanism criterion.

**Inconclusive if.**

If a complete-path arm falls below its bar, the contingency applies. That arm and full are re-run with a0_contingency_entrypoint at 4000 steps, with the same seeds and each arm's selected lr. The 4000-step pair then decides that contrast under the common rule, recorded as 'decided at 4000 steps'. If the arm is still below the bar, the contrast is INCONCLUSIVE: an optimisation failure, not evidence that the mechanism is needed.

A restricted-class arm below its bar is recorded as failed-to-train, and its contrasts are INCONCLUSIVE.

### no-semantic-slots-masked (M001 primary: typed slot content, with admission held equal; comparator full)

**Supports mechanism if.**

Endpoint: composite A = mean accuracy over ood_compose_epi, ood_compose_regime and ood_distractors. Common rule with delta_min = 0.02.

The verdict text is: 'typed slot content improves OOD routing over index-tagged raw binding in A0'.

Prior on record: SUPPORTS 0.35, FALSIFIES 0.30, INCONCLUSIVE 0.35.

**Falsifies mechanism if.**

The CI upper bound is < 0.02: a benefit of 2 points or more from typed slot content is excluded. If the upper bound is < 0 as well, the verdict is HARMFUL: the raw-only path is better.

**Inconclusive if.**

Any other outcome, or the learnability or contingency path fails.

Reported without a verdict: test_iid, ood_validity, ood_payload (by construction), the heldout and transfer subsets, and the validation-curve AUC.

### no-semantic-slots (M001 secondary: typed content plus admission; comparator full)

**Supports mechanism if.**

Endpoint: composite A. Common rule with delta_min = 0.02.

A share of any gap is by construction, because 0.208 of IID examples are validity-decisive and the mask handles them exactly. The report therefore states the mask share, (full - no-semantic-slots) - (full - no-semantic-slots-masked) on composite A, next to the verdict.

Prior: SUPPORTS 0.45.

**Falsifies mechanism if.**

The CI upper bound is < 0.02; HARMFUL if it is < 0.

**Inconclusive if.**

Any other outcome.

Reported without a verdict: the mask effect, accuracy(no-semantic-slots-masked) - accuracy(no-semantic-slots), on the validity-decisive subset of test_iid and on ood_validity.

### raw-blind (manipulation check for M002)

**Supports mechanism if.**

PASS (the raw path is used) if, for full - raw-blind on test_iid, m >= 0.03 and delta_s > 0 for all 5 seeds.

**Falsifies mechanism if.**

FAIL otherwise. The no-typed-attention verdict then becomes NOT EXERCISED, whatever its numbers.

**Inconclusive if.**

Never: this check always yields PASS or FAIL. The leak alarm is defined under the study-wide gates.

### no-typed-attention (M002: necessity given QK; comparator full)

**Supports mechanism if.**

Endpoint: composite B = mean accuracy over ood_distractors and ood_compose_regime. Common rule with delta_min = 0.02.

The prediction on record is a null: FALSIFIES with probability 0.55.

**Falsifies mechanism if.**

The CI upper bound is < 0.02. The verdict text is: 'the rank-1 typed bias adds nothing measurable beyond QK over metadata-bearing slots in A0'.

**Inconclusive if.**

Any other outcome. The verdict is instead NOT EXERCISED if the raw-blind check fails.

Reported without a verdict: test_iid, and the regime- and budget-decisive subsets.

### typed-attention sufficiency (M002, tier 2: blind-query-k0 vs blind-query-k0-no-typed-attention)

**Supports mechanism if.**

Endpoint: test_iid accuracy on the regime-decisive subset (about 660 examples). Common rule with delta_min = 0.03.

The verdict text is: 'when it is the only metadata-dependent read, the typed bias learns role-gated reading'.

Prior: SUPPORTS 0.40.

**Falsifies mechanism if.**

The CI upper bound is < 0.03.

**Inconclusive if.**

Any other outcome.

Blindness check: if blind-query-k0-no-typed-attention's 5-seed mean test_iid exceeds the bound additive reference + 0.05, blindness failed and the verdict is INCONCLUSIVE.

If the budget rule dropped this pair, it is recorded as 'not run (budget rule)'.

### no-latent-recurrence (M003: latent-0 vs full, with latent-linear for attribution)

**Supports mechanism if.**

Endpoint: test_iid. The verdict SUPPORTS-NONLINEARITY requires both:
- the common rule, with delta_min = 0.02, gives SUPPORTS for full - latent-0;
- it also gives SUPPORTS for full - latent-linear.
Together these mean the per-slot gelu is what is used, not the extra affine parameters.

Prior: 0.65.

**Falsifies mechanism if.**

The common rule gives FALSIFIES for full - latent-0 (CI upper bound < 0.02). The verdict text is: 'attention alone implements the per-fact gating at this scale'.

**Inconclusive if.**

If full - latent-0 gives SUPPORTS but full - latent-linear does not, the verdict is 'INCONCLUSIVE: the gain is not attributable to the nonlinearity'. Any other outcome, or a learnability failure, is also INCONCLUSIVE.

latent-linear - latent-0 is reported without a verdict; it is expected to be about 0.

### latent depth (M003: recurrence proper, full with K = 2 vs latent-1 at identical parameters; latent-4 secondary)

**Supports mechanism if.**

Endpoint: composite C = mean accuracy over test_iid, ood_compose_epi and ood_compose_regime, with delta = full - latent-1. Common rule with delta_min = 0.02.

Prior: 0.20.

Secondary (tier 2), no verdict: the dose response is called 'monotone' if latent-4 - full >= 0 on composite C in at least 4 of 5 seeds.

**Falsifies mechanism if.**

The CI upper bound is < 0.02: applying the tied refinement a second time gives no measurable benefit at equal parameters.

**Inconclusive if.**

Any other outcome.

### no-router (M004: frozen-router as a negative control; comparator full)

**Supports mechanism if.**

Endpoint: test_iid.

The predicted outcome is EQUIVALENT: the CI lies strictly inside (-0.02, 0.02). Prior 0.70.

If the common rule gives SUPPORTS, the result is recorded as 'learning the readout matters (probable optimisation artefact, not evidence for routing)'. The study summary must then add that the paired design detected a difference between two models that are equivalent up to reparameterisation, so effects near 0.02 elsewhere must be read with that in mind.

Secondary, deterministic references: 'learned routing beats the designer's deterministic router on IID' holds if the full arm's minimum over the 5 seeds on test_iid is >= hand-weighted + 0.05.

**Falsifies mechanism if.**

Not a mechanism test. HARMFUL (the frozen arm is better) is recorded if the CI upper bound is < 0. For the secondary comparison, the claim fails if the full arm's maximum over seeds on test_iid is < hand-weighted.

**Inconclusive if.**

Any outcome other than EQUIVALENT, SUPPORTS or HARMFUL. Nothing about M004's 'LLM-only + deterministic router' comparison may be stated.

### no-verifier-head

**Supports mechanism if.**

Not applicable: PtrA0 has no verifier head.

**Falsifies mechanism if.**

Not applicable.

**Inconclusive if.**

Always recorded as 'not tested: no such module in PtrA0'. It is never reported as a null result.

### DESCRIPTIVE TRANSFER STATEMENTS (no verdicts; wording fixed now)

**Supports mechanism if.**

An arm 'transfers to the new role x regime cell' if its 5-seed minimum accuracy on the strict transfer subset of ood_compose_regime is >= 0.30.

For comparison, every no-transfer reference scores 0 there by construction, and the training prior of Causal under the interventional regime is 0.096.

An arm 'transfers to held-out role x epistemic pairs' if its 5-seed minimum accuracy on the heldout subset of ood_compose_epi is >= the heldout-as-Inferred reference on that subset.

**Falsifies mechanism if.**

'Does not transfer' is recorded if the arm's 5-seed maximum on the subset is below the bar.

**Inconclusive if.**

Otherwise the statement is 'mixed'. None of these statements changes a verdict.

## Compute estimate

These are estimates, not measurements. Calibration measures ms/step before the freeze, and the budget rule is resolved from that measurement alone.

PER STEP
About 20 ms, with a range of 10-40 ms. Basis: single-threaded release build; B = 128, T = 36, D = 32, S = 6.
- Arithmetic: about 27M multiply-adds forward, and about 60M forward plus backward on the live path.
- burn-flex has no rayon in this lockfile.
- Burn's matmul transform policy keeps [128, 36, 32] x [32, 32] as 128 per-example gemm calls, because rows = 36 and merging would not make the output squarer.
- Roughly 60 elementwise ops over 147k floats each.
- The dead raw->slot branch is computed forward but has no backward.
- For scale, the existing debug pilot ran at about 7 ms/step on tiny tensors.

PER ARM-RUN AT S* = 2000
About 40 s of training, plus about 3 s for 10 validation passes of 2,000 examples and 18,000 test examples (half of them at T = 72). About 43 s in total. Each process also spends about 1 s loading 36k TSV records and re-deriving their labels.

RUN COUNTS AT S* = 2000 WITH EVERY TIER
- Sweep: 12 arms x 3 lrs = 36 arm-runs, in 12 processes.
- Evaluation: 12 arms x 5 seeds = 60 arm-runs, in 20 processes.
- Rerun: 1 arm-run.
- Total: 97 arm-runs, about 70 core-minutes, about 23 minutes on 3 workers.

Under the 20-minute budget rule at 20 ms/step:
- dropping latent-4 leaves 89 arm-runs, about 21 minutes;
- also dropping the blind-query pair leaves 73 arm-runs, about 17.5 minutes.
At 16 ms/step or less, all tiers fit in about 19 minutes. At 40 ms/step, ladder steps 1-3 bring the study to about 30 minutes. The hard cap is 45 minutes.

ONE-OFF AND CONTINGENT COSTS (not in the figures above)
- Release build of the example into model/burn-a0/target-a0-study: 5-10 minutes. It uses its own target directory, so it never touches the shared model/burn-a0/target.
- Calibration: at most 6,500 steps, about 2.2 minutes. Each revision-ladder attempt costs up to that again.
- Data generation in pure Python, 36k JSONL and TSV records: 30-60 s.
- references.py: under 2 minutes. The exact raw-blind posterior takes about 15 s; the SGD linear references about 1 minute.
- Self-test and cargo tests: a few minutes.
- Contingency, only if an arm fails learnability: 10 arm-runs at 4,000 steps, about 14 core-minutes, about 5 minutes of wall time.
- Scoring and aggregation: under 1 minute.

MEMORY AND STORAGE
- Memory: under 300 MB per process.
- Committed output: about 1.3 MB. The 21 eval-phase run records are about 20-80 KB each, dominated by PRED hex strings (18,000 characters per arm-seed). The 12 sweep records are about 3 KB each. The aggregates, a0_internal_metrics.json per experiment and results.json are each under 100 KB.
- Generated data stays gitignored.

## Threats to validity

1. Designer alignment. The same designer wrote the label rule and knows A0. The rule is a sum over admitted facts of one vector per fact, which is exactly the form of A0's router. This helps A0 as a whole and applies to every arm alike, but it means positive results show only what A0 can do on data shaped like its readout.
2. Pre-freeze tuning. The generator constants were tuned in a Python prototype for label balance and decisive rates; no A0 model was trained. The changes were: Evidence resampled under the interventional regime, and p_live moved from 0.85 to 0.78 to restore the validity-decisive rate to 0.208. The calibration run then sees the full arm's train and val accuracy and may apply a revision ladder written in advance. It never sees an ablated arm or a test split, but it is a degree of freedom, and it is disclosed with its outputs.
3. Learning-rate selection uses one salted sweep seed and a three-point grid. The chosen lr is noisy, and an edge pick is flagged but not re-swept. If the budget rule halves the sweep length, the choice leans toward larger learning rates.
4. Index-tagged raw tokens make binding far easier than natural text. A typed-slot win here is conservative; a null here does not show typed slots are useless for text.
5. The confidence representation differs between arms. The raw-only arms get exactly the bucket the rule uses; typed arms get a continuous value in the slot plus the same bucket token in raw. This is an inductive-bias difference, not an information difference.
6. The Evidence exclusion under the interventional regime gives every arm a regime cue inside the slots in training-type splits: Evidence present means the regime is not interventional. It is part of the held-out-cell design. The exact raw-blind ceiling accounts for it. In ood_compose_regime the cue contradicts the raw regime token.
7. Payloads are SlotEncoding V1 identity hashes. The study tests typed metadata and admission, not semantic slot contents.
8. The raw->slot attention branch (raw_query, slot_key, slot_value, raw_output: 4,224 parameters) never reaches router_logits in any arm. 'Matched' means matched modules and identical initial weights. Effective parameter counts differ (6,444 to 13,292) and are reported for each arm.
9. Model scale: one block, one attention head per direction, D = 32, about 13k parameters receiving gradient. Effects may not carry over to deeper or wider models.
10. Power. Five paired seeds give a 95% t half-width of 1.24 x SD(delta). If the seed-to-seed SD of delta exceeds about 0.016, a 2-point effect can be neither confirmed nor excluded, and the verdict will honestly read INCONCLUSIVE. The variance comes only from initialisation and batch order on one fixed dataset draw.
11. Multiplicity: 6 to 7 verdict-bearing contrasts, each at 95%. This is mitigated, not corrected: SUPPORTS also needs all 5 seeds positive (one-sided sign p = 0.031) and m >= delta_min. The frozen-router negative control gives an empirical check of the rule's false-positive behaviour.
12. Redundant paths inside A0 make single-switch nulls ambiguous: QK against the typed bias, and attention softmax against the latent gelu. The blind-query pair and latent-linear exist to separate 'redundant' from 'useless', but they are design choices themselves.
13. Running several arms sequentially in one process relies on device.seed() before each init leaving no other shared state. Self-test T7 and gate G2 (the full arm alone must equal the full arm run first of three) guard this.
14. Floating-point determinism depends on the gemm dispatch path on this host (AVX-512F/BW/DQ/VL/VNNI). Bitwise reproduction is guaranteed only on the recorded hardware with the same binary.
15. The CPU is shared with other work, so timing is noisy and latency is not reported as a result. Noisy timing can move the budget-rule outcome, but it cannot touch an outcome-relevant quantity.
16. Near-ties (about 7% of IID examples have margin < 0.1; exact ties are 1.0% on IID and 4.1% on ood_validity) lower every arm's achievable accuracy and add noise to paired differences.
17. The new switches (typed_query, latent_nonlinearity, frozen_router) are constants or init-time choices, not recorded in the checkpoint header, just like typed_attention and latent_steps. The study saves no checkpoints; anyone reusing the switches must carry them in their own header.
18. ood_compose_regime concentrates labels on Causal (0.534). Its accuracy mixes transfer with class-prior effects, which is why the strict transfer subset is reported separately. In training, Causal is the label of 9.6% of interventional examples.

## What this study cannot show

1. Anything about language-model quality, natural-language understanding or reasoning over real text. The task is synthetic, and A0 has no LM head.
2. M001's actual claim: typed slots against a matched plain backbone. research/baselines/plain_model is unpinned (owner decision O2). The no-semantic-slots arms are A0-internal content-free-query variants, and the manifests stay 'planned'.
3. M002's claims about hard violations and constraint retention in generation. Calibration is covered only as NLL and ECE of an 11-way router.
4. M003's claim about saving explicit reasoning tokens, because A0 generates no tokens.
5. M004's comparison against LLM-only reasoning with a deterministic router. There is no LLM. The frozen router is a negative control, and the hand routers are synthetic reference lines.
6. Anything about a verifier head, an action head or UncertaintyKind. A0 has none of them.
7. Whether learned or semantic slot encodings help. The V1 encoding is an identity hash, and payloads carry no label information here.
8. Anything about the raw->slot direction of A0's attention, which never reaches the output.
9. Generalisation to unseen slot counts, unseen vocabulary, longer compositions, real routing costs, or other data draws. The costs and affinities are invented, and there is one benchmark seed.
10. Scaling behaviour, GPU performance, or latency.
11. Anything about the two imported bundles requested by recommendation 1.2 (typed_agent_behavior_v0_2, podwire_native_protocol_v0_1). They are chat-format data with JSON-string targets and no operator labels, so this study does not use them.
12. Anything about runtime integration. A0 is not reachable from ptr-runtime (owner decision O6).
13. Even a clean positive result would show only that a mechanism helps this A0 (D=32, one block) on operator-routing v1. It would not be evidence for the PTR thesis beyond A0-internal mechanism behaviour.

## Implementation plan

Steps 1-7 are implementation. Step 8 is the pre-freeze phase and step 9 the freeze. No declared-seed model run happens before step 10.

1. Measured hardware profile (recommendation 1.4)
- Add hardware/a0-cpu-4core.toml, filled from lscpu, /proc/meminfo and uname: Intel Xeon @ 2.80 GHz, 4 vCPU, AVX2 and AVX-512F/BW/CD/DQ/VL/VNNI, 15 GiB RAM, Linux 6.18. Its fields match hardware/default.toml.
- The preregistration commit points the M001-M004 manifests' hardware_profile at this file, so the runner records its contents.

2. Benchmark suite (recommendation 1.5), in benchmarks/operator-routing/
- generator.py: splitmix64, the vocabulary, sampling, the label rule, the tags, the JSONL and TSV writers, and a --check mode against splits.lock.json. Output goes to datasets/generated/operator_routing_v1/.
- score.py: a label re-derivation written separately from the generator (it shares no code with it); route_accuracy, cost_adjusted_regret, task_success, NLL/ECE passthrough and the subset accuracies. It reads PRED lines from run records.
- references.py: every reference line, the exact raw-blind posterior, the data bands of G0, and the nuisance-leakage predictor. It writes references.json.
- suite.toml: metric definitions, split sizes, generator_version = 1, the codebook fingerprint.
- README.md: the rule table and split definitions.
- tests/: digest stability; at least 12 hand-checked rule cases, including ties and budget cases; HELD_OUT cells absent outside compose_epi; Evidence x interventional absent outside compose_regime; at least one Live fact per example; generator and scorer label agreement; the TSV/JSONL round trip; the data bands.
- Also add datasets/samples/operator_route_v1.jsonl (one record per split), datasets/cards/operator_routing_v1.md, and a datasets/registry.toml entry.
- Fix datasets/schemas/operator_route.schema.json: type_codebook_version becomes an integer, and type_codebook_fingerprint and the new fields become optional properties. The committed sample currently violates the schema. Check it with training/src/ptr_training/validate_dataset.py.
- The scratch prototype is scratchpad/synth/gen2.py and refs2.py. It uses Python's random module, so it is a reference only.

3. Golden test, committed before any lib.rs change
- model/burn-a0/tests/golden_logits.rs records router_logits as f32 bit patterns at the current HEAD (c9a13f7 behaviour) for a fixed seed and input with latent_steps 2. This is T1: every default path must stay bit-identical.

4. Model switches in model/burn-a0/src/lib.rs
- Add PtrA0Config::with_typed_query(bool), default true. When false, slot_query reads slot_values.values() (lib.rs:741).
- Add with_latent_nonlinearity(bool), default true. When false, the latent loop drops the gelu.
- Add with_frozen_router(bool), default false. It is applied in init() after materialize(), never in init_lazy.
- Reuse the existing typed_attention field and materialize(). Add no modules, so the record, the checkpoint and the ptr-runtime fixture stay unchanged.
- New tests, each RNG-sensitive test in its own test file, as seeded_init.rs already does:
  - T2 (tests/ablation_switches.rs): each non-default switch changes the logits at init, except frozen_router. For frozen_router, one Adam step leaves the router bitwise unchanged while slot_query changes.
  - T3: switch completeness. With K = 0, logits are invariant to latent_refine weights. With typed_query, typed_attention and K all off, logits(role a) - logits(role b) is the same for different raw inputs (tolerance 1e-5). The existing metadata_bias invariance test stays.
  - T4: in every typed arm config, logits are bitwise invariant to the role, epistemic state, confidence, payload and provenance of non-Live slots.
  - T5: raw_query, slot_key, slot_value and raw_output are absent from GradientsParams in every arm config. This pins the dead branch, so a future fix is visible.
  - T6 (tests/seeded_init_arms.rs): all 12 arm configs give bitwise-identical parameters after device.seed(s) and init.
- Update model/burn-a0/README.md (the switches and the dead-branch finding) and config.toml.

5. Study binary: model/burn-a0/examples/a0_ablation.rs (split into modules under examples/a0_ablation/ if it grows). No new dependencies.
- TSV loader, the FNV-1a-64 check, and the Rust label re-derivation, which halts on any mismatch.
- Three batch builders (typed, content-free, raw-blind), all built through CodeGrid, SlotValues, SlotEncoding::V1, ValidityMask and admission_bias. SlotVectors are pre-encoded once per entity or index.
- The arm table: 12 arms with their switches, tier and experiment.
- The training loop: warm-up plus cosine schedule, seeded Fisher-Yates epochs, and device.seed before each arm.
- Evaluation, NLL and 15-bin ECE.
- Phases: calibrate (seed and steps free; train and val only), sweep (val only, salted seeds), eval, list-arms (JSON), and self-test. self-test runs T7, arm-order invariance and a double run on a tiny fixture, and T8, the Rust rule on the hand-checked cases.
- Output: hand-formatted JSON rows and PRED lines on stdout; timing on stderr.
- make a0 already compiles examples, so the binary stays under clippy and fmt.

6. Configuration (recommendation 1.1)
- model/configs/ablations.toml: add a0_arm and a0_status to each entry:
  - no-semantic-slots maps to no-semantic-slots and the masked control;
  - no-typed-attention maps to no-typed-attention and the blind-query pair;
  - no-latent-recurrence maps to latent-0, latent-linear, latent-1 and latent-4;
  - no-router maps to frozen-router, as a negative control;
  - no-verifier-head is not-applicable.
- New model/configs/a0_ablation_study.toml: the arms, switches, tiers, experiment membership, hyperparameters, lr grid, S* candidates, revision ladder and budget rule.
- scripts/tests/test_a0_ablation_config.py checks that the binary's list-arms output equals both files. It is skipped when the binary is not built, and the driver runs it as part of G6.

7. Driver, selection and aggregation
- scripts/run_a0_ablation_study.py is a resumable driver with explicit phases:
  - prefreeze: generate, --check, references and G0 bands, build, self-test, calibrate, then the budget rule. It writes into research/falsification/A0-ablations-v1/.
  - sweep: 12 runner invocations on 3 workers.
  - select: writes lr_selection.tsv and .json.
  - eval: 20 processes plus the rerun, on 3 workers, longest first.
  - contingency: only if the learnability criterion triggers it.
  - aggregate.
  - It never edits criteria.toml, and it refuses to start the eval phase unless the worktree is clean and the G4 diff condition holds.
- scripts/aggregate_a0_ablation.py:
  - recompute everything via score.py;
  - check G0-G6 and the leak alarm;
  - apply criteria.toml mechanically;
  - cross-check against the stock `run_experiment.py aggregate` output;
  - write results/a0_internal_metrics.json in each of M001-M004 (deliberately not metrics.json, which stays reserved for the real baseline comparison);
  - write research/falsification/A0-ablations-v1/results.json with every verdict, its reason and its statistics.
- scripts/tests/test_a0_ablation_aggregate.py uses synthetic records to exercise every branch: SUPPORTS, FALSIFIES, HARMFUL, INCONCLUSIVE, NOT EXERCISED, EQUIVALENT, each gate failure, the leak alarm, learnability failure and the contingency path, NaN retry, and the tier-2 'not run' case.
- Confirm that run_experiment.py validate, check_research_gates.py and check_repo.py still pass.

8. Pre-freeze phase, from a clean commit
- Generate the splits and pin their digests.
- Compute the references and check the G0 bands. A failed band means a generator bug, which is fixed before the freeze, and the fix is documented.
- Build once into model/burn-a0/target-a0-study with cargo +1.95.0 --release --locked --offline.
- Run the self-test and cargo test.
- Run calibration (and the revision ladder if needed), then the budget rule.

9. Preregistration commit, committed alone and tagged a0-ablation-prereg-v1
- research/falsification/A0-ablations-v1/ receives:
  - PREREGISTRATION.md: this design, the information accounting, the priors with probabilities, the recomputed references, the calibration and budget outcomes, and the split digests;
  - criteria.toml;
  - references.json, calibration.json and budget.json.
- Also in this commit:
  - benchmarks/operator-routing/splits.lock.json;
  - the M001-M004 experiment.toml files, which gain the a0_* entrypoint keys (literals for S*, arms and FNV) and the hardware_profile pointer, with status left 'planned';
  - a scope paragraph in each M00x README: 'A0-internal ablation on synthetic operator-routing v1; not this manifest's baseline comparison';
  - the M001 pilot marked superseded in its README. Its files are kept. Re-running it today gives typed 1.0 and ablated 1.0, not the recorded 0.75.

10. Sweep through the runner, then select, then the lr-selection commit (lr_selection.* plus the sweep records only).

11. Evaluation, the rerun and, if triggered, the contingency runs, all from the lr-selection commit with a clean worktree.

12. Aggregate, then the results commit
- Commit the run records, the stock aggregates, a0_internal_metrics.json and results.json, and a RESULTS.md that reports every verdict, including nulls, HARMFUL results and INCONCLUSIVE ones.
- Add a research/falsification note for each FALSIFIES or HARMFUL verdict and for a failed competence gate.
- Update docs/components/STATUS.md and PRIORITIES.md to say 'A0-internal evidence only'.
