//! The CPU reference model: parameters, typed embeddings and the forward pass.
//!
//! A deterministic `f32` reference of the PTR-Core forward pass. It is randomly
//! initialised (or loaded from a checkpoint) and is **not** a trained model:
//! nothing here says anything about task quality, and it is not numerically
//! compared with `model/burn-a0`.

use crate::config::CoreConfig;
use crate::epistemic_workspace::EpistemicWorkspace;
use crate::error::CoreError;
use crate::latent_reasoner::{HaltReason, RecurrenceOptions, RefinementBlock, StepTrace};
use crate::layers::Linear;
use crate::semantic_slots::SemanticSlot;
use crate::tensor::{sigmoid, uniform_matrix, xavier_uniform, Matrix, SeededRng, ShapeError};
use crate::typed_attention::AttentionBias;
use ptr_types::{
    Codebook, CognitiveType, EpistemicState, ReasoningOperator, SemanticRole, UncertaintyKind,
    ValidityMask,
};

/// Codebook the reference model's embedding tables are sized and indexed by.
pub const CODEBOOK: Codebook = Codebook::V1;

/// Half-width of the uniform initialiser of embedding tables.
const EMBED_INIT: f32 = 0.5;

/// Stable code of a type-kernel member under [`CODEBOOK`].
pub(crate) fn code_index<T: CognitiveType>(member: T) -> Result<usize, CoreError> {
    CODEBOOK
        .code_of(member)
        .map(|c| usize::from(c.index()))
        .map_err(|e| CoreError::Codebook { code: e.code() })
}

/// Everything one forward pass returns.
#[derive(Clone, Debug, PartialEq)]
pub struct ForwardOutput {
    /// `slots x d_model` final slot states (excluded slots are still computed
    /// but never read by any other slot and never pooled).
    pub states: Matrix,
    /// Mean of the admitted slots' states.
    pub pooled: Vec<f32>,
    /// One entry per executed recurrence step.
    pub traces: Vec<StepTrace>,
    /// Why the recurrence stopped.
    pub halt: HaltReason,
    /// Attention weights of the last executed step, one matrix per head
    /// (empty when no step ran).
    pub last_attention: Vec<Matrix>,
    /// The admission mask that was applied (slot validity, narrowed by any
    /// caller mask).
    pub mask: ValidityMask,
}

impl ForwardOutput {
    /// Number of recurrence steps that ran.
    pub fn steps_used(&self) -> u32 {
        self.traces.len() as u32
    }
}

/// The PTR-Core CPU reference model.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreModel {
    pub(crate) config: CoreConfig,
    pub(crate) role_emb: Matrix,
    pub(crate) epistemic_emb: Matrix,
    pub(crate) uncertainty_emb: Matrix,
    pub(crate) block: RefinementBlock,
    pub(crate) halt: Linear,
    pub(crate) router: Linear,
    pub(crate) verifier: Linear,
    pub(crate) action_w: Matrix,
    pub(crate) diff_tok: Matrix,
    pub(crate) diff_pos: Matrix,
    pub(crate) diff_out: Linear,
}

impl CoreModel {
    /// Instantiate with deterministic seeded parameters.
    pub fn new(config: CoreConfig) -> Result<Self, CoreError> {
        config.validate()?;
        let d = config.d_model;
        let s = config.seed;
        let roles = usize::from(CODEBOOK.cardinality_of::<SemanticRole>());
        let epistemic = usize::from(CODEBOOK.cardinality_of::<EpistemicState>());
        let uncertainty = usize::from(CODEBOOK.cardinality_of::<UncertaintyKind>());
        let operators = usize::from(CODEBOOK.cardinality_of::<ReasoningOperator>());
        let table = |name: &str, rows: usize, cols: usize| -> Result<Matrix, ShapeError> {
            uniform_matrix(rows, cols, EMBED_INIT, &mut SeededRng::for_name(s, name))
        };
        Ok(Self {
            role_emb: table("embed.role", roles, d)?,
            epistemic_emb: table("embed.epistemic", epistemic, d)?,
            uncertainty_emb: table("embed.uncertainty", uncertainty, d)?,
            block: RefinementBlock::init(s, d, config.heads, config.d_ff, roles)?,
            halt: Linear::init(s, "halt", d, 1)?,
            router: Linear::init(s, "head.router", d, operators)?,
            verifier: Linear::init(s, "head.verifier", d, 1)?,
            action_w: xavier_uniform(d, d, &mut SeededRng::for_name(s, "head.action"))?,
            diff_tok: table("diff.tok", config.diff_vocab, d)?,
            diff_pos: table("diff.pos", config.max_slots, d)?,
            diff_out: Linear::init(s, "diff.out", d, config.diff_vocab)?,
            config,
        })
    }

    /// The configuration this model was built with.
    pub fn config(&self) -> &CoreConfig {
        &self.config
    }

    /// Every parameter with a stable name, in checkpoint order.
    pub fn named_params(&self) -> Vec<(&'static str, &Matrix)> {
        let b = &self.block;
        vec![
            ("embed.role", &self.role_emb),
            ("embed.epistemic", &self.epistemic_emb),
            ("embed.uncertainty", &self.uncertainty_emb),
            ("attn.wq.w", &b.attn.wq.w),
            ("attn.wq.b", &b.attn.wq.b),
            ("attn.wk.w", &b.attn.wk.w),
            ("attn.wk.b", &b.attn.wk.b),
            ("attn.wv.w", &b.attn.wv.w),
            ("attn.wv.b", &b.attn.wv.b),
            ("attn.wo.w", &b.attn.wo.w),
            ("attn.wo.b", &b.attn.wo.b),
            ("attn.role_bias", &b.attn.role_bias),
            ("block.ln1.g", &b.ln1_g),
            ("block.ln1.b", &b.ln1_b),
            ("block.ff1.w", &b.ff1.w),
            ("block.ff1.b", &b.ff1.b),
            ("block.ff2.w", &b.ff2.w),
            ("block.ff2.b", &b.ff2.b),
            ("block.ln2.g", &b.ln2_g),
            ("block.ln2.b", &b.ln2_b),
            ("halt.w", &self.halt.w),
            ("halt.b", &self.halt.b),
            ("head.router.w", &self.router.w),
            ("head.router.b", &self.router.b),
            ("head.verifier.w", &self.verifier.w),
            ("head.verifier.b", &self.verifier.b),
            ("head.action.w", &self.action_w),
            ("diff.tok", &self.diff_tok),
            ("diff.pos", &self.diff_pos),
            ("diff.out.w", &self.diff_out.w),
            ("diff.out.b", &self.diff_out.b),
        ]
    }

    /// Mutable view of [`Self::named_params`], same names and order.
    pub(crate) fn named_params_mut(&mut self) -> Vec<(&'static str, &mut Matrix)> {
        let b = &mut self.block;
        vec![
            ("embed.role", &mut self.role_emb),
            ("embed.epistemic", &mut self.epistemic_emb),
            ("embed.uncertainty", &mut self.uncertainty_emb),
            ("attn.wq.w", &mut b.attn.wq.w),
            ("attn.wq.b", &mut b.attn.wq.b),
            ("attn.wk.w", &mut b.attn.wk.w),
            ("attn.wk.b", &mut b.attn.wk.b),
            ("attn.wv.w", &mut b.attn.wv.w),
            ("attn.wv.b", &mut b.attn.wv.b),
            ("attn.wo.w", &mut b.attn.wo.w),
            ("attn.wo.b", &mut b.attn.wo.b),
            ("attn.role_bias", &mut b.attn.role_bias),
            ("block.ln1.g", &mut b.ln1_g),
            ("block.ln1.b", &mut b.ln1_b),
            ("block.ff1.w", &mut b.ff1.w),
            ("block.ff1.b", &mut b.ff1.b),
            ("block.ff2.w", &mut b.ff2.w),
            ("block.ff2.b", &mut b.ff2.b),
            ("block.ln2.g", &mut b.ln2_g),
            ("block.ln2.b", &mut b.ln2_b),
            ("halt.w", &mut self.halt.w),
            ("halt.b", &mut self.halt.b),
            ("head.router.w", &mut self.router.w),
            ("head.router.b", &mut self.router.b),
            ("head.verifier.w", &mut self.verifier.w),
            ("head.verifier.b", &mut self.verifier.b),
            ("head.action.w", &mut self.action_w),
            ("diff.tok", &mut self.diff_tok),
            ("diff.pos", &mut self.diff_pos),
            ("diff.out.w", &mut self.diff_out.w),
            ("diff.out.b", &mut self.diff_out.b),
        ]
    }

    /// Total number of scalar parameters.
    pub fn parameter_count(&self) -> usize {
        self.named_params()
            .iter()
            .map(|(_, m)| m.as_slice().len())
            .sum()
    }

    /// The router head (`d_model -> operators`).
    pub fn router_head(&self) -> &Linear {
        &self.router
    }

    /// Typed embedding: `content + role[r] + epistemic[e] + uncertainty[u]`.
    ///
    /// Each axis has its own learnable table indexed by the ptr-types
    /// [`CODEBOOK`] code, so the three axes stay separable in the sum.
    pub fn embed(&self, slots: &[SemanticSlot]) -> Result<Matrix, CoreError> {
        self.check_slot_count(slots.len())?;
        let d = self.config.d_model;
        let mut out = Matrix::zeros(slots.len(), d)?;
        for (i, slot) in slots.iter().enumerate() {
            if slot.value.len() != d {
                return Err(CoreError::SlotWidth {
                    slot: slot.id,
                    expected: d,
                    found: slot.value.len(),
                });
            }
            if !slot.value.iter().all(|v| v.is_finite()) {
                return Err(CoreError::NonFinite { what: "slot value" });
            }
            let role = self.role_emb.row(code_index(slot.role)?);
            let epi = self.epistemic_emb.row(code_index(slot.epistemic)?);
            let unc = self.uncertainty_emb.row(code_index(slot.uncertainty)?);
            let (Some(role), Some(epi), Some(unc), Some(dst)) = (role, epi, unc, out.row_mut(i))
            else {
                return Err(CoreError::Codebook {
                    code: "PTR_CORE_EMBED_ROW",
                });
            };
            for (j, o) in dst.iter_mut().enumerate() {
                *o = slot.value[j] + role[j] + epi[j] + unc[j];
            }
        }
        Ok(out)
    }

    fn check_slot_count(&self, n: usize) -> Result<(), CoreError> {
        if n == 0 {
            return Err(CoreError::NoSlots);
        }
        if n > self.config.max_slots {
            return Err(CoreError::TooManySlots {
                got: n,
                max: self.config.max_slots,
            });
        }
        Ok(())
    }

    /// Forward pass: typed embedding, then the shared block applied up to the
    /// step cap with ACT-style halting. Slot validity alone defines the mask.
    pub fn forward(
        &self,
        slots: &[SemanticSlot],
        options: &RecurrenceOptions,
    ) -> Result<ForwardOutput, CoreError> {
        self.forward_inner(slots, options, None)
    }

    /// Like [`Self::forward`], additionally narrowing the admission by `extra`.
    /// The mask can only exclude more slots, never re-admit a non-live one.
    pub fn forward_masked(
        &self,
        slots: &[SemanticSlot],
        options: &RecurrenceOptions,
        extra: &ValidityMask,
    ) -> Result<ForwardOutput, CoreError> {
        self.forward_inner(slots, options, Some(extra))
    }

    /// Forward over a workspace; adds the executed step count to `workspace.step`.
    pub fn reason_workspace(
        &self,
        workspace: &mut EpistemicWorkspace,
        options: &RecurrenceOptions,
    ) -> Result<ForwardOutput, CoreError> {
        let out = self.forward(&workspace.slots, options)?;
        workspace.step = workspace.step.saturating_add(out.steps_used());
        Ok(out)
    }

    fn forward_inner(
        &self,
        slots: &[SemanticSlot],
        options: &RecurrenceOptions,
        extra: Option<&ValidityMask>,
    ) -> Result<ForwardOutput, CoreError> {
        if !(0.0..1.0).contains(&options.epsilon) {
            return Err(CoreError::EpsilonOutOfRange);
        }
        let validities: Vec<_> = slots.iter().map(|s| s.validity).collect();
        let mut mask = ValidityMask::from_validities(&validities);
        if let Some(extra) = extra {
            mask.narrow(extra).map_err(|_| CoreError::MaskLength {
                expected: slots.len(),
                found: extra.len(),
            })?;
        }
        let mut x = self.embed(slots)?;
        if mask.admitted_count() == 0 {
            return Err(CoreError::NoAdmittedSlots);
        }
        let roles = slots
            .iter()
            .map(|s| code_index(s.role))
            .collect::<Result<Vec<_>, _>>()?;
        let key_bias = mask.attention_bias();
        let bias = AttentionBias {
            key_bias: &key_bias,
            roles: &roles,
            use_role_bias: self.config.role_bias,
        };
        let cap = options.budget.steps().min(self.config.max_latent_steps);
        let mut halt = if cap == 0 {
            HaltReason::NoBudget
        } else {
            HaltReason::BudgetExhausted
        };
        let mut traces = Vec::new();
        let mut last_attention = Vec::new();
        let mut cumulative = 0.0_f32;
        for step in 1..=cap {
            let (next, weights) = self.block.apply(&x, &bias)?;
            let delta_l2 = next
                .as_slice()
                .iter()
                .zip(x.as_slice())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            x = next;
            last_attention = weights;
            let pooled = pool(&x, &mask)?;
            let logit = self.halt.forward_vec(&pooled)?;
            let p = sigmoid(logit[0]);
            cumulative += p;
            traces.push(StepTrace {
                step,
                halt_probability: p,
                cumulative_halt: cumulative,
                delta_l2,
            });
            if cumulative >= 1.0 - options.epsilon {
                halt = HaltReason::Halted;
                break;
            }
        }
        if !x.is_finite() {
            return Err(CoreError::NonFinite { what: "states" });
        }
        let pooled = pool(&x, &mask)?;
        Ok(ForwardOutput {
            states: x,
            pooled,
            traces,
            halt,
            last_attention,
            mask,
        })
    }
}

/// Mean of the admitted rows of `states`.
pub(crate) fn pool(states: &Matrix, mask: &ValidityMask) -> Result<Vec<f32>, CoreError> {
    let mut acc = vec![0.0_f32; states.cols()];
    let mut count = 0_usize;
    for i in mask.admitted_slots() {
        let Some(row) = states.row(i) else {
            return Err(CoreError::MaskLength {
                expected: states.rows(),
                found: mask.len(),
            });
        };
        for (a, v) in acc.iter_mut().zip(row) {
            *a += v;
        }
        count += 1;
    }
    if count == 0 {
        return Err(CoreError::NoAdmittedSlots);
    }
    let inv = 1.0 / count as f32;
    acc.iter_mut().for_each(|a| *a *= inv);
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::latent_reasoner::ReasoningBudget;
    use ptr_types::{Generation, Probability, Validity};

    pub(crate) fn slot(id: u32, role: SemanticRole, value: Vec<f32>) -> SemanticSlot {
        SemanticSlot {
            id,
            role,
            epistemic: EpistemicState::Observed,
            uncertainty: UncertaintyKind::Point,
            generation: Generation(1),
            validity: Validity::Live,
            confidence: Probability::default(),
            value,
            provenance: Vec::new(),
        }
    }

    fn small() -> CoreModel {
        CoreModel::new(CoreConfig {
            d_model: 8,
            heads: 2,
            d_ff: 16,
            max_slots: 6,
            ..CoreConfig::default()
        })
        .unwrap()
    }

    fn slots(n: usize) -> Vec<SemanticSlot> {
        let roles = [
            SemanticRole::Goal,
            SemanticRole::Claim,
            SemanticRole::Evidence,
            SemanticRole::Action,
            SemanticRole::Resource,
            SemanticRole::Relation,
        ];
        (0..n)
            .map(|i| {
                slot(
                    i as u32,
                    roles[i % roles.len()],
                    vec![0.1 * (i as f32 + 1.0); 8],
                )
            })
            .collect()
    }

    #[test]
    fn param_name_lists_agree_and_are_unique() {
        let mut m = small();
        let names: Vec<_> = m
            .named_params()
            .iter()
            .map(|(n, s)| (*n, s.shape()))
            .collect();
        let names_mut: Vec<_> = m
            .named_params_mut()
            .into_iter()
            .map(|(n, s)| (n, s.shape()))
            .collect();
        assert_eq!(names, names_mut);
        let mut uniq: Vec<_> = names.iter().map(|(n, _)| *n).collect();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), names.len());
        assert_eq!(
            m.parameter_count(),
            names.iter().map(|(_, s)| s.0 * s.1).sum::<usize>()
        );
    }

    #[test]
    fn embedding_is_content_plus_three_typed_tables() {
        let m = small();
        let s = slot(0, SemanticRole::Evidence, vec![1.0; 8]);
        let e = m.embed(std::slice::from_ref(&s)).unwrap();
        let r = m
            .role_emb
            .row(code_index(SemanticRole::Evidence).unwrap())
            .unwrap();
        let p = m
            .epistemic_emb
            .row(code_index(EpistemicState::Observed).unwrap())
            .unwrap();
        let u = m
            .uncertainty_emb
            .row(code_index(UncertaintyKind::Point).unwrap())
            .unwrap();
        for j in 0..8 {
            assert_eq!(e.row(0).unwrap()[j], 1.0 + r[j] + p[j] + u[j]);
        }
        // Each axis changes the embedding independently.
        let mut s2 = s.clone();
        s2.role = SemanticRole::Goal;
        assert_ne!(m.embed(&[s2]).unwrap(), e);
        let mut s3 = s.clone();
        s3.epistemic = EpistemicState::Verified;
        assert_ne!(m.embed(&[s3]).unwrap(), e);
        let mut s4 = s;
        s4.uncertainty = UncertaintyKind::Distribution;
        assert_ne!(m.embed(&[s4]).unwrap(), e);
    }

    #[test]
    fn embed_refusals_are_typed() {
        let m = small();
        assert_eq!(m.embed(&[]), Err(CoreError::NoSlots));
        assert_eq!(
            m.embed(&slots(6).into_iter().cycle().take(7).collect::<Vec<_>>()),
            Err(CoreError::TooManySlots { got: 7, max: 6 })
        );
        assert!(matches!(
            m.embed(&[slot(9, SemanticRole::Goal, vec![0.0; 3])]),
            Err(CoreError::SlotWidth {
                slot: 9,
                expected: 8,
                found: 3
            })
        ));
        let mut bad = vec![0.0; 8];
        bad[2] = f32::NAN;
        assert!(matches!(
            m.embed(&[slot(0, SemanticRole::Goal, bad)]),
            Err(CoreError::NonFinite { .. })
        ));
    }

    #[test]
    fn forward_is_deterministic_and_finite() {
        let m = small();
        let o = RecurrenceOptions::default();
        let a = m.forward(&slots(4), &o).unwrap();
        let b = m.forward(&slots(4), &o).unwrap();
        assert_eq!(a, b);
        assert!(a.states.is_finite());
        assert_eq!(a.pooled.len(), 8);
        let c = CoreModel::new(CoreConfig {
            seed: 99,
            d_model: 8,
            heads: 2,
            d_ff: 16,
            max_slots: 6,
            ..CoreConfig::default()
        })
        .unwrap();
        assert_ne!(c.forward(&slots(4), &o).unwrap().pooled, a.pooled);
    }

    #[test]
    fn zero_budget_returns_unrefined_embedding() {
        let m = small();
        let o = RecurrenceOptions {
            budget: ReasoningBudget::None,
            epsilon: 0.01,
        };
        let out = m.forward(&slots(3), &o).unwrap();
        assert_eq!(out.halt, HaltReason::NoBudget);
        assert!(out.traces.is_empty());
        assert!(out.last_attention.is_empty());
        assert_eq!(out.states, m.embed(&slots(3)).unwrap());
    }

    #[test]
    fn budget_caps_steps_and_epsilon_controls_halting() {
        let m = small();
        // epsilon 0 needs cumulative >= 1; p ~ 0.5 so it takes a few steps but
        // never more than the budget.
        let full = m
            .forward(
                &slots(3),
                &RecurrenceOptions {
                    budget: ReasoningBudget::Max(3),
                    epsilon: 0.0,
                },
            )
            .unwrap();
        assert!(full.steps_used() <= 3);
        // The configured cap (8) bounds an oversized budget.
        let capped = m
            .forward(
                &slots(3),
                &RecurrenceOptions {
                    budget: ReasoningBudget::Max(1000),
                    epsilon: 0.0,
                },
            )
            .unwrap();
        assert!(capped.steps_used() <= 8);
        // Any threshold halts at step 1 when 1 - epsilon <= p.
        let early = m
            .forward(
                &slots(3),
                &RecurrenceOptions {
                    budget: ReasoningBudget::High,
                    epsilon: 0.999_999,
                },
            )
            .unwrap();
        assert_eq!(early.steps_used(), 1);
        assert_eq!(early.halt, HaltReason::Halted);
    }

    #[test]
    fn halting_follows_the_cumulative_rule_and_bounds_trace() {
        let mut m = small();
        // Force p = sigmoid(0 + b) with b = 0 -> 0.5 exactly.
        m.halt.w.as_mut_slice().iter_mut().for_each(|w| *w = 0.0);
        m.halt.b.as_mut_slice()[0] = 0.0;
        let out = m
            .forward(
                &slots(3),
                &RecurrenceOptions {
                    budget: ReasoningBudget::High,
                    epsilon: 0.01,
                },
            )
            .unwrap();
        // 0.5 + 0.5 = 1.0 >= 0.99 -> halts after exactly 2 steps.
        assert_eq!(out.steps_used(), 2);
        assert_eq!(out.halt, HaltReason::Halted);
        assert_eq!(out.traces[0].cumulative_halt, 0.5);
        assert_eq!(out.traces[1].cumulative_halt, 1.0);
        assert_eq!(out.traces[1].step, 2);
        // A strongly negative bias never halts: all budget steps run.
        m.halt.b.as_mut_slice()[0] = -20.0;
        let never = m
            .forward(
                &slots(3),
                &RecurrenceOptions {
                    budget: ReasoningBudget::Medium,
                    epsilon: 0.01,
                },
            )
            .unwrap();
        assert_eq!(never.steps_used(), 4);
        assert_eq!(never.halt, HaltReason::BudgetExhausted);
    }

    #[test]
    fn epsilon_out_of_range_is_refused() {
        let m = small();
        for eps in [-0.1, 1.0, f32::NAN] {
            let r = m.forward(
                &slots(2),
                &RecurrenceOptions {
                    budget: ReasoningBudget::Low,
                    epsilon: eps,
                },
            );
            assert_eq!(r.err(), Some(CoreError::EpsilonOutOfRange));
        }
    }

    #[test]
    fn revoked_slot_has_zero_attention_and_no_influence_on_the_pool() {
        let m = small();
        let o = RecurrenceOptions {
            budget: ReasoningBudget::Medium,
            epsilon: 0.0,
        };
        let mut a = slots(4);
        a[1].validity = Validity::Revoked;
        let mut b = a.clone();
        b[1].value = vec![5.0; 8]; // change only the revoked slot's content
        b[1].role = SemanticRole::Constraint;
        let (oa, ob) = (m.forward(&a, &o).unwrap(), m.forward(&b, &o).unwrap());
        for w in &oa.last_attention {
            for i in 0..4 {
                assert_eq!(w.row(i).unwrap()[1], 0.0);
            }
        }
        assert_eq!(oa.pooled, ob.pooled);
        assert_eq!(oa.mask.admitted_count(), 3);
    }

    #[test]
    fn all_revoked_and_extra_mask_cannot_readmit() {
        let m = small();
        let o = RecurrenceOptions::default();
        let mut s = slots(2);
        s[0].validity = Validity::Revoked;
        s[1].validity = Validity::Disputed;
        assert_eq!(m.forward(&s, &o).err(), Some(CoreError::NoAdmittedSlots));
        // Admitting-all extra mask does not revive a revoked slot.
        let mut s = slots(3);
        s[0].validity = Validity::Revoked;
        let out = m
            .forward_masked(&s, &o, &ValidityMask::admitting_all(3))
            .unwrap();
        assert!(!out.mask.admits(0));
        // A narrower extra mask excludes more.
        let mut extra = ValidityMask::admitting_all(3);
        extra.exclude(2).unwrap();
        let out = m.forward_masked(&s, &o, &extra).unwrap();
        assert_eq!(out.mask.admitted_count(), 1);
        // Length mismatch is typed.
        assert!(matches!(
            m.forward_masked(&s, &o, &ValidityMask::admitting_all(2)),
            Err(CoreError::MaskLength { .. })
        ));
    }

    #[test]
    fn workspace_step_accounting_uses_executed_steps() {
        let m = small();
        let mut ws = EpistemicWorkspace {
            slots: slots(3),
            step: 10,
        };
        let out = m
            .reason_workspace(
                &mut ws,
                &RecurrenceOptions {
                    budget: ReasoningBudget::Max(2),
                    epsilon: 0.0,
                },
            )
            .unwrap();
        assert_eq!(ws.step, 10 + out.steps_used());
    }

    #[test]
    fn invalid_config_is_refused_at_construction() {
        let r = CoreModel::new(CoreConfig {
            heads: 3,
            ..CoreConfig::default()
        });
        assert!(matches!(r, Err(CoreError::Config(_))));
    }
}
