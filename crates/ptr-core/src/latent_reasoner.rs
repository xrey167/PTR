use crate::epistemic_workspace::EpistemicWorkspace;
use crate::error::CoreError;
use crate::layers::Linear;
use crate::tensor::{Matrix, ShapeError};
use crate::typed_attention::{AttentionBias, MultiHeadAttention};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReasoningBudget {
    None,
    Low,
    Medium,
    High,
    Max(u32),
}

impl ReasoningBudget {
    pub fn steps(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Low => 1,
            Self::Medium => 4,
            Self::High => 8,
            Self::Max(v) => v,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LatentReasoner;

impl LatentReasoner {
    pub fn advance(&self, workspace: &mut EpistemicWorkspace, budget: ReasoningBudget) {
        workspace.step = workspace.step.saturating_add(budget.steps());
    }
}

/// Layer-norm epsilon used by the reference block.
pub(crate) const LN_EPS: f32 = 1e-5;

/// Options of the recurrent refinement loop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecurrenceOptions {
    /// Step allowance; the effective cap is `min(budget.steps(), max_latent_steps)`.
    pub budget: ReasoningBudget,
    /// ACT threshold: stop once the cumulative halting probability reaches
    /// `1 - epsilon`. Must lie in `[0, 1)`.
    pub epsilon: f32,
}

impl Default for RecurrenceOptions {
    fn default() -> Self {
        Self {
            budget: ReasoningBudget::High,
            epsilon: 0.01,
        }
    }
}

/// Why the recurrence stopped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HaltReason {
    /// The step allowance was zero; the embedding was returned unrefined.
    NoBudget,
    /// The cumulative halting probability reached `1 - epsilon`.
    Halted,
    /// The step cap was reached first.
    BudgetExhausted,
}

/// What one refinement step did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepTrace {
    /// 1-based step index.
    pub step: u32,
    /// Halting probability emitted after this step.
    pub halt_probability: f32,
    /// Running sum of halting probabilities.
    pub cumulative_halt: f32,
    /// L2 norm of the change of the slot states in this step.
    pub delta_l2: f32,
}

/// The shared refinement block: attention and MLP, each with residual and layer
/// norm (post-norm). The same weights are applied at every recurrence step.
#[derive(Clone, Debug, PartialEq)]
pub struct RefinementBlock {
    pub(crate) attn: MultiHeadAttention,
    pub(crate) ln1_g: Matrix,
    pub(crate) ln1_b: Matrix,
    pub(crate) ff1: Linear,
    pub(crate) ff2: Linear,
    pub(crate) ln2_g: Matrix,
    pub(crate) ln2_b: Matrix,
}

impl RefinementBlock {
    pub(crate) fn init(
        seed: u64,
        d_model: usize,
        heads: usize,
        d_ff: usize,
        roles: usize,
    ) -> Result<Self, ShapeError> {
        Ok(Self {
            attn: MultiHeadAttention::init(seed, d_model, heads, roles)?,
            ln1_g: Matrix::filled(1, d_model, 1.0)?,
            ln1_b: Matrix::zeros(1, d_model)?,
            ff1: Linear::init(seed, "ff1", d_model, d_ff)?,
            ff2: Linear::init(seed, "ff2", d_ff, d_model)?,
            ln2_g: Matrix::filled(1, d_model, 1.0)?,
            ln2_b: Matrix::zeros(1, d_model)?,
        })
    }

    /// One application. Returns the new states and every head's attention weights.
    pub(crate) fn apply(
        &self,
        x: &Matrix,
        bias: &AttentionBias<'_>,
    ) -> Result<(Matrix, Vec<Matrix>), CoreError> {
        let att = self.attn.forward(x, bias)?;
        let h = x
            .add(&att.output)?
            .layer_norm(&self.ln1_g, &self.ln1_b, LN_EPS)?;
        let ff = self.ff2.forward(&self.ff1.forward(&h)?.gelu())?;
        let y = h.add(&ff)?.layer_norm(&self.ln2_g, &self.ln2_b, LN_EPS)?;
        Ok((y, att.weights))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_steps_are_unchanged() {
        assert_eq!(ReasoningBudget::None.steps(), 0);
        assert_eq!(ReasoningBudget::Low.steps(), 1);
        assert_eq!(ReasoningBudget::Medium.steps(), 4);
        assert_eq!(ReasoningBudget::High.steps(), 8);
        assert_eq!(ReasoningBudget::Max(13).steps(), 13);
    }

    #[test]
    fn block_preserves_shape_and_normalises_rows() {
        let b = RefinementBlock::init(1, 8, 2, 16, 9).unwrap();
        let x = crate::tensor::uniform_matrix(3, 8, 1.0, &mut crate::tensor::SeededRng::new(4))
            .unwrap();
        let (y, w) = b
            .apply(
                &x,
                &AttentionBias {
                    key_bias: &[0.0; 3],
                    roles: &[0, 1, 2],
                    use_role_bias: true,
                },
            )
            .unwrap();
        assert_eq!(y.shape(), (3, 8));
        assert_eq!(w.len(), 2);
        for i in 0..3 {
            let row = y.row(i).unwrap();
            assert!((row.iter().sum::<f32>() / 8.0).abs() < 1e-4);
        }
    }
}
