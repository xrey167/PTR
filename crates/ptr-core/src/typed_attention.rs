use crate::error::CoreError;
use crate::layers::Linear;
use crate::tensor::{Matrix, ShapeError};

#[derive(Clone, Debug)]
pub struct TypedAttentionConfig {
    pub model_width: usize,
    pub heads: usize,
    pub type_bias: bool,
    pub epistemic_bias: bool,
    pub validity_mask: bool,
}

impl Default for TypedAttentionConfig {
    fn default() -> Self {
        Self {
            model_width: 512,
            heads: 8,
            type_bias: true,
            epistemic_bias: true,
            validity_mask: true,
        }
    }
}

/// Per-call typing information for [`MultiHeadAttention::forward`].
#[derive(Clone, Copy, Debug)]
pub struct AttentionBias<'a> {
    /// Additive per-key bias, normally `ValidityMask::attention_bias()`
    /// (`0.0` admitted, `-inf` excluded).
    pub key_bias: &'a [f32],
    /// Semantic-role code of each slot, indexing the learned role bias.
    pub roles: &'a [usize],
    /// Whether to add the learned per-(role, role) bias.
    pub use_role_bias: bool,
}

/// Result of one attention application.
#[derive(Clone, Debug)]
pub struct AttentionOutput {
    /// `slots x d_model` attended values after the output projection.
    pub output: Matrix,
    /// Attention weights, one `slots x slots` matrix per head; rows sum to 1.
    pub weights: Vec<Matrix>,
}

/// Multi-head scaled dot-product attention with a typed additive bias.
///
/// The score of query `i` on key `j` is
/// `q_i . k_j / sqrt(d_head) + key_bias[j] + role_bias[role_i][role_j]`.
/// A `-inf` key bias is applied before the softmax, so an excluded slot gets
/// exactly zero weight in every row (not a small weight).
#[derive(Clone, Debug, PartialEq)]
pub struct MultiHeadAttention {
    pub(crate) heads: usize,
    pub(crate) wq: Linear,
    pub(crate) wk: Linear,
    pub(crate) wv: Linear,
    pub(crate) wo: Linear,
    /// Learned `roles x roles` bias shared by all heads; zero-initialised.
    pub(crate) role_bias: Matrix,
}

impl MultiHeadAttention {
    pub(crate) fn init(
        seed: u64,
        d_model: usize,
        heads: usize,
        roles: usize,
    ) -> Result<Self, ShapeError> {
        Ok(Self {
            heads,
            wq: Linear::init(seed, "attn.wq", d_model, d_model)?,
            wk: Linear::init(seed, "attn.wk", d_model, d_model)?,
            wv: Linear::init(seed, "attn.wv", d_model, d_model)?,
            wo: Linear::init(seed, "attn.wo", d_model, d_model)?,
            role_bias: Matrix::zeros(roles, roles)?,
        })
    }

    /// Attend over the rows of `x` (`slots x d_model`).
    pub fn forward(
        &self,
        x: &Matrix,
        bias: &AttentionBias<'_>,
    ) -> Result<AttentionOutput, CoreError> {
        let n = x.rows();
        if bias.key_bias.len() != n {
            return Err(CoreError::MaskLength {
                expected: n,
                found: bias.key_bias.len(),
            });
        }
        if bias.roles.len() != n {
            return Err(CoreError::MaskLength {
                expected: n,
                found: bias.roles.len(),
            });
        }
        if !bias.key_bias.iter().any(|b| b.is_finite()) {
            return Err(CoreError::NoAdmittedSlots);
        }
        let d = x.cols();
        let dh = d / self.heads;
        let q = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut ctx = Matrix::zeros(n, d)?;
        let mut weights = Vec::with_capacity(self.heads);
        for h in 0..self.heads {
            let (lo, hi) = (h * dh, (h + 1) * dh);
            let qh = q.cols_range(lo, hi)?;
            let kh = k.cols_range(lo, hi)?;
            let vh = v.cols_range(lo, hi)?;
            let mut scores = qh.matmul(&kh.transpose())?.scale(scale);
            for (i, &role_i) in bias.roles.iter().enumerate() {
                let row = scores
                    .row_mut(i)
                    .ok_or(ShapeError::RowOutOfRange { row: i, rows: n })?;
                for ((s, kb), &role_j) in row.iter_mut().zip(bias.key_bias).zip(bias.roles) {
                    *s += kb;
                    if bias.use_role_bias {
                        *s += self.role_bias.get(role_i, role_j).ok_or(
                            ShapeError::RowOutOfRange {
                                row: role_i.max(role_j),
                                rows: self.role_bias.rows(),
                            },
                        )?;
                    }
                }
            }
            let w = scores.softmax_rows();
            ctx.set_cols(lo, &w.matmul(&vh)?)?;
            weights.push(w);
        }
        let output = self.wo.forward(&ctx)?;
        Ok(AttentionOutput { output, weights })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{uniform_matrix, SeededRng};
    use ptr_types::{Validity, ValidityMask};

    fn identity_attention(d: usize, heads: usize, roles: usize) -> MultiHeadAttention {
        let id = || Linear {
            w: Matrix::identity(d).unwrap(),
            b: Matrix::zeros(1, d).unwrap(),
        };
        MultiHeadAttention {
            heads,
            wq: id(),
            wk: id(),
            wv: id(),
            wo: id(),
            role_bias: Matrix::zeros(roles, roles).unwrap(),
        }
    }

    /// Independent hand computation: x = I2, one head, identity projections.
    /// scores = x x^T / sqrt(2) = [[a, 0], [0, a]] with a = 1/sqrt(2) = 0.70710678.
    /// softmax row 0 = [e^a, 1] / (e^a + 1) with e^a = 2.02811498
    ///   = [0.669756, 0.330244]; the value rows are [1,0] and [0,1], so the output
    /// row 0 is [0.669756, 0.330244] and row 1 is its mirror image.
    #[test]
    fn two_by_two_attention_matches_hand_computation() {
        let att = identity_attention(2, 1, 1);
        let x = Matrix::identity(2).unwrap();
        let out = att
            .forward(
                &x,
                &AttentionBias {
                    key_bias: &[0.0, 0.0],
                    roles: &[0, 0],
                    use_role_bias: false,
                },
            )
            .unwrap();
        let w = out.weights[0].as_slice();
        assert!((w[0] - 0.669_756).abs() < 1e-5, "{w:?}");
        assert!((w[1] - 0.330_244).abs() < 1e-5);
        assert!((w[2] - 0.330_244).abs() < 1e-5);
        assert!((w[3] - 0.669_756).abs() < 1e-5);
        let o = out.output.as_slice();
        assert!((o[0] - 0.669_756).abs() < 1e-5);
        assert!((o[1] - 0.330_244).abs() < 1e-5);
    }

    #[test]
    fn excluded_slots_get_exactly_zero_weight_in_every_row_and_head() {
        let mut rng = SeededRng::new(5);
        let att = MultiHeadAttention::init(3, 8, 2, 9).unwrap();
        let x = uniform_matrix(5, 8, 1.0, &mut rng).unwrap();
        let mask = ValidityMask::from_validities(&[
            Validity::Live,
            Validity::Revoked,
            Validity::Live,
            Validity::Disputed,
            Validity::Superseded,
        ]);
        let key_bias = mask.attention_bias();
        let bias = AttentionBias {
            key_bias: &key_bias,
            roles: &[0, 1, 2, 3, 4],
            use_role_bias: true,
        };
        let out = att.forward(&x, &bias).unwrap();
        for w in &out.weights {
            for i in 0..5 {
                let row = w.row(i).unwrap();
                assert_eq!(row[1], 0.0);
                assert_eq!(row[3], 0.0);
                assert_eq!(row[4], 0.0);
                assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            }
        }
        // An excluded slot's content cannot leak into the other slots' outputs.
        let mut x2 = x.clone();
        x2.row_mut(1).unwrap().iter_mut().for_each(|v| *v += 3.0);
        let out2 = att.forward(&x2, &bias).unwrap();
        for i in [0, 2, 3, 4] {
            assert_eq!(out.output.row(i), out2.output.row(i));
        }
    }

    #[test]
    fn rows_sum_to_one_without_mask() {
        let att = MultiHeadAttention::init(1, 8, 4, 9).unwrap();
        let x = uniform_matrix(6, 8, 1.0, &mut SeededRng::new(2)).unwrap();
        let out = att
            .forward(
                &x,
                &AttentionBias {
                    key_bias: &[0.0; 6],
                    roles: &[0; 6],
                    use_role_bias: true,
                },
            )
            .unwrap();
        assert_eq!(out.weights.len(), 4);
        for w in &out.weights {
            for i in 0..6 {
                assert!((w.row(i).unwrap().iter().sum::<f32>() - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn role_bias_shifts_weights_toward_the_favoured_role() {
        let mut att = identity_attention(2, 1, 2);
        // Query role 0 favours key role 1 by +5.
        att.role_bias = Matrix::new(2, 2, vec![0.0, 5.0, 0.0, 0.0]).unwrap();
        let x = Matrix::zeros(2, 2).unwrap(); // all raw scores equal 0
        let run = |use_role_bias| {
            att.forward(
                &x,
                &AttentionBias {
                    key_bias: &[0.0, 0.0],
                    roles: &[0, 1],
                    use_role_bias,
                },
            )
            .unwrap()
            .weights[0]
                .row(0)
                .unwrap()
                .to_vec()
        };
        assert_eq!(run(false), vec![0.5, 0.5]);
        let biased = run(true);
        assert!(biased[1] > 0.99 && biased[0] < 0.01);
    }

    #[test]
    fn all_excluded_and_length_mismatches_are_refused() {
        let att = identity_attention(2, 1, 1);
        let x = Matrix::identity(2).unwrap();
        let call = |kb: &[f32], roles: &[usize]| {
            att.forward(
                &x,
                &AttentionBias {
                    key_bias: kb,
                    roles,
                    use_role_bias: false,
                },
            )
            .map(|_| ())
        };
        assert_eq!(
            call(&[f32::NEG_INFINITY; 2], &[0, 0]),
            Err(CoreError::NoAdmittedSlots)
        );
        assert!(matches!(
            call(&[0.0], &[0, 0]),
            Err(CoreError::MaskLength { .. })
        ));
        assert!(matches!(
            call(&[0.0; 2], &[0]),
            Err(CoreError::MaskLength { .. })
        ));
    }

    #[test]
    fn out_of_range_role_is_refused_not_panicking() {
        let att = identity_attention(2, 1, 1);
        let x = Matrix::identity(2).unwrap();
        let r = att.forward(
            &x,
            &AttentionBias {
                key_bias: &[0.0; 2],
                roles: &[0, 7],
                use_role_bias: true,
            },
        );
        assert!(matches!(r, Err(CoreError::Shape(_))));
    }
}
