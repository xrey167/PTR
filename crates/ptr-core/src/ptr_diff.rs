//! PTR-Diff reference: discrete masked denoising on a small vocabulary.
//!
//! Only the inference-time step and a deterministic unmasking schedule exist.
//! There is **no training objective**: the logits come from randomly initialised
//! (or checkpoint-loaded) weights, so the unmasked tokens carry no learned
//! meaning. What is tested is the mechanics: positions are only ever unmasked,
//! never re-masked or rewritten, and the loop terminates.

use crate::core_model::CoreModel;
use crate::error::CoreError;
use crate::tensor::{softmax, Matrix};
use crate::typed_attention::AttentionBias;

#[derive(Clone, Debug)]
pub struct PtrDiffConfig {
    pub blocks: usize,
    pub canvas_slots: usize,
    pub denoise_steps: u32,
}
impl Default for PtrDiffConfig {
    fn default() -> Self {
        Self {
            blocks: 12,
            canvas_slots: 64,
            denoise_steps: 8,
        }
    }
}

/// Largest accepted denoising step count (Invariant 8).
pub const MAX_DENOISE_STEPS: u32 = 1024;

/// Positions to unmask this step: `ceil(masked / steps_remaining)`, so the last
/// remaining step unmasks everything left. `0` when nothing is masked.
pub fn unmask_schedule(masked: usize, steps_remaining: u32) -> usize {
    if masked == 0 || steps_remaining == 0 {
        return 0;
    }
    masked.div_ceil(steps_remaining as usize)
}

/// One denoising step.
#[derive(Clone, Debug, PartialEq)]
pub struct DenoiseStep {
    /// The canvas after the step.
    pub tokens: Vec<u32>,
    /// Positions unmasked in this step, most confident first.
    pub unmasked: Vec<usize>,
    /// Confidence (max softmax probability) of each unmasked position.
    pub confidences: Vec<f32>,
}

/// A complete denoising run.
#[derive(Clone, Debug, PartialEq)]
pub struct DenoiseRun {
    /// Final canvas.
    pub tokens: Vec<u32>,
    /// Number of masked positions before each executed step, then after the last.
    pub masked_history: Vec<usize>,
    /// Steps executed.
    pub steps_taken: u32,
}

impl CoreModel {
    /// The reserved mask token: the last id of the diffusion vocabulary.
    pub fn mask_token(&self) -> u32 {
        (self.config.diff_vocab - 1) as u32
    }

    fn check_canvas(&self, tokens: &[u32]) -> Result<(), CoreError> {
        if tokens.is_empty() || tokens.len() > self.config.max_slots {
            return Err(CoreError::CanvasLength {
                len: tokens.len(),
                max: self.config.max_slots,
            });
        }
        let vocab = self.config.diff_vocab;
        match tokens.iter().position(|t| *t as usize >= vocab) {
            Some(position) => Err(CoreError::TokenOutOfVocab {
                position,
                token: tokens[position],
                vocab,
            }),
            None => Ok(()),
        }
    }

    /// Per-position vocabulary logits (`positions x diff_vocab`) from slot states:
    /// token + position embeddings, one pass of the shared refinement block with
    /// every position admitted, then the output projection.
    pub fn denoise_logits(&self, tokens: &[u32]) -> Result<Matrix, CoreError> {
        self.check_canvas(tokens)?;
        let d = self.config.d_model;
        let mut x = Matrix::zeros(tokens.len(), d)?;
        for (i, tok) in tokens.iter().enumerate() {
            let (Some(t), Some(p), Some(dst)) = (
                self.diff_tok.row(*tok as usize),
                self.diff_pos.row(i),
                x.row_mut(i),
            ) else {
                return Err(CoreError::Codebook {
                    code: "PTR_CORE_DIFF_ROW",
                });
            };
            for ((o, a), b) in dst.iter_mut().zip(t).zip(p) {
                *o = a + b;
            }
        }
        let zeros = vec![0.0_f32; tokens.len()];
        let roles = vec![0_usize; tokens.len()];
        let (h, _) = self.block.apply(
            &x,
            &AttentionBias {
                key_bias: &zeros,
                roles: &roles,
                use_role_bias: false,
            },
        )?;
        let logits = self.diff_out.forward(&h)?;
        if logits.is_finite() {
            Ok(logits)
        } else {
            Err(CoreError::NonFinite {
                what: "diffusion logits",
            })
        }
    }

    /// Unmask the `k` most confident masked positions (ties: lower index first).
    ///
    /// The predicted token at a position is the argmax over the non-mask
    /// vocabulary (the mask id is excluded, so a step never writes a mask) and
    /// its confidence is that token's softmax probability. Unmasked positions are
    /// returned unchanged. `k` larger than the masked count unmasks all of them.
    pub fn denoise_step(&self, tokens: &[u32], k: usize) -> Result<DenoiseStep, CoreError> {
        let logits = self.denoise_logits(tokens)?;
        let mask = self.mask_token();
        let mut candidates: Vec<(usize, u32, f32)> = Vec::new();
        for (i, tok) in tokens.iter().enumerate() {
            if *tok != mask {
                continue;
            }
            let Some(row) = logits.row(i) else { continue };
            let mut row = row.to_vec();
            row[mask as usize] = f32::NEG_INFINITY;
            let probs = softmax(&row);
            let (best, conf) = probs
                .iter()
                .enumerate()
                .fold(
                    (0, probs[0]),
                    |(bi, bp), (j, p)| {
                        if *p > bp {
                            (j, *p)
                        } else {
                            (bi, bp)
                        }
                    },
                );
            candidates.push((i, best as u32, conf));
        }
        candidates.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
        candidates.truncate(k);
        let mut out = tokens.to_vec();
        let mut unmasked = Vec::with_capacity(candidates.len());
        let mut confidences = Vec::with_capacity(candidates.len());
        for (pos, tok, conf) in candidates {
            out[pos] = tok;
            unmasked.push(pos);
            confidences.push(conf);
        }
        Ok(DenoiseStep {
            tokens: out,
            unmasked,
            confidences,
        })
    }

    /// Run up to `steps` denoising steps with [`unmask_schedule`].
    ///
    /// Every executed step unmasks at least one position while any remain, and the
    /// schedule unmasks everything on the last step, so a run with `steps >= 1`
    /// always ends fully unmasked.
    pub fn denoise(&self, tokens: &[u32], steps: u32) -> Result<DenoiseRun, CoreError> {
        if steps == 0 || steps > MAX_DENOISE_STEPS {
            return Err(CoreError::DenoiseSteps { steps });
        }
        self.check_canvas(tokens)?;
        let mask = self.mask_token();
        let count = |t: &[u32]| t.iter().filter(|x| **x == mask).count();
        let mut current = tokens.to_vec();
        let mut history = vec![count(&current)];
        let mut taken = 0;
        for s in 0..steps {
            let masked = count(&current);
            if masked == 0 {
                break;
            }
            let k = unmask_schedule(masked, steps - s);
            current = self.denoise_step(&current, k)?.tokens;
            history.push(count(&current));
            taken += 1;
        }
        Ok(DenoiseRun {
            tokens: current,
            masked_history: history,
            steps_taken: taken,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CoreConfig;

    fn model() -> CoreModel {
        CoreModel::new(CoreConfig {
            d_model: 8,
            heads: 2,
            d_ff: 16,
            max_slots: 12,
            diff_vocab: 10,
            ..CoreConfig::default()
        })
        .unwrap()
    }

    #[test]
    fn schedule_is_ceiling_division_and_finishes() {
        assert_eq!(unmask_schedule(0, 3), 0);
        assert_eq!(unmask_schedule(7, 3), 3);
        assert_eq!(unmask_schedule(4, 1), 4);
        assert_eq!(unmask_schedule(5, 0), 0);
        // Simulating the schedule always reaches zero by the last step.
        for masked0 in 1..40 {
            for steps in 1..10_u32 {
                let mut m = masked0;
                for s in 0..steps {
                    m -= unmask_schedule(m, steps - s).min(m);
                }
                assert_eq!(m, 0, "{masked0} masked, {steps} steps");
            }
        }
    }

    #[test]
    fn step_unmasks_top_k_by_confidence_and_leaves_others() {
        let m = model();
        let mask = m.mask_token();
        let canvas = vec![3, mask, mask, 5, mask, mask];
        let step = m.denoise_step(&canvas, 2).unwrap();
        assert_eq!(step.unmasked.len(), 2);
        assert!(step.confidences[0] >= step.confidences[1]);
        // Known tokens untouched; exactly two masks resolved; never writes a mask.
        assert_eq!(step.tokens[0], 3);
        assert_eq!(step.tokens[3], 5);
        let still = step.tokens.iter().filter(|t| **t == mask).count();
        assert_eq!(still, 2);
        for p in &step.unmasked {
            assert_ne!(step.tokens[*p], mask);
        }
        // The chosen positions really are the two most confident.
        let all = m.denoise_step(&canvas, 4).unwrap();
        let mut ranked: Vec<(usize, f32)> = all
            .unmasked
            .iter()
            .copied()
            .zip(all.confidences.iter().copied())
            .collect();
        ranked.truncate(2);
        assert_eq!(
            step.unmasked,
            ranked.iter().map(|r| r.0).collect::<Vec<_>>()
        );
        // k = 0 is a no-op; k beyond the masked count unmasks all.
        assert_eq!(m.denoise_step(&canvas, 0).unwrap().tokens, canvas);
        assert_eq!(m.denoise_step(&canvas, 99).unwrap().unmasked.len(), 4);
    }

    #[test]
    fn run_is_monotone_never_rewrites_and_terminates() {
        let m = model();
        let mask = m.mask_token();
        let start = vec![mask, 2, mask, mask, 7, mask, mask, mask, 1, mask];
        let run = m.denoise(&start, 4).unwrap();
        assert!(run.masked_history.windows(2).all(|w| w[1] < w[0]));
        assert_eq!(*run.masked_history.last().unwrap(), 0);
        assert!(run.steps_taken <= 4);
        assert!(!run.tokens.contains(&mask));
        for (a, b) in start.iter().zip(&run.tokens) {
            if *a != mask {
                assert_eq!(a, b, "a known token was rewritten");
            }
        }
        assert_eq!(run, m.denoise(&start, 4).unwrap());
    }

    #[test]
    fn single_step_budget_unmasks_everything_and_no_mask_is_a_no_op() {
        let m = model();
        let mask = m.mask_token();
        let run = m.denoise(&[mask; 5], 1).unwrap();
        assert_eq!(run.steps_taken, 1);
        assert!(!run.tokens.contains(&mask));
        let done = m.denoise(&[1, 2, 3], 3).unwrap();
        assert_eq!(done.steps_taken, 0);
        assert_eq!(done.tokens, vec![1, 2, 3]);
    }

    #[test]
    fn refusals_are_typed() {
        let m = model();
        let mask = m.mask_token();
        assert_eq!(
            m.denoise(&[mask], 0).err(),
            Some(CoreError::DenoiseSteps { steps: 0 })
        );
        assert!(matches!(
            m.denoise(&[mask], MAX_DENOISE_STEPS + 1),
            Err(CoreError::DenoiseSteps { .. })
        ));
        assert!(matches!(
            m.denoise(&[], 1),
            Err(CoreError::CanvasLength { len: 0, .. })
        ));
        assert!(matches!(
            m.denoise(&[mask; 13], 1),
            Err(CoreError::CanvasLength { len: 13, max: 12 })
        ));
        assert_eq!(
            m.denoise(&[1, 10], 1).err(),
            Some(CoreError::TokenOutOfVocab {
                position: 1,
                token: 10,
                vocab: 10
            })
        );
    }
}
