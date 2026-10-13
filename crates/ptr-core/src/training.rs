//! Toy training of **one head**: the linear operator-router head.
//!
//! This is not task training. The backbone is frozen at its random seeded
//! initialisation, the data is a seeded synthetic set whose label is a
//! deterministic function of the slot roles, and only the router head's weights
//! and bias are updated, by plain full-batch gradient descent on softmax
//! cross-entropy with the exact analytic gradient. It exists to prove the
//! gradient is right and that the head can be fitted, nothing more.

use crate::core_model::{code_index, CoreModel};
use crate::error::CoreError;
use crate::latent_reasoner::{ReasoningBudget, RecurrenceOptions};
use crate::layers::Linear;
use crate::operator_router::operator_classes;
use crate::semantic_slots::SemanticSlot;
use crate::tensor::{softmax, Matrix, SeededRng};
use ptr_types::{
    CognitiveType, EpistemicState, Generation, Probability, SemanticRole, UncertaintyKind, Validity,
};

/// Largest accepted training-set size or epoch count.
pub const MAX_TOY_TRAINING: usize = 100_000;

/// One frozen-backbone feature vector and its operator class.
#[derive(Clone, Debug, PartialEq)]
pub struct RouterExample {
    /// Pooled backbone state (`d_model` wide).
    pub features: Vec<f32>,
    /// Index into [`operator_classes`].
    pub target: usize,
}

/// Options of the toy router-head training.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToyRouterOptions {
    /// Number of synthetic examples.
    pub examples: usize,
    /// Full-batch gradient steps.
    pub epochs: usize,
    /// Step size.
    pub learning_rate: f32,
    /// Seed of the synthetic data (the backbone seed is the model's own).
    pub data_seed: u64,
}

impl Default for ToyRouterOptions {
    fn default() -> Self {
        Self {
            examples: 96,
            epochs: 60,
            learning_rate: 0.5,
            data_seed: 17,
        }
    }
}

/// Outcome of the toy training.
#[derive(Clone, Debug, PartialEq)]
pub struct ToyRouterReport {
    /// Mean cross-entropy before each update, one entry per epoch.
    pub losses: Vec<f32>,
    /// Mean cross-entropy after the last update.
    pub final_loss: f32,
    /// Training-set accuracy before the first update.
    pub initial_accuracy: f32,
    /// Training-set accuracy after the last update.
    pub final_accuracy: f32,
}

/// Loss and exact gradient of one example.
#[derive(Clone, Debug, PartialEq)]
pub struct RouterGradient {
    /// Cross-entropy `-ln softmax(xW + b)[target]`.
    pub loss: f32,
    /// `d loss / d W`, shape `d_model x classes`: `x_i (p_j - [j == target])`.
    pub d_weight: Matrix,
    /// `d loss / d b`, shape `1 x classes`: `p_j - [j == target]`.
    pub d_bias: Matrix,
}

fn shape_ok(head: &Linear, features: &[f32], target: usize) -> Result<(), CoreError> {
    if features.len() != head.w.rows() {
        return Err(CoreError::PooledWidth {
            expected: head.w.rows(),
            found: features.len(),
        });
    }
    if target >= head.w.cols() {
        return Err(CoreError::TrainingInput {
            what: "target class out of range",
        });
    }
    Ok(())
}

/// Exact analytic gradient of softmax cross-entropy for a linear head.
pub fn router_gradient(
    head: &Linear,
    features: &[f32],
    target: usize,
) -> Result<RouterGradient, CoreError> {
    shape_ok(head, features, target)?;
    let logits = head.forward_vec(features)?;
    let p = softmax(&logits);
    let loss = -(p[target].max(f32::MIN_POSITIVE)).ln();
    let mut dz = p;
    dz[target] -= 1.0;
    let classes = dz.len();
    let mut dw = Vec::with_capacity(features.len() * classes);
    for &x in features {
        dw.extend(dz.iter().map(|g| x * g));
    }
    Ok(RouterGradient {
        loss,
        d_weight: Matrix::new(features.len(), classes, dw)?,
        d_bias: Matrix::new(1, classes, dz)?,
    })
}

/// Cross-entropy of one example evaluated entirely in `f64`, for finite-difference
/// checks of [`router_gradient`].
pub fn router_loss_f64(head: &Linear, features: &[f32], target: usize) -> Result<f64, CoreError> {
    shape_ok(head, features, target)?;
    let classes = head.w.cols();
    let logits: Vec<f64> = (0..classes)
        .map(|j| {
            features
                .iter()
                .enumerate()
                .map(|(i, x)| f64::from(*x) * f64::from(head.w.as_slice()[i * classes + j]))
                .sum::<f64>()
                + f64::from(head.b.as_slice()[j])
        })
        .collect();
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lse = max + logits.iter().map(|z| (z - max).exp()).sum::<f64>().ln();
    Ok(lse - logits[target])
}

/// Label rule of the synthetic task: the operator whose code equals the code of
/// the most frequent role (ties go to the lowest code).
pub fn target_class(roles: &[SemanticRole]) -> Result<usize, CoreError> {
    let mut counts = vec![0_usize; operator_classes().len()];
    for role in roles {
        let c = code_index(*role)?;
        if let Some(slot) = counts.get_mut(c) {
            *slot += 1;
        }
    }
    let mut best = 0;
    for (i, c) in counts.iter().enumerate() {
        if *c > counts[best] {
            best = i;
        }
    }
    Ok(best)
}

/// Seeded synthetic slot sets with their label, backbone features pooled by the
/// frozen model.
pub fn synthetic_examples(
    model: &CoreModel,
    count: usize,
    seed: u64,
) -> Result<Vec<RouterExample>, CoreError> {
    if count == 0 || count > MAX_TOY_TRAINING {
        return Err(CoreError::TrainingInput {
            what: "example count",
        });
    }
    let roles = SemanticRole::table(crate::core_model::CODEBOOK.version()).unwrap_or(&[]);
    let epistemic = EpistemicState::table(crate::core_model::CODEBOOK.version()).unwrap_or(&[]);
    let uncertainty = UncertaintyKind::table(crate::core_model::CODEBOOK.version()).unwrap_or(&[]);
    let d = model.config().d_model;
    let options = RecurrenceOptions {
        budget: ReasoningBudget::Medium,
        epsilon: 0.01,
    };
    let mut rng = SeededRng::new(seed);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let n = (3 + rng.below(4)).min(model.config().max_slots);
        let mut slots = Vec::with_capacity(n);
        let mut slot_roles = Vec::with_capacity(n);
        for id in 0..n {
            let role = roles[rng.below(roles.len())];
            slot_roles.push(role);
            slots.push(SemanticSlot {
                id: id as u32,
                role,
                epistemic: epistemic[rng.below(epistemic.len())],
                uncertainty: uncertainty[rng.below(uncertainty.len())],
                generation: Generation(0),
                validity: Validity::Live,
                confidence: Probability::default(),
                value: (0..d).map(|_| rng.uniform(-0.05, 0.05)).collect(),
                provenance: Vec::new(),
            });
        }
        let features = model.forward(&slots, &options)?.pooled;
        out.push(RouterExample {
            features,
            target: target_class(&slot_roles)?,
        });
    }
    Ok(out)
}

fn mean_loss_and_accuracy(head: &Linear, data: &[RouterExample]) -> Result<(f32, f32), CoreError> {
    let mut loss = 0.0_f32;
    let mut correct = 0_usize;
    for ex in data {
        loss += router_gradient(head, &ex.features, ex.target)?.loss;
        let logits = head.forward_vec(&ex.features)?;
        let argmax = logits
            .iter()
            .enumerate()
            .fold(0, |b, (i, v)| if *v > logits[b] { i } else { b });
        correct += usize::from(argmax == ex.target);
    }
    let n = data.len() as f32;
    Ok((loss / n, correct as f32 / n))
}

/// Fit the router head on the synthetic task with the backbone frozen.
pub fn train_router_head(
    model: &mut CoreModel,
    options: &ToyRouterOptions,
) -> Result<ToyRouterReport, CoreError> {
    if options.epochs == 0 || options.epochs > MAX_TOY_TRAINING {
        return Err(CoreError::TrainingInput { what: "epochs" });
    }
    if !(options.learning_rate.is_finite() && options.learning_rate > 0.0) {
        return Err(CoreError::TrainingInput {
            what: "learning rate",
        });
    }
    let data = synthetic_examples(model, options.examples, options.data_seed)?;
    let (_, initial_accuracy) = mean_loss_and_accuracy(&model.router, &data)?;
    let mut losses = Vec::with_capacity(options.epochs);
    let inv = 1.0 / data.len() as f32;
    for _ in 0..options.epochs {
        let mut gw = Matrix::zeros(model.router.w.rows(), model.router.w.cols())?;
        let mut gb = Matrix::zeros(1, model.router.b.cols())?;
        let mut total = 0.0_f32;
        for ex in &data {
            let g = router_gradient(&model.router, &ex.features, ex.target)?;
            total += g.loss;
            for (a, b) in gw.as_mut_slice().iter_mut().zip(g.d_weight.as_slice()) {
                *a += b;
            }
            for (a, b) in gb.as_mut_slice().iter_mut().zip(g.d_bias.as_slice()) {
                *a += b;
            }
        }
        losses.push(total * inv);
        let step = options.learning_rate * inv;
        for (w, g) in model.router.w.as_mut_slice().iter_mut().zip(gw.as_slice()) {
            *w -= step * g;
        }
        for (b, g) in model.router.b.as_mut_slice().iter_mut().zip(gb.as_slice()) {
            *b -= step * g;
        }
    }
    let (final_loss, final_accuracy) = mean_loss_and_accuracy(&model.router, &data)?;
    if !final_loss.is_finite() {
        return Err(CoreError::NonFinite { what: "loss" });
    }
    Ok(ToyRouterReport {
        losses,
        final_loss,
        initial_accuracy,
        final_accuracy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CoreConfig;

    fn model() -> CoreModel {
        CoreModel::new(CoreConfig {
            d_model: 16,
            heads: 2,
            d_ff: 32,
            max_slots: 8,
            ..CoreConfig::default()
        })
        .unwrap()
    }

    #[test]
    fn label_rule_is_plurality_role_with_lowest_code_tiebreak() {
        use SemanticRole::*;
        let g = code_index(Goal).unwrap();
        assert_eq!(
            target_class(&[Claim, Claim, Goal]).unwrap(),
            code_index(Claim).unwrap()
        );
        assert_eq!(
            target_class(&[Goal, Claim]).unwrap(),
            g.min(code_index(Claim).unwrap())
        );
        assert_eq!(
            target_class(&[Action]).unwrap(),
            code_index(Action).unwrap()
        );
    }

    #[test]
    fn analytic_gradient_matches_f64_finite_differences() {
        let m = model();
        let data = synthetic_examples(&m, 5, 3).unwrap();
        // Perturb the head away from its init so gradients are not special.
        let mut head = m.router.clone();
        for (i, w) in head.w.as_mut_slice().iter_mut().enumerate() {
            *w += 0.01 * ((i % 7) as f32 - 3.0);
        }
        for (j, b) in head.b.as_mut_slice().iter_mut().enumerate() {
            *b += 0.05 * j as f32;
        }
        let eps = 1e-3_f64;
        for ex in &data {
            let g = router_gradient(&head, &ex.features, ex.target).unwrap();
            let mut checked = 0;
            for idx in 0..head.w.as_slice().len() {
                let numeric = {
                    let mut hp = head.clone();
                    hp.w.as_mut_slice()[idx] += eps as f32;
                    let mut hm = head.clone();
                    hm.w.as_mut_slice()[idx] -= eps as f32;
                    (router_loss_f64(&hp, &ex.features, ex.target).unwrap()
                        - router_loss_f64(&hm, &ex.features, ex.target).unwrap())
                        / (2.0 * f64::from(eps as f32))
                };
                let analytic = f64::from(g.d_weight.as_slice()[idx]);
                let rel = (analytic - numeric).abs() / analytic.abs().max(numeric.abs()).max(1e-3);
                assert!(rel < 1e-3, "w[{idx}] analytic {analytic} numeric {numeric}");
                checked += 1;
            }
            assert_eq!(checked, head.w.as_slice().len());
            for j in 0..head.b.as_slice().len() {
                let mut hp = head.clone();
                hp.b.as_mut_slice()[j] += eps as f32;
                let mut hm = head.clone();
                hm.b.as_mut_slice()[j] -= eps as f32;
                let numeric = (router_loss_f64(&hp, &ex.features, ex.target).unwrap()
                    - router_loss_f64(&hm, &ex.features, ex.target).unwrap())
                    / (2.0 * f64::from(eps as f32));
                let analytic = f64::from(g.d_bias.as_slice()[j]);
                let rel = (analytic - numeric).abs() / analytic.abs().max(numeric.abs()).max(1e-3);
                assert!(rel < 1e-3, "b[{j}] analytic {analytic} numeric {numeric}");
            }
            // f32 loss agrees with the f64 loss.
            let l64 = router_loss_f64(&head, &ex.features, ex.target).unwrap();
            assert!((f64::from(g.loss) - l64).abs() < 1e-4);
        }
    }

    #[test]
    fn gradient_rows_sum_to_zero_per_input() {
        let m = model();
        let ex = &synthetic_examples(&m, 1, 1).unwrap()[0];
        let g = router_gradient(&m.router, &ex.features, ex.target).unwrap();
        assert!(g.d_bias.as_slice().iter().sum::<f32>().abs() < 1e-6);
        let classes = g.d_bias.cols();
        for row in g.d_weight.as_slice().chunks_exact(classes) {
            assert!(row.iter().sum::<f32>().abs() < 1e-5);
        }
    }

    #[test]
    fn training_decreases_loss_and_updates_only_the_router() {
        let mut m = model();
        let before = m.clone();
        let report = train_router_head(&mut m, &ToyRouterOptions::default()).unwrap();
        assert_eq!(report.losses.len(), 60);
        assert!(report.losses[0].is_finite());
        assert!(
            report.final_loss < report.losses[0] * 0.9,
            "{:?}",
            (report.losses[0], report.final_loss)
        );
        // Monotone on this deterministic full-batch run with a small step.
        assert!(report.losses.windows(2).all(|w| w[1] <= w[0] + 1e-6));
        // Backbone frozen: every parameter except the router is bit-identical.
        for ((name, a), (_, b)) in before.named_params().iter().zip(m.named_params().iter()) {
            if name.starts_with("head.router") {
                assert_ne!(a, b, "{name} should have changed");
            } else {
                assert_eq!(a, b, "{name} must be frozen");
            }
        }
    }

    #[test]
    fn training_is_deterministic() {
        let (mut a, mut b) = (model(), model());
        let o = ToyRouterOptions {
            epochs: 5,
            ..ToyRouterOptions::default()
        };
        assert_eq!(
            train_router_head(&mut a, &o).unwrap(),
            train_router_head(&mut b, &o).unwrap()
        );
        assert_eq!(a, b);
    }

    #[test]
    fn bad_options_are_typed() {
        let mut m = model();
        let bad = |o: ToyRouterOptions| train_router_head(&mut model(), &o).err();
        let d = ToyRouterOptions::default();
        assert!(matches!(
            bad(ToyRouterOptions { epochs: 0, ..d }),
            Some(CoreError::TrainingInput { .. })
        ));
        assert!(matches!(
            bad(ToyRouterOptions { examples: 0, ..d }),
            Some(CoreError::TrainingInput { .. })
        ));
        assert!(matches!(
            bad(ToyRouterOptions {
                learning_rate: f32::NAN,
                ..d
            }),
            Some(CoreError::TrainingInput { .. })
        ));
        assert!(matches!(
            router_gradient(&m.router, &[0.0; 16], 99),
            Err(CoreError::TrainingInput { .. })
        ));
        assert!(matches!(
            router_gradient(&m.router, &[0.0; 3], 0),
            Err(CoreError::PooledWidth { .. })
        ));
        m.router.b.as_mut_slice()[0] = 0.0;
    }
}
