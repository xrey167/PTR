//! Public-API tests of the deterministic CPU reference forward pass.

mod common;

use common::{fixed_slots, tiny_config};
use ptr_core::action_head::ActionIr;
use ptr_core::epistemic_workspace::EpistemicWorkspace;
use ptr_core::latent_reasoner::{HaltReason, ReasoningBudget, RecurrenceOptions};
use ptr_core::operator_router::operator_classes;
use ptr_core::training::{
    router_gradient, router_loss_f64, synthetic_examples, train_router_head, ToyRouterOptions,
};
use ptr_core::{CoreConfig, CoreError, CoreModel};
use ptr_types::{CapabilityId, Effect, Generation, Revision, TypeId, Validity};

fn medium() -> RecurrenceOptions {
    RecurrenceOptions {
        budget: ReasoningBudget::Medium,
        epsilon: 0.01,
    }
}

/// Golden values of the tiny config (d_model 4, 2 heads, d_ff 8, seed 7) on the
/// fixed four slots. They were produced once by this implementation and are pinned
/// to catch unintended numerical drift; they are not an independent reference. The
/// independent hand check lives next to the attention code (a 2x2 attention worked
/// out from e^(1/sqrt 2) = 2.02811498). Tolerance 1e-5 allows for libm differences
/// in `exp`/`tanh` across platforms.
const GOLDEN_POOLED: [f32; 4] = [-1.014_180_8, -0.215_652_36, 1.628_248_9, -0.398_416_04];
const GOLDEN_HALT_PROBS: [f32; 3] = [0.452_025_6, 0.451_338_95, 0.490_325_1];
const GOLDEN_VERIFIER: f32 = 0.361_186_98;

#[test]
fn golden_forward_values_are_pinned() {
    let m = CoreModel::new(tiny_config()).unwrap();
    let out = m.forward(&fixed_slots(), &medium()).unwrap();
    for (got, want) in out.pooled.iter().zip(GOLDEN_POOLED) {
        assert!((got - want).abs() < 1e-5, "pooled {got} vs {want}");
    }
    assert_eq!(out.steps_used(), 3);
    assert_eq!(out.halt, HaltReason::Halted);
    for (t, want) in out.traces.iter().zip(GOLDEN_HALT_PROBS) {
        assert!((t.halt_probability - want).abs() < 1e-5);
    }
    // 0.4520 + 0.4513 = 0.9034 < 0.99 after two steps; 1.3937 >= 0.99 after three.
    assert!(out.traces[1].cumulative_halt < 0.99);
    assert!(out.traces[2].cumulative_halt >= 0.99);
    let v = m.verify_probability(&out.pooled).unwrap().get();
    assert!((v - GOLDEN_VERIFIER).abs() < 1e-5);
    // Post layer-norm rows have zero mean, so the pooled mean of a full-live
    // workspace must too: an independent structural check on the golden vector.
    assert!(GOLDEN_POOLED.iter().sum::<f32>().abs() < 1e-5);
    let probs = m.router_probabilities(&out.pooled).unwrap();
    assert_eq!(probs.len(), operator_classes().len());
    assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
}

#[test]
fn forward_is_reproducible_across_independent_instances() {
    let a = CoreModel::new(tiny_config()).unwrap();
    let b = CoreModel::new(tiny_config()).unwrap();
    assert_eq!(a, b);
    assert_eq!(
        a.forward(&fixed_slots(), &medium()).unwrap(),
        b.forward(&fixed_slots(), &medium()).unwrap()
    );
}

#[test]
fn revoked_slot_receives_exactly_zero_attention_through_the_public_api() {
    let m = CoreModel::new(tiny_config()).unwrap();
    let mut slots = fixed_slots();
    slots[2].validity = Validity::Revoked;
    let out = m.forward(&slots, &medium()).unwrap();
    assert!(!out.last_attention.is_empty());
    for head in &out.last_attention {
        for row in 0..slots.len() {
            let r = head.row(row).unwrap();
            assert_eq!(r[2], 0.0);
            assert!((r.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }
}

#[test]
fn budget_is_respected_and_accounted_in_the_workspace() {
    let m = CoreModel::new(tiny_config()).unwrap();
    let mut ws = EpistemicWorkspace {
        slots: fixed_slots(),
        step: 0,
    };
    let opts = RecurrenceOptions {
        budget: ReasoningBudget::Max(2),
        epsilon: 0.0,
    };
    let out = m.reason_workspace(&mut ws, &opts).unwrap();
    assert!(out.steps_used() <= 2);
    assert_eq!(ws.step, out.steps_used());
    let none = RecurrenceOptions {
        budget: ReasoningBudget::None,
        epsilon: 0.0,
    };
    let out = m.reason_workspace(&mut ws, &none).unwrap();
    assert_eq!(out.halt, HaltReason::NoBudget);
    // The config cap (4) bounds an oversized budget.
    let big = RecurrenceOptions {
        budget: ReasoningBudget::Max(500),
        epsilon: 0.0,
    };
    assert!(m.forward(&fixed_slots(), &big).unwrap().steps_used() <= 4);
}

fn action(op: &str) -> ActionIr {
    ActionIr {
        operation: op.into(),
        target: format!("target-{op}"),
        capability: CapabilityId::from("cap"),
        effect: Effect::Pure,
        input_type: TypeId::from("t"),
        generation: Generation(2),
        revision: Revision(5),
        payload: Vec::new(),
    }
}

#[test]
fn learned_action_score_cannot_override_a_deterministic_rejection() {
    let m = CoreModel::new(tiny_config()).unwrap();
    let out = m.forward(&fixed_slots(), &medium()).unwrap();
    let candidates: Vec<ActionIr> = ["a", "b", "c", "d", "e"]
        .iter()
        .map(|o| action(o))
        .collect();
    let favourite = m
        .rank_actions(&out.pooled, &candidates, |_| true)
        .unwrap()
        .index;
    let vetoed = candidates[favourite].clone();
    let choice = m
        .rank_actions(&out.pooled, &candidates, |a| *a != vetoed)
        .unwrap();
    assert_ne!(choice.index, favourite);
    assert_eq!(
        m.rank_actions(&out.pooled, &candidates, |_| false).err(),
        Some(CoreError::AllCandidatesRejected)
    );
}

#[test]
fn toy_router_training_lowers_loss_and_raises_accuracy() {
    let mut m = CoreModel::new(CoreConfig {
        d_model: 16,
        heads: 2,
        d_ff: 32,
        max_slots: 8,
        ..CoreConfig::default()
    })
    .unwrap();
    let report = train_router_head(&mut m, &ToyRouterOptions::default()).unwrap();
    assert!(report.final_loss < report.losses[0] * 0.8);
    assert!(report.final_accuracy > report.initial_accuracy);
    // The fitted head is what the public router now uses.
    let ex = &synthetic_examples(&m, 1, 99).unwrap()[0];
    let probs = m.router_probabilities(&ex.features).unwrap();
    assert_eq!(probs.len(), operator_classes().len());
}

#[test]
fn public_bias_gradient_is_softmax_minus_onehot() {
    let m = CoreModel::new(tiny_config()).unwrap();
    let out = m.forward(&fixed_slots(), &medium()).unwrap();
    let head = m.router_head();
    let target = 3;
    let g = router_gradient(head, &out.pooled, target).unwrap();
    let probs = m.router_probabilities(&out.pooled).unwrap();
    for (j, p) in probs.iter().enumerate() {
        let expected = f64::from(*p) - if j == target { 1.0 } else { 0.0 };
        assert!((f64::from(g.d_bias.as_slice()[j]) - expected).abs() < 1e-5);
    }
    let l = router_loss_f64(head, &out.pooled, target).unwrap();
    assert!((f64::from(g.loss) - l).abs() < 1e-4);
}
