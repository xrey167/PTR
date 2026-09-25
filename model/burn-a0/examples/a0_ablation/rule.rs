//! The operator-routing v1 label rule, written again here rather than read from
//! the generator: the loader re-derives every label with it and refuses to run on
//! any disagreement, so a generator bug cannot become a silently wrong label.
//!
//! z_k = sum over Live facts with cb > 0 of
//!       G[cb] * (W[e] * A[r,R]_k + U[e] * [k = Probabilistic]) - LAMBDA * max(0, COST_k - BUDGET_B)
//! label = argmax_k z_k; values within 1e-9 of the maximum tie, and a tie goes to
//! the lowest COST, then the lowest codebook code. Accumulation is in f64, facts
//! in slot order and operators in codebook-code order, as in the generator.

use ptr_types::{Codebook, EpistemicState, ReasoningOperator, SemanticRole, Validity};

pub const OPERATORS: usize = 11;
const TIE_TOLERANCE: f64 = 1e-9;
const LAMBDA: f64 = 1.0;
const BUDGET: [f64; 3] = [0.4, 0.7, 1.0];
const GAIN: [f64; 5] = [0.0, 0.5, 1.0, 1.0, 1.25];
const PRIMARY: f64 = 2.0;
const SECONDARY: f64 = 0.9;

/// One fact as the rule sees it.
#[derive(Clone, Copy, Debug)]
pub struct RuleFact {
    pub role: SemanticRole,
    pub epistemic: EpistemicState,
    /// floor(5c).
    pub bucket: usize,
    pub validity: Validity,
}

fn epistemic_weight(e: EpistemicState) -> (f64, f64) {
    match e {
        EpistemicState::Unknown => (0.30, 0.35),
        EpistemicState::Assumed => (0.55, 0.60),
        EpistemicState::Hypothesis => (0.80, 0.85),
        EpistemicState::Observed => (1.30, 0.0),
        EpistemicState::Inferred => (1.05, 0.0),
        EpistemicState::Verified => (1.55, 0.0),
    }
}

pub fn cost(op: ReasoningOperator) -> f64 {
    match op {
        ReasoningOperator::Semantic => 0.2,
        ReasoningOperator::Deductive => 0.3,
        ReasoningOperator::Probabilistic => 0.5,
        ReasoningOperator::Statistical => 0.5,
        ReasoningOperator::Temporal => 0.5,
        ReasoningOperator::Causal => 0.6,
        ReasoningOperator::Search => 0.6,
        ReasoningOperator::Optimization => 0.8,
        ReasoningOperator::Simulation => 0.8,
        ReasoningOperator::Symbolic => 0.4,
        ReasoningOperator::ExternalPod => 0.9,
    }
}

/// op(R): tabular, temporal, interventional, textual.
fn regime_operator(regime: usize) -> ReasoningOperator {
    [
        ReasoningOperator::Statistical,
        ReasoningOperator::Temporal,
        ReasoningOperator::Causal,
        ReasoningOperator::Semantic,
    ][regime]
}

fn affinity(role: SemanticRole, regime: usize) -> (ReasoningOperator, ReasoningOperator) {
    use ReasoningOperator as O;
    match role {
        SemanticRole::Goal => (O::Search, O::Optimization),
        SemanticRole::Constraint => (O::Optimization, O::Symbolic),
        SemanticRole::Claim => (O::Deductive, regime_operator(regime)),
        SemanticRole::Evidence => (regime_operator(regime), O::Probabilistic),
        SemanticRole::Resource => (O::ExternalPod, O::Search),
        SemanticRole::Capability => (O::Simulation, O::ExternalPod),
        SemanticRole::Relation => (O::Causal, O::Deductive),
        SemanticRole::Procedure => (O::Symbolic, O::Simulation),
        SemanticRole::Action => (O::ExternalPod, O::Temporal),
    }
}

/// Every operator, indexed by its codebook code.
pub fn operators_by_code() -> [ReasoningOperator; OPERATORS] {
    use ReasoningOperator as O;
    let mut out = [O::Semantic; OPERATORS];
    for op in [
        O::Semantic,
        O::Deductive,
        O::Probabilistic,
        O::Statistical,
        O::Temporal,
        O::Causal,
        O::Search,
        O::Optimization,
        O::Simulation,
        O::Symbolic,
        O::ExternalPod,
    ] {
        let code = Codebook::V1.code_of(op).expect("v1 operator").index();
        out[usize::from(code)] = op;
    }
    out
}

/// z_k for every operator, by codebook code.
pub fn utilities(facts: &[RuleFact], regime: usize, budget: usize) -> [f64; OPERATORS] {
    let ops = operators_by_code();
    let mut z = [0.0_f64; OPERATORS];
    for fact in facts {
        if fact.validity != Validity::Live {
            continue;
        }
        let gain = GAIN[fact.bucket];
        if gain == 0.0 {
            continue;
        }
        let (primary, secondary) = affinity(fact.role, regime);
        let (w, u) = epistemic_weight(fact.epistemic);
        for (k, op) in ops.iter().enumerate() {
            let a = if *op == primary {
                PRIMARY
            } else if *op == secondary {
                SECONDARY
            } else {
                0.0
            };
            let bonus = if *op == ReasoningOperator::Probabilistic {
                u
            } else {
                0.0
            };
            let penalty = LAMBDA * (cost(*op) - BUDGET[budget]).max(0.0);
            z[k] += gain * (w * a + bonus) - penalty;
        }
    }
    z
}

/// The rule's argmax with its tie-break, as a codebook code.
pub fn decide(z: &[f64; OPERATORS]) -> usize {
    let ops = operators_by_code();
    let best = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (0..OPERATORS)
        .filter(|&k| z[k] >= best - TIE_TOLERANCE)
        .min_by(|&a, &b| {
            cost(ops[a])
                .partial_cmp(&cost(ops[b]))
                .expect("finite costs")
                .then(a.cmp(&b))
        })
        .expect("eleven operators")
}

pub fn label(facts: &[RuleFact], regime: usize, budget: usize) -> usize {
    decide(&utilities(facts, regime, budget))
}
