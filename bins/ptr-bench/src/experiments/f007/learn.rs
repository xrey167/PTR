//! Training of the two learned arms, on pilot-seed traffic only, under the same tuning budget.
//!
//! * The VCW arm's router is the repository's `RoutingLearner` (a Beta-Bernoulli posterior per
//!   (finding code, Pod) arm, Thompson sampling), trained on traffic routed uniformly inside the
//!   true finding map, tuned over four priors, and promoted only through `PromotionGate`.
//! * The gate baseline is an MoE-style softmax gate over all Pods whose logits are the verifier-
//!   reward means per (step type, Pod), with the finding code deliberately not an input. It is
//!   trained on traffic routed uniformly over all Pods, tuned over four temperatures, and
//!   promoted by the same rule (`compare_policies`, the function `PromotionGate` uses).
//!
//! Both read the same reward (the verifier's verdict through `classify`), the same stream sizes
//! (A: statistics, B: tuning, C: promotion) and the same selection estimator (doubly robust on
//! stream B). Neither sees a confirmatory seed: the streams are generated from pilot seeds only.

use std::collections::BTreeMap;
use std::sync::Arc;

use ptr_router::{
    compare_policies, estimate_policy, AdmissibleScorer, Confidence, Estimator, LearnedPolicy,
    LearnerConfig, LoggedDecision, PolicyBook, PromotionGate, PromotionRule, RewardModel,
    RouteClass, RouteError, RoutingLearner, ScoringContext, UniformScorer, Verdict,
};
use ptr_types::PodId;

use super::lane::{Lane, LaneConfig, RouteScorer};
use super::params::{self, Scale};
use super::task::{Kind, Task, KINDS};
use super::world::{RegistryMode, WorldId};

/// The step a routing decision is for, the only thing the gate conditions on besides the kind.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Step {
    Propose,
    Repair,
}

fn step_of(class: &RouteClass) -> Step {
    match class {
        RouteClass::Repair(_) => Step::Repair,
        RouteClass::Regenerate(_) => Step::Propose,
    }
}

/// An MoE-style gate: softmax over the admissible Pods of `mean / tau`, where `mean` is the
/// verifier-reward mean of (step, Pod) in the training log. It cannot see a finding code.
#[derive(Clone, Debug)]
pub struct GateScorer {
    tau: f64,
    means: BTreeMap<(Step, PodId), f64>,
}

impl GateScorer {
    pub fn fit(log: &[LoggedDecision], tau_permille: u64) -> Self {
        let mut counts: BTreeMap<(Step, PodId), (u64, u64)> = BTreeMap::new();
        for decision in log {
            let entry = counts
                .entry((step_of(decision.class()), decision.chosen().clone()))
                .or_default();
            entry.1 += 1;
            entry.0 += u64::from(decision.reward().passed());
        }
        Self {
            tau: tau_permille as f64 / 1000.0,
            means: counts
                .into_iter()
                .map(|(key, (cleared, total))| (key, (cleared as f64 + 1.0) / (total as f64 + 2.0)))
                .collect(),
        }
    }

    pub fn mean(&self, class: &RouteClass, pod: &PodId) -> f64 {
        self.means
            .get(&(step_of(class), pod.clone()))
            .copied()
            .unwrap_or(0.5)
    }

    pub fn tau_permille(&self) -> u64 {
        (self.tau * 1000.0).round() as u64
    }
}

impl AdmissibleScorer for GateScorer {
    fn weights(&self, context: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
        // Relative to the best admissible Pod, so that the best always has weight 1 and a low
        // temperature cannot underflow every weight.
        let means: Vec<f64> = context
            .admissible
            .iter()
            .map(|pod| self.mean(context.class, pod))
            .collect();
        let best = means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Ok(context
            .admissible
            .iter()
            .zip(means)
            .map(|(pod, mean)| (pod.clone(), ((mean - best) / self.tau).exp().max(1e-12)))
            .collect())
    }
}

impl RewardModel for GateScorer {
    fn predict(&self, class: &RouteClass, pod: &PodId) -> f64 {
        self.mean(class, pod)
    }
}

/// What training produced for one lane.
pub struct KindTraining {
    pub kind: Kind,
    pub vcw_scorer: ptr_router::ActiveScorer,
    pub vcw_promoted: bool,
    pub vcw_prior: u64,
    pub vcw_lcb: f64,
    pub vcw_difference: f64,
    pub vcw_records: usize,
    pub vcw_digest: [u8; 32],
    pub vcw_candidates: Vec<(u64, f64)>,
    pub gate: Arc<GateScorer>,
    pub gate_promoted: bool,
    pub gate_tau: u64,
    pub gate_lcb: f64,
    pub gate_difference: f64,
    pub gate_records: usize,
    pub gate_candidates: Vec<(u64, f64)>,
    /// The gate-weighted proposers of best-of-N: tuned and promoted like the gate, on the
    /// proposal decisions alone.
    pub bon_gate: Arc<GateScorer>,
    pub bon_promoted: bool,
    pub bon_tau: u64,
    pub bon_lcb: f64,
    pub bon_difference: f64,
    pub bon_candidates: Vec<(u64, f64)>,
    /// Whether a tuned parameter sits at the edge of its grid with a materially better value than
    /// its neighbour (the grid was too narrow).
    pub vcw_edge: bool,
    /// The edge candidate's advantage over its inner neighbour on stream B (DR value), whatever its
    /// significance.
    pub vcw_edge_excess: f64,
    pub gate_edge_excess: f64,
    pub bon_edge_excess: f64,
    /// The paired standard error of each edge decision (0 where the winner is not at an edge).
    pub vcw_edge_se: f64,
    pub gate_edge_se: f64,
    pub bon_edge_se: f64,
    pub gate_edge: bool,
    pub bon_edge: bool,
    /// Sensitivity of the edge test: it flags the tuned gate against its worst candidate (a
    /// materially worse policy). False means the test is blind, which is as invalid as an edge.
    pub gate_edge_test_sees: bool,
    pub bon_edge_test_sees: bool,
    pub training_requests: u64,
    pub failures: Failures,
    /// The pilot seeds the streams were drawn from.
    pub seeds: [u64; 3],
}

pub struct Trained {
    pub kinds: Vec<KindTraining>,
}

impl Trained {
    pub fn of(&self, kind: Kind) -> &KindTraining {
        &self.kinds[kind.index()]
    }
}

/// Serve a training stream. A loop error or panic is a product defect, counted and carried to the
/// run's counters (`Failures`), not a reason to stop the harness.
fn run_stream(
    lane: &mut Lane,
    seed: u64,
    kind: Kind,
    tag: &str,
    n: usize,
    failures: &mut Failures,
) {
    for index in 0..n {
        let task = Arc::new(Task::generate(seed, index as u64, kind));
        let id = format!("train-{tag}-{}-{index}", kind.name());
        match lane.serve(&id, &task).tag {
            super::lane::EndTag::Error => failures.loop_errors += 1,
            super::lane::EndTag::Panic => failures.panics += 1,
            _ => {}
        }
    }
}

/// Product defects seen while training.
#[derive(Clone, Copy, Debug, Default)]
pub struct Failures {
    pub loop_errors: u64,
    pub panics: u64,
}

fn lane_config(
    world: WorldId,
    kind: Kind,
    registry: RegistryMode,
    seed: u64,
    learner: &Arc<RoutingLearner>,
) -> LaneConfig {
    let book = PolicyBook::new(kind.name());
    LaneConfig {
        world,
        kind,
        registry,
        scorer: RouteScorer::Active(book.scorer()),
        hints_steer: false,
        schema: false,
        l3: false,
        broadcast: true,
        sim_seed: seed,
        router_seed: 31,
        learner: Some(Arc::clone(learner)),
        calls: params::VERIFIER_CALL_BUDGET,
        shared_first_draft: false,
    }
}

fn learner_for(kind: Kind, prior: u64, seed: u64) -> Result<Arc<RoutingLearner>, String> {
    let mut config = LearnerConfig::new(kind.name(), seed);
    config.prior = prior as f64;
    RoutingLearner::new(config)
        .map(Arc::new)
        .map_err(|error| format!("learner: {error}"))
}

fn dr_value(
    log: &[LoggedDecision],
    policy: &dyn AdmissibleScorer,
    model: &dyn RewardModel,
) -> Result<f64, String> {
    estimate_policy(log, policy, model)
        .map(|estimate| estimate.value(Estimator::DoublyRobust))
        .map_err(|error| format!("selection estimate: {error}"))
}

fn rule() -> PromotionRule {
    PromotionRule::new(
        params::PROMOTION_MARGIN_PERMILLE as f64 / 1000.0,
        params::PROMOTION_MIN_RECORDS,
        params::PROMOTION_MIN_ESS as f64,
    )
}

/// The inner neighbour of the chosen grid index when it sits at an edge of the (ascending) grid.
pub fn edge_neighbour(len: usize, chosen: usize) -> Option<usize> {
    if len < 2 {
        None
    } else if chosen == 0 {
        Some(1)
    } else if chosen + 1 == len {
        Some(len - 2)
    } else {
        None
    }
}

/// Whether the optimum may lie outside the grid: the candidate chosen at an edge beats its inner
/// neighbour on stream B with a lower bound, at `EDGE_Z_MILLI` standard errors of their paired
/// difference (the same stream, so the same noise), above zero. Returns the excess and the paired
/// standard error too.
fn edge_excess(
    log_b: &[LoggedDecision],
    edge: (&dyn AdmissibleScorer, &dyn RewardModel),
    neighbour: (&dyn AdmissibleScorer, &dyn RewardModel),
) -> Result<(bool, f64, f64), String> {
    let z = Confidence::from_quantile(params::EDGE_Z_MILLI as f64 / 1000.0)
        .map_err(|error| format!("edge confidence: {error}"))?;
    // The reward model is the edge candidate's own; the neighbour is judged as a policy.
    let comparison = compare_policies(
        log_b,
        edge.0,
        neighbour.0,
        edge.1,
        Estimator::DoublyRobust,
        z,
    )
    .map_err(|error| format!("edge comparison: {error}"))?;
    Ok((
        comparison.difference_lcb > 0.0,
        comparison.difference,
        comparison.difference_se,
    ))
}

/// What tuning and promoting a softmax gate produced.
struct GateTuning {
    gate: GateScorer,
    promoted: bool,
    lcb: f64,
    difference: f64,
    candidates: Vec<(u64, f64)>,
    edge: bool,
    edge_excess: f64,
    edge_se: f64,
    edge_test_sees: bool,
}

/// Tune the temperature on stream B and promote on stream C, exactly as for the learned router. With
/// `proposals_only` the gate is fitted and judged on proposal decisions alone (the proposers of
/// best-of-N).
fn tune_gate(
    log_a: &[LoggedDecision],
    log_b: &[LoggedDecision],
    log_c: &[LoggedDecision],
    proposals_only: bool,
) -> Result<GateTuning, String> {
    let keep = |log: &[LoggedDecision]| -> Vec<LoggedDecision> {
        log.iter()
            .filter(|d| !proposals_only || matches!(d.class(), RouteClass::Regenerate(_)))
            .cloned()
            .collect()
    };
    let (log_a, log_b, log_c) = (keep(log_a), keep(log_b), keep(log_c));
    let mut candidates = Vec::new();
    let mut gates: Vec<GateScorer> = Vec::new();
    let mut best: Option<(usize, f64)> = None;
    for tau in params::GATE_TAU_PERMILLE {
        let gate = GateScorer::fit(&log_a, tau);
        let value = dr_value(&log_b, &gate, &gate)?;
        candidates.push((tau, value));
        if best.is_none_or(|(_, top)| value > top) {
            best = Some((gates.len(), value));
        }
        gates.push(gate);
    }
    let (chosen, _) = best.ok_or("no gate candidate")?;
    let (edge, edge_excess_value, edge_se) = match edge_neighbour(gates.len(), chosen) {
        Some(inner) => edge_excess(
            &log_b,
            (&gates[chosen], &gates[chosen]),
            (&gates[inner], &gates[inner]),
        )?,
        None => (false, 0.0, 0.0),
    };
    // The test must be able to see a materially worse policy: the tuned gate against the worst
    // candidate of the grid.
    let worst = candidates
        .iter()
        .enumerate()
        .min_by(|a, b| a.1 .1.total_cmp(&b.1 .1))
        .map_or(chosen, |(index, _)| index);
    let edge_test_sees = worst == chosen
        || edge_excess(
            &log_b,
            (&gates[chosen], &gates[chosen]),
            (&gates[worst], &gates[worst]),
        )?
        .0;
    let gate = gates.swap_remove(chosen);
    let comparison = compare_policies(
        &log_c,
        &gate,
        &UniformScorer,
        &gate,
        Estimator::DoublyRobust,
        Confidence::ONE_SIDED_95,
    )
    .map_err(|error| format!("gate comparison: {error}"))?;
    let rule = rule();
    let promoted = log_c.len() >= rule.min_records
        && comparison.target.effective_sample_size >= rule.min_effective_sample
        && comparison.difference_lcb > rule.margin;
    Ok(GateTuning {
        gate,
        promoted,
        lcb: comparison.difference_lcb,
        difference: comparison.difference,
        candidates,
        edge,
        edge_excess: edge_excess_value,
        edge_se,
        edge_test_sees,
    })
}

/// Train both learned arms of one lane of `world` for the run with seed `seed`.
pub fn train_kind(
    world: WorldId,
    kind: Kind,
    seed: u64,
    scale: Scale,
) -> Result<KindTraining, String> {
    let [a, b, c] = scale.training();
    let seeds = params::training_seeds(seed);
    let world_salt = world as u64;
    let mut trained_requests = 0_u64;
    let mut failures = Failures::default();

    // ---- the VCW arm's router: true map, the priors on identical traffic -------------------
    let mut candidates: Vec<(u64, LearnedPolicy)> = Vec::new();
    let mut kept: Option<(Lane, Arc<RoutingLearner>)> = None;
    for prior in params::PRIORS {
        let learner = learner_for(kind, prior, seeds[0])?;
        let mut lane = Lane::new(lane_config(
            world,
            kind,
            RegistryMode::True,
            seeds[0] ^ world_salt,
            &learner,
        ));
        run_stream(&mut lane, seeds[0], kind, "a", a, &mut failures);
        trained_requests += a as u64;
        candidates.push((prior, learner.snapshot()));
        if prior == params::PRIORS[0] {
            kept = Some((lane, learner));
        }
    }
    let (mut lane, learner) = kept.ok_or("no first prior")?;
    let cutoff_a = candidates[0].1.cutoff();
    run_stream(&mut lane, seeds[1], kind, "b", b, &mut failures);
    let after_b = learner.log_from(cutoff_a);
    let len_b = after_b.len();
    run_stream(&mut lane, seeds[2], kind, "c", c, &mut failures);
    trained_requests += (b + c) as u64;
    let both = learner.log_from(cutoff_a);
    let log_b = &both[..len_b];
    let log_c = &both[len_b..];
    let mut vcw_candidates = Vec::new();
    let mut best: Option<(u64, &LearnedPolicy, f64)> = None;
    for (prior, policy) in &candidates {
        let value = dr_value(log_b, policy, policy)?;
        vcw_candidates.push((*prior, value));
        if best.is_none_or(|(_, _, top)| value > top) {
            best = Some((*prior, policy, value));
        }
    }
    let (vcw_prior, chosen, _) = best.ok_or("no candidate")?;
    let chosen_index = candidates
        .iter()
        .position(|(p, _)| *p == vcw_prior)
        .unwrap_or(0);
    let (vcw_edge, vcw_edge_excess, vcw_edge_se) =
        match edge_neighbour(candidates.len(), chosen_index) {
            Some(inner) => edge_excess(
                log_b,
                (&candidates[chosen_index].1, &candidates[chosen_index].1),
                (&candidates[inner].1, &candidates[inner].1),
            )?,
            None => (false, 0.0, 0.0),
        };
    let mut book = PolicyBook::new(kind.name());
    let promotion = PromotionGate::new(rule()).map_err(|e| format!("gate: {e}"))?;
    let assessment = promotion
        .assess(log_c, chosen, &book.scorer(), book.active_digest())
        .map_err(|error| format!("promotion assessment: {error}"))?;
    let (vcw_difference, vcw_lcb) = (
        assessment.comparison.difference,
        assessment.comparison.difference_lcb,
    );
    let vcw_promoted = match assessment.verdict {
        Verdict::Promote(evidence) => book
            .promote(chosen.clone(), evidence)
            .map(|_| true)
            .map_err(|error| format!("promote: {error:?}"))?,
        Verdict::Refuse(_) => false,
    };
    let vcw_digest = book.active_digest();
    let vcw_scorer = book.scorer();

    // ---- the gate: generic registry, one pass of the same three streams ---------------------
    let learner = learner_for(kind, 1, seeds[0])?;
    let mut lane = Lane::new(lane_config(
        world,
        kind,
        RegistryMode::Generic,
        seeds[0] ^ world_salt,
        &learner,
    ));
    run_stream(&mut lane, seeds[0], kind, "a", a, &mut failures);
    let log_a = learner.log_from(0);
    let cutoff = learner.snapshot().cutoff();
    run_stream(&mut lane, seeds[1], kind, "b", b, &mut failures);
    let len_b = learner.log_from(cutoff).len();
    run_stream(&mut lane, seeds[2], kind, "c", c, &mut failures);
    trained_requests += (a + b + c) as u64;
    let both = learner.log_from(cutoff);
    let log_b = &both[..len_b];
    let log_c = &both[len_b..];
    let tuned = tune_gate(&log_a, log_b, log_c, false)?;
    let bon = tune_gate(&log_a, log_b, log_c, true)?;
    Ok(KindTraining {
        kind,
        vcw_scorer,
        vcw_promoted,
        vcw_prior,
        vcw_lcb,
        vcw_difference,
        vcw_records: log_c.len(),
        vcw_digest,
        vcw_candidates,
        gate_tau: tuned.gate.tau_permille(),
        gate: Arc::new(tuned.gate),
        gate_promoted: tuned.promoted,
        gate_lcb: tuned.lcb,
        gate_difference: tuned.difference,
        gate_records: log_c.len(),
        gate_candidates: tuned.candidates,
        bon_tau: bon.gate.tau_permille(),
        bon_gate: Arc::new(bon.gate),
        bon_promoted: bon.promoted,
        bon_lcb: bon.lcb,
        bon_difference: bon.difference,
        bon_candidates: bon.candidates,
        vcw_edge,
        vcw_edge_excess,
        gate_edge_excess: tuned.edge_excess,
        bon_edge_excess: bon.edge_excess,
        vcw_edge_se,
        gate_edge_se: tuned.edge_se,
        bon_edge_se: bon.edge_se,
        gate_edge: tuned.edge,
        bon_edge: bon.edge,
        gate_edge_test_sees: tuned.edge_test_sees,
        bon_edge_test_sees: bon.edge_test_sees,
        training_requests: trained_requests,
        failures,
        seeds,
    })
}

pub fn train(world: WorldId, seed: u64, scale: Scale) -> Result<Trained, String> {
    let mut kinds = Vec::new();
    for kind in KINDS {
        kinds.push(train_kind(world, kind, seed, scale)?);
    }
    Ok(Trained { kinds })
}
