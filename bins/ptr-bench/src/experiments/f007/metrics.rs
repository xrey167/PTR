//! The rule lists of F007 (coverage, falsifiers, invalidity controls) and the controls the
//! harness evaluates per seed.
//!
//! `criteria.toml` holds the same three lists; a harness test pins that file to these functions,
//! in order. The grammar is the repository's: `<left> <op> <right>`, integers only.
//!
//! * `falsifiers` are product defects observed from events. Any one trips, FALSIFIED.
//! * `invalid_if` are the controls that show the experiment could discriminate. Any one trips,
//!   INVALID (a statement about the experiment, not the hypothesis).
//! * `coverage` are the prerequisites; unmet, the seed is INVALID.

use super::arms::{Arm, Pair, WorldRun};
use super::params;

/// The product counters, in the order of the rule list.
pub const FALSIFIER_COUNTERS: [&str; 13] = [
    "false_accepts",
    "unverified_returns",
    "double_verifications",
    "budget_violations",
    "routing_outside_admissible",
    "verdict_mismatches",
    "verdict_overrides",
    "verifier_call_mismatches",
    "calls_after_acceptance",
    "l3_unadmitted",
    "loop_errors",
    "product_panics",
    "exhausted_as_success",
];

/// The control counters, in the order of the rule list.
pub const CONTROL_COUNTERS: [&str; 11] = [
    "control_shuffled",
    "control_null",
    "control_baseline",
    "control_workspace",
    "control_ctx",
    "control_training",
    "control_grid_edge",
    "control_oracle",
    "control_bon",
    "control_l3",
    "control_first_draft",
];

/// The per-rule shortfall counters, in the order of the coverage rules.
pub const COVERAGE_COUNTERS: [&str; 5] = [
    "coverage_unmet_kind_share",
    "coverage_unmet_hard_events",
    "coverage_unmet_repairs",
    "coverage_unmet_exhausted",
    "coverage_unmet_births",
];

pub fn falsifiers() -> Vec<String> {
    FALSIFIER_COUNTERS
        .iter()
        .map(|name| format!("{name} <= 0"))
        .collect()
}

pub fn invalid_if() -> Vec<String> {
    CONTROL_COUNTERS
        .iter()
        .map(|name| format!("{name} <= 0"))
        .collect()
}

pub fn coverage() -> Vec<String> {
    vec![
        format!(
            "coverage.min_kind_share_permille >= {}",
            params::MIN_REQUESTS_PER_KIND_PERMILLE
        ),
        format!(
            "coverage.min_hard_events_per_code >= {}",
            params::MIN_HARD_EVENTS_PER_CODE
        ),
        format!(
            "coverage.repairs_applied >= {}",
            params::MIN_REPAIRS_APPLIED
        ),
        format!("coverage.exhausted_runs >= {}", params::MIN_EXHAUSTED_RUNS),
        format!("coverage.births >= {}", params::MIN_BIRTHS),
    ]
}

/// The outcome of the harness controls for one seed: `(counter, failures)` and the observed
/// gaps, in permille of requests (positive means the first arm is ahead).
pub struct Controls {
    pub counters: Vec<(&'static str, i64)>,
    pub observed: Vec<(String, i64)>,
}

fn gap_permille(a: &super::arms::ArmResult, b: &super::arms::ArmResult) -> i64 {
    let n = a.requests.max(1) as i64;
    (a.successes as i64 - b.successes as i64) * 1000 / n
}

/// `a - b >= threshold` in permille of requests, exactly.
fn at_least(a: &super::arms::ArmResult, b: &super::arms::ArmResult, threshold: i64) -> bool {
    (a.successes as i64 - b.successes as i64) * 1000 >= threshold * a.requests as i64
}

fn within(a: &super::arms::ArmResult, b: &super::arms::ArmResult, band: i64) -> bool {
    let diff = (a.successes as i64 - b.successes as i64).abs();
    diff * 1000 <= band * a.requests as i64
}

pub fn controls(worlds: &[WorldRun], probe: (u64, u64, u64)) -> Controls {
    let world = |name: &str| worlds.iter().find(|w| w.world.name() == name);
    let mut counters = Vec::new();
    let mut observed = Vec::new();
    let hetero = world("hetero");
    let homog = world("homog");
    let ctx = world("ctx");

    // The wrong finding map must lose clearly, or the experiment cannot see a bad map.
    let mut shuffled = 0;
    if let Some(w) = hetero {
        let (full, wrong) = (w.arm(Arm::Full), w.arm(Arm::Shuffled));
        observed.push((
            "hetero.full_minus_shuffled".to_string(),
            gap_permille(full, wrong),
        ));
        if !at_least(full, wrong, params::SHUFFLED_MIN_LOSS_PERMILLE) {
            shuffled += 1;
        }
    }
    counters.push(("control_shuffled", shuffled));

    // With every Pod equally good, finding routing has nothing to exploit: ties.
    let mut null = 0;
    if let Some(w) = homog {
        let routed = w.arm(Arm::NoL3);
        for other in [Arm::Gate, Arm::NoFinding, Arm::Shuffled] {
            let other = w.arm(other);
            observed.push((
                format!("homog.no_l3_minus_{}", other.arm.name()),
                gap_permille(routed, other),
            ));
            if !within(routed, other, params::NULL_MAX_GAP_PERMILLE) {
                null += 1;
            }
        }
    }
    counters.push(("control_null", null));

    // The gate must be a trained baseline, not a straw man: it beats the untrained pool.
    let mut baseline = 0;
    if let Some(w) = hetero {
        let (gate, pool) = (w.arm(Arm::Gate), w.arm(Arm::NoFinding));
        observed.push((
            "hetero.gate_minus_no_finding".to_string(),
            gap_permille(gate, pool),
        ));
        if !at_least(gate, pool, params::BASELINE_MIN_GAIN_PERMILLE) {
            baseline += 1;
        }
    }
    counters.push(("control_baseline", baseline));

    // Where no Pod reads the broadcast, withholding it changes no outcome at all.
    let mut workspace = 0;
    for w in [hetero, homog].into_iter().flatten() {
        let pair = Pair::of(w.arm(Arm::Full), w.arm(Arm::NoWorkspace));
        observed.push((
            format!("{}.full_vs_no_workspace_discordant", w.world.name()),
            (pair.only_a + pair.only_b) as i64,
        ));
        if pair.only_a + pair.only_b != 0 {
            workspace += 1;
        }
    }
    counters.push(("control_workspace", workspace));

    // The broadcast path is a working mechanism, checked without reference to any success rate (the
    // success-rate contrast is the workspace secondary hypothesis): in the probe a Pod reads the
    // broadcast and delivering it changes the candidates produced; withheld, it is never read; in the
    // `ctx` world the full arm reads it and the no-workspace arm never does.
    let mut ctx_failures = 0;
    let (reads_on, reads_off, changed) = probe;
    observed.push(("ctx.probe_reads".to_string(), reads_on as i64));
    observed.push(("ctx.probe_requests_changed".to_string(), changed as i64));
    if reads_on == 0 || reads_off != 0 || changed == 0 {
        ctx_failures += 1;
    }
    if let Some(w) = ctx {
        let (full, blind) = (w.arm(Arm::Full), w.arm(Arm::NoWorkspace));
        observed.push((
            "ctx.full_broadcast_reads".to_string(),
            full.broadcast_reads as i64,
        ));
        if full.broadcast_reads == 0 || blind.broadcast_reads != 0 {
            ctx_failures += 1;
        }
    }
    counters.push(("control_ctx", ctx_failures));

    // No learned part may be silently replaced by uniform routing, and no tuned parameter may sit
    // at the edge of a grid whose neighbour is materially worse, and the edge test must be able to see a
    // materially worse policy (a blind test is as invalid as a flagged edge). Hetero only.
    let (mut unpromoted, mut edges) = (0, 0);
    if let Some(w) = hetero {
        for kind in &w.trained.kinds {
            unpromoted += i64::from(!kind.vcw_promoted)
                + i64::from(!kind.gate_promoted)
                + i64::from(!kind.bon_promoted);
            edges += i64::from(kind.vcw_edge)
                + i64::from(kind.gate_edge)
                + i64::from(kind.bon_edge)
                + i64::from(!kind.gate_edge_test_sees)
                + i64::from(!kind.bon_edge_test_sees);
        }
    }
    counters.push(("control_training", unpromoted));
    counters.push(("control_grid_edge", edges));

    // The oracle fixed-Pod reference is an upper reference: not below the full loop by more than
    // the tolerance (the full loop has L3, the oracle none).
    let mut oracle = 0;
    if let Some(w) = hetero {
        let (reference, full) = (w.arm(Arm::Oracle), w.arm(Arm::Full));
        observed.push((
            "hetero.oracle_minus_full".to_string(),
            gap_permille(reference, full),
        ));
        if !at_least(reference, full, -params::ORACLE_TOLERANCE_PERMILLE) {
            oracle += 1;
        }
    }
    counters.push(("control_oracle", oracle));

    // Best-of-N must beat single shot: it is not a crippled baseline.
    let mut bon = 0;
    if let Some(w) = hetero {
        let arm = w.arm(Arm::BestOfN);
        let gain =
            (arm.successes as i64 - arm.first_pass as i64) * 1000 / arm.requests.max(1) as i64;
        observed.push(("hetero.bon_minus_single_shot".to_string(), gain));
        if gain < params::BON_MIN_GAIN_PERMILLE {
            bon += 1;
        }
    }
    counters.push(("control_bon", bon));

    // The L3 ablation is wired: with and without L3 the loop differs on some request.
    let mut l3 = 0;
    if let Some(w) = hetero {
        let pair = Pair::of(w.arm(Arm::Full), w.arm(Arm::NoL3));
        observed.push((
            "hetero.full_vs_no_l3_discordant".to_string(),
            (pair.only_a + pair.only_b) as i64,
        ));
        if pair.only_a + pair.only_b == 0 {
            l3 += 1;
        }
    }
    counters.push(("control_l3", l3));

    // The first draft is the same in every arm (a function of the request, not of the arm's state), so
    // the paired contrasts and the L3 population start from one candidate. A difference means arm
    // state reached round 0: a harness defect affecting every paired contrast.
    let mut first_draft = 0;
    for w in worlds {
        let differing = w.first_draft_differences();
        observed.push((
            format!("{}.first_candidates_differ", w.world.name()),
            differing as i64,
        ));
        if differing != 0 {
            first_draft += 1;
        }
    }
    counters.push(("control_first_draft", first_draft));
    Controls { counters, observed }
}
