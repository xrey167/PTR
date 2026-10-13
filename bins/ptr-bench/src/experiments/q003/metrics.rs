//! The metric list of Q003 and the two judgements of a run: the coverage prerequisites (a run that
//! misses one is vacuous) and the falsifiers (a run that trips one falsifies Q003).
//!
//! The rules are strings in the grammar of `criteria.toml` (see `layered_kv::emit`): every
//! "< N" prerequisite of the manifest becomes `>= N`, every "> 0" violation `<= 0`. The harness
//! judges its own output with them, and a test pins `criteria.toml` to exactly these lists, in
//! order. Fractions are integer permille; every metric is a JSON integer (a grouped metric an
//! object of integers).

use super::params;
use crate::experiments::layered_kv::emit::{MetricList, Tally};

/// The metrics of the manifest, in its order (a test compares them with the file).
pub const METRIC_NAMES: [&str; 49] = [
    "queries",
    "answerable_queries",
    "unanswerable_queries",
    "probe_queries",
    "superseded_targets_queried",
    "revoked_targets_queried",
    "recall10_pg_permille",
    "recall10_reference_permille",
    "recall10_strong_permille",
    "recall10_pg_on_strong_subset_permille",
    "ndcg10_pg_permille",
    "ndcg10_reference_permille",
    "ndcg10_strong_permille",
    "ndcg10_pg_on_strong_subset_permille",
    "recall10_pg_deficit_vs_reference_permille",
    "recall10_pg_deficit_vs_exact_permille",
    "recall10_pg_deficit_vs_strong_permille",
    "recall_violation_vs_reference",
    "recall_violation_vs_exact",
    "recall10_pg_lexical_only_permille",
    "recall10_pg_vector_only_permille",
    "recall10_exact_cosine_permille",
    "recall10_bm25_only_permille",
    "recall_paired_se_vs_reference_permille",
    "strong_discordant_queries",
    "strong_discordance_bound_permille",
    "main_query_lexical_hits",
    "live_probe_lexical_self_hits",
    "lexical_probe_violation",
    "stale_hits",
    "stale_hits_by_list",
    "planted_stale_rows",
    "planted_stale_returned",
    "projector_residual_rows",
    "late_write_attempts",
    "late_write_accepted",
    "late_write_wrong_refusal",
    "product_errors",
    "lifecycle_projection_mismatches",
    "product_panics",
    "nondeterministic_searches",
    "retrieval_hits",
    "control_planted_returned",
    "control_vector_returned",
    "reference_queries",
    "strong_queries",
    "workload_digest_mismatches",
    "latency_ns",
    "index_build_ns",
];

/// The counters whose sum is `hard_failures` and that `tests/mutations.toml` names. Each is a
/// metric above; `recall_violation_vs_reference` restates the recall rule as a count, and
/// `workload_digest_mismatches` is a harness-integrity guard (the aggregator judges a seed that
/// trips it INVALID, not falsified).
pub const HARD_COUNTERS: [&str; 13] = [
    "stale_hits",
    "planted_stale_returned",
    "projector_residual_rows",
    "late_write_accepted",
    "late_write_wrong_refusal",
    "product_errors",
    "lifecycle_projection_mismatches",
    "product_panics",
    "nondeterministic_searches",
    "recall_violation_vs_reference",
    "recall_violation_vs_exact",
    "lexical_probe_violation",
    "workload_digest_mismatches",
];

pub const LIST: MetricList = MetricList {
    names: &METRIC_NAMES,
    grouped: &["stale_hits_by_list", "retrieval_hits", "latency_ns"],
};

/// The metrics that need the strong arm; a run without it declares them unsupported.
pub const STRONG_METRICS: [&str; 8] = [
    "recall10_strong_permille",
    "recall10_pg_on_strong_subset_permille",
    "ndcg10_strong_permille",
    "ndcg10_pg_on_strong_subset_permille",
    "recall10_pg_deficit_vs_strong_permille",
    "strong_discordant_queries",
    "strong_discordance_bound_permille",
    "strong_queries",
];

/// The coverage prerequisites: each unmet one makes the run vacuous.
pub fn coverage() -> Vec<String> {
    vec![
        format!("planted_stale_rows >= {}", params::MIN_PLANTED_STALE_ROWS),
        format!(
            "revoked_targets_queried >= {}",
            params::MIN_REVOKED_TARGETS_QUERIED
        ),
        format!(
            "superseded_targets_queried >= {}",
            params::MIN_SUPERSEDED_TARGETS_QUERIED
        ),
        format!("answerable_queries >= {}", params::MIN_ANSWERABLE_QUERIES),
        format!(
            "retrieval_hits.lexical >= {}",
            params::MIN_RETRIEVAL_HITS_PER_MODE
        ),
        format!(
            "retrieval_hits.vector >= {}",
            params::MIN_RETRIEVAL_HITS_PER_MODE
        ),
        format!(
            "retrieval_hits.fused >= {}",
            params::MIN_RETRIEVAL_HITS_PER_MODE
        ),
        format!("late_write_attempts >= {}", params::MIN_LATE_WRITE_ATTEMPTS),
        format!("probe_queries >= {}", params::MIN_PROBE_QUERIES),
        format!(
            "control_planted_returned >= {}",
            params::MIN_CONTROL_PLANTED_RETURNED
        ),
        format!(
            "control_vector_returned >= {}",
            params::MIN_CONTROL_VECTOR_RETURNED
        ),
        format!(
            "live_probe_lexical_self_hits >= {}",
            params::MIN_LIVE_PROBE_LEXICAL_SELF_HITS
        ),
        format!("reference_queries >= {}", params::MIN_REFERENCE_QUERIES),
        format!("strong_queries >= {}", params::MIN_STRONG_QUERIES),
        format!(
            "recall10_reference_permille >= {}",
            params::MIN_RECALL10_REFERENCE_PERMILLE
        ),
        format!(
            "workload_digest_mismatches <= {}",
            params::MAX_WORKLOAD_DIGEST_MISMATCHES
        ),
    ]
}

/// The falsifiers: each tripped one falsifies Q003 on this seed.
pub fn falsifiers() -> Vec<String> {
    vec![
        format!("stale_hits <= {}", params::MAX_STALE_HITS),
        format!(
            "planted_stale_returned <= {}",
            params::MAX_PLANTED_STALE_RETURNED
        ),
        format!(
            "projector_residual_rows <= {}",
            params::MAX_PROJECTOR_RESIDUAL_ROWS
        ),
        format!("late_write_accepted <= {}", params::MAX_LATE_WRITE_ACCEPTED),
        format!(
            "late_write_wrong_refusal <= {}",
            params::MAX_LATE_WRITE_WRONG_REFUSAL
        ),
        format!("product_errors <= {}", params::MAX_PRODUCT_ERRORS),
        format!(
            "lifecycle_projection_mismatches <= {}",
            params::MAX_LIFECYCLE_PROJECTION_MISMATCHES
        ),
        format!("product_panics <= {}", params::MAX_PRODUCT_PANICS),
        format!(
            "nondeterministic_searches <= {}",
            params::MAX_NONDETERMINISTIC_SEARCHES
        ),
        format!(
            "recall10_pg_deficit_vs_reference_permille <= {}",
            params::RECALL_MARGIN_PERMILLE
        ),
        format!(
            "recall10_pg_deficit_vs_exact_permille <= {}",
            params::RECALL_EXACT_MARGIN_PERMILLE
        ),
    ]
}

/// Set the derived recall metrics and the violation counts, from the per-arm recalls already in
/// `tally` (`recall10_*_permille`). The strong ones exist only when the strong arm ran, and compare
/// the strong arm with the product on the queries the strong arm answered.
pub fn fill_derived(tally: &mut Tally, strong_ran: bool) {
    let pg = tally.get("recall10_pg_permille");
    let reference = tally.get("recall10_reference_permille");
    let deficit = (reference - pg).max(0);
    tally.set("recall10_pg_deficit_vs_reference_permille", deficit);
    tally.set(
        "recall_violation_vs_reference",
        i64::from(deficit > params::RECALL_MARGIN_PERMILLE),
    );
    let exact = (tally.get("recall10_exact_cosine_permille") - pg).max(0);
    tally.set("recall10_pg_deficit_vs_exact_permille", exact);
    tally.set(
        "recall_violation_vs_exact",
        i64::from(exact > params::RECALL_EXACT_MARGIN_PERMILLE),
    );
    let live_probes = tally.get("live_probe_queries");
    tally.set(
        "lexical_probe_violation",
        i64::from(
            tally.get("live_probe_lexical_self_hits") * 100
                < live_probes * params::LEXICAL_SELF_HIT_PERCENT,
        ),
    );
    if strong_ran {
        let on_subset = tally.get("recall10_pg_on_strong_subset_permille");
        let deficit = (tally.get("recall10_strong_permille") - on_subset).max(0);
        tally.set("recall10_pg_deficit_vs_strong_permille", deficit);
    }
}

/// The sum of the hard counters.
pub fn hard_failures(tally: &Tally) -> i64 {
    HARD_COUNTERS.iter().map(|name| tally.get(name)).sum()
}
