//! The metric list of R006 and the two judgements of a run: the coverage prerequisites (a run
//! that misses one is vacuous) and the falsifiers (a run that trips one falsifies R006).
//!
//! The rules are strings in the grammar of `criteria.toml` (see `layered_kv::emit`), derived
//! from the manifest's falsification text and the thresholds of `config.toml`: every "< N"
//! prerequisite becomes `>= N`, every "> 0" violation becomes `<= 0`, every "!= x" violation
//! becomes `== x`. A grouped metric has one rule per member, because the grammar has no
//! quantifier: `reuse_attempts_per_shift.shift_64 >= 1`, `invalidation_matrix_stale_hits.<cell>
//! <= 0`. The harness (not built yet) judges its own output with them, and a test pins
//! `criteria.toml` to exactly these lists.
//!
//! Two rules compare two printed values instead of a literal, as the manifest does:
//! `fresh_prefills_completed_after_revocation >= upstream_revocation_cases` and the identity-set
//! equality of the two case-id digests.
//!
//! Staleness is structural here (owner decision 5, `config.toml` `stale_hit_definition` and
//! `escaped_output_definition`): in a one-layer model a stale page is bit-identical to a fresh
//! one, so every rule about revocation, supersession, omission and escape counts lookups that
//! returned a dead run and tokens or inserts after the event step, never an output difference.

use super::params;
use crate::experiments::layered_kv::emit::MetricList;

/// The metrics of the manifest, in its order (a test compares them with the file).
pub const METRIC_NAMES: [&str; 39] = [
    "bitwise_mismatch_count",
    "reuse_latency_ms",
    "reuse_attempts_per_shift",
    "secondary_index_lookups",
    "nonaligned_insertion_cases",
    "upstream_revocation_cases",
    "revoked_upstream_page_reachable_count",
    "fresh_prefill_after_revocation",
    "reused_tokens_per_shift",
    "secondary_index_reuse_cases_per_shift",
    "post_revocation_secondary_lookups_matching_content",
    "old_upstream_pages_seeded_before_revocation",
    "dead_original_upstream_shift_attempts",
    "shift_hits_with_dead_original_upstream",
    "omitted_upstream_segment_cases",
    "omitted_upstream_admissions",
    "upstream_supersession_cases",
    "superseded_upstream_page_reachable_count",
    "selection_digest_mismatches",
    "procedure_mismatches",
    "source_generation_mismatch_cases",
    "source_digest_mismatch_cases",
    "source_lineage_mismatch_admissions",
    "fresh_prefills_completed_after_revocation",
    "fresh_prefill_output_mismatches",
    "post_supersession_secondary_lookups_matching_content",
    "invalidated_case_id_set_digest",
    "fresh_prefill_completed_case_id_set_digest",
    "invalidation_matrix_cases",
    "invalidation_matrix_stale_hits",
    "shifted_reuse_inflight_race_cases",
    "shifted_reuse_inflight_outputs",
    "shifted_reuse_inflight_cache_inserts",
    "exact_shifted_parity_cases",
    "tainted_exact_results",
    "non_bit_exact_exact_results",
    "raw_ancestor_omission_cases",
    "reuse_after_raw_ancestor_removed",
    "model_pin_mismatches",
];

/// Metrics printed as an object built from the dotted counters under their name.
pub const GROUPED: [&str; 8] = [
    "bitwise_mismatch_count",
    "reuse_attempts_per_shift",
    "reused_tokens_per_shift",
    "secondary_index_reuse_cases_per_shift",
    "nonaligned_insertion_cases",
    "invalidation_matrix_cases",
    "invalidation_matrix_stale_hits",
    "shifted_reuse_inflight_race_cases",
];

pub const LIST: MetricList = MetricList {
    names: &METRIC_NAMES,
    grouped: &GROUPED,
};

/// The metrics that no rule reads: `reuse_latency_ms` is timing only (an object of per-shift
/// milliseconds, floats), and the preregistration makes no wall-clock claim.
pub const UNJUDGED: [&str; 1] = ["reuse_latency_ms"];

/// The metrics each case of the draft preregistration produces (a metric can be fed by several
/// cases). The map is checked against the manifest: every metric has a case, and no case names
/// a metric the manifest lacks. C10 (the negative controls) produces no manifest metric.
pub const CASE_MAP: [(&str, &[&str]); 11] = [
    (
        "C1",
        &[
            "bitwise_mismatch_count",
            "reuse_latency_ms",
            "reuse_attempts_per_shift",
            "reused_tokens_per_shift",
            "secondary_index_reuse_cases_per_shift",
            "secondary_index_lookups",
            "exact_shifted_parity_cases",
            "tainted_exact_results",
            "non_bit_exact_exact_results",
        ],
    ),
    (
        "C2",
        &[
            "nonaligned_insertion_cases",
            "bitwise_mismatch_count",
            "secondary_index_lookups",
            "exact_shifted_parity_cases",
        ],
    ),
    (
        "C3",
        &[
            "upstream_revocation_cases",
            "old_upstream_pages_seeded_before_revocation",
            "revoked_upstream_page_reachable_count",
            "post_revocation_secondary_lookups_matching_content",
            "fresh_prefill_after_revocation",
            "fresh_prefills_completed_after_revocation",
            "fresh_prefill_output_mismatches",
            "invalidated_case_id_set_digest",
            "fresh_prefill_completed_case_id_set_digest",
        ],
    ),
    (
        "C4",
        &[
            "upstream_supersession_cases",
            "superseded_upstream_page_reachable_count",
            "post_supersession_secondary_lookups_matching_content",
            "invalidated_case_id_set_digest",
            "fresh_prefill_completed_case_id_set_digest",
        ],
    ),
    (
        "C5",
        &[
            "dead_original_upstream_shift_attempts",
            "shift_hits_with_dead_original_upstream",
        ],
    ),
    (
        "C6",
        &[
            "omitted_upstream_segment_cases",
            "omitted_upstream_admissions",
        ],
    ),
    (
        "C7",
        &[
            "source_generation_mismatch_cases",
            "source_digest_mismatch_cases",
            "source_lineage_mismatch_admissions",
        ],
    ),
    (
        "C8",
        &[
            "raw_ancestor_omission_cases",
            "reuse_after_raw_ancestor_removed",
        ],
    ),
    (
        "C9",
        &[
            "invalidation_matrix_cases",
            "invalidation_matrix_stale_hits",
            "shifted_reuse_inflight_race_cases",
            "shifted_reuse_inflight_outputs",
            "shifted_reuse_inflight_cache_inserts",
            "invalidated_case_id_set_digest",
            "fresh_prefill_completed_case_id_set_digest",
        ],
    ),
    ("C10", &[]),
    (
        "C11",
        &[
            "model_pin_mismatches",
            "procedure_mismatches",
            "selection_digest_mismatches",
        ],
    ),
];

/// One rule per pinned shift of the per-shift metric `name`: `<name>.shift_<s> <op> <bound>`.
fn per_shift(name: &str, op: &str, bound: u64) -> Vec<String> {
    params::POSITION_SHIFT_TOKENS
        .iter()
        .map(|shift| format!("{name}.{} {op} {bound}", params::shift_member(*shift)))
        .collect()
}

/// One rule per cell of `invalidation_coverage_matrix` for the per-cell metric `name`.
fn per_cell(name: &str, op: &str, bound: u64) -> Vec<String> {
    params::INVALIDATION_COVERAGE_MATRIX_CELLS
        .iter()
        .map(|cell| format!("{name}.{cell} {op} {bound}"))
        .collect()
}

/// The coverage prerequisites of the manifest: each unmet one makes the run vacuous, in the
/// order of the falsification text.
pub fn coverage() -> Vec<String> {
    let mut rules: Vec<String> = Vec::new();
    rules.extend(per_shift(
        "reuse_attempts_per_shift",
        ">=",
        params::MIN_REUSE_ATTEMPTS_PER_SHIFT,
    ));
    rules.extend(per_shift(
        "reused_tokens_per_shift",
        ">=",
        params::MIN_REUSED_TOKENS_PER_SHIFT,
    ));
    rules.extend(per_shift(
        "secondary_index_reuse_cases_per_shift",
        ">=",
        params::MIN_SECONDARY_INDEX_REUSE_CASES_PER_SHIFT,
    ));
    rules.push(format!(
        "secondary_index_lookups >= {}",
        params::MIN_SECONDARY_INDEX_LOOKUPS
    ));
    for tokens in params::NONALIGNED_INSERTION_TOKENS {
        rules.push(format!(
            "nonaligned_insertion_cases.{} >= {}",
            params::insertion_member(tokens),
            params::MIN_NONALIGNED_INSERTION_CASES_PER_SIZE
        ));
    }
    rules.push(format!(
        "upstream_revocation_cases >= {}",
        params::MIN_UPSTREAM_REVOCATION_CASES
    ));
    rules.push(format!(
        "old_upstream_pages_seeded_before_revocation >= {}",
        params::MIN_OLD_UPSTREAM_PAGES_SEEDED_BEFORE_REVOCATION
    ));
    rules.push(format!(
        "fresh_prefill_after_revocation >= {}",
        params::MIN_FRESH_PREFILL_AFTER_REVOCATION
    ));
    rules.push(format!(
        "post_revocation_secondary_lookups_matching_content >= {}",
        params::MIN_POST_REVOCATION_SECONDARY_LOOKUPS_MATCHING_CONTENT
    ));
    rules.push(format!(
        "dead_original_upstream_shift_attempts >= {}",
        params::MIN_DEAD_ORIGINAL_UPSTREAM_SHIFT_ATTEMPTS
    ));
    rules.push(format!(
        "omitted_upstream_segment_cases >= {}",
        params::MIN_OMITTED_UPSTREAM_SEGMENT_CASES
    ));
    rules.push(format!(
        "upstream_supersession_cases >= {}",
        params::MIN_UPSTREAM_SUPERSESSION_CASES
    ));
    rules.push(format!(
        "selection_digest_mismatches <= {}",
        params::MAX_SELECTION_DIGEST_MISMATCHES
    ));
    rules.push(format!(
        "procedure_mismatches <= {}",
        params::MAX_PROCEDURE_MISMATCHES
    ));
    rules.push(format!(
        "source_generation_mismatch_cases >= {}",
        params::MIN_SOURCE_GENERATION_MISMATCH_CASES
    ));
    rules.push(format!(
        "source_digest_mismatch_cases >= {}",
        params::MIN_SOURCE_DIGEST_MISMATCH_CASES
    ));
    // Every revoked case completed via a fresh prefill: at least as many completions as cases.
    rules
        .push("fresh_prefills_completed_after_revocation >= upstream_revocation_cases".to_string());
    rules.push(format!(
        "post_supersession_secondary_lookups_matching_content >= {}",
        params::MIN_POST_SUPERSESSION_SECONDARY_LOOKUPS_MATCHING_CONTENT
    ));
    // Exact identity-set equality, not count equality (`identity_set_checks`).
    rules.push(
        "invalidated_case_id_set_digest == fresh_prefill_completed_case_id_set_digest".to_string(),
    );
    rules.extend(per_cell(
        "invalidation_matrix_cases",
        ">=",
        params::MIN_INVALIDATION_MATRIX_CASES_PER_CELL,
    ));
    for member in params::inflight_members() {
        rules.push(format!(
            "shifted_reuse_inflight_race_cases.{member} >= {}",
            params::MIN_SHIFTED_REUSE_INFLIGHT_RACE_CASES_PER_CELL
        ));
    }
    rules.push(format!(
        "exact_shifted_parity_cases >= {}",
        params::MIN_EXACT_SHIFTED_PARITY_CASES
    ));
    rules.push(format!(
        "raw_ancestor_omission_cases >= {}",
        params::MIN_RAW_ANCESTOR_OMISSION_CASES
    ));
    rules.push(format!(
        "model_pin_mismatches <= {}",
        params::MAX_MODEL_PIN_MISMATCHES
    ));
    rules
}

/// The falsifiers of the manifest: each tripped one falsifies R006 on this seed, in the order
/// of the falsification text.
pub fn falsifiers() -> Vec<String> {
    let mut rules: Vec<String> = Vec::new();
    rules.extend(per_shift(
        "bitwise_mismatch_count",
        "<=",
        params::MAX_BITWISE_MISMATCH_COUNT_PER_SHIFT,
    ));
    rules.push(format!(
        "revoked_upstream_page_reachable_count <= {}",
        params::MAX_REVOKED_UPSTREAM_PAGE_REACHABLE_COUNT
    ));
    rules.push(format!(
        "shift_hits_with_dead_original_upstream <= {}",
        params::MAX_SHIFT_HITS_WITH_DEAD_ORIGINAL_UPSTREAM
    ));
    rules.push(format!(
        "omitted_upstream_admissions <= {}",
        params::MAX_OMITTED_UPSTREAM_ADMISSIONS
    ));
    rules.push(format!(
        "superseded_upstream_page_reachable_count <= {}",
        params::MAX_SUPERSEDED_UPSTREAM_PAGE_REACHABLE_COUNT
    ));
    rules.push(format!(
        "source_lineage_mismatch_admissions <= {}",
        params::MAX_SOURCE_LINEAGE_MISMATCH_ADMISSIONS
    ));
    rules.push(format!(
        "fresh_prefill_output_mismatches <= {}",
        params::FRESH_PREFILL_OUTPUT_MISMATCHES_MAX
    ));
    rules.extend(per_cell(
        "invalidation_matrix_stale_hits",
        "<=",
        params::MAX_INVALIDATION_MATRIX_STALE_HITS_PER_CELL,
    ));
    rules.push(format!(
        "shifted_reuse_inflight_outputs <= {}",
        params::MAX_SHIFTED_REUSE_INFLIGHT_OUTPUTS
    ));
    rules.push(format!(
        "shifted_reuse_inflight_cache_inserts <= {}",
        params::MAX_SHIFTED_REUSE_INFLIGHT_CACHE_INSERTS
    ));
    rules.push(format!(
        "tainted_exact_results <= {}",
        params::MAX_TAINTED_EXACT_RESULTS
    ));
    rules.push(format!(
        "non_bit_exact_exact_results <= {}",
        params::MAX_NON_BIT_EXACT_EXACT_RESULTS
    ));
    rules.push(format!(
        "reuse_after_raw_ancestor_removed <= {}",
        params::MAX_REUSE_AFTER_RAW_ANCESTOR_REMOVED
    ));
    rules
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::emit::{judge, Json, Status};
    use std::collections::BTreeSet;

    const CRITERIA: &str = include_str!(
        "../../../../../experiments/runtime/R006-positional-reuse-oracle/criteria.toml"
    );

    fn is_literal(token: &str) -> bool {
        let digits = token.strip_prefix('-').unwrap_or(token);
        !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
    }

    fn tokens(rule: &str) -> (String, String, String) {
        let parts: Vec<&str> = rule.split(' ').collect();
        assert_eq!(parts.len(), 3, "{rule}");
        (
            parts[0].to_string(),
            parts[1].to_string(),
            parts[2].to_string(),
        )
    }

    /// Set `path` (dotted) in a root object, creating the objects on the way.
    fn put(root: &mut Vec<(String, Json)>, path: &str, value: Json) {
        let (head, rest) = match path.split_once('.') {
            Some((head, rest)) => (head, Some(rest)),
            None => (path, None),
        };
        let at = root.iter().position(|(key, _)| key == head);
        match (rest, at) {
            (None, Some(index)) => root[index].1 = value,
            (None, None) => root.push((head.to_string(), value)),
            (Some(rest), Some(index)) => {
                if let Json::Obj(inner) = &mut root[index].1 {
                    put(inner, rest, value);
                } else {
                    panic!("{head} is not an object");
                }
            }
            (Some(rest), None) => {
                let mut inner = Vec::new();
                put(&mut inner, rest, value);
                root.push((head.to_string(), Json::Obj(inner)));
            }
        }
    }

    fn get(root: &[(String, Json)], path: &str) -> Option<Json> {
        crate::experiments::layered_kv::emit::resolve(&Json::Obj(root.to_vec()), path).cloned()
    }

    /// A result in which every rule holds exactly at its threshold.
    fn passing() -> Vec<(String, Json)> {
        let mut pairs = Vec::new();
        let all: Vec<String> = coverage().into_iter().chain(falsifiers()).collect();
        for rule in &all {
            let (left, _, right) = tokens(rule);
            if is_literal(&right) {
                put(&mut pairs, &left, Json::Int(right.parse().unwrap()));
            }
        }
        for rule in &all {
            let (left, op, right) = tokens(rule);
            if !is_literal(&right) {
                let value = if op == "==" {
                    Json::str("sha256:aa")
                } else {
                    get(&pairs, &right).expect("the right side was set by its own rule")
                };
                put(&mut pairs, &left, value.clone());
                if op == "==" {
                    put(&mut pairs, &right, value);
                }
            }
        }
        pairs
    }

    fn failing_rules(pairs: &[(String, Json)], rules: &[String]) -> Vec<String> {
        judge(&Json::Obj(pairs.to_vec()), &[], rules)
            .into_iter()
            .filter(|check| check.status == Status::Fail)
            .map(|check| check.rule)
            .collect()
    }

    #[test]
    fn the_metric_names_are_the_manifests() {
        let manifest: toml::Table = params::EXPERIMENT_TOML
            .parse()
            .expect("experiment.toml parses");
        let listed: Vec<&str> = manifest["metrics"]
            .as_str()
            .expect("metrics is a string")
            .split(", ")
            .collect();
        assert_eq!(listed, METRIC_NAMES);
        assert_eq!(METRIC_NAMES.len(), 39);
        let unique: BTreeSet<&str> = METRIC_NAMES.iter().copied().collect();
        assert_eq!(unique.len(), METRIC_NAMES.len());
        for grouped in GROUPED {
            assert!(METRIC_NAMES.contains(&grouped), "{grouped}");
        }
    }

    #[test]
    fn every_manifest_metric_has_a_case_and_no_case_names_a_stranger() {
        let mut produced = BTreeSet::new();
        for (case, metrics) in CASE_MAP {
            for metric in metrics {
                assert!(METRIC_NAMES.contains(metric), "{case}: {metric}");
                produced.insert(*metric);
            }
        }
        let all: BTreeSet<&str> = METRIC_NAMES.iter().copied().collect();
        assert_eq!(produced, all);
        let cases: Vec<&str> = CASE_MAP.iter().map(|(case, _)| *case).collect();
        assert_eq!(
            cases,
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11"]
        );
    }

    #[test]
    fn the_rule_counts_follow_the_pinned_axes() {
        let shifts = params::POSITION_SHIFT_TOKENS.len();
        let cells = params::INVALIDATION_COVERAGE_MATRIX_CELLS.len();
        assert_eq!((shifts, cells), (5, 52));
        // Coverage: 3 per-shift metrics, 2 insertion sizes, 52 cells, 6 in-flight members and
        // 18 single rules (including the two comparing two printed values).
        assert_eq!(coverage().len(), 3 * shifts + 2 + cells + 6 + 18);
        // Falsifiers: bitwise per shift, 52 stale-hit cells and 11 single rules.
        assert_eq!(falsifiers().len(), shifts + cells + 11);
        for list in [coverage(), falsifiers()] {
            let unique: BTreeSet<&String> = list.iter().collect();
            assert_eq!(unique.len(), list.len());
        }
    }

    #[test]
    fn the_criteria_file_is_exactly_the_rules_the_harness_judges_in_their_order() {
        let table: toml::Table = CRITERIA.parse().expect("criteria.toml is TOML");
        let rules = |key: &str| -> Vec<String> {
            table[key]
                .as_array()
                .unwrap_or_else(|| panic!("{key} is an array"))
                .iter()
                .map(|rule| rule.as_str().expect("a rule is a string").to_string())
                .collect()
        };
        assert_eq!(rules("coverage"), coverage());
        assert_eq!(rules("falsifiers"), falsifiers());
        assert_eq!(table["experiment"].as_str(), Some("R006"));
        assert_eq!(
            table["seed_aggregation"].as_str(),
            Some(params::SEED_AGGREGATION)
        );
    }

    #[test]
    fn a_result_at_every_threshold_holds_every_rule() {
        let pairs = passing();
        assert!(failing_rules(&pairs, &coverage()).is_empty());
        assert!(failing_rules(&pairs, &falsifiers()).is_empty());
    }

    /// No rule is vacuous: some value of its own counter fails it, and only it.
    #[test]
    fn every_rule_can_fail_and_fails_alone() {
        for (rules, label) in [(coverage(), "coverage"), (falsifiers(), "falsifier")] {
            for rule in &rules {
                let (left, op, right) = tokens(rule);
                let mut pairs = passing();
                let current = get(&pairs, &left).expect("the counter is set");
                let broken = match (&current, op.as_str()) {
                    (Json::Str(_), "==") => Json::str("sha256:bb"),
                    (Json::Int(_), ">=") => {
                        let bound = if is_literal(&right) {
                            right.parse::<i64>().unwrap()
                        } else {
                            match get(&pairs, &right) {
                                Some(Json::Int(value)) => value,
                                other => panic!("{right}: {other:?}"),
                            }
                        };
                        Json::Int(bound - 1)
                    }
                    (Json::Int(value), "<=") => Json::Int(value + 1),
                    other => panic!("{rule}: {other:?}"),
                };
                put(&mut pairs, &left, broken);
                assert_eq!(
                    failing_rules(&pairs, &rules),
                    vec![rule.clone()],
                    "{label}: {rule}"
                );
            }
        }
    }

    #[test]
    fn a_missing_null_or_mistyped_counter_or_member_is_not_a_pass() {
        for rule in [
            "bitwise_mismatch_count.shift_4096 <= 0",
            "revoked_upstream_page_reachable_count <= 0",
        ] {
            for damage in [
                None,
                Some(Json::Null),
                Some(Json::str("0")),
                Some(Json::Float(0.0)),
            ] {
                let (left, _, _) = tokens(rule);
                let mut pairs = passing();
                match damage {
                    None => {
                        if let Some((head, rest)) = left.split_once('.') {
                            let index = pairs.iter().position(|(key, _)| key == head).unwrap();
                            if let Json::Obj(inner) = &mut pairs[index].1 {
                                inner.retain(|(key, _)| key != rest);
                            }
                        } else {
                            pairs.retain(|(key, _)| *key != left);
                        }
                    }
                    Some(value) => put(&mut pairs, &left, value),
                }
                let failed = failing_rules(&pairs, &falsifiers());
                assert!(failed.contains(&rule.to_string()), "{rule}: {failed:?}");
            }
        }
        // The digest rule fails on a number and on another digest.
        let digest_rule =
            "invalidated_case_id_set_digest == fresh_prefill_completed_case_id_set_digest";
        let mut pairs = passing();
        put(&mut pairs, "invalidated_case_id_set_digest", Json::Int(5));
        assert_eq!(
            failing_rules(&pairs, &coverage()),
            vec![digest_rule.to_string()]
        );
    }

    #[test]
    fn the_grouped_rules_are_per_member_and_cover_every_pinned_member() {
        let all: Vec<String> = coverage().into_iter().chain(falsifiers()).collect();
        for shift in params::POSITION_SHIFT_TOKENS {
            let member = params::shift_member(shift);
            for metric in [
                "reuse_attempts_per_shift",
                "reused_tokens_per_shift",
                "secondary_index_reuse_cases_per_shift",
            ] {
                assert!(
                    all.contains(&format!("{metric}.{member} >= 1")),
                    "{metric} {member}"
                );
            }
            assert!(all.contains(&format!("bitwise_mismatch_count.{member} <= 0")));
        }
        for cell in params::INVALIDATION_COVERAGE_MATRIX_CELLS {
            assert!(
                all.contains(&format!("invalidation_matrix_cases.{cell} >= 1")),
                "{cell}"
            );
            assert!(
                all.contains(&format!("invalidation_matrix_stale_hits.{cell} <= 0")),
                "{cell}"
            );
        }
        // A member name is one path segment.
        for rule in &all {
            let (left, _, _) = tokens(rule);
            assert!(left.split('.').count() <= 2, "{rule}");
        }
        // A missing cell is not a pass: a stale-hit counter absent for one cell fails its rule.
        let mut pairs = passing();
        let cell = params::INVALIDATION_COVERAGE_MATRIX_CELLS[17];
        let index = pairs
            .iter()
            .position(|(key, _)| key == "invalidation_matrix_stale_hits")
            .unwrap();
        if let Json::Obj(inner) = &mut pairs[index].1 {
            inner.retain(|(key, _)| key != cell);
        }
        assert_eq!(
            failing_rules(&pairs, &falsifiers()),
            vec![format!("invalidation_matrix_stale_hits.{cell} <= 0")]
        );
    }

    /// Every rule is a reading of a sentence of the manifest's falsification text, and every
    /// sentence that bounds a metric has a rule.
    #[test]
    fn the_rules_mirror_the_falsification_text_of_the_manifest() {
        let manifest: toml::Table = params::EXPERIMENT_TOML
            .parse()
            .expect("experiment.toml parses");
        let text = manifest["falsification"].as_str().expect("falsification");
        let base = |name: &str| name.split('.').next().unwrap().to_string();
        let ruled = |name: &str| {
            coverage().into_iter().chain(falsifiers()).any(|rule| {
                let (left, _, right) = tokens(&rule);
                base(&left) == name || base(&right) == name
            })
        };
        for rule in coverage().iter().chain(&falsifiers()) {
            let (left, op, right) = tokens(rule);
            let metric = base(&left);
            let snippet = match (metric.as_str(), op.as_str()) {
                ("reuse_attempts_per_shift", ">=") => {
                    "reuse_attempts_per_shift records fewer than 1 reuse result".to_string()
                }
                ("reused_tokens_per_shift", ">=") => "reused_tokens_per_shift == 0".to_string(),
                (_, "==") => format!("{left} != {right}"),
                // A literal bound is the text's own number, a reference bound the other counter.
                (_, ">=") => format!("{metric} < {}", base(&right)),
                (_, "<=") => format!("{metric} > {right}"),
                _ => unreachable!("{rule}"),
            };
            assert!(
                text.contains(&snippet),
                "{rule}: no '{snippet}' in the text"
            );
        }
        // Every metric the text bounds has a rule, and every metric of the manifest is judged
        // by some rule except the timing one.
        for name in METRIC_NAMES {
            if text.contains(&format!("{name} <"))
                || text.contains(&format!("{name} >"))
                || text.contains(&format!("{name} =="))
                || text.contains(&format!("{name} !="))
            {
                assert!(ruled(name), "{name}");
            }
            if UNJUDGED.contains(&name) {
                assert!(!ruled(name), "{name} is timing only");
            } else {
                assert!(ruled(name), "{name} has no rule");
            }
        }
        // The pinned shifts the text names are the config's.
        assert!(text.contains("position_shift_tokens = [1, 7, 64, 513, 4096]"));
        assert!(text.contains("config.toml enumerates all 52 cells"));
        // The in-flight members of the text are the three kinds x two types the config names.
        assert!(text.contains(
            "{semantic lineage segment, memory lineage segment, base model} x {revocation, supersession}"
        ));
        assert_eq!(
            params::INFLIGHT_RACE_REQUIRED_KINDS,
            [
                "semantic_lineage_segment",
                "memory_lineage_segment",
                "base_model"
            ]
        );
    }

    #[test]
    fn no_rule_reads_a_timing() {
        for rule in coverage().iter().chain(&falsifiers()) {
            assert!(!rule.contains("latency"), "{rule}");
            assert!(!rule.contains("timing"), "{rule}");
            assert!(!rule.contains("elapsed"), "{rule}");
        }
    }
}
