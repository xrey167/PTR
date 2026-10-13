//! The metric list of R007 and the two judgements of a run: the coverage prerequisites (a run
//! that misses one is vacuous) and the falsifiers (a run that trips one falsifies R007).
//!
//! The rules are strings in the grammar of `criteria.toml` (see `layered_kv::emit`), derived
//! from the manifest's falsification text and the thresholds of `config.toml`: every
//! "< N" prerequisite becomes `>= N`, every "> 0" violation becomes `<= 0`, every "!= x"
//! violation becomes `== x`. The harness judges its own output with them, and a test pins
//! `criteria.toml` to exactly these lists.
//!
//! Two rules read a second printed value instead of a literal: the digest rule compares the
//! recorded canary identity-set digest with `canary_identity.pinned` (the pinned digest of
//! `config.toml`, echoed by the harness and checked against the file by the tests and by the
//! aggregator's own test), and `recompute_tokens == tokens_after_span` compares two measured
//! counters, as the manifest does.

use super::params;
use crate::experiments::layered_kv::emit::MetricList;

/// The metrics of the manifest, in its order (a test compares them with the file).
pub const METRIC_NAMES: [&str; 17] = [
    "bitwise_mismatch_count",
    "canary_logit_shift",
    "recompute_tokens",
    "span_insertions",
    "span_evictions",
    "downstream_recomputations",
    "canary_probes",
    "stale_downstream_rows_reused",
    "tokens_after_span",
    "canary_span_tokens",
    "downstream_rows_before_forget",
    "prefix_reuse_tokens",
    "post_forget_probes",
    "selection_digest_mismatches",
    "procedure_mismatches",
    "canary_identity_set_digest",
    "model_pin_mismatches",
];

pub const LIST: MetricList = MetricList {
    names: &METRIC_NAMES,
    grouped: &[],
};

/// The coverage prerequisites of the manifest: each unmet one makes the run vacuous, in the
/// order of the falsification text.
pub fn coverage() -> Vec<String> {
    vec![
        format!("span_insertions >= {}", params::MIN_SPAN_INSERTIONS),
        format!("span_evictions >= {}", params::MIN_SPAN_EVICTIONS),
        format!(
            "downstream_recomputations >= {}",
            params::MIN_DOWNSTREAM_RECOMPUTATIONS
        ),
        format!("canary_probes >= {}", params::MIN_CANARY_PROBES),
        format!("canary_span_tokens >= {}", params::MIN_CANARY_SPAN_TOKENS),
        format!(
            "downstream_rows_before_forget >= {}",
            params::MIN_DOWNSTREAM_ROWS_BEFORE_FORGET
        ),
        format!("prefix_reuse_tokens >= {}", params::MIN_PREFIX_REUSE_TOKENS),
        format!("post_forget_probes >= {}", params::MIN_POST_FORGET_PROBES),
        format!(
            "selection_digest_mismatches <= {}",
            params::MAX_SELECTION_DIGEST_MISMATCHES
        ),
        format!(
            "procedure_mismatches <= {}",
            params::MAX_PROCEDURE_MISMATCHES
        ),
        // The 32 pinned canaries: fewer probes than that means the set is not the pinned one.
        format!("canary_probes >= {}", params::CANARY_PROBES),
        format!(
            "model_pin_mismatches <= {}",
            params::MAX_MODEL_PIN_MISMATCHES
        ),
        "canary_identity_set_digest == canary_identity.pinned".to_string(),
    ]
}

/// The falsifiers of the manifest: each tripped one falsifies R007 on this seed.
pub fn falsifiers() -> Vec<String> {
    vec![
        "bitwise_mismatch_count <= 0".to_string(),
        format!("canary_logit_shift <= {}", params::MAX_CANARY_LOGIT_SHIFT),
        format!(
            "stale_downstream_rows_reused <= {}",
            params::MAX_STALE_DOWNSTREAM_ROWS_REUSED
        ),
        "recompute_tokens == tokens_after_span".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::emit::{judge, Json, Status};

    const DIGEST: &str = "65f9b8cd60e57cb38a8588c23d5bb494c7817fc10511353980f8be1c690efe19";

    /// A result in which every rule holds exactly at its threshold.
    fn passing() -> Vec<(String, Json)> {
        let mut pairs = Vec::new();
        for name in METRIC_NAMES {
            let value = match name {
                "canary_identity_set_digest" => Json::str(DIGEST),
                "canary_probes" => Json::Int(32),
                "recompute_tokens" | "tokens_after_span" => Json::Int(2880),
                "bitwise_mismatch_count"
                | "canary_logit_shift"
                | "stale_downstream_rows_reused"
                | "selection_digest_mismatches"
                | "procedure_mismatches"
                | "model_pin_mismatches" => Json::Int(0),
                _ => Json::Int(1),
            };
            pairs.push((name.to_string(), value));
        }
        pairs.push((
            "canary_identity".to_string(),
            Json::obj([("pinned", Json::str(DIGEST))]),
        ));
        pairs
    }

    fn failing_rules(root: &Json, rules: &[String]) -> Vec<String> {
        judge(root, &[], rules)
            .into_iter()
            .filter(|check| check.status == Status::Fail)
            .map(|check| check.rule)
            .collect()
    }

    fn set(pairs: &mut [(String, Json)], name: &str, value: Json) {
        pairs.iter_mut().find(|(key, _)| key == name).unwrap().1 = value;
    }

    #[test]
    fn a_result_at_every_threshold_holds_every_rule() {
        let root = Json::Obj(passing());
        assert!(failing_rules(&root, &coverage()).is_empty());
        assert!(failing_rules(&root, &falsifiers()).is_empty());
    }

    #[test]
    fn each_coverage_rule_fails_when_its_counter_leaves_the_threshold() {
        for rule in coverage() {
            let name = rule.split(' ').next().unwrap();
            let mut pairs = passing();
            let slot = pairs.iter_mut().find(|(key, _)| key == name).unwrap();
            let bound: i64 = rule.rsplit(' ').next().unwrap().parse().unwrap_or(0);
            slot.1 = match (&slot.1, rule.contains(">=")) {
                (Json::Int(_), true) => Json::Int(bound - 1),
                (Json::Int(value), false) => Json::Int(value + 1),
                (Json::Str(_), _) => Json::str("0".repeat(64)),
                _ => unreachable!(),
            };
            let failed = failing_rules(&Json::Obj(pairs), &coverage());
            // `canary_probes` has two rules, and a value below the smaller fails both.
            assert!(failed.contains(&rule), "{rule}: {failed:?}");
            assert!(failed.len() <= 2, "{rule}: {failed:?}");
        }
        let mut pairs = passing();
        set(&mut pairs, "canary_probes", Json::Int(31));
        assert_eq!(
            failing_rules(&Json::Obj(pairs), &coverage()),
            vec!["canary_probes >= 32".to_string()]
        );
    }

    #[test]
    fn each_falsifier_trips_on_a_single_violation() {
        for rule in falsifiers() {
            let name = rule.split(' ').next().unwrap();
            let mut pairs = passing();
            let value = if name == "recompute_tokens" { 2881 } else { 1 };
            set(&mut pairs, name, Json::Int(value));
            let failed = failing_rules(&Json::Obj(pairs), &falsifiers());
            assert_eq!(failed, vec![rule.clone()], "{rule}");
        }
        // The other side of the equality: fewer recomputed tokens than tokens after the span.
        let mut pairs = passing();
        set(&mut pairs, "tokens_after_span", Json::Int(2881));
        assert_eq!(
            failing_rules(&Json::Obj(pairs), &falsifiers()),
            vec!["recompute_tokens == tokens_after_span".to_string()]
        );
    }

    #[test]
    fn a_missing_null_or_mistyped_counter_is_not_a_pass() {
        for name in [
            "bitwise_mismatch_count",
            "canary_logit_shift",
            "recompute_tokens",
        ] {
            for damage in [None, Some(Json::Null), Some(Json::str("0"))] {
                let mut pairs = passing();
                match damage {
                    None => pairs.retain(|(key, _)| key != name),
                    Some(value) => set(&mut pairs, name, value),
                }
                let failed = failing_rules(&Json::Obj(pairs), &falsifiers());
                assert!(!failed.is_empty(), "{name}");
            }
        }
        let mut pairs = passing();
        pairs.retain(|(key, _)| key != "canary_identity_set_digest");
        assert_eq!(
            failing_rules(&Json::Obj(pairs), &coverage()),
            vec!["canary_identity_set_digest == canary_identity.pinned".to_string()]
        );
    }

    #[test]
    fn the_digest_rule_fails_on_any_other_digest_and_on_a_number() {
        let mut pairs = passing();
        set(&mut pairs, "canary_identity_set_digest", Json::str("abc"));
        assert_eq!(failing_rules(&Json::Obj(pairs), &coverage()).len(), 1);
        let mut pairs = passing();
        set(&mut pairs, "canary_identity_set_digest", Json::Int(5));
        assert_eq!(failing_rules(&Json::Obj(pairs), &coverage()).len(), 1);
    }
}
