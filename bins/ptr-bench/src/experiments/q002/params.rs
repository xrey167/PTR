//! The preregistered parameters of Q002, compiled in.
//!
//! The constants mirror the keys of `[pinned_thresholds]` in the experiment's `config.toml`,
//! which is read at compile time; the drift test fails when a constant differs from the file,
//! and when the file holds a key no constant mirrors. Q002 is `planned`: nothing here is
//! frozen.

// The thresholds complete the compiled table the drift test compares with the file; the harness
// evaluates most of them in `metrics`, and the rest only echo the table.
#![cfg_attr(not(test), allow(dead_code))]

/// The experiment's configuration file, read at compile time.
pub const CONFIG_TOML: &str =
    include_str!("../../../../../experiments/retrieval/Q002-evidence-promotion/config.toml");

/// The experiment's manifest, read at compile time (the metric names come from it).
pub const EXPERIMENT_TOML: &str =
    include_str!("../../../../../experiments/retrieval/Q002-evidence-promotion/experiment.toml");

/// The subcommand of the harness (`ptr-bench <BENCHMARK> <seed>`).
pub const BENCHMARK: &str = "evidence-promotion";

/// The runner's command, as the preregistration states it. `experiment.toml` carries it from
/// the freeze on (until then its `entrypoint` is empty).
pub const ENTRYPOINT: &str =
    "cargo run --release --locked -p ptr-bench -- evidence-promotion <seed>";

pub const SEED_AGGREGATION: &str = "every_seed_must_pass";
pub const SEEDS: [u64; 5] = [17, 29, 43, 71, 101];

/// The world: capsules, their lifecycle, the facts they state.
pub const CAPSULES: usize = 36;
pub const HEALTHY_CAPSULES: usize = 26;
pub const REVOKED_CAPSULES: usize = 5;
pub const DISPUTED_CAPSULES: usize = 5;
pub const MAX_GENERATIONS: usize = 3;
pub const FACTS_PER_CAPSULE: usize = 6;
pub const ENTITIES: usize = 200;
pub const TWIN_PAIRS: usize = 60;
pub const PREDICATES: [&str; 6] = ["status", "tier", "region", "owner", "color", "phase"];
pub const VALUES_PER_PREDICATE: usize = 12;
pub const ABSENT_QUESTIONS: usize = 40;
pub const MIRRORS_PER_QUARANTINED_EVENT: usize = 4;
pub const NEGATIONS_PER_HEALTHY_EVENT: usize = 2;
pub const STALE_VALUE_CHANGE_PERMILLE: u64 = 700;

/// The hits: how many of each poison class the retriever returns (a class with fewer candidates
/// than its quota gets all of them; P7 and P8 take every statement of the quarantined events).
pub const QUOTA_P1_FABRICATED_SOURCE: usize = 60;
pub const QUOTA_P2_SHIFTED_SPAN: usize = 24;
pub const QUOTA_P2_SPAN_PAST_END: usize = 24;
pub const QUOTA_P2_FORGED_QUOTE: usize = 24;
pub const QUOTA_P3_TAMPERED_DIGEST: usize = 40;
pub const QUOTA_P4_LOOKALIKE_SUBJECT: usize = 48;
pub const QUOTA_P5_CONTRADICTING_VALUE: usize = 30;
pub const QUOTA_P5_NEGATION: usize = 30;
pub const QUOTA_P6_SUPERSEDED: usize = 60;
pub const QUOTA_P2_EMPTY_SPAN: usize = 24;
pub const QUOTA_P2_TWO_LINE_SPAN: usize = 24;
pub const QUOTA_P2_PADDED_SPAN: usize = 24;
pub const QUOTA_P4_PREDICATE_ONLY: usize = 24;
pub const QUOTA_P4_CASE_ONLY: usize = 24;
pub const QUOTA_P5_TRUNCATED_NEGATION: usize = 24;
pub const QUOTA_P9_RELABELLED_SUPERSEDED: usize = 30;
pub const QUOTA_P9_RELABELLED_QUARANTINED: usize = 30;
/// How many honest hits of each hard kind the world has (multibyte value, value containing
/// `not`, repeated statement, one-line event, span at the event start, span at the event end).
pub const HARD_HONEST_PER_VARIANT: usize = 6;
pub const HONEST_SCORE_MIN: u64 = 400;
pub const HONEST_SCORE_MAX: u64 = 900;
pub const POISON_SCORE_MIN: u64 = 300;
pub const POISON_SCORE_MAX: u64 = 950;

/// The adversarial stage-skip attempts of each kind (K1..K5, K6a, K6b, K7..K11).
pub const STAGE_SKIP_ATTEMPTS_PER_KIND: usize = 24;

/// Coverage prerequisites (a run that misses one is vacuous) and falsifier bounds.
pub const MIN_POISON_ATTEMPTS_PER_CLASS: i64 = 20;
pub const MIN_TRUE_CLAIMS: i64 = 100;
pub const MIN_UNANSWERABLE_QUESTIONS: i64 = 50;
pub const MIN_STAGE_SKIP_ATTEMPTS_PER_KIND: i64 = 20;
pub const MIN_HARD_HONEST_PRESENTED: i64 = 30;
pub const MAX_HARD_HONEST_REFUSED: i64 = 0;
pub const MIN_FALSE_KNOWN_BASELINE_DIRECT: i64 = 1;
pub const MIN_FALSE_KNOWN_GEN_FILTER_ONLY: i64 = 1;
pub const MIN_FALSE_KNOWN_BOOL_TRUST: i64 = 1;
pub const MAX_ORACLE_LABEL_MISMATCHES: i64 = 0;
pub const MAX_CORPUS_SELECTION_DIGEST_MISMATCHES: i64 = 0;
pub const MAX_FALSE_KNOWN_MECHANISM: i64 = 0;
pub const MAX_STAGE_SKIP_ACCEPTED: i64 = 0;
pub const MAX_STAGE_SKIP_UNEXPECTED: i64 = 0;
pub const MAX_STALE_OR_REVOKED_PROMOTED: i64 = 0;
pub const MAX_WRONG_REFUSAL_REASON: i64 = 0;
pub const MAX_PRODUCT_PANICS: i64 = 0;
pub const MIN_RECALL_PERMILLE: i64 = 950;

/// The SHA-256 of the generated corpus (raw events, replicas, questions and hits) of each
/// declared seed at the pinned shape. A run whose corpus digest differs counts one
/// `corpus_selection_digest_mismatches`.
pub const CORPUS_DIGESTS: [(u64, &str); 5] = [
    (
        17,
        "f4057e61ddf1736e13d988607e014f49dc74fb300f8e90cd513e61a2e2dd48b0",
    ),
    (
        29,
        "d4df6e680dd55be21eb9a5972d1d441f512cf7dfb785eaef963048d5bd61b413",
    ),
    (
        43,
        "52d000e8d03207717d21fd1eac79cf6fa3b6928f86081425511f0eaef7eafa7e",
    ),
    (
        71,
        "9389830e75c361d1f443616977bfbeb4608c344ca400267742541df206ad9814",
    ),
    (
        101,
        "a0bda1f8e89570385480cd9af8925e01bc77662fddb15dc6472bd71c6dd337cc",
    ),
];

/// The pinned corpus digest of a seed, if the seed is declared.
pub fn pinned_digest(seed: u64) -> Option<&'static str> {
    CORPUS_DIGESTS
        .iter()
        .find(|(declared, _)| *declared == seed)
        .map(|(_, digest)| *digest)
}

/// The shape of the world: how many capsules of each lifecycle state, and how many questions
/// have no source at all.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub healthy: usize,
    pub revoked: usize,
    pub disputed: usize,
    pub absent: usize,
}

impl Shape {
    pub fn capsules(&self) -> usize {
        self.healthy + self.revoked + self.disputed
    }

    /// The pinned shape.
    pub const PINNED: Shape = Shape {
        healthy: HEALTHY_CAPSULES,
        revoked: REVOKED_CAPSULES,
        disputed: DISPUTED_CAPSULES,
        absent: ABSENT_QUESTIONS,
    };

    /// The shape of `capsules` capsules in the pinned proportions (the mutation checker's
    /// first argument); at least one capsule of each state, at most 100 capsules.
    pub fn scaled(capsules: usize) -> Result<Shape, String> {
        if !(3..=100).contains(&capsules) {
            return Err(format!(
                "capsules must be between 3 and 100, got {capsules}"
            ));
        }
        let revoked = (capsules * REVOKED_CAPSULES / CAPSULES).max(1);
        let disputed = (capsules * DISPUTED_CAPSULES / CAPSULES).max(1);
        let healthy = capsules - revoked - disputed;
        Ok(Shape {
            healthy,
            revoked,
            disputed,
            absent: (capsules * ABSENT_QUESTIONS / CAPSULES).max(1),
        })
    }
}

/// The scale of a run: only the pinned one is the experiment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scale {
    Pinned,
    /// Used by the unit tests: a small world of every capsule state.
    #[cfg_attr(not(test), allow(dead_code))]
    Reduced,
    /// This many capsules in the pinned proportions (the mutation checker's first argument);
    /// no evidence, also when the number is the pinned one.
    Custom(usize),
}

impl Scale {
    pub fn name(self) -> &'static str {
        match self {
            Self::Pinned => "pinned",
            Self::Reduced => "reduced",
            Self::Custom(_) => "custom",
        }
    }

    pub fn shape(self) -> Result<Shape, String> {
        match self {
            Self::Pinned => Ok(Shape::PINNED),
            Self::Reduced => Shape::scaled(9),
            Self::Custom(capsules) => Shape::scaled(capsules),
        }
    }
}

/// One value of the compiled table, comparable with a value of `config.toml`.
#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Leaf {
    Int(i64),
    Bool(bool),
    Text(String),
    Ints(Vec<i64>),
    Texts(Vec<String>),
}

/// The compiled table, one entry per key of `[pinned_thresholds]`.
#[cfg(test)]
pub fn compiled() -> Vec<(String, Leaf)> {
    fn int(value: impl TryInto<i64>) -> Leaf {
        Leaf::Int(value.try_into().ok().expect("fits"))
    }
    let mut table: Vec<(String, Leaf)> = vec![
        (
            "seed_aggregation".into(),
            Leaf::Text(SEED_AGGREGATION.into()),
        ),
        (
            "seeds".into(),
            Leaf::Ints(SEEDS.iter().map(|seed| *seed as i64).collect()),
        ),
        ("capsules".into(), int(CAPSULES)),
        ("healthy_capsules".into(), int(HEALTHY_CAPSULES)),
        ("revoked_capsules".into(), int(REVOKED_CAPSULES)),
        ("disputed_capsules".into(), int(DISPUTED_CAPSULES)),
        ("max_generations".into(), int(MAX_GENERATIONS)),
        ("facts_per_capsule".into(), int(FACTS_PER_CAPSULE)),
        ("entities".into(), int(ENTITIES)),
        ("twin_pairs".into(), int(TWIN_PAIRS)),
        (
            "predicates".into(),
            Leaf::Texts(PREDICATES.iter().map(|p| p.to_string()).collect()),
        ),
        ("values_per_predicate".into(), int(VALUES_PER_PREDICATE)),
        ("absent_questions".into(), int(ABSENT_QUESTIONS)),
        (
            "mirrors_per_quarantined_event".into(),
            int(MIRRORS_PER_QUARANTINED_EVENT),
        ),
        (
            "negations_per_healthy_event".into(),
            int(NEGATIONS_PER_HEALTHY_EVENT),
        ),
        (
            "stale_value_change_permille".into(),
            int(STALE_VALUE_CHANGE_PERMILLE),
        ),
        (
            "quota_p1_fabricated_source".into(),
            int(QUOTA_P1_FABRICATED_SOURCE),
        ),
        ("quota_p2_shifted_span".into(), int(QUOTA_P2_SHIFTED_SPAN)),
        ("quota_p2_span_past_end".into(), int(QUOTA_P2_SPAN_PAST_END)),
        ("quota_p2_forged_quote".into(), int(QUOTA_P2_FORGED_QUOTE)),
        (
            "quota_p3_tampered_digest".into(),
            int(QUOTA_P3_TAMPERED_DIGEST),
        ),
        (
            "quota_p4_lookalike_subject".into(),
            int(QUOTA_P4_LOOKALIKE_SUBJECT),
        ),
        (
            "quota_p5_contradicting_value".into(),
            int(QUOTA_P5_CONTRADICTING_VALUE),
        ),
        ("quota_p5_negation".into(), int(QUOTA_P5_NEGATION)),
        ("quota_p6_superseded".into(), int(QUOTA_P6_SUPERSEDED)),
        ("quota_p2_empty_span".into(), int(QUOTA_P2_EMPTY_SPAN)),
        ("quota_p2_two_line_span".into(), int(QUOTA_P2_TWO_LINE_SPAN)),
        ("quota_p2_padded_span".into(), int(QUOTA_P2_PADDED_SPAN)),
        (
            "quota_p4_predicate_only".into(),
            int(QUOTA_P4_PREDICATE_ONLY),
        ),
        ("quota_p4_case_only".into(), int(QUOTA_P4_CASE_ONLY)),
        (
            "quota_p5_truncated_negation".into(),
            int(QUOTA_P5_TRUNCATED_NEGATION),
        ),
        (
            "quota_p9_relabelled_superseded".into(),
            int(QUOTA_P9_RELABELLED_SUPERSEDED),
        ),
        (
            "quota_p9_relabelled_quarantined".into(),
            int(QUOTA_P9_RELABELLED_QUARANTINED),
        ),
        (
            "hard_honest_per_variant".into(),
            int(HARD_HONEST_PER_VARIANT),
        ),
        ("honest_score_min".into(), int(HONEST_SCORE_MIN)),
        ("honest_score_max".into(), int(HONEST_SCORE_MAX)),
        ("poison_score_min".into(), int(POISON_SCORE_MIN)),
        ("poison_score_max".into(), int(POISON_SCORE_MAX)),
        (
            "stage_skip_attempts_per_kind".into(),
            int(STAGE_SKIP_ATTEMPTS_PER_KIND),
        ),
        (
            "min_poison_attempts_per_class".into(),
            int(MIN_POISON_ATTEMPTS_PER_CLASS),
        ),
        ("min_true_claims".into(), int(MIN_TRUE_CLAIMS)),
        (
            "min_unanswerable_questions".into(),
            int(MIN_UNANSWERABLE_QUESTIONS),
        ),
        (
            "min_stage_skip_attempts_per_kind".into(),
            int(MIN_STAGE_SKIP_ATTEMPTS_PER_KIND),
        ),
        (
            "min_hard_honest_presented".into(),
            int(MIN_HARD_HONEST_PRESENTED),
        ),
        (
            "max_hard_honest_refused".into(),
            int(MAX_HARD_HONEST_REFUSED),
        ),
        (
            "min_false_known_baseline_direct".into(),
            int(MIN_FALSE_KNOWN_BASELINE_DIRECT),
        ),
        (
            "min_false_known_gen_filter_only".into(),
            int(MIN_FALSE_KNOWN_GEN_FILTER_ONLY),
        ),
        (
            "min_false_known_bool_trust".into(),
            int(MIN_FALSE_KNOWN_BOOL_TRUST),
        ),
        (
            "max_oracle_label_mismatches".into(),
            int(MAX_ORACLE_LABEL_MISMATCHES),
        ),
        (
            "max_corpus_selection_digest_mismatches".into(),
            int(MAX_CORPUS_SELECTION_DIGEST_MISMATCHES),
        ),
        (
            "max_false_known_mechanism".into(),
            int(MAX_FALSE_KNOWN_MECHANISM),
        ),
        (
            "max_stage_skip_accepted".into(),
            int(MAX_STAGE_SKIP_ACCEPTED),
        ),
        (
            "max_stage_skip_unexpected".into(),
            int(MAX_STAGE_SKIP_UNEXPECTED),
        ),
        (
            "max_stale_or_revoked_promoted".into(),
            int(MAX_STALE_OR_REVOKED_PROMOTED),
        ),
        (
            "max_wrong_refusal_reason".into(),
            int(MAX_WRONG_REFUSAL_REASON),
        ),
        ("max_product_panics".into(), int(MAX_PRODUCT_PANICS)),
        ("min_recall_permille".into(), int(MIN_RECALL_PERMILLE)),
        ("timing_gated".into(), Leaf::Bool(false)),
    ];
    for (seed, digest) in CORPUS_DIGESTS {
        table.push((
            format!("corpus_digest_seed_{seed}"),
            Leaf::Text(digest.to_string()),
        ));
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn leaf_of(value: &toml::Value) -> Leaf {
        match value {
            toml::Value::Integer(number) => Leaf::Int(*number),
            toml::Value::Boolean(flag) => Leaf::Bool(*flag),
            toml::Value::String(text) => Leaf::Text(text.clone()),
            toml::Value::Array(items) if items.iter().all(toml::Value::is_integer) => Leaf::Ints(
                items
                    .iter()
                    .map(|item| item.as_integer().expect("integer"))
                    .collect(),
            ),
            toml::Value::Array(items) => Leaf::Texts(
                items
                    .iter()
                    .map(|item| item.as_str().expect("string").to_string())
                    .collect(),
            ),
            other => panic!("unexpected value {other:?}"),
        }
    }

    /// The keys on which the compiled constants and `config` disagree, one reason each.
    fn drift(config: &str) -> Vec<String> {
        let parsed: toml::Table = config.parse().expect("config.toml parses");
        let file: BTreeMap<String, Leaf> = parsed["pinned_thresholds"]
            .as_table()
            .expect("[pinned_thresholds]")
            .iter()
            .map(|(key, value)| (key.clone(), leaf_of(value)))
            .collect();
        let compiled: BTreeMap<String, Leaf> = compiled().into_iter().collect();
        let mut problems = Vec::new();
        for (key, value) in &file {
            match compiled.get(key) {
                None => problems.push(format!("{key}: no constant")),
                Some(constant) if constant != value => problems.push(format!("{key}: drifted")),
                Some(_) => {}
            }
        }
        for key in compiled.keys().filter(|key| !file.contains_key(*key)) {
            problems.push(format!("{key}: no key in the file"));
        }
        problems
    }

    #[test]
    fn the_compiled_constants_are_the_pinned_thresholds_of_config_toml() {
        assert_eq!(drift(CONFIG_TOML), Vec::<String>::new());
    }

    #[test]
    fn a_changed_added_or_removed_key_of_config_toml_is_drift() {
        let changed = CONFIG_TOML.replace("capsules = 36\n", "capsules = 37\n");
        assert_ne!(changed, CONFIG_TOML);
        assert_eq!(drift(&changed), ["capsules: drifted"]);
        let added = CONFIG_TOML.replace(
            "[pinned_thresholds]\n",
            "[pinned_thresholds]\nbrand_new_threshold = 1\n",
        );
        assert_eq!(drift(&added), ["brand_new_threshold: no constant"]);
        let removed = CONFIG_TOML.replace("capsules = 36\n", "");
        assert_ne!(removed, CONFIG_TOML);
        assert_eq!(drift(&removed), ["capsules: no key in the file"]);
    }

    #[test]
    fn the_compiled_table_has_no_duplicate_key() {
        let all = compiled();
        let unique: std::collections::BTreeSet<&str> =
            all.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(unique.len(), all.len());
    }

    #[test]
    fn the_pinned_shape_adds_up_and_the_manifest_seeds_are_the_constants() {
        assert_eq!(
            HEALTHY_CAPSULES + REVOKED_CAPSULES + DISPUTED_CAPSULES,
            CAPSULES
        );
        assert_eq!(Shape::PINNED.capsules(), CAPSULES);
        assert_eq!(Shape::scaled(CAPSULES), Ok(Shape::PINNED));
        let manifest: toml::Table = EXPERIMENT_TOML.parse().expect("experiment.toml parses");
        let seeds: Vec<u64> = manifest["seeds"]
            .as_array()
            .expect("seeds")
            .iter()
            .map(|seed| seed.as_integer().expect("integer") as u64)
            .collect();
        assert_eq!(seeds, SEEDS);
        let digests: Vec<u64> = CORPUS_DIGESTS.iter().map(|(seed, _)| *seed).collect();
        assert_eq!(digests, SEEDS);
    }

    #[test]
    fn the_scales_are_named_and_a_bad_capsule_count_is_refused() {
        assert_eq!(Scale::Pinned.name(), "pinned");
        assert_eq!(Scale::Pinned.shape(), Ok(Shape::PINNED));
        assert_ne!(Scale::Reduced.shape(), Ok(Shape::PINNED));
        assert_eq!(Scale::Custom(5).name(), "custom");
        for bad in [0, 1, 2, 101] {
            assert!(Shape::scaled(bad).is_err(), "{bad}");
        }
        for capsules in 3..=100 {
            let shape = Shape::scaled(capsules).expect("a shape");
            assert_eq!(shape.capsules(), capsules);
            assert!(shape.healthy >= 1 && shape.revoked >= 1 && shape.disputed >= 1);
        }
    }

    #[test]
    fn pinned_digests_are_looked_up_by_declared_seed_only() {
        assert!(pinned_digest(17).is_some());
        assert!(pinned_digest(18).is_none());
    }
}
