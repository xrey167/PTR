use super::*;
use crate::experiments::layered_kv::emit::{self, judge, Json, Status};

const CRITERIA: &str =
    include_str!("../../../../../experiments/retrieval/Q003-postgres-hybrid/criteria.toml");
const MUTATIONS: &str =
    include_str!("../../../../../experiments/retrieval/Q003-postgres-hybrid/tests/mutations.toml");
const DRAFT: &str =
    include_str!("../../../../../experiments/retrieval/Q003-postgres-hybrid/PREREGISTRATION.md");

fn rules(table: &toml::Table, key: &str) -> Vec<String> {
    table[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} is an array"))
        .iter()
        .map(|rule| rule.as_str().expect("a rule is a string").to_string())
        .collect()
}

fn is_literal(token: &str) -> bool {
    let digits = token.strip_prefix('-').unwrap_or(token);
    !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
}

fn int(json: &Json, path: &str) -> i64 {
    match emit::resolve(json, path) {
        Some(Json::Int(value)) => *value,
        other => panic!("{path} is {other:?}"),
    }
}

/// `json` with the member `path` (dotted) replaced.
fn with(json: &Json, path: &str, value: Json) -> Json {
    fn set(at: &mut Json, keys: &[&str], value: Json) {
        let Json::Obj(pairs) = at else {
            panic!("not an object")
        };
        let slot = pairs
            .iter_mut()
            .find(|(name, _)| name == keys[0])
            .expect("member");
        if keys.len() == 1 {
            slot.1 = value;
        } else {
            set(&mut slot.1, &keys[1..], value);
        }
    }
    let mut copy = json.clone();
    set(&mut copy, &path.split('.').collect::<Vec<_>>(), value);
    copy
}

fn args(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

fn no_environment(_: &str) -> Option<String> {
    None
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
    assert_eq!(listed, metrics::METRIC_NAMES);
}

#[test]
fn the_criteria_file_is_exactly_the_rules_the_harness_judges_in_its_order() {
    let table: toml::Table = CRITERIA.parse().expect("criteria.toml is TOML");
    assert_eq!(rules(&table, "coverage"), metrics::coverage());
    assert_eq!(rules(&table, "falsifiers"), metrics::falsifiers());
    assert_eq!(table["experiment"].as_str(), Some("Q003"));
    assert_eq!(
        table["seed_aggregation"].as_str(),
        Some(params::SEED_AGGREGATION)
    );
}

/// Every rule is a reading of a sentence of the manifest's falsification text.
#[test]
fn the_rules_mirror_the_falsification_text_of_the_manifest() {
    let manifest: toml::Table = params::EXPERIMENT_TOML
        .parse()
        .expect("experiment.toml parses");
    let text = manifest["falsification"].as_str().expect("falsification");
    for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
        let tokens: Vec<&str> = rule.split(' ').collect();
        let sentence = match tokens[1] {
            ">=" => format!("{} < {}", tokens[0], tokens[2]),
            "<=" => format!("{} > {}", tokens[0], tokens[2]),
            _ => format!("{} != {}", tokens[0], tokens[2]),
        };
        assert!(
            text.contains(&sentence),
            "{rule}: no '{sentence}' in the text"
        );
    }
    assert_eq!(metrics::falsifiers().len(), 11);
    assert_eq!(metrics::coverage().len(), 16);
}

#[test]
fn the_entrypoint_is_empty_until_the_freeze_and_then_the_subcommand() {
    assert_eq!(params::BENCHMARK, "postgres-hybrid");
    let manifest: toml::Table = params::EXPERIMENT_TOML
        .parse()
        .expect("experiment.toml parses");
    let entrypoint = manifest["entrypoint"].as_str().expect("entrypoint");
    assert!(entrypoint.is_empty() || entrypoint == params::ENTRYPOINT);
    assert!(DRAFT.contains(params::ENTRYPOINT));
    let required: Vec<&str> = manifest["required_artifacts"]
        .as_array()
        .expect("required_artifacts")
        .iter()
        .map(|name| name.as_str().expect("a name"))
        .collect();
    assert!(required.contains(&"mutations.json"));
}

/// The compiled constants are exactly the keys of `[pinned_thresholds]`.
#[test]
fn the_compiled_parameters_are_the_pinned_thresholds_of_the_config() {
    let config: toml::Table = params::CONFIG_TOML.parse().expect("config.toml parses");
    let pinned = config["pinned_thresholds"]
        .as_table()
        .expect("pinned_thresholds");
    let int_of = |value: usize| toml::Value::Integer(value as i64);
    let signed = toml::Value::Integer;
    let text = |value: &str| toml::Value::String(value.to_string());
    let mut compiled: Vec<(String, toml::Value)> = vec![
        ("seed_aggregation".into(), text(params::SEED_AGGREGATION)),
        (
            "seeds".into(),
            toml::Value::Array(params::SEEDS.iter().map(|s| signed(*s as i64)).collect()),
        ),
        ("embedding_model".into(), text(params::EMBEDDING_MODEL)),
        (
            "embedding_revision".into(),
            text(params::EMBEDDING_REVISION),
        ),
        (
            "embedding_dimensions".into(),
            int_of(params::EMBEDDING_DIMENSIONS),
        ),
        (
            "strong_dimensions".into(),
            int_of(params::STRONG_DIMENSIONS),
        ),
        ("pool_path".into(), text(params::POOL_PATH)),
        ("pool_sha256".into(), text(params::POOL_SHA256)),
        ("pool_rows".into(), int_of(params::POOL_ROWS)),
        ("vectors_path".into(), text(params::VECTORS_PATH)),
        ("vectors_sha256".into(), text(params::VECTORS_SHA256)),
        (
            "strong_vectors_path".into(),
            text(params::STRONG_VECTORS_PATH),
        ),
        (
            "strong_vectors_sha256".into(),
            text(params::STRONG_VECTORS_SHA256),
        ),
        ("capsules".into(), int_of(params::CAPSULES)),
        ("stay_permille".into(), int_of(params::STAY_PERMILLE)),
        (
            "superseded_once_permille".into(),
            int_of(params::SUPERSEDED_ONCE_PERMILLE),
        ),
        (
            "superseded_twice_permille".into(),
            int_of(params::SUPERSEDED_TWICE_PERMILLE),
        ),
        (
            "revoked_at_1_permille".into(),
            int_of(params::REVOKED_AT_1_PERMILLE),
        ),
        (
            "revoked_at_2_permille".into(),
            int_of(params::REVOKED_AT_2_PERMILLE),
        ),
        ("queries".into(), int_of(params::QUERIES)),
        ("strong_queries".into(), int_of(params::STRONG_QUERIES)),
        (
            "first_checkpoint_percent".into(),
            int_of(params::FIRST_CHECKPOINT_PERCENT),
        ),
        (
            "revoked_target_percent".into(),
            int_of(params::REVOKED_TARGET_PERCENT),
        ),
        (
            "extra_plant_percent".into(),
            int_of(params::EXTRA_PLANT_PERCENT),
        ),
        (
            "stale_probe_percent".into(),
            int_of(params::STALE_PROBE_PERCENT),
        ),
        (
            "live_probe_percent".into(),
            int_of(params::LIVE_PROBE_PERCENT),
        ),
        (
            "late_superseded_percent".into(),
            int_of(params::LATE_SUPERSEDED_PERCENT),
        ),
        (
            "late_revoked_percent".into(),
            int_of(params::LATE_REVOKED_PERCENT),
        ),
        (
            "late_future_percent".into(),
            int_of(params::LATE_FUTURE_PERCENT),
        ),
        (
            "late_ghost_percent".into(),
            int_of(params::LATE_GHOST_PERCENT),
        ),
        ("repeat_queries".into(), int_of(params::REPEAT_QUERIES)),
        (
            "search_limit".into(),
            signed(i64::from(params::SEARCH_LIMIT)),
        ),
        (
            "lexical_weight_permille".into(),
            signed(i64::from(params::LEXICAL_WEIGHT_PERMILLE)),
        ),
        (
            "vector_weight_permille".into(),
            signed(i64::from(params::VECTOR_WEIGHT_PERMILLE)),
        ),
        (
            "rank_constant".into(),
            signed(i64::from(params::RANK_CONSTANT)),
        ),
        ("k".into(), int_of(params::K)),
        (
            "min_planted_stale_rows".into(),
            signed(params::MIN_PLANTED_STALE_ROWS),
        ),
        (
            "min_revoked_targets_queried".into(),
            signed(params::MIN_REVOKED_TARGETS_QUERIED),
        ),
        (
            "min_superseded_targets_queried".into(),
            signed(params::MIN_SUPERSEDED_TARGETS_QUERIED),
        ),
        (
            "min_answerable_queries".into(),
            signed(params::MIN_ANSWERABLE_QUERIES),
        ),
        (
            "min_retrieval_hits_per_mode".into(),
            signed(params::MIN_RETRIEVAL_HITS_PER_MODE),
        ),
        (
            "min_late_write_attempts".into(),
            signed(params::MIN_LATE_WRITE_ATTEMPTS),
        ),
        (
            "min_probe_queries".into(),
            signed(params::MIN_PROBE_QUERIES),
        ),
        (
            "min_control_planted_returned".into(),
            signed(params::MIN_CONTROL_PLANTED_RETURNED),
        ),
        (
            "min_control_vector_returned".into(),
            signed(params::MIN_CONTROL_VECTOR_RETURNED),
        ),
        (
            "min_reference_queries".into(),
            signed(params::MIN_REFERENCE_QUERIES),
        ),
        (
            "min_strong_queries".into(),
            signed(params::MIN_STRONG_QUERIES),
        ),
        (
            "min_recall10_reference_permille".into(),
            signed(params::MIN_RECALL10_REFERENCE_PERMILLE),
        ),
        (
            "pilot_recall_paired_se_permille".into(),
            signed(params::PILOT_RECALL_PAIRED_SE_PERMILLE),
        ),
        (
            "max_workload_digest_mismatches".into(),
            signed(params::MAX_WORKLOAD_DIGEST_MISMATCHES),
        ),
        ("max_stale_hits".into(), signed(params::MAX_STALE_HITS)),
        (
            "max_planted_stale_returned".into(),
            signed(params::MAX_PLANTED_STALE_RETURNED),
        ),
        (
            "max_projector_residual_rows".into(),
            signed(params::MAX_PROJECTOR_RESIDUAL_ROWS),
        ),
        (
            "max_nondeterministic_searches".into(),
            signed(params::MAX_NONDETERMINISTIC_SEARCHES),
        ),
        (
            "min_live_probe_lexical_self_hits".into(),
            signed(params::MIN_LIVE_PROBE_LEXICAL_SELF_HITS),
        ),
        (
            "lexical_self_hit_percent".into(),
            signed(params::LEXICAL_SELF_HIT_PERCENT),
        ),
        (
            "recall_exact_margin_permille".into(),
            signed(params::RECALL_EXACT_MARGIN_PERMILLE),
        ),
        (
            "max_late_write_accepted".into(),
            signed(params::MAX_LATE_WRITE_ACCEPTED),
        ),
        (
            "max_late_write_wrong_refusal".into(),
            signed(params::MAX_LATE_WRITE_WRONG_REFUSAL),
        ),
        (
            "max_product_errors".into(),
            signed(params::MAX_PRODUCT_ERRORS),
        ),
        (
            "max_lifecycle_projection_mismatches".into(),
            signed(params::MAX_LIFECYCLE_PROJECTION_MISMATCHES),
        ),
        (
            "max_product_panics".into(),
            signed(params::MAX_PRODUCT_PANICS),
        ),
        (
            "recall_margin_permille".into(),
            signed(params::RECALL_MARGIN_PERMILLE),
        ),
        ("timing_gated".into(), toml::Value::Boolean(false)),
    ];
    for (seed, digest) in params::WORKLOAD_DIGESTS {
        compiled.push((format!("workload_digest_seed_{seed}"), text(digest)));
    }
    for (key, value) in &compiled {
        assert_eq!(pinned.get(key), Some(value), "{key}");
    }
    let mut left: Vec<&str> = pinned.keys().map(String::as_str).collect();
    let mut right: Vec<&str> = compiled.iter().map(|(key, _)| key.as_str()).collect();
    left.sort_unstable();
    right.sort_unstable();
    assert_eq!(
        left, right,
        "a key of the config no constant mirrors, or the reverse"
    );
}

#[test]
fn the_pinned_inputs_agree_with_the_baselines_own_configuration() {
    let root = data::repo_root();
    let read = |path: &str| std::fs::read_to_string(root.join(path)).expect(path);
    let reference: toml::Table = read("research/baselines/rag_reference/config.toml")
        .parse()
        .unwrap();
    assert_eq!(
        reference["hybrid"]["embedding_model"].as_str(),
        Some(params::EMBEDDING_MODEL)
    );
    assert_eq!(
        reference["hybrid"]["embedding_revision"].as_str(),
        Some(params::EMBEDDING_REVISION)
    );
    let strong: toml::Table = read("research/baselines/strong_retrieval/config.toml")
        .parse()
        .unwrap();
    assert_eq!(
        strong["dense"]["vectors"].as_str(),
        Some(params::STRONG_VECTORS_PATH)
    );
    assert_eq!(strong["dense"]["pool"].as_str(), Some(params::POOL_PATH));
    assert_eq!(
        strong["dense"]["dimension"].as_integer(),
        Some(params::STRONG_DIMENSIONS as i64)
    );
    assert_eq!(
        strong["fusion"]["k"].as_float(),
        Some(f64::from(params::RANK_CONSTANT))
    );
}

#[test]
fn the_hard_counters_of_the_mutation_plan_are_the_harnesss_and_each_is_expected_by_a_mutation() {
    let plan: toml::Table = MUTATIONS.parse().expect("mutations.toml is TOML");
    let listed: Vec<String> = plan["hard_counters"]
        .as_array()
        .expect("hard_counters")
        .iter()
        .map(|name| name.as_str().expect("a name").to_string())
        .collect();
    assert_eq!(listed, metrics::HARD_COUNTERS);
    assert_eq!(plan["subcommand"].as_str(), Some(params::BENCHMARK));
    assert_eq!(plan["cases"].as_integer(), Some(400));
    let mutations = plan["mutation"].as_array().expect("mutations");
    assert_eq!(mutations.len(), 19);
    for counter in metrics::HARD_COUNTERS {
        assert!(
            mutations.iter().any(|m| m["expect"]
                .as_array()
                .expect("expect")
                .iter()
                .any(|name| name.as_str() == Some(counter))),
            "{counter} is expected by no mutation"
        );
    }
    // Every falsifier names a hard counter or is restated by one, and every hard counter is a metric.
    for counter in metrics::HARD_COUNTERS {
        assert!(metrics::METRIC_NAMES.contains(&counter), "{counter}");
    }
}

#[test]
fn the_arguments_are_a_seed_or_queries_and_a_seed_and_the_development_flags() {
    let pinned = parse_args(&args(&["17"]), &no_environment).unwrap();
    assert_eq!(pinned.seed, 17);
    assert_eq!(pinned.scale, Scale::Pinned);
    assert_eq!(
        pinned.options.strong,
        StrongMode::Full,
        "the pinned entrypoint runs the full strong arm"
    );
    let custom = parse_args(&args(&["400", "17"]), &no_environment).unwrap();
    assert_eq!(custom.scale, Scale::Custom(400));
    assert_eq!(custom.options.strong, StrongMode::Off);
    let off = parse_args(&args(&["17", "--no-strong"]), &no_environment).unwrap();
    assert_eq!(off.options.strong, StrongMode::Off);
    let flags = parse_args(
        &args(&[
            "--strong-python",
            "/x/python",
            "17",
            "--models-root",
            "/m",
            "--strong-threads",
            "2",
        ]),
        &no_environment,
    )
    .unwrap();
    assert_eq!(flags.options.strong_python.as_deref(), Some("/x/python"));
    assert_eq!(
        flags.options.models_root,
        Some(std::path::PathBuf::from("/m"))
    );
    assert_eq!(flags.options.strong_threads, Some(2));
    let from_environment = parse_args(&args(&["17"]), &|name| match name {
        "PTR_Q003_STRONG_PYTHON" => Some("/env/python".into()),
        "PTR_Q003_MODELS_ROOT" => Some("/env/models".into()),
        _ => None,
    })
    .unwrap();
    assert_eq!(
        from_environment.options.strong_python.as_deref(),
        Some("/env/python")
    );
    assert!(
        parse_args(&args(&["17", "--print-workload-digest"]), &no_environment)
            .unwrap()
            .digest_only
    );
    for bad in [
        args(&[]),
        args(&["x"]),
        args(&["-1"]),
        args(&["400", "x"]),
        args(&["1", "2", "3"]),
        args(&["17", "--nope"]),
        args(&["17", "--models-root"]),
        args(&["17", "--strong-threads", "x"]),
    ] {
        assert!(parse_args(&bad, &no_environment).is_err(), "{bad:?}");
    }
}

#[test]
fn a_shape_the_generator_cannot_serve_stops_the_run_before_anything_starts() {
    let options = Options {
        strong: StrongMode::Off,
        strong_python: None,
        models_root: None,
        strong_threads: None,
        resume_dir: None,
    };
    assert!(execute(17, Scale::Custom(2), &options).is_err());
    assert!(execute(17, Scale::Custom(5000), &options).is_err());
}

// The tests below drive a real server through PTR_PG_EXPERIMENT_DSN and the Python drivers, so they
// are ignored by default: `cargo test -p ptr-bench --features postgres-experiments q003 --
// --include-ignored`.

fn development_options() -> Options {
    Options {
        strong: StrongMode::NoRerank,
        strong_python: None,
        models_root: None,
        strong_threads: None,
        resume_dir: None,
    }
}

/// The shared reduced run (a small world, the strong arm without its cross-encoder).
fn reduced_report() -> &'static Report {
    static REPORT: std::sync::OnceLock<Report> = std::sync::OnceLock::new();
    REPORT
        .get_or_init(|| execute(17, Scale::Reduced, &development_options()).expect("a reduced run"))
}

#[test]
#[ignore = "needs PTR_PG_EXPERIMENT_DSN and python3"]
fn a_reduced_run_assembles_one_json_object_with_every_metric() {
    let report = reduced_report();
    assert!(!report.line.contains('\n'));
    assert!(report
        .line
        .starts_with("{\"benchmark\":\"postgres-hybrid\",\"experiment\":\"Q003\""));
    assert!(report.line.contains("\"harness_gaps\":[]"), "no gap");
    assert_eq!(int(&report.json, "seed"), 17);
    for name in metrics::METRIC_NAMES {
        assert!(
            emit::resolve(&report.json, name).is_some_and(|v| !matches!(v, Json::Null)),
            "{name}"
        );
    }
    assert!(matches!(
        emit::resolve(&report.json, "scale"),
        Some(Json::Str(scale)) if scale == "reduced"
    ));
    assert!(matches!(
        emit::resolve(&report.json, "procedure.shape_is_pinned"),
        Some(Json::Bool(false))
    ));
    assert!(matches!(
        emit::resolve(&report.json, "procedure.strong_arm.mode"),
        Some(Json::Str(mode)) if mode == "no-rerank"
    ));
}

#[test]
#[ignore = "needs PTR_PG_EXPERIMENT_DSN and python3"]
fn every_rule_names_an_integer_the_harness_prints() {
    let report = reduced_report();
    for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
        let tokens: Vec<&str> = rule.split(' ').collect();
        assert_eq!(tokens.len(), 3, "{rule}");
        for token in [tokens[0], tokens[2]] {
            if !is_literal(token) {
                assert!(
                    matches!(emit::resolve(&report.json, token), Some(Json::Int(_))),
                    "{rule}: {token} does not resolve to an integer in the harness JSON"
                );
            }
        }
    }
}

/// Every rule can fail: set the counter it names to a violating value in the emitted JSON and the
/// harness's own judge flags that rule.
#[test]
#[ignore = "needs PTR_PG_EXPERIMENT_DSN and python3"]
fn every_rule_fails_when_the_emitted_counter_is_set_to_a_violation() {
    let report = reduced_report();
    for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
        let tokens: Vec<&str> = rule.split(' ').collect();
        let current = int(&report.json, tokens[0]);
        let violating = match tokens[1] {
            ">=" if is_literal(tokens[2]) => tokens[2].parse::<i64>().unwrap() - 1,
            "<=" if is_literal(tokens[2]) => tokens[2].parse::<i64>().unwrap() + 1,
            ">=" => int(&report.json, tokens[2]) - 1,
            _ => int(&report.json, tokens[2]) + 1,
        };
        assert_ne!(current, violating, "{rule}");
        let broken = with(&report.json, tokens[0], Json::Int(violating));
        let failed: Vec<String> = judge(&broken, &[], std::slice::from_ref(rule))
            .into_iter()
            .filter(|check| check.status == Status::Fail)
            .map(|check| check.rule)
            .collect();
        assert_eq!(failed, vec![rule.clone()], "{rule}");
    }
}

#[test]
#[ignore = "needs PTR_PG_EXPERIMENT_DSN and python3"]
fn the_product_holds_the_lifecycle_rules_on_a_reduced_world_and_the_control_shows_the_defect() {
    let json = &reduced_report().json;
    for counter in [
        "stale_hits",
        "planted_stale_returned",
        "projector_residual_rows",
        "late_write_accepted",
        "late_write_wrong_refusal",
        "product_errors",
        "lifecycle_projection_mismatches",
        "product_panics",
        "nondeterministic_searches",
        "recall_violation_vs_exact",
        "lexical_probe_violation",
    ] {
        assert_eq!(int(json, counter), 0, "{counter}");
    }
    assert!(int(json, "planted_stale_rows") > 0);
    assert!(int(json, "late_write_attempts") > 0);
    assert!(int(json, "control_planted_returned") > 0);
    assert!(
        int(json, "retrieval_hits.lexical") > 0,
        "the probes make the lexical mode return hits"
    );
    assert!(int(json, "retrieval_hits.vector") > 0);
    assert_eq!(int(json, "queries"), 36);
    assert!(matches!(
        emit::resolve(json, "controls.no_live_join.required_rule_tripped"),
        Some(Json::Bool(_))
    ));
}

#[test]
#[ignore = "needs PTR_PG_EXPERIMENT_DSN and python3"]
fn the_same_seed_gives_the_same_json_apart_from_the_timings() {
    let strip = |json: &Json| {
        let Json::Obj(pairs) = json else {
            panic!("object")
        };
        Json::Obj(
            pairs
                .iter()
                .filter(|(k, _)| {
                    !matches!(k.as_str(), "timing_ns" | "latency_ns" | "index_build_ns")
                })
                .cloned()
                .collect(),
        )
        .render()
    };
    let first = strip(&reduced_report().json);
    let again = strip(
        &execute(17, Scale::Reduced, &development_options())
            .expect("a second run")
            .json,
    );
    assert_eq!(first, again);
}

#[test]
#[ignore = "needs PTR_PG_EXPERIMENT_DSN and python3"]
fn timings_are_in_their_own_fields_and_no_rule_reads_them() {
    let json = &reduced_report().json;
    assert!(int(json, "timing_ns.elapsed_ns") > 0);
    assert!(int(json, "latency_ns.pg_search_p50") > 0);
    assert!(int(json, "index_build_ns") > 0);
    for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
        assert!(!rule.contains("timing") && !rule.contains("_ns"), "{rule}");
    }
}

#[test]
#[ignore = "needs PTR_PG_EXPERIMENT_DSN and python3"]
fn a_second_run_resumes_every_strong_answer_from_the_first() {
    let dir = std::env::temp_dir().join(format!("q003-resume-run-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let options = Options {
        resume_dir: Some(dir.clone()),
        ..development_options()
    };
    let first = execute(17, Scale::Reduced, &options).expect("a first run");
    let second = execute(17, Scale::Reduced, &options).expect("a resumed run");
    let strong = int(&first.json, "strong_queries");
    assert!(strong > 0);
    assert_eq!(int(&first.json, "procedure.strong_arm.resumed_queries"), 0);
    assert_eq!(
        int(&second.json, "procedure.strong_arm.resumed_queries"),
        strong
    );
    for name in [
        "recall10_strong_permille",
        "strong_discordant_queries",
        "strong_queries",
    ] {
        assert_eq!(int(&first.json, name), int(&second.json, name), "{name}");
    }
    // The answers are on disk, one line a query after the header.
    let files: Vec<_> = std::fs::read_dir(&dir).unwrap().collect();
    assert_eq!(files.len(), 1);
    let text = std::fs::read_to_string(files[0].as_ref().unwrap().path()).unwrap();
    assert_eq!(text.lines().count() as i64, strong + 1);
    let _ = std::fs::remove_dir_all(&dir);
}
