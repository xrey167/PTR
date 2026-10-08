//! Experiment R007, span forget, as a harness: `ptr-bench span-forget <seed>` runs the cases
//! C1 to C7 of the experiment's preregistration once for one seed and prints exactly one JSON
//! object on stdout.
//!
//! One run is one seed's evidence, judged mechanically by the experiment's `aggregate.py`
//! under `criteria.toml`; a run at a reduced scale (the smoke test) is no evidence. The code
//! under test is the CPU reference KV backend (`ReferenceKvReuseBackend`: its catalog with
//! parent-chained chunk keys, lineage revocation, and the bitwise probes of R007 WP1), driven
//! only through its public API. The mechanism is lineage revocation plus parent-chained
//! chunk keys, not a mid-sequence eviction primitive (the catalog has none).
//!
//! The object holds, in this order: the identity of the run, every metric the manifest names
//! (a JSON integer, or the digest string), the timings (never read by a rule), the procedure
//! parameters, the model pins, the producer environment, the corpus selection digest, the
//! canary identity, the per-case counters, the two judgements (`coverage` and `falsifiers`),
//! and the counters that belong to no metric (`internal`). The exit status is 1 when a
//! coverage prerequisite is unmet, a falsifier is tripped or a metric has no value and no
//! declared reason, and 2 when the harness itself cannot run a case (a refusal that is not the
//! expected one, an insert refusal on the honest path, a negative control that does not trip).

pub mod cases;
pub mod metrics;
pub mod params;

use super::layered_kv::emit::{count, judge, Check, Gaps, Json, Status, Tally};
use super::layered_kv::lab::Res;
use super::layered_kv::pins::{self, PinChecks};
use params::Shape;
use std::time::Instant;

/// The scale of a run: only the pinned one is the experiment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scale {
    Pinned,
    /// Used by the smoke test only: the same span and canaries in a short context.
    #[cfg_attr(not(test), allow(dead_code))]
    Reduced,
}

impl Scale {
    pub fn name(self) -> &'static str {
        match self {
            Self::Pinned => "pinned",
            Self::Reduced => "reduced",
        }
    }

    pub fn shape(self) -> Shape {
        match self {
            Self::Pinned => params::PINNED_SHAPE,
            Self::Reduced => params::REDUCED_SHAPE,
        }
    }
}

fn selection(tally: &mut Tally) -> Json {
    let computed = cases::selection_digest();
    let matches = computed == params::SELECTION_DIGEST;
    tally.set("selection_digest_mismatches", i64::from(!matches));
    Json::obj([
        ("corpus", Json::str(params::CORPUS)),
        ("rule", Json::str(params::CORPUS_SELECTION_RULE)),
        (
            "generator_seed",
            Json::Int(params::CORPUS_GENERATOR_SEED as i64),
        ),
        ("domain", Json::str(params::SELECTION_DOMAIN)),
        ("computed", Json::str(computed)),
        ("pinned", Json::str(params::SELECTION_DIGEST)),
        ("matches", Json::Bool(matches)),
        ("sequences", Json::Int(params::SELECTION_SEQUENCES as i64)),
        (
            "sequence_tokens",
            Json::Int(params::SELECTION_SEQUENCE_TOKENS as i64),
        ),
        ("per_seed_role", Json::str(params::PER_SEED_ROLE)),
    ])
}

/// The run's identity and the choices of the harness, as `procedure` records them next to the
/// observed parameters (which the cases fill in).
fn chosen_json(seed: u64, scale: Scale) -> Json {
    let shape = scale.shape();
    let seeds = params::declared_seeds();
    Json::obj([
        ("scale", Json::str(scale.name())),
        ("seed", Json::Int(seed as i64)),
        ("seed_is_declared", Json::Bool(seeds.contains(&seed))),
        (
            "shape_is_pinned",
            Json::Bool(shape == params::PINNED_SHAPE),
        ),
        (
            "shape",
            Json::obj([
                ("system", Json::Int(shape.system as i64)),
                ("head", Json::Int(shape.head as i64)),
                ("span", Json::Int(shape.span as i64)),
                ("tail", Json::Int(shape.tail as i64)),
                ("question", Json::Int(shape.question as i64)),
            ]),
        ),
        ("prompt_tokens_with_span", Json::Int(shape.with_span() as i64)),
        (
            "prompt_tokens_without_span",
            Json::Int(shape.without_span() as i64),
        ),
        (
            "max_prompt_tokens",
            Json::Int(cases::limits(shape).max_prompt_tokens as i64),
        ),
        ("max_new_tokens", Json::Int(cases::limits(shape).max_new_tokens as i64)),
        (
            "catalog_limits",
            Json::str("max_trie_nodes 65536, max_kv_page_bytes 2^32, max_metadata_bytes 2^30 (an insert refusal on the honest path is exit 2)"),
        ),
        (
            "reference_limits_raised_in_harness_only",
            Json::Bool(params::REFERENCE_LIMITS_RAISED_IN_HARNESS_ONLY),
        ),
    ])
}

/// The model pins: the dense checks of the backend (digest of the configuration it runs,
/// tokenizer, shape, chain id, cold run reproduced by an independent stack) on a backend with
/// the raised limits, and that the raised limits left the pinned configuration untouched.
fn verify_pins(
    tally: &mut Tally,
    backend: &ptr_runtime::ReferenceKvReuseBackend,
    shape: Shape,
) -> Res<pins::Pins> {
    let observed = cases::observed_config_digest(backend);
    if observed != params::TINY_MODEL_CONFIG_DIGEST {
        return Err(format!(
            "the raised limits changed the tiny-model configuration digest: {observed}"
        ));
    }
    let default_backend = crate::experiments::layered_kv::lab::backend()?;
    if default_backend.chain_id() != backend.chain_id() {
        return Err("the raised limits changed the KV chain id".into());
    }
    let mut checks = PinChecks::default();
    let build = pins::base_checks(
        &mut checks,
        backend,
        &observed,
        params::TINY_MODEL_CONFIG_DIGEST,
        params::TOKENIZER_PIN,
    )?;
    Ok(checks.finish(
        tally,
        Json::obj([
            (
                "tiny_model_config_digest",
                Json::str(params::TINY_MODEL_CONFIG_DIGEST),
            ),
            ("tokenizer_pin", Json::str(params::TOKENIZER_PIN)),
            ("moe_topology", Json::str(params::MOE_TOPOLOGY)),
            ("weights", Json::str(params::WEIGHTS)),
            ("weight_seed", Json::Int(params::WEIGHT_SEED as i64)),
            ("base_checkpoint", Json::str("none (seeded random)")),
        ]),
        Json::obj([
            (
                "intermediate_size",
                Json::Int(pins::INTERMEDIATE_SIZE as i64),
            ),
            ("rope_theta", Json::str("10000.0")),
            ("rms_norm_eps", Json::str("1e-6")),
            ("qk_norm", Json::str("disabled")),
            ("key_encoding", Json::str("PreRope")),
            ("max_new_tokens", Json::Int(pins::MAX_NEW_TOKENS as i64)),
            (
                "max_prompt_tokens",
                Json::Int(cases::limits(shape).max_prompt_tokens as i64),
            ),
            (
                "default_max_prompt_tokens",
                Json::Int(default_backend.limits().max_prompt_tokens as i64),
            ),
        ]),
        build,
    ))
}

/// What a run produced, before it is printed.
pub struct Report {
    pub line: String,
    /// The same object as `line`, before it is rendered (read by the tests).
    #[cfg_attr(not(test), allow(dead_code))]
    pub json: Json,
    pub coverage: Vec<Check>,
    pub falsifiers: Vec<Check>,
    pub gaps: Gaps,
}

impl Report {
    pub fn coverage_unmet(&self) -> usize {
        count(&self.coverage, Status::Fail)
    }

    pub fn falsifiers_tripped(&self) -> usize {
        count(&self.falsifiers, Status::Fail)
    }

    /// Whether the run is a clean pass: nothing unmet, nothing tripped, no gap.
    pub fn clean(&self) -> bool {
        self.coverage_unmet() == 0 && self.falsifiers_tripped() == 0 && self.gaps.is_empty()
    }
}

fn note(text: &str) -> Json {
    Json::str(text)
}

/// Run every case once for `seed` at `scale` and assemble the one JSON object. A case the
/// harness cannot run is an `Err` with the reason.
pub fn execute(seed: u64, scale: Scale) -> Result<Report, String> {
    let started = Instant::now();
    let shape = scale.shape();
    let mut tally = Tally::new(metrics::LIST);
    let mut lap = Instant::now();
    let mut phases: Vec<(String, Json)> = Vec::new();
    let mut phase = |name: &str, lap: &mut Instant| {
        phases.push((
            name.to_string(),
            Json::Float(lap.elapsed().as_secs_f64() * 1000.0),
        ));
        *lap = Instant::now();
    };

    let pin_backend = cases::backend(shape)?;
    let pins = verify_pins(&mut tally, &pin_backend, shape)?;
    drop(pin_backend);
    let environment = pins::environment(&pins);
    let selection = selection(&mut tally);
    let chosen = chosen_json(seed, scale);
    phase("pins_and_selection", &mut lap);

    let findings = cases::run_all(&mut tally, seed, shape)?;
    phase("cases", &mut lap);

    let (printed, gaps) = tally.metrics();
    let unsupported_names = tally.unsupported_names();
    let mut procedure = match findings.procedure {
        Json::Obj(pairs) => pairs,
        other => return Err(format!("the procedure record is not an object: {other:?}")),
    };
    if let Json::Obj(extra) = chosen {
        procedure.extend(extra);
    }
    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str(params::BENCHMARK)),
        ("experiment".to_string(), Json::str("R007")),
        (
            "status_note".to_string(),
            note("one seed of the R007 harness: evidence only when run at the pinned scale from the frozen commit and judged by aggregate.py under criteria.toml; the exit status and the coverage and falsifiers lists below are the harness's own reading of the same rules"),
        ),
        ("seed".to_string(), Json::Int(seed as i64)),
        ("scale".to_string(), Json::str(scale.name())),
    ];
    pairs.extend(printed);
    pairs.push((
        "timing_ms".to_string(),
        Json::obj([
            (
                "elapsed",
                Json::Float(started.elapsed().as_secs_f64() * 1000.0),
            ),
            ("phases", Json::Obj(phases)),
            ("runs", findings.timings),
        ]),
    ));
    pairs.push(("procedure".to_string(), Json::Obj(procedure)));
    pairs.push(("model_pins".to_string(), pins.json));
    pairs.push(("environment".to_string(), environment));
    pairs.push(("selection".to_string(), selection));
    pairs.push(("canary_identity".to_string(), findings.canary_identity));
    pairs.push(("corpus_roles".to_string(), findings.roles));
    pairs.push(("cases".to_string(), Json::Obj(findings.cases)));
    pairs.push(("unsupported".to_string(), tally.unsupported_json()));
    pairs.push((
        "measurement_notes".to_string(),
        Json::obj([
            (
                "mechanism",
                note("lineage revocation plus parent-chained chunk keys: the span is a revocable Derived PromptSegment, forget revokes its lineage identity and runs the prompt without it; the catalog has no mid-sequence eviction primitive, and this run does not test one"),
            ),
            (
                "counts",
                note("prefix_reuse_tokens, recompute_tokens, downstream_recomputations and the row counts are read from the runtime's outcome (hit_tokens, prefilled_tokens, inserted_chunks, the revocation's retired count); tokens_after_span is summed from the segments of the request built; none is asserted from a constant"),
            ),
            (
                "bitwise_mismatch_count",
                note("one per differing KV-state digest plus one per differing logits entry (f32 bit patterns), for the forget run against the cold baseline and for each of the 32 probes (forget against baseline); the decode digest is compared too but counted under internal only"),
            ),
            (
                "canary_logit_shift",
                note("an integer: the largest distance, in representable f32 steps (ulps), between the forget run's and the baseline's logit over all 256 entries and all 32 probes; 0 iff the maximum absolute difference is exactly 0 (the float maximum is printed under cases.C5.max_abs_f32, never read by a rule)"),
            ),
            (
                "span_insertions_and_evictions",
                note("1 each: the warm run inserted a row for every chunk that cites the span; the revocation retired exactly the rows that cite it (not fewer)"),
            ),
            (
                "stale_downstream_rows_reused",
                note("catalog rows (16-token chunks) served to the forget run beyond the prefix before the span; every row in that catalog is the with-span run's"),
            ),
            (
                "negative_control",
                note("a third backend under IndexUpdate::Suppressed serves the donor's rows after the span as the prompt's own (StaleDonor); the four falsifiers are judged on its counters and all must trip, else exit 2. It shows the criteria can fail; it is no metric"),
            ),
            (
                "weak_metrics",
                note("model_pin_mismatches and procedure_mismatches are weak (base checkpoint, topology, generator seed are constants here; the rest is read from the backend); canary_probes and post_forget_probes count probes whose two runtime calls returned"),
            ),
            (
                "limits",
                note("tiny seeded random model, CPU reference path, one machine profile, bitwise on f32 bit patterns with no tolerance; probes are logits after a 16-byte suffix at one position; a trace outside the logits and the KV-state digest (timing, memory) is not covered"),
            ),
        ]),
    ));
    pairs.push(("internal".to_string(), tally.internal()));
    let root = Json::Obj(pairs.clone());
    let coverage = judge(&root, &unsupported_names, &metrics::coverage());
    let falsifiers = judge(&root, &unsupported_names, &metrics::falsifiers());
    pairs.push((
        "coverage".to_string(),
        Json::Arr(coverage.iter().map(Check::json).collect()),
    ));
    pairs.push((
        "falsifiers".to_string(),
        Json::Arr(falsifiers.iter().map(Check::json).collect()),
    ));
    pairs.push((
        "summary".to_string(),
        Json::obj([
            (
                "coverage_unmet",
                Json::Int(count(&coverage, Status::Fail) as i64),
            ),
            (
                "coverage_unsupported",
                Json::Int(count(&coverage, Status::Unsupported) as i64),
            ),
            (
                "falsifiers_tripped",
                Json::Int(count(&falsifiers, Status::Fail) as i64),
            ),
            (
                "falsifiers_unsupported",
                Json::Int(count(&falsifiers, Status::Unsupported) as i64),
            ),
        ]),
    ));
    pairs.push((
        "harness_gaps".to_string(),
        Json::Arr(
            gaps.iter()
                .map(|(metric, reason)| {
                    Json::obj([
                        ("metric", Json::str(metric.clone())),
                        ("reason", Json::str(reason.clone())),
                    ])
                })
                .collect(),
        ),
    ));
    let json = Json::Obj(pairs);
    Ok(Report {
        line: json.render(),
        json,
        coverage,
        falsifiers,
        gaps,
    })
}

/// Run the experiment's harness for `seed` and print its one JSON object. Exits 1 when the
/// run is not clean and 2 when the harness cannot run a case.
pub fn run(seed: u64) {
    let seeds = params::declared_seeds();
    if !seeds.contains(&seed) {
        eprintln!("seed {seed} is not one of the manifest's seeds {seeds:?}; this is a dry run");
    }
    match execute(seed, Scale::Pinned) {
        Ok(report) => {
            println!("{}", report.line);
            for check in report.coverage.iter().chain(&report.falsifiers) {
                if check.status == Status::Fail {
                    eprintln!("{} ({})", check.rule, check.observed.render());
                }
            }
            for (metric, reason) in &report.gaps {
                eprintln!("harness gap {metric}: {reason}");
            }
            if !report.clean() {
                std::process::exit(1);
            }
        }
        Err(reason) => {
            eprintln!("{}: {reason}", params::BENCHMARK);
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::emit;

    const PREREGISTRATION: &str =
        include_str!("../../../../../experiments/runtime/R007-span-forget/PREREGISTRATION.md");
    const CRITERIA: &str =
        include_str!("../../../../../experiments/runtime/R007-span-forget/criteria.toml");

    /// The shared reduced run (every case once at the reduced scale: the full 32 canaries and
    /// the 256-token span in a 64-token context; a few tens of seconds in a debug build).
    fn reduced_report() -> &'static Report {
        static REPORT: std::sync::OnceLock<Report> = std::sync::OnceLock::new();
        REPORT.get_or_init(|| execute(17, Scale::Reduced).expect("a reduced run"))
    }

    fn rules(table: &toml::Table, key: &str) -> Vec<String> {
        table[key]
            .as_array()
            .unwrap_or_else(|| panic!("{key} is an array"))
            .iter()
            .map(|rule| rule.as_str().expect("a rule is a string").to_string())
            .collect()
    }

    fn int(json: &Json, path: &str) -> i64 {
        match emit::resolve(json, path) {
            Some(Json::Int(value)) => *value,
            other => panic!("{path} is {other:?}"),
        }
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
    fn the_pinned_selection_digest_is_what_the_corpus_generator_produces() {
        assert_eq!(cases::selection_digest(), params::SELECTION_DIGEST);
        assert!(PREREGISTRATION.contains(params::SELECTION_DIGEST));
    }

    #[test]
    fn the_criteria_file_is_exactly_the_rules_the_harness_judges_in_its_order() {
        let table: toml::Table = CRITERIA.parse().expect("criteria.toml is TOML");
        assert_eq!(rules(&table, "coverage"), metrics::coverage());
        assert_eq!(rules(&table, "falsifiers"), metrics::falsifiers());
        assert_eq!(table["experiment"].as_str(), Some("R007"));
        assert_eq!(
            table["seed_aggregation"].as_str(),
            Some("every_seed_must_pass")
        );
        assert_eq!(
            table["seed_aggregation"].as_str(),
            Some(params::SEED_AGGREGATION)
        );
    }

    #[test]
    fn every_rule_is_three_tokens_and_names_a_path_the_harness_prints() {
        let report = reduced_report();
        for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
            let tokens: Vec<&str> = rule.split(' ').collect();
            assert_eq!(tokens.len(), 3, "{rule}");
            for token in [tokens[0], tokens[2]] {
                let literal = token
                    .strip_prefix('-')
                    .unwrap_or(token)
                    .bytes()
                    .all(|byte| byte.is_ascii_digit());
                if !literal {
                    assert!(
                        matches!(
                            emit::resolve(&report.json, token),
                            Some(Json::Int(_) | Json::Str(_))
                        ),
                        "{rule}: {token} does not resolve to an integer or a digest in the harness JSON"
                    );
                }
            }
        }
    }

    /// Every rule is a reading of a sentence of the manifest's falsification text, and every
    /// sentence that bounds a metric has a rule.
    #[test]
    fn the_rules_mirror_the_falsification_text_of_the_manifest() {
        let manifest: toml::Table = params::EXPERIMENT_TOML
            .parse()
            .expect("experiment.toml parses");
        let text = manifest["falsification"].as_str().expect("falsification");
        let ruled = |name: &str| {
            metrics::coverage()
                .into_iter()
                .chain(metrics::falsifiers())
                .any(|rule| rule.split(' ').next() == Some(name))
        };
        // Coverage: "<metric> < N" in the text.
        for rule in metrics::coverage() {
            let (metric, op, bound) = {
                let tokens: Vec<&str> = rule.split(' ').collect();
                (
                    tokens[0].to_string(),
                    tokens[1].to_string(),
                    tokens[2].to_string(),
                )
            };
            match op.as_str() {
                ">=" => assert!(
                    text.contains(&format!("{metric} < {bound}")),
                    "{rule}: no '{metric} < {bound}' in the text"
                ),
                "<=" => assert!(
                    text.contains(&format!("{metric} > {bound}")),
                    "{rule}: no '{metric} > {bound}' in the text"
                ),
                _ => assert!(
                    text.contains(
                        "canary_identity_set_digest differs from the pinned expected digest"
                    ),
                    "{rule}"
                ),
            }
        }
        // Falsifiers: "> 0" and "!= x" in the text.
        assert!(text.contains("bitwise_mismatch_count > 0"));
        assert!(text.contains("canary_logit_shift != 0"));
        assert!(text.contains("stale_downstream_rows_reused > 0"));
        assert!(text.contains("recompute_tokens != tokens_after_span"));
        assert_eq!(metrics::falsifiers().len(), 4);
        // Every "<metric> <" or "<metric> >" of the text has a rule.
        for name in metrics::METRIC_NAMES {
            if text.contains(&format!("{name} <")) || text.contains(&format!("{name} >")) {
                assert!(ruled(name), "{name}");
            }
        }
        // Every metric of the manifest is judged by some rule or is a bare counter the text
        // reads only through another (tokens_after_span through recompute_tokens).
        for name in metrics::METRIC_NAMES {
            assert!(ruled(name) || name == "tokens_after_span", "{name}");
        }
    }

    #[test]
    fn the_entrypoint_is_empty_until_the_freeze_and_then_the_subcommand() {
        assert_eq!(params::BENCHMARK, "span-forget");
        let manifest: toml::Table = params::EXPERIMENT_TOML
            .parse()
            .expect("experiment.toml parses");
        let entrypoint = manifest["entrypoint"].as_str().expect("entrypoint");
        assert!(entrypoint.is_empty() || entrypoint == params::ENTRYPOINT);
        assert!(PREREGISTRATION.contains(params::ENTRYPOINT));
    }

    #[test]
    fn the_harness_echoes_the_pinned_canary_digest_of_config_toml() {
        let config: toml::Table = params::CONFIG_TOML.parse().expect("config.toml parses");
        let pinned = config["pinned_thresholds"]["canary_expected_digest"]
            .as_str()
            .expect("canary_expected_digest");
        assert!(matches!(
            emit::resolve(&reduced_report().json, "canary_identity.pinned"),
            Some(Json::Str(text)) if text == pinned
        ));
    }

    #[test]
    fn a_reduced_run_assembles_one_json_object_with_every_metric() {
        let report = reduced_report();
        assert!(!report.line.contains('\n'));
        assert!(report
            .line
            .starts_with("{\"benchmark\":\"span-forget\",\"experiment\":\"R007\""));
        for name in metrics::METRIC_NAMES {
            assert!(report.line.contains(&format!("\"{name}\":")), "{name}");
        }
        assert!(report.line.contains("\"harness_gaps\":[]"), "no gap");
        assert_eq!(int(&report.json, "seed"), 17);
        assert!(matches!(
            emit::resolve(&report.json, "scale"),
            Some(Json::Str(scale)) if scale == "reduced"
        ));
        // Every metric is an integer, the digest a string: no float, no null.
        for name in metrics::METRIC_NAMES {
            let value = emit::resolve(&report.json, name).expect(name);
            if name == "canary_identity_set_digest" {
                assert!(
                    matches!(value, Json::Str(text) if text.len() == 64),
                    "{name}"
                );
            } else {
                assert!(matches!(value, Json::Int(_)), "{name}: {value:?}");
            }
        }
    }

    /// The reduced run holds every coverage rule and trips no falsifier: a clean pass (exit
    /// status 0), with the counts the shape implies measured from the runtime.
    #[test]
    fn a_reduced_run_holds_every_coverage_rule_and_trips_no_falsifier() {
        let report = reduced_report();
        let unmet: Vec<&str> = report
            .coverage
            .iter()
            .filter(|check| check.status == Status::Fail)
            .map(|check| check.rule.as_str())
            .collect();
        assert!(unmet.is_empty(), "{unmet:?}");
        let tripped: Vec<&str> = report
            .falsifiers
            .iter()
            .filter(|check| check.status == Status::Fail)
            .map(|check| check.rule.as_str())
            .collect();
        assert!(tripped.is_empty(), "{tripped:?}");
        let shape = params::REDUCED_SHAPE;
        assert_eq!(
            int(&report.json, "prefix_reuse_tokens"),
            shape.prefix() as i64
        );
        assert_eq!(
            int(&report.json, "recompute_tokens"),
            shape.after_span() as i64
        );
        assert_eq!(
            int(&report.json, "tokens_after_span"),
            shape.after_span() as i64
        );
        assert_eq!(
            int(&report.json, "downstream_recomputations"),
            (shape.after_span() / 16) as i64
        );
        assert_eq!(
            int(&report.json, "downstream_rows_before_forget"),
            (shape.after_span() / 16) as i64
        );
        assert_eq!(int(&report.json, "canary_probes"), 32);
        assert_eq!(int(&report.json, "post_forget_probes"), 32);
        assert_eq!(int(&report.json, "canary_span_tokens"), 256);
        assert_eq!(int(&report.json, "span_insertions"), 1);
        assert_eq!(int(&report.json, "span_evictions"), 1);
        assert_eq!(int(&report.json, "bitwise_mismatch_count"), 0);
        assert_eq!(int(&report.json, "canary_logit_shift"), 0);
        assert_eq!(int(&report.json, "stale_downstream_rows_reused"), 0);
        assert!(report.clean(), "{}", report.line);
        // The negative control tripped every falsifier on its own counters.
        assert_eq!(int(&report.json, "cases.C6.falsifiers_tripped"), 4);
        assert!(int(&report.json, "internal.negative_control_bitwise") > 0);
        assert!(int(&report.json, "internal.negative_control_stale_rows") > 0);
        assert!(int(&report.json, "internal.negative_control_ulps") > 0);
    }

    #[test]
    fn the_procedure_records_every_parameter_the_config_lists() {
        let procedure = emit::resolve(&reduced_report().json, "procedure").expect("procedure");
        for name in params::PROCEDURE_PARAMETERS_RECORDED_PER_CELL {
            assert!(emit::resolve(procedure, name).is_some(), "{name}");
        }
        assert!(matches!(
            emit::resolve(procedure, "differing"),
            Some(Json::Arr(items)) if items.is_empty()
        ));
    }

    #[test]
    fn the_model_pins_hold_with_the_raised_limits() {
        let mut tally = Tally::new(metrics::LIST);
        let backend = cases::backend(params::PINNED_SHAPE).unwrap();
        let pins = verify_pins(&mut tally, &backend, params::PINNED_SHAPE).unwrap();
        assert_eq!(
            tally.get("model_pin_mismatches"),
            0,
            "{}",
            pins.json.render()
        );
    }

    #[test]
    fn every_case_ran_and_is_reported_with_its_counters() {
        let json = &reduced_report().json;
        for case in ["C1", "C2", "C3", "C4", "C5", "C6"] {
            assert!(
                matches!(
                    emit::resolve(json, &format!("cases.{case}")),
                    Some(Json::Obj(_))
                ),
                "{case}"
            );
        }
    }

    #[test]
    fn timings_are_in_their_own_fields_and_no_rule_reads_them() {
        let json = &reduced_report().json;
        assert!(matches!(
            emit::resolve(json, "timing_ms.elapsed"),
            Some(Json::Float(_))
        ));
        for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
            assert!(!rule.contains("timing"), "{rule}");
            assert!(!rule.contains("elapsed"), "{rule}");
        }
    }

    #[test]
    fn the_scales_are_named_and_only_the_pinned_one_has_the_pinned_shape() {
        assert_eq!(Scale::Pinned.name(), "pinned");
        assert_eq!(Scale::Reduced.name(), "reduced");
        assert_eq!(Scale::Pinned.shape(), params::PINNED_SHAPE);
        assert_ne!(Scale::Reduced.shape(), params::PINNED_SHAPE);
    }

    /// The full R007 shape for one seed: about five minutes in release and far longer in a
    /// debug build, so it is opt-in: `cargo test --release -p ptr-bench -- --ignored r007`.
    #[test]
    #[ignore = "about five minutes in release; run with --ignored"]
    fn r007_the_pinned_shape_holds_every_rule_with_the_expected_exact_counts() {
        let report = execute(17, Scale::Pinned).expect("a pinned run");
        assert!(report.clean(), "{}", report.line);
        assert_eq!(
            int(&report.json, "prefix_reuse_tokens") as u64,
            params::PREFIX_REUSE_TOKENS_EXPECTED
        );
        assert_eq!(
            int(&report.json, "recompute_tokens") as u64,
            params::TOKENS_AFTER_SPAN_EXPECTED
        );
        assert_eq!(int(&report.json, "downstream_recomputations"), 180);
        assert_eq!(int(&report.json, "downstream_rows_before_forget"), 180);
        assert_eq!(int(&report.json, "canary_probes"), 32);
    }
}
