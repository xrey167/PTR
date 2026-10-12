//! Experiment R013, authority over dormant seeds, lineage and in-flight calls, and a minimal
//! attestor mechanism, as a harness: `ptr-bench self-modification-authority <seed>` runs the
//! experiment's cases once for one seed and prints exactly one JSON object on stdout.
//! `self-modification-authority --preflight` checks, without recording anything, that the
//! harness can run on this machine at all.
//!
//! One run is one seed's evidence, judged mechanically by the experiment's `aggregate.py` under
//! `criteria.toml`; a run at another scale (the unit tests, or the mutation checker's
//! `self-modification-authority <cases> <seed>`) is no evidence. The code under test is
//! `ptr_runtime::PtrRuntime` (its expert registry, the dependency index, the authority epoch,
//! the use-time `expert_route_admission` and the attested admission) over a durable
//! `FileLedger`, driven only through its public API, and, for the in-flight cases, the R010
//! `KvSession` on the tiny seeded MoE. The fixtures are R011's (`r011::cases::fixture`). There
//! is no router and no training; "route" is a probe of `expert_route_admission`, and "direct
//! append" is the public `PtrRuntime` surface (no claim against a raw ledger writer).
//!
//! The exit status is 1 when a coverage prerequisite is unmet or a falsifier is tripped, and 2
//! when the harness itself cannot run: a refusal that is not the expected typed error, a
//! fixture that does not build, a positive control that fails. Every panic of the code under
//! test is contained and counted (`product_panics`) and falsifies.

pub mod cases;
pub mod metrics;
pub mod params;

use super::layered_kv::emit::{count, judge, Check, Gaps, Json, Status};
use super::layered_kv::{corpus, lab};
use cases::Cx;
use params::Scale;
use std::time::Instant;

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

/// The sequences of the suite's selection digest, in listed order.
fn selection_sequences() -> Vec<Vec<u32>> {
    params::ADVERSARIAL_CASES
        .iter()
        .map(|name| name.bytes().map(u32::from).collect())
        .collect()
}

pub fn selection_digest() -> String {
    corpus::selection_digest(params::SELECTION_DOMAIN, &selection_sequences())
}

/// What the pins say: the suite is the pinned one and the backend is the pinned tiny model.
/// Returns the number of pins that differ.
fn procedure_mismatches() -> Res<i64> {
    let mut mismatches = i64::from(selection_digest() != params::SELECTION_DIGEST);
    let backend = lab::backend()?;
    let config = backend.stack_config().clone();
    let r011 = super::r011::params::TEST_MODEL_LAYERS;
    let ok = config.num_layers as u64 == r011
        && config.layer.hidden_size as u64 == super::r011::params::HIDDEN_SIZE
        && config.layer.num_attention_heads as u64 == super::r011::params::NUM_HEADS
        && config.layer.num_key_value_heads as u64 == super::r011::params::NUM_KV_HEADS
        && (config.layer.hidden_size / config.layer.num_attention_heads) as u64
            == super::r011::params::HEAD_DIM
        && config.vocab_size as u64 == super::r011::params::VOCAB_SIZE;
    mismatches += i64::from(!ok);
    Ok(mismatches)
}

type Res<T> = Result<T, String>;

/// What `--preflight` checks: the harness can run on this machine. It records nothing and
/// reads no seed: one attested admission and one compound invalidation end to end, and the pins.
pub fn preflight() -> Result<String, String> {
    use crate::experiments::r011::cases::fixture::{Rt, Setup};
    use cases::{admit_attested, Attestor};
    use ptr_pods::{ExpertInputs, StructuralOp};
    use ptr_runtime::expert::InputInvalidation;
    use ptr_types::Generation;

    let pins = procedure_mismatches()?;
    if pins > 0 {
        return Err(format!("{pins} pins of the harness differ from the build"));
    }
    let mut rt = Rt::fresh()?;
    let attestor = Attestor::new("preflight", 1);
    let op = StructuralOp::Birth {
        result: rt.spec("p"),
    };
    admit_attested(
        &mut rt,
        op,
        ExpertInputs::default(),
        attestor,
        1,
        Setup::PRE,
    )
    .map_err(|error| format!("the attested admission of the preflight: {error:?}"))?;
    if !rt.routable("p", 1) {
        return Err("the attested expert of the preflight does not route".into());
    }
    rt.invalidate(InputInvalidation::revoked(
        attestor.identity(),
        Generation(attestor.epoch),
        attestor.digest(),
    ))
    .map_err(|error| format!("the invalidation of the preflight: {error:?}"))?;
    if rt.routable("p", 1) {
        return Err("the expert of the preflight routes after its attestor was revoked".into());
    }
    Ok(format!(
        "{{\"preflight\":\"ok\",\"benchmark\":\"{}\"}}",
        params::BENCHMARK
    ))
}

/// Run every case once for `seed` at `scale` and assemble the one JSON object. A run the
/// harness cannot make is an `Err` with the reason.
pub fn execute(seed: u64, scale: Scale) -> Result<Report, String> {
    let started = Instant::now();
    let mut cx = Cx::new(seed);
    cx.tally
        .seed_group("adversarial_attempts_per_case", &metrics::attempt_members());
    cx.tally
        .seed_group("structural_inflight_race_cases", &params::INFLIGHT_KINDS);
    cx.tally.seed_group(
        "transitive_authority_revocation_cases",
        &params::AUTHORITY_KINDS,
    );
    cx.tally
        .seed_group("revoke_while_retired_cases", &params::DORMANT_KINDS);
    cx.tally
        .seed_group("reactivation_controls_accepted", &params::DORMANT_KINDS);
    cx.tally.seed_group(
        "superseded_source_revocation_cases",
        &["input", "attestor", "gate_config", "evaluator"],
    );
    cx.tally
        .seed_group("attestation_controls_by_operation", &["birth"]);
    cx.tally
        .add("procedure_mismatches", procedure_mismatches()?);
    let order = cases::run_all(&mut cx)?;
    let hard = metrics::hard_failures(&cx.tally);

    let (printed, gaps) = cx.tally.metrics();
    let unsupported_names = cx.tally.unsupported_names();
    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str(params::BENCHMARK)),
        ("experiment".to_string(), Json::str(params::EXPERIMENT_ID)),
        (
            "status_note".to_string(),
            note("one seed of the R013 harness: evidence only when run at the pinned scale from the frozen commit and judged by aggregate.py under criteria.toml; the exit status and the coverage and falsifiers lists below are the harness's own reading of the same rules"),
        ),
        ("seed".to_string(), Json::Int(seed as i64)),
        ("scale".to_string(), Json::str(scale.name())),
    ];
    pairs.extend(printed);
    pairs.push(("hard_failures".to_string(), Json::Int(hard)));
    pairs.push((
        "timing_ns".to_string(),
        Json::obj([("elapsed_ns", Json::Int(started.elapsed().as_nanos() as i64))]),
    ));
    pairs.push((
        "procedure".to_string(),
        Json::obj([
            ("seed_is_declared", Json::Bool(params::SEEDS.contains(&seed))),
            ("shape_is_pinned", Json::Bool(scale == Scale::Pinned)),
            ("case_order", Json::Arr(order)),
            ("selection_digest", Json::str(selection_digest())),
            ("selection_digest_pinned", Json::str(params::SELECTION_DIGEST)),
            (
                "expectation",
                note("a seed retired before its authority or input was revoked is not reactivated on the evidence of what was revoked, whatever block names it (an unaffected seed is, and so is the compromised seed on evidence under the standing that replaced the authority); every descendant of a consumed source whose input, attestor or authority is revoked is fenced or unroutable and no bystander is closed; a call in flight when the base model or an adapter it rests on is invalidated is cancelled at its next use and nothing is emitted after the event; an attestation that names another proposal or another artifact is refused, a grant that requires attestation refuses an admission without one, a run id is single-use across attestors, restarts and the plain path, a proposal queued across the revocation of its attestor is refused, an expert admitted under a revoked attestor is fenced for good, and a new attestor epoch admits a retrained expert"),
            ),
            (
                "baselines",
                note("B0 is the positive control of every cell: the same operation on an unaffected seed, on evidence under the current standing, with a fresh attestor or a fresh run, or without the requirement of attestation, which must be admitted (a control that is refused ends the run with status 2); the mutation plan (tests/mutations.toml) is the ablation: each planted defect removes one mechanism and the harness must fire the counter that names it"),
            ),
        ]),
    ));
    pairs.push((
        "environment".to_string(),
        Json::obj([(
            "runtime",
            Json::str("PtrRuntime over a FileLedger in a temporary directory; the CPU reference path with a tiny seeded model"),
        )]),
    ));
    pairs.push((
        "measurement_notes".to_string(),
        Json::obj([
            (
                "attempts",
                note("an adversarial attempt counts only after the runtime returned either Ok (a violation) or the expected typed error; any other error is a harness error and ends the run with exit 2, and so does a failing positive control"),
            ),
            (
                "route",
                note("a route is a probe of PtrRuntime::expert_route_admission; no production router consumes it and no end-to-end MoE routing is claimed; a probe of an expert that must be refused is kind-checked (Fenced after a fence), and any other refusal is a harness error"),
            ),
            (
                "compromised_seed",
                note("for an input or an attestor, any reactivation of the seed is owed a refusal (revoked data is not laundered by retraining); for an authority kind, a reactivation on the evidence of the revoked authority is owed a refusal as stale evidence, and a reactivation on evidence under the standing that replaced it is a re-evaluation of the same weights under the current authority, which is the positive control; this is an a-priori definition of this experiment, stated in its preregistration"),
            ),
            (
                "transitive_authority",
                note("weak by construction in this runtime: an authority's epochs are monotone and a revocation fences every serving expert admitted under an earlier epoch, so a descendant admitted before the revocation is fenced by its own epoch; the cases ask that no path (merge, split, reactivation; normal, reopen, compacted restore) leaves one serving, and they see the defects that break the epoch bookkeeping, not a missing lineage rule"),
            ),
            (
                "adapter",
                note("weak: the runtime has no adapter role; an adapter is an input identity (adapter:lora-1) in the training block, so the adapter cells exercise the same index as a training input under another identity; the base-model cells fence all sixteen experts of the MoE and so check the cancellation, the dead dependencies and the refusal of a later request only (no router with every expert masked exists for the replacement path)"),
            ),
            (
                "attestor",
                note("the attestor and the run are input identities (attestor:{name}, run:{hex}) recorded in the dependency block, so revocation, terminality and lineage are the existing dependency index's; the attestation itself is the host's assertion, not recorded and not signed: replay re-checks the attestor's liveness and the single use of the run id (both in the block) and does not re-check the artifact or proposal binding or that an attestation was presented; the artifact digest it binds is the one the operation determines, not weights"),
            ),
            (
                "race",
                note("every race is a sequenced interleaving in a single-writer runtime (prepare, change the authority, commit); the in-flight race point is drawn by the seed in 1..=total steps so at least one step runs before the event"),
            ),
        ]),
    ));
    let mut internal = match cx.tally.internal() {
        Json::Obj(pairs) => pairs,
        other => vec![("counters".to_string(), other)],
    };
    internal.push(("unsupported".to_string(), cx.tally.unsupported_json()));
    pairs.push(("internal".to_string(), Json::Obj(internal)));
    pairs.push(("fixture".to_string(), Json::Obj(cx.fixture.clone())));
    pairs.push(("cases".to_string(), Json::Obj(cx.cases.clone())));
    pairs.push(("subcases".to_string(), cx.subcases_json()));
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

fn print_and_exit(result: Result<Report, String>) {
    match result {
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

/// What the arguments after the subcommand ask for.
#[derive(Debug, Eq, PartialEq)]
enum Request {
    Run(u64, Scale),
    Preflight,
}

/// The seed and scale, or the preflight; strict, so a typo stops the run.
fn parse_args(args: &[String]) -> Result<Request, String> {
    let parse = |text: &String, what: &str| {
        text.parse::<u64>()
            .map_err(|_| format!("{what} must be a non-negative integer, got {text:?}"))
    };
    match args {
        [flag] if flag == "--preflight" => Ok(Request::Preflight),
        [seed] => parse(seed, "seed").map(|seed| Request::Run(seed, Scale::Pinned)),
        [cases, seed] => parse(cases, "cases").and_then(|cases| {
            if cases != 1 {
                return Err("cases must be 1: the plan has one pass".to_string());
            }
            parse(seed, "seed").map(|seed| Request::Run(seed, Scale::Custom))
        }),
        _ => Err(
            "usage: self-modification-authority <seed> | self-modification-authority 1 <seed> | self-modification-authority --preflight"
                .to_string(),
        ),
    }
}

/// Run the experiment's harness and print its one JSON object. `args` are the arguments after
/// the subcommand: `<seed>` (the pinned scale), `1 <seed>` (the mutation checker's form: the
/// same plan, no evidence) or `--preflight`. Exits 1 when the run is not clean and 2 when the
/// harness cannot run.
pub fn run(args: &[String]) {
    static QUIET: std::sync::Once = std::sync::Once::new();
    QUIET.call_once(|| {});
    let (seed, scale) = match parse_args(args) {
        Ok(Request::Preflight) => match preflight() {
            Ok(line) => {
                println!("{line}");
                return;
            }
            Err(reason) => {
                eprintln!("{}: preflight: {reason}", params::BENCHMARK);
                std::process::exit(2);
            }
        },
        Ok(Request::Run(seed, scale)) => (seed, scale),
        Err(reason) => {
            eprintln!("{}: {reason}", params::BENCHMARK);
            std::process::exit(2);
        }
    };
    if !params::SEEDS.contains(&seed) {
        eprintln!(
            "seed {seed} is not one of the manifest's seeds {:?}; this is a dry run",
            params::SEEDS
        );
    }
    print_and_exit(execute(seed, scale));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::emit;

    const CRITERIA: &str = include_str!(
        "../../../../../experiments/runtime/R013-deferred-self-modification-authority/criteria.toml"
    );
    const MUTATIONS: &str = include_str!(
        "../../../../../experiments/runtime/R013-deferred-self-modification-authority/tests/mutations.toml"
    );
    const PREREGISTRATION: &str = include_str!(
        "../../../../../experiments/runtime/R013-deferred-self-modification-authority/PREREGISTRATION.md"
    );

    /// The shared run (the whole plan once, at the unit-test scale).
    fn shared_report() -> &'static Report {
        static REPORT: std::sync::OnceLock<Report> = std::sync::OnceLock::new();
        REPORT.get_or_init(|| execute(1001, Scale::Custom).expect("a run"))
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

    fn is_literal(token: &str) -> bool {
        let digits = token.strip_prefix('-').unwrap_or(token);
        !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
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

    #[test]
    fn the_suite_is_the_pinned_one() {
        assert_eq!(selection_digest(), params::SELECTION_DIGEST);
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
        assert_eq!(table["experiment"].as_str(), Some(params::EXPERIMENT_ID));
        assert_eq!(
            table["seed_aggregation"].as_str(),
            Some(params::SEED_AGGREGATION)
        );
    }

    #[test]
    fn every_rule_is_three_tokens_and_names_an_integer_the_harness_prints() {
        let report = shared_report();
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

    /// Every rule can fail: set the counter it names to a violating value in the emitted JSON
    /// and the harness's own judge flags that rule.
    #[test]
    fn every_rule_fails_when_the_emitted_counter_is_set_to_a_violation() {
        let report = shared_report();
        for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
            let tokens: Vec<&str> = rule.split(' ').collect();
            let current = int(&report.json, tokens[0]);
            let violating = match tokens[1] {
                ">=" => {
                    let below = tokens[2].parse::<i64>().unwrap() - 1;
                    if below == current {
                        below - 1
                    } else {
                        below
                    }
                }
                "<=" => tokens[2].parse::<i64>().unwrap() + 1,
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
        assert_eq!(metrics::falsifiers().len(), 16);
    }

    #[test]
    fn the_entrypoint_and_the_preflight_are_the_subcommand_and_the_status_is_planned_or_later() {
        assert_eq!(params::BENCHMARK, "self-modification-authority");
        let manifest: toml::Table = params::EXPERIMENT_TOML
            .parse()
            .expect("experiment.toml parses");
        assert_eq!(
            manifest["preflight"].as_str().expect("preflight"),
            params::PREFLIGHT
        );
        assert!(PREREGISTRATION.contains(params::ENTRYPOINT));
        assert!(PREREGISTRATION.contains(params::PREFLIGHT));
        assert!(matches!(
            manifest["status"].as_str(),
            Some("planned" | "prepared" | "completed" | "failed")
        ));
        assert_eq!(manifest["id"].as_str(), Some(params::EXPERIMENT_ID));
        // Until the freeze the manifest names no entrypoint; from then on it is this one.
        let entrypoint = manifest["entrypoint"].as_str().expect("entrypoint");
        assert!(entrypoint.is_empty() || entrypoint == params::ENTRYPOINT);
    }

    #[test]
    fn the_mutation_plan_names_the_harnesss_hard_counters_and_each_is_expected_by_one() {
        let plan: toml::Table = MUTATIONS.parse().expect("mutations.toml is TOML");
        let listed: Vec<String> = plan["hard_counters"]
            .as_array()
            .expect("hard_counters")
            .iter()
            .map(|name| name.as_str().expect("a name").to_string())
            .collect();
        assert_eq!(listed, metrics::HARD_COUNTERS);
        assert_eq!(plan["subcommand"].as_str(), Some(params::BENCHMARK));
        let mutations = plan["mutation"].as_array().expect("mutations");
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
    }

    #[test]
    fn a_run_assembles_one_json_object_with_every_metric() {
        let report = shared_report();
        assert!(!report.line.contains('\n'));
        assert!(report
            .line
            .starts_with("{\"benchmark\":\"self-modification-authority\",\"experiment\":\"R013\""));
        assert!(report.line.contains("\"harness_gaps\":[]"), "no gap");
        assert_eq!(int(&report.json, "seed"), 1001);
        for name in metrics::METRIC_NAMES {
            if !metrics::GROUPED.contains(&name) {
                assert!(
                    matches!(emit::resolve(&report.json, name), Some(Json::Int(_))),
                    "{name}"
                );
            }
        }
        assert!(matches!(
            emit::resolve(&report.json, "scale"),
            Some(Json::Str(scale)) if scale == "custom"
        ));
    }

    /// The real product trips no falsifier on the run, and the run really exercises what it
    /// names.
    #[test]
    fn a_run_trips_no_falsifier_and_meets_every_coverage_rule() {
        let report = shared_report();
        let tripped: Vec<&str> = report
            .falsifiers
            .iter()
            .chain(&report.coverage)
            .filter(|check| check.status == Status::Fail)
            .map(|check| check.rule.as_str())
            .collect();
        assert!(tripped.is_empty(), "{tripped:?}");
        assert_eq!(int(&report.json, "hard_failures"), 0);
    }

    #[test]
    fn the_same_seed_gives_the_same_json_apart_from_the_timings_and_another_seed_does_not() {
        let strip = |json: &Json| {
            let Json::Obj(pairs) = json else {
                panic!("object")
            };
            Json::Obj(
                pairs
                    .iter()
                    .filter(|(k, _)| k != "timing_ns" && k != "seed")
                    .cloned()
                    .collect(),
            )
            .render()
        };
        let first = strip(&shared_report().json);
        let again = strip(&execute(1001, Scale::Custom).expect("a second run").json);
        assert_eq!(first, again);
        let other = strip(&execute(1002, Scale::Custom).expect("another seed").json);
        assert_ne!(
            first, other,
            "the seed orders the cells and draws the race points"
        );
    }

    #[test]
    fn timings_are_in_their_own_field_and_no_rule_reads_them() {
        let json = &shared_report().json;
        assert!(int(json, "timing_ns.elapsed_ns") > 0);
        for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
            assert!(!rule.contains("timing") && !rule.contains("_ns"), "{rule}");
        }
    }

    #[test]
    fn the_arguments_are_a_seed_or_one_and_a_seed_or_the_preflight_and_nothing_else() {
        let args = |items: &[&str]| items.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            parse_args(&args(&["17"])),
            Ok(Request::Run(17, Scale::Pinned))
        );
        assert_eq!(
            parse_args(&args(&["1", "17"])),
            Ok(Request::Run(17, Scale::Custom))
        );
        assert_eq!(parse_args(&args(&["--preflight"])), Ok(Request::Preflight));
        for bad in [
            args(&[]),
            args(&["x"]),
            args(&["-1"]),
            args(&["0", "17"]),
            args(&["2", "17"]),
            args(&["1", "x"]),
            args(&["1", "2", "3"]),
            args(&["--preflight", "17"]),
            args(&["--prefight"]),
        ] {
            assert!(parse_args(&bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_preflight_passes_on_a_machine_that_can_run_the_harness() {
        let line = preflight().expect("the preflight");
        assert!(line.contains("\"preflight\":\"ok\""), "{line}");
    }
}
