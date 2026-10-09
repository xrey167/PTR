//! Experiment Q002, evidence promotion, as a harness: `ptr-bench evidence-promotion <seed>`
//! runs the four arms over a seeded world once for one seed and prints exactly one JSON object
//! on stdout.
//!
//! One run is one seed's evidence, judged mechanically by the experiment's `aggregate.py` under
//! `criteria.toml`; a run at another scale (the unit tests, or the mutation checker's
//! `evidence-promotion <capsules> <seed>`) is no evidence. The code under test is
//! `ptr_search`: `ExactSourceResolver`, `verify_claim` and the typed promotion path of
//! `SearchHit` (arm M), against three baselines (B0 the top hit is the answer, B1 the lifecycle
//! filter `retain_live` alone, B2 today's boolean API fed `true`); the lifecycle authority is a
//! real `PtrRuntime` (`authority.rs`), the truth is the generator's table (`world.rs`) checked
//! by a naive independent reading (`oracle.rs`).
//!
//! The object holds, in this order: the identity of the run, every metric the manifest names (a
//! JSON integer or an object of integers), `hard_failures`, the timings (never read by a rule),
//! the procedure, the negative controls, the per-class breakdown, the notes, the counters that
//! belong to no metric (`internal`), and the two judgements (`coverage` and `falsifiers`). The
//! exit status is 1 when a coverage prerequisite is unmet, a falsifier is tripped or a metric
//! has no value and no declared reason, and 2 when the harness itself cannot run: the world
//! does not generate, the runtime or the store (neither is under test) refuses the world's own
//! setup, or the negative control B0 (direct retrieval, which runs no code under test) produces
//! no false Known; B2, the boolean API, runs code under test, so its failing to hallucinate is
//! the coverage rule `false_known_bool_trust >= 1`. Everything the code under test does wrong,
//! a panic included, is a counter and a falsifier; a disagreement between the generator's labels
//! and the naive reading is the counter `oracle_label_mismatches`, which the grammar can only
//! express as a failed coverage rule (the seed is then vacuous).

pub mod arms;
pub mod authority;
pub mod metrics;
pub mod oracle;
pub mod params;
pub mod world;

use super::layered_kv::emit::{count, judge, Check, Gaps, Json, Status, Tally};
use arms::Arms;
use authority::{Authority, Source};
use params::Scale;
use std::time::Instant;
use world::{Class, World};

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

/// Compare every label of the generator with the naive reading of the corpus; each hit or
/// question on which they disagree is one `oracle_label_mismatches`.
fn cross_check(world: &World, tally: &mut Tally) {
    for hit in &world.hits {
        if oracle::naive_eligible(world, hit) != hit.class.eligible() {
            tally.add("oracle_label_mismatches", 1);
            tally.add(&format!("label_mismatch_hits.{}", hit.class.name()), 1);
        }
    }
    for question in &world.questions {
        if oracle::naive_answer(world, question) != question.expected {
            tally.add("oracle_label_mismatches", 1);
            tally.add("label_mismatch_questions", 1);
        }
    }
}

/// A negative control: a baseline that must produce a false Known on this corpus, else the
/// corpus cannot show the hallucination it is made to show. B0 uses no code under test, so a
/// control that fails is the harness's fault and `exit 2`; B2 runs the boolean API of the
/// code under test, so the same failure may be a defect of the product (an over-refusing
/// `observe`), and is left to the coverage rule `false_known_bool_trust >= 1` (a vacuous seed,
/// counted as falsified).
fn control(tally: &Tally, counter: &str) -> Json {
    let false_known = tally.get(counter);
    Json::obj([
        ("required_rule", Json::str(format!("{counter} >= 1"))),
        ("required_rule_tripped", Json::Bool(false_known >= 1)),
        ("false_known", Json::Int(false_known)),
        (
            "false_known_untrue_claim",
            Json::Int(tally.get(&counter.replace("false_known_", "false_known_untrue_"))),
        ),
    ])
}

fn class_breakdown(tally: &Tally) -> Json {
    let rows: Vec<(String, Json)> = [Class::Honest, Class::HardHonest]
        .into_iter()
        .chain(Class::POISON)
        .map(|class| {
            let name = class.name();
            let get = |prefix: &str| tally.get(&format!("{prefix}.{name}"));
            (
                name.to_string(),
                Json::obj([
                    ("presented", Json::Int(get("hits_by_class"))),
                    ("refused_as_expected", Json::Int(get("poison_attempts"))),
                    ("promoted", Json::Int(get("poison_promoted"))),
                ]),
            )
        })
        .collect();
    Json::Obj(rows)
}

fn world_json(world: &World) -> Json {
    let count_status = |status| {
        world
            .capsules
            .iter()
            .filter(|capsule| capsule.status == status)
            .count() as i64
    };
    let kind = |kind| {
        world
            .questions
            .iter()
            .filter(|question| question.kind == kind)
            .count() as i64
    };
    Json::obj([
        ("capsules", Json::Int(world.capsules.len() as i64)),
        (
            "healthy_capsules",
            Json::Int(count_status(world::Status::Healthy)),
        ),
        (
            "revoked_capsules",
            Json::Int(count_status(world::Status::Revoked)),
        ),
        (
            "disputed_capsules",
            Json::Int(count_status(world::Status::Disputed)),
        ),
        ("raw_events", Json::Int(world.events().count() as i64)),
        ("corrupt_replicas", Json::Int(world.replicas.len() as i64)),
        ("hits", Json::Int(world.hits.len() as i64)),
        (
            "answerable_questions",
            Json::Int(kind(world::QuestionKind::Answerable)),
        ),
        (
            "quarantined_questions",
            Json::Int(kind(world::QuestionKind::Quarantined)),
        ),
        (
            "absent_questions",
            Json::Int(kind(world::QuestionKind::Absent)),
        ),
    ])
}

/// Run the world of `seed` at `scale` through every arm and assemble the one JSON object. A run
/// the harness cannot make is an `Err` with the reason.
pub fn execute(seed: u64, scale: Scale) -> Result<Report, String> {
    let started = Instant::now();
    let shape = scale.shape()?;
    let world = World::generate(seed, shape)?;
    let store = Source::store_of(&world)?;
    let source = Source::new(&store, &world);
    let authority = Authority::build(&world)?;

    let mut tally = Tally::new(metrics::LIST);
    cross_check(&world, &mut tally);
    let digest = world.corpus_digest();
    let pinned = if shape == params::Shape::PINNED {
        params::pinned_digest(seed)
    } else {
        None
    };
    if let Some(pinned) = pinned {
        if pinned != digest {
            tally.add("corpus_selection_digest_mismatches", 1);
        }
    }

    let mut arms = Arms::new(&world, &authority, &source, tally);
    arms.run();
    let mut tally = arms.tally;
    tally.set("lifecycle_queries", authority.queries());
    metrics::fill(
        &mut tally,
        world.questions.len() as i64,
        world.answerable() as i64,
    );
    if tally.get("false_known_baseline_direct") < 1 {
        return Err(
            "the negative control B0 (direct retrieval) produced no false Known: the corpus does not show the hallucination it exists to show"
                .to_string(),
        );
    }
    let baseline_control = control(&tally, "false_known_baseline_direct");
    let bool_control = control(&tally, "false_known_bool_trust");
    let hard = metrics::hard_failures(&tally);

    let (printed, gaps) = tally.metrics();
    let unsupported_names = tally.unsupported_names();
    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str(params::BENCHMARK)),
        ("experiment".to_string(), Json::str("Q002")),
        (
            "status_note".to_string(),
            note("one seed of the Q002 harness: evidence only when run at the pinned scale from the frozen commit and judged by aggregate.py under criteria.toml; the exit status and the coverage and falsifiers lists below are the harness's own reading of the same rules"),
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
            (
                "seed_is_declared",
                Json::Bool(params::SEEDS.contains(&seed)),
            ),
            ("shape_is_pinned", Json::Bool(scale == Scale::Pinned)),
            ("corpus_digest", Json::str(digest.clone())),
            (
                "pinned_corpus_digest",
                pinned.map_or(Json::Null, Json::str),
            ),
            ("world", world_json(&world)),
            (
                "false_known",
                note("a Known is false when the hit's generator label is not eligible: the baselines make one Known assertion per question (their top hit, B1 and B2 the top hit the lifecycle lets through), the mechanism is judged on every hit it promotes; false_known_untrue_* is the subset whose claim is also false in the world"),
            ),
            (
                "poison_attempts",
                note("poison_attempts.Pk counts the class-k hits that the mechanism refused with the refusal the generator expects for that class, at the API boundary; a hit refused for another reason is wrong_refusal_reason, a hit promoted is false_known_mechanism"),
            ),
        ]),
    ));
    pairs.push((
        "controls".to_string(),
        Json::obj([
            ("baseline_direct", baseline_control),
            ("bool_trust", bool_control),
        ]),
    ));
    pairs.push(("classes".to_string(), class_breakdown(&tally)));
    pairs.push(("unsupported".to_string(), tally.unsupported_json()));
    pairs.push((
        "measurement_notes".to_string(),
        Json::obj([
            (
                "lifecycle",
                note("generation validity is PtrRuntime::generation_validity over a ledger the harness committed (CapsuleCommitted, CapsuleSuperseded, Revoked), except P8: the runtime never answers Disputed, so the generations the generator marked disputed are answered Disputed by the harness's overlay and every other by the runtime"),
            ),
            (
                "ownership",
                note("the source double owns the ownership record: every raw event belongs to the capsule generation the generator wrote it for, a corrupt replica to the one of the event it damages (so the digest stays the only barrier of P3); the ownership is the generator's table, a trust root of the resolver like the source itself, and P9 and K11 show that a hit is observed only with an event of its own capsule generation"),
            ),
            (
                "hard_honest",
                note("hard honest hits are exact and must be promoted: a multibyte value, a value containing `not`, a statement an event states twice (the second occurrence is cited), a one-line event, a span at the start or at the end of the event; hard_honest_refused is a zero-tolerance falsifier beside the 950 per mille recall floor"),
            ),
            (
                "tampered_digest",
                note("a KnowledgeStore cannot hold an event whose bytes disagree with its digest, so P3 hits cite corrupt replicas served beside the store by a RawEventSource double; the resolver's digest check is defence in depth against such a source"),
            ),
            (
                "labels",
                note("the generator's class labels and answers are cross-checked by a naive reading of the corpus (substring search, string equality with the canonical rendering, the generator's lifecycle table, a scan of live healthy events); a disagreement is oracle_label_mismatches"),
            ),
            (
                "stage_skip",
                note("an attempt counts only when the API returned the refusal the stage rules prescribe and the hit's stage, source and claim are unchanged; an accepted call is stage_skip_accepted, any other outcome stage_skip_unexpected; a setup that the harness could not build from honest hits counts into stage_skip_unexpected"),
            ),
            (
                "wall_time",
                note("timing_ns is recorded and never judged"),
            ),
        ]),
    ));
    let internal = tally.internal();
    pairs.push(("internal".to_string(), internal));
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

/// The seed and scale from the arguments after the subcommand; strict, so a typo stops the run.
fn parse_args(args: &[String]) -> Result<(u64, Scale), String> {
    let parse = |text: &String, what: &str| {
        text.parse::<u64>()
            .map_err(|_| format!("{what} must be a non-negative integer, got {text:?}"))
    };
    match args {
        [seed] => parse(seed, "seed").map(|seed| (seed, Scale::Pinned)),
        [capsules, seed] => parse(capsules, "capsules").and_then(|capsules| {
            parse(seed, "seed").map(|seed| (seed, Scale::Custom(capsules as usize)))
        }),
        _ => Err(
            "usage: evidence-promotion <seed> | evidence-promotion <capsules> <seed>".to_string(),
        ),
    }
}

/// Run the experiment's harness and print its one JSON object. `args` are the arguments after
/// the subcommand: `<seed>` (the pinned scale) or `<capsules> <seed>` (the mutation checker's
/// form: that many capsules in the pinned proportions, no evidence). Exits 1 when the run is
/// not clean and 2 when the harness cannot run.
pub fn run(args: &[String]) {
    let (seed, scale) = match parse_args(args) {
        Ok(parsed) => parsed,
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

    const CRITERIA: &str =
        include_str!("../../../../../experiments/retrieval/Q002-evidence-promotion/criteria.toml");
    const MUTATIONS: &str = include_str!(
        "../../../../../experiments/retrieval/Q002-evidence-promotion/tests/mutations.toml"
    );
    const PREREGISTRATION: &str = include_str!(
        "../../../../../experiments/retrieval/Q002-evidence-promotion/PREREGISTRATION.md"
    );

    /// The shared reduced run (a small world of every capsule state).
    fn reduced_report() -> &'static Report {
        static REPORT: std::sync::OnceLock<Report> = std::sync::OnceLock::new();
        REPORT.get_or_init(|| execute(17, Scale::Reduced).expect("a reduced run"))
    }

    /// The pinned run of one seed (a second or two in a debug build).
    fn pinned_report(seed: u64) -> Report {
        execute(seed, Scale::Pinned).expect("a pinned run")
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
        assert_eq!(table["experiment"].as_str(), Some("Q002"));
        assert_eq!(
            table["seed_aggregation"].as_str(),
            Some(params::SEED_AGGREGATION)
        );
    }

    #[test]
    fn every_rule_is_three_tokens_and_names_an_integer_the_harness_prints() {
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

    /// Every rule can fail: set the counter it names to a violating value in the emitted JSON
    /// and the harness's own judge flags that rule.
    #[test]
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
        assert_eq!(metrics::coverage().len(), 29);
    }

    #[test]
    fn the_entrypoint_is_empty_until_the_freeze_and_then_the_subcommand() {
        assert_eq!(params::BENCHMARK, "evidence-promotion");
        let manifest: toml::Table = params::EXPERIMENT_TOML
            .parse()
            .expect("experiment.toml parses");
        let entrypoint = manifest["entrypoint"].as_str().expect("entrypoint");
        assert!(entrypoint.is_empty() || entrypoint == params::ENTRYPOINT);
        assert!(PREREGISTRATION.contains(params::ENTRYPOINT));
    }

    #[test]
    fn the_hard_counters_of_the_mutation_plan_are_the_harnesss_and_each_is_expected_by_a_mutation()
    {
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
    fn a_reduced_run_assembles_one_json_object_with_every_metric() {
        let report = reduced_report();
        assert!(!report.line.contains('\n'));
        assert!(report
            .line
            .starts_with("{\"benchmark\":\"evidence-promotion\",\"experiment\":\"Q002\""));
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
    }

    /// The real mechanism trips no falsifier on any pinned seed, and each pinned seed meets
    /// every coverage rule except the digest pin that is filled in at the freeze.
    #[test]
    fn the_mechanism_holds_every_rule_on_every_pinned_seed() {
        for seed in params::SEEDS {
            let report = pinned_report(seed);
            let tripped: Vec<&str> = report
                .falsifiers
                .iter()
                .filter(|check| check.status == Status::Fail)
                .map(|check| check.rule.as_str())
                .collect();
            assert!(tripped.is_empty(), "seed {seed}: {tripped:?}");
            let unmet: Vec<&str> = report
                .coverage
                .iter()
                .filter(|check| check.status == Status::Fail)
                .map(|check| check.rule.as_str())
                .collect();
            assert!(unmet.is_empty(), "seed {seed}: {unmet:?}");
            let json = &report.json;
            assert_eq!(int(json, "hard_failures"), 0, "seed {seed}");
            assert_eq!(int(json, "false_known_mechanism"), 0);
            assert_eq!(int(json, "stage_skip_accepted"), 0);
            assert_eq!(int(json, "stale_or_revoked_promoted"), 0);
            assert_eq!(int(json, "wrong_refusal_reason"), 0);
            assert_eq!(int(json, "oracle_label_mismatches"), 0);
            assert_eq!(int(json, "hard_honest_refused"), 0);
            assert_eq!(int(json, "task_success_violation"), 0);
            assert!(int(json, "recall_permille") >= 950);
            assert!(int(json, "task_success_mechanism") >= int(json, "task_success_baseline"));
            assert_eq!(int(json, "product_panics"), 0);
            assert_eq!(int(json, "corpus_selection_digest_mismatches"), 0);
        }
    }

    /// The baselines hallucinate: each produces false Known on the poison, and the mechanism
    /// answers at least as many questions correctly as the direct baseline.
    #[test]
    fn the_baselines_produce_false_known_and_the_mechanism_does_not() {
        let report = pinned_report(17);
        let json = &report.json;
        assert!(int(json, "false_known_baseline_direct") >= 1);
        assert!(int(json, "false_known_gen_filter_only") >= 1);
        assert!(int(json, "false_known_bool_trust") >= 1);
        assert!(
            int(json, "false_known_baseline_direct") > int(json, "false_known_gen_filter_only")
        );
        assert!(
            int(json, "task_success_mechanism") > int(json, "task_success_baseline"),
            "{} vs {}",
            int(json, "task_success_mechanism"),
            int(json, "task_success_baseline")
        );
        for control in ["baseline_direct", "bool_trust"] {
            assert!(matches!(
                emit::resolve(json, &format!("controls.{control}.required_rule_tripped")),
                Some(Json::Bool(true))
            ));
        }
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
                    .filter(|(k, _)| k != "timing_ns")
                    .cloned()
                    .collect(),
            )
            .render()
        };
        let first = strip(&reduced_report().json);
        let again = strip(&execute(17, Scale::Reduced).expect("a second run").json);
        assert_eq!(first, again);
        let other = strip(&execute(29, Scale::Reduced).expect("another seed").json);
        assert_ne!(first, other);
    }

    #[test]
    fn timings_are_in_their_own_field_and_no_rule_reads_them() {
        let json = &reduced_report().json;
        assert!(int(json, "timing_ns.elapsed_ns") > 0);
        for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
            assert!(!rule.contains("timing") && !rule.contains("_ns"), "{rule}");
        }
    }

    #[test]
    fn the_arguments_are_a_seed_or_capsules_and_a_seed_and_nothing_else() {
        let args = |items: &[&str]| items.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(parse_args(&args(&["17"])), Ok((17, Scale::Pinned)));
        assert_eq!(
            parse_args(&args(&["36", "17"])),
            Ok((17, Scale::Custom(36)))
        );
        for bad in [
            args(&[]),
            args(&["x"]),
            args(&["-1"]),
            args(&["36", "x"]),
            args(&["1", "2", "3"]),
        ] {
            assert!(parse_args(&bad).is_err(), "{bad:?}");
        }
        assert!(execute(17, Scale::Custom(2)).is_err());
    }

    #[test]
    fn a_custom_scale_run_is_marked_and_still_judges_with_the_same_rules() {
        let report = execute(17, Scale::Custom(36)).expect("a custom run");
        assert!(matches!(
            emit::resolve(&report.json, "scale"),
            Some(Json::Str(scale)) if scale == "custom"
        ));
        assert!(matches!(
            emit::resolve(&report.json, "procedure.shape_is_pinned"),
            Some(Json::Bool(false))
        ));
        assert_eq!(int(&report.json, "hard_failures"), 0);
    }

    #[test]
    fn a_corpus_digest_that_differs_from_the_pin_is_counted_at_the_pinned_shape_only() {
        let pinned = pinned_report(17);
        let digest = match emit::resolve(&pinned.json, "procedure.corpus_digest") {
            Some(Json::Str(digest)) => digest.clone(),
            other => panic!("{other:?}"),
        };
        assert_eq!(digest.len(), 64);
        // The reduced shape has no pin: nothing is compared.
        assert_eq!(
            int(&reduced_report().json, "corpus_selection_digest_mismatches"),
            0
        );
        assert!(matches!(
            emit::resolve(&reduced_report().json, "procedure.pinned_corpus_digest"),
            Some(Json::Null)
        ));
    }
}
