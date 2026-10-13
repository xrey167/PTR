//! Experiment M008, fast-weight memory, as a harness: `ptr-bench fast-weight-memory <seed>` runs
//! one seeded session of facts, updates and queries through the arms once and prints exactly one
//! JSON object on stdout.
//!
//! One run is one seed's evidence, judged mechanically by the experiment's `aggregate.py` under
//! `criteria.toml`; a run at another scale (the unit tests, or the mutation checker's
//! `fast-weight-memory <cases> <seed>`) is no evidence. The code under test is `ptr_fastmem`
//! (`fast.rs`: write, read, decode, the key-projection binding); the baselines are a recency buffer
//! of the last N writes (`arms.rs`) and the pinned `rag_reference` hybrid retrieval stack, live and
//! append-only (`hybrid.rs`, through `rag_bridge`). Every text and every vector comes from the
//! frozen pool (`pool.rs`); the truth is the harness's own table (`oracle.rs`).
//!
//! The object holds, in this order: the identity of the run, every metric the manifest names (a
//! JSON integer or an object of integers), `hard_failures`, the timings (never read by a rule), the
//! procedure, the negative controls, the notes, the counters that belong to no metric (`internal`),
//! and the two judgements (`coverage` and `falsifiers`). The exit status is 1 when a coverage
//! prerequisite is unmet, a falsifier is tripped or a metric has no value and no declared reason,
//! and 2 when the harness itself cannot run: the frozen pool or vectors are not the pinned ones,
//! the token budget and window do not follow from the pool, the Python driver cannot be spawned or
//! speaks another protocol, or a negative control is not separated from the arms. Everything the
//! code under test does wrong, a panic included, is a counter and a falsifier.
//!
//! The thresholds of the arms were set on the pilot seeds only (`calibrate.rs`,
//! `fast-weight-memory calibrate <pilot seeds>`), and are pinned in `params.rs` and in
//! `config.toml`.

pub mod arms;
pub mod calibrate;
pub mod descriptive;
pub mod fast;
pub mod hybrid;
pub mod metrics;
pub mod oracle;
pub mod params;
pub mod pool;
pub mod score;
pub mod session;
pub mod stats;

use super::layered_kv::emit::{count, judge, Check, Gaps, Json, Status, Tally};
use super::rag_bridge::{Bridge, Expect};
use arms::{Policy, Ranked};
use metrics::Scores;
use oracle::Truth;
use params::Scale;
use pool::Pool;
use session::Session;
use std::path::PathBuf;
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

pub(crate) fn driver_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../research/baselines/rag_reference/q003_driver.py")
}

pub(crate) fn spawn_driver() -> Result<Bridge, String> {
    let script = driver_script();
    Bridge::spawn(
        "python3",
        &[&script.to_string_lossy()],
        &Expect {
            driver: "rag_reference",
            embedding_model: Some(params::EMBEDDING_MODEL),
        },
    )
    .map_err(|error| error.to_string())
}

/// A negative control: an arm that must separate from the arms under test on this corpus, else the
/// scoring or the corpus cannot show what it is made to show.
fn control(name: &str, rule: String, observed: i64, tripped: bool) -> (String, Json) {
    (
        name.to_string(),
        Json::obj([
            ("required_rule", Json::str(rule)),
            ("required_rule_tripped", Json::Bool(tripped)),
            ("observed_permille", Json::Int(observed)),
        ]),
    )
}

fn per_query(total_ns: u128, queries: i64) -> i64 {
    if queries == 0 {
        0
    } else {
        (total_ns / queries as u128) as i64
    }
}

/// Run the session of `seed` at `scale` through every arm and assemble the one JSON object. A run
/// the harness cannot make is an `Err` with the reason.
pub fn execute(seed: u64, scale: Scale) -> Result<Report, String> {
    let started = Instant::now();
    let shape = scale.shape()?;
    let pool = Pool::load()?;
    if pool.max_fact_tokens != params::MAX_FACT_TOKENS {
        return Err(format!(
            "the longest fact of the pool is {} tokens, the pinned value is {}",
            pool.max_fact_tokens,
            params::MAX_FACT_TOKENS
        ));
    }
    if params::TOKEN_BUDGET / pool.max_fact_tokens != params::RECENCY_WINDOW {
        return Err(format!(
            "the recency window is floor({} / {}) = {}, the pinned value is {}",
            params::TOKEN_BUDGET,
            pool.max_fact_tokens,
            params::TOKEN_BUDGET / pool.max_fact_tokens,
            params::RECENCY_WINDOW
        ));
    }
    let session = Session::generate(&pool, seed, shape)?;
    let truth = Truth::fold(&session.events);
    let mut tally = Tally::new(metrics::LIST);
    tally.set("written_facts", session.events.len() as i64);
    tally.set(
        "oracle_label_mismatches",
        oracle::mismatches(&session, &truth),
    );

    // Fast memory (the code under test).
    let fast_run = fast::run(
        &pool,
        &session,
        &fast::FastConfig::pinned(),
        Policy::milli(params::FAST_MIN_SCORE_MILLI, params::FAST_MIN_MARGIN_MILLI),
        &mut tally,
    );
    let fast_answers: Vec<Vec<usize>> = fast_run
        .queries
        .iter()
        .map(|query| query.product.clone().unwrap_or_default())
        .collect();
    tally.set("fast_candidates", fast_run.candidates as i64);

    // The recency buffer.
    let recency_started = Instant::now();
    let (recency_ranked, window_len) = arms::recency(&pool, &session);
    let recency_ns = recency_started.elapsed().as_nanos();
    let recency_policy = Policy::milli(params::RECENCY_MIN_COSINE_MILLI, 0);
    let recency_answers: Vec<Vec<usize>> = recency_ranked
        .iter()
        .map(|ranked| recency_policy.apply(ranked))
        .collect();
    tally.set("recency_window_facts", window_len as i64);
    tally.set(
        "window_budget_violations",
        i64::from(window_len * params::MAX_FACT_TOKENS > params::TOKEN_BUDGET),
    );

    // Hybrid retrieval through the pinned driver.
    let mut bridge = spawn_driver()?;
    let live = hybrid::run(&mut bridge, &pool, &session, hybrid::Index::Live)
        .map_err(|error| error.to_string())?;
    let append = hybrid::run(&mut bridge, &pool, &session, hybrid::Index::Append)
        .map_err(|error| error.to_string())?;
    tally.set("hybrid_live_documents", live.documents as i64);
    tally.set("hybrid_append_documents", append.documents as i64);
    let stale_hits: usize = live
        .queries
        .iter()
        .flat_map(|ranked| &ranked.rows)
        .filter(|row| truth.current(pool.row(**row).cue) != Some(**row))
        .count();
    tally.set(
        "stale_in_live_index",
        (stale_hits + live.documents.saturating_sub(truth.len())) as i64,
    );
    let answers_of = |runs: &[Ranked], threshold: i64| -> Vec<Vec<usize>> {
        let policy = Policy::milli(threshold, 0);
        runs.iter().map(|ranked| policy.apply(ranked)).collect()
    };
    let live_answers = answers_of(&live.queries, params::HYBRID_LIVE_MIN_COSINE_MILLI);
    let append_answers = answers_of(&append.queries, params::HYBRID_APPEND_MIN_COSINE_MILLI);

    // Controls and scoring.
    let controls = arms::controls(&session, &truth, seed);
    let scores = Scores {
        fast: score::score(&pool, &session, &truth, &fast_answers),
        recency: score::score(&pool, &session, &truth, &recency_answers),
        hybrid_live: score::score(&pool, &session, &truth, &live_answers),
        hybrid_append: score::score(&pool, &session, &truth, &append_answers),
        oracle_cheat: score::score(&pool, &session, &truth, &controls.oracle_cheat),
        random: score::score(&pool, &session, &truth, &controls.random),
        stale_first: score::score(&pool, &session, &truth, &controls.stale_first),
    };
    let queries = scores.fast.queries;
    for (arm, nanos) in [
        ("fast", fast_run.query_ns),
        ("recency", recency_ns),
        ("hybrid_live", live.query_ns),
        ("hybrid_append", append.query_ns),
    ] {
        tally.set(
            &format!("latency_ns_per_query.{arm}"),
            per_query(nanos, queries),
        );
    }
    metrics::fill(&mut tally, &scores, seed);

    // Descriptive evidence (no rule reads it): no abstention, the cue text as the query, the
    // threshold sweep and the capacity table.
    let fast_ranked: Vec<Ranked> = fast_run.queries.iter().map(|q| q.ranked.clone()).collect();
    let exact_tops: Vec<Option<usize>> = fast_run.queries.iter().map(|q| q.exact_cue_top).collect();
    for (arm, ranked) in [
        ("fast", &fast_ranked),
        ("recency", &recency_ranked),
        ("hybrid_live", &live.queries),
        ("hybrid_append", &append.queries),
    ] {
        tally.set(
            &format!("recall_at_1_unthresholded_permille.{arm}"),
            descriptive::unthresholded_recall(&pool, &session, &truth, ranked),
        );
    }
    tally.set(
        "fast_exact_cue_recall_at_1_permille",
        descriptive::exact_cue_recall(&session, &truth, &exact_tops),
    );
    for (arm, ranked) in [
        ("fast", &fast_ranked),
        ("recency", &recency_ranked),
        ("hybrid_live", &live.queries),
    ] {
        descriptive::sweep(&pool, &session, &truth, arm, ranked, &mut tally);
    }
    descriptive::capacity(&pool, seed, &mut bridge, &mut tally)?;
    for (arm, key) in [
        ("fast", "fast"),
        ("recency", "recency"),
        ("hybrid_live", "hybrid_live"),
    ] {
        let value = tally.get(&format!("recall_at_1_unthresholded_permille.{arm}"));
        tally.set(
            &format!("capacity_recall_at_1_permille.{key}_c{:03}", shape.written),
            value,
        );
    }
    let exact = tally.get("fast_exact_cue_recall_at_1_permille");
    tally.set(
        &format!(
            "capacity_recall_at_1_permille.fast_exact_cue_c{:03}",
            shape.written
        ),
        exact,
    );
    bridge.quit().map_err(|error| error.to_string())?;

    let oracle_recall = tally.get("control_oracle_cheat_recall_permille");
    let random_recall = tally.get("control_random_recall_permille");
    let stale_false_recall = tally.get("control_stale_first_false_recall_permille");
    let controls_json = vec![
        control(
            "oracle_cheat",
            format!(
                "control_oracle_cheat_recall_permille >= {}",
                params::MIN_CONTROL_ORACLE_CHEAT_RECALL_PERMILLE
            ),
            oracle_recall,
            oracle_recall >= params::MIN_CONTROL_ORACLE_CHEAT_RECALL_PERMILLE,
        ),
        control(
            "random",
            format!(
                "control_random_recall_permille <= {}",
                params::MAX_CONTROL_RANDOM_RECALL_PERMILLE
            ),
            random_recall,
            random_recall <= params::MAX_CONTROL_RANDOM_RECALL_PERMILLE,
        ),
        control(
            "stale_first",
            format!(
                "control_stale_first_false_recall_permille >= {}",
                params::MIN_CONTROL_STALE_FIRST_FALSE_RECALL_PERMILLE
            ),
            stale_false_recall,
            stale_false_recall >= params::MIN_CONTROL_STALE_FIRST_FALSE_RECALL_PERMILLE,
        ),
        control(
            "positive",
            "control_positive_gain_vs_recency_lo_permille >= 1, control_positive_gain_vs_hybrid_live_lo_permille >= 1 and control_positive_false_recall_headroom_permille >= 0"
                .to_string(),
            tally.get("control_positive_gain_vs_recency_lo_permille"),
            tally.get("control_positive_gain_vs_recency_lo_permille")
                >= params::MIN_CONTROL_POSITIVE_GAIN_LO_PERMILLE
                && tally.get("control_positive_gain_vs_hybrid_live_lo_permille")
                    >= params::MIN_CONTROL_POSITIVE_GAIN_LO_PERMILLE
                && tally.get("control_positive_false_recall_headroom_permille")
                    >= params::MIN_CONTROL_POSITIVE_FALSE_RECALL_HEADROOM_PERMILLE,
        ),
    ];
    for (name, json) in &controls_json {
        if !matches!(
            json,
            Json::Obj(pairs) if pairs.iter().any(|(key, value)| key == "required_rule_tripped" && *value == Json::Bool(true))
        ) {
            return Err(format!(
                "the negative control {name} is not separated from the arms under test: the scoring or the corpus cannot show what the controls exist to show"
            ));
        }
    }
    let hard = metrics::hard_failures(&tally);
    let (printed, gaps) = tally.metrics();

    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str(params::BENCHMARK)),
        ("experiment".to_string(), Json::str(params::EXPERIMENT)),
        (
            "status_note".to_string(),
            note("one seed of the M008 harness: evidence only when run at the pinned scale from the frozen commit and judged by aggregate.py under criteria.toml; the exit status and the coverage and falsifiers lists below are the harness's own reading of the same rules"),
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
            ("seed_is_pilot", Json::Bool(params::PILOT_SEEDS.contains(&seed))),
            ("shape_is_pinned", Json::Bool(scale == Scale::Pinned)),
            ("pool_sha256", Json::str(pool.pool_sha256.clone())),
            ("vectors_sha256", Json::str(pool.vectors_sha256.clone())),
            ("embedding_model", Json::str(params::EMBEDDING_MODEL)),
            ("embedding_revision", Json::str(params::EMBEDDING_REVISION)),
            ("token_pattern", Json::str(params::TOKEN_PATTERN)),
            ("token_budget", Json::Int(params::TOKEN_BUDGET as i64)),
            ("max_fact_tokens", Json::Int(pool.max_fact_tokens as i64)),
            ("recency_window", Json::Int(params::RECENCY_WINDOW as i64)),
            ("ranked_limit", Json::Int(params::RANKED_LIMIT as i64)),
            (
                "memory",
                Json::obj([
                    ("heads", Json::Int(params::HEADS as i64)),
                    ("key_dim", Json::Int(params::KEY_DIM as i64)),
                    ("value_dim", Json::Int(params::VALUE_DIM as i64)),
                    ("beta_permille", Json::Int(params::BETA_PERMILLE)),
                    ("decay_permille", Json::Int(params::DECAY_PERMILLE)),
                ]),
            ),
            (
                "operating_points_milli",
                Json::obj([
                    ("fast_min_score", Json::Int(params::FAST_MIN_SCORE_MILLI)),
                    ("fast_min_margin", Json::Int(params::FAST_MIN_MARGIN_MILLI)),
                    ("recency_min_cosine", Json::Int(params::RECENCY_MIN_COSINE_MILLI)),
                    ("hybrid_live_min_cosine", Json::Int(params::HYBRID_LIVE_MIN_COSINE_MILLI)),
                    (
                        "hybrid_append_min_cosine",
                        Json::Int(params::HYBRID_APPEND_MIN_COSINE_MILLI),
                    ),
                    (
                        "matched_false_recall_permille",
                        Json::Int(params::MATCHED_FALSE_RECALL_PERMILLE),
                    ),
                ]),
            ),
            (
                "bootstrap",
                Json::obj([
                    ("seed", Json::Int(params::BOOTSTRAP_SEED as i64)),
                    ("resamples", Json::Int(params::BOOTSTRAP_RESAMPLES as i64)),
                    (
                        "lower_quantile_permille",
                        Json::Int(params::BOOTSTRAP_LOWER_QUANTILE_PERMILLE as i64),
                    ),
                ]),
            ),
            (
                "false_recall",
                note("a query is a false recall when the arm's top fact is not the current fact of the queried cue (a stale version, or another cue) or when the query is unanswerable and the arm answers at all; the rate is over all queries; unanswerable_answered_permille is the same on the unanswerable queries alone, the rate the thresholds were matched to on the pilot seeds"),
            ),
        ]),
    ));
    pairs.push(("controls".to_string(), Json::Obj(controls_json)));
    pairs.push((
        "measurement_notes".to_string(),
        Json::obj([
            (
                "keys_and_values",
                note("fast memory is keyed by the frozen embedding of the cue text (paraphrase 0 of the cue) projected by a seeded orthogonal projection, and stores the identifier code of the fact (capsule m008:c:<cue> at generation <version>); queries are paraphrases 1..3, so no query repeats the cue text; the recency and hybrid arms see only the fact texts and the query texts; the memory is therefore given a cue extractor that the baselines do not have, which favours the memory"),
            ),
            (
                "recency_buffer",
                note("the last RECENCY_WINDOW writes (old versions included, the buffer knows nothing of cues), answered by the cosine between the query and each buffered fact on the frozen vectors, the newer write first on a tie; it can answer only for facts written inside the window"),
            ),
            (
                "hybrid",
                note("rag_reference BM25(k1=1.5,b=0.75) + cosine, RRF k=60, run by research/baselines/rag_reference/q003_driver.py through rag_bridge; the driver's rank score is ordinal, so the abstention rule is the cosine between the query and the top hit; hybrid_live keeps the current version of each cue only, hybrid_append keeps every version (reported, not gated)"),
            ),
            (
                "decode_oracle",
                note("decode_contract_violations counts queries on which the product's decode_readout differs from the harness's own scoring of the readout against the candidates' codes under the same rule; readback_failures counts writes that do not read back through their own key; stale_residual_violations counts exact-key overwrites that leave the stale version above the residual bound; all three are read from the readout, not from the product's verdict"),
            ),
            (
                "matched_operating_point",
                note("the abstention thresholds of every arm were set on the pilot seeds (1001..1005, disjoint from the confirmatory seeds) so that each arm answers at most 5 percent of the unanswerable pilot queries; the confirmatory seeds apply them unchanged; unanswerable_answered_permille reports the rate each arm realised"),
            ),
            (
                "wall_time",
                note("timing_ns and latency_ns_per_query are recorded and never judged"),
            ),
        ]),
    ));
    pairs.push(("internal".to_string(), tally.internal()));
    let root = Json::Obj(pairs.clone());
    let unsupported_names = tally.unsupported_names();
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
#[derive(Debug, PartialEq)]
pub enum Request {
    Run(u64, Scale),
    /// `calibrate <pilot seed>...`: set the operating points on the pilot seeds.
    Calibrate(Vec<u64>),
}

/// The request from the arguments after the subcommand; strict, so a typo stops the run.
pub fn parse_args(args: &[String]) -> Result<Request, String> {
    let parse = |text: &String, what: &str| {
        text.parse::<u64>()
            .map_err(|_| format!("{what} must be a non-negative integer, got {text:?}"))
    };
    match args {
        [first, seeds @ ..] if first == "calibrate" => {
            if seeds.is_empty() {
                return Err("usage: fast-weight-memory calibrate <pilot seed>...".to_string());
            }
            seeds
                .iter()
                .map(|seed| parse(seed, "pilot seed"))
                .collect::<Result<Vec<_>, _>>()
                .map(Request::Calibrate)
        }
        [seed] => parse(seed, "seed").map(|seed| Request::Run(seed, Scale::Pinned)),
        [cases, seed] => parse(cases, "cases")
            .and_then(|cases| parse(seed, "seed").map(|seed| Request::Run(seed, Scale::Custom(cases as usize)))),
        _ => Err(
            "usage: fast-weight-memory <seed> | fast-weight-memory <cases> <seed> | fast-weight-memory calibrate <pilot seed>..."
                .to_string(),
        ),
    }
}

/// Run the experiment's harness and print its one JSON object. `args` are the arguments after the
/// subcommand: `<seed>` (the pinned scale), `<cases> <seed>` (the mutation checker's form) or
/// `calibrate <pilot seed>...`. Exit 0 on a clean pass, 1 on an unmet coverage rule or a tripped
/// falsifier, 2 when the harness cannot run.
pub fn run(args: &[String]) {
    match parse_args(args) {
        Ok(Request::Run(seed, scale)) => print_and_exit(execute(seed, scale)),
        Ok(Request::Calibrate(seeds)) => match calibrate::run(&seeds) {
            Ok(line) => println!("{line}"),
            Err(reason) => {
                eprintln!("{}: {reason}", params::BENCHMARK);
                std::process::exit(2);
            }
        },
        Err(reason) => {
            eprintln!("{}: {reason}", params::BENCHMARK);
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::emit::{evaluate, resolve};
    use std::sync::OnceLock;

    fn experiment_file(name: &str) -> toml::Table {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../experiments/model/M008-fast-weight-memory")
            .join(name);
        std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
            .parse()
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
    }

    fn strings(table: &toml::Table, key: &str) -> Vec<String> {
        table[key]
            .as_array()
            .unwrap_or_else(|| panic!("{key} is an array"))
            .iter()
            .map(|item| item.as_str().expect("string").to_string())
            .collect()
    }

    /// One pilot seed at the pinned scale, run once for the whole test binary: it is the
    /// ordinary run (pool, memory, recency, the real Python driver) with the confirmatory seeds
    /// left alone.
    fn pilot_run() -> &'static Report {
        static REPORT: OnceLock<Report> = OnceLock::new();
        REPORT
            .get_or_init(|| execute(1001, Scale::Pinned).expect("the harness runs on a pilot seed"))
    }

    fn with(root: &Json, path: &str, value: Json) -> Json {
        fn set(at: &mut Json, keys: &[&str], value: Json) {
            let Json::Obj(pairs) = at else {
                panic!("object")
            };
            let slot = pairs
                .iter_mut()
                .find(|(k, _)| k == keys[0])
                .expect("member");
            if keys.len() == 1 {
                slot.1 = value;
            } else {
                set(&mut slot.1, &keys[1..], value);
            }
        }
        let mut copy = root.clone();
        set(&mut copy, &path.split('.').collect::<Vec<_>>(), value);
        copy
    }

    #[test]
    fn criteria_toml_is_the_harnesss_rule_lists_in_order() {
        let criteria = experiment_file("criteria.toml");
        assert_eq!(strings(&criteria, "coverage"), metrics::coverage());
        assert_eq!(strings(&criteria, "falsifiers"), metrics::falsifiers());
        assert_eq!(criteria["experiment"].as_str(), Some("M008"));
    }

    #[test]
    fn the_manifest_names_the_harness_metrics_in_order_and_its_entrypoint_is_planned() {
        let manifest = experiment_file("experiment.toml");
        let names: Vec<&str> = manifest["metrics"].as_str().unwrap().split(", ").collect();
        assert_eq!(names, metrics::METRIC_NAMES);
        // Planned before the freeze, prepared after it, then completed or falsified.
        assert!(matches!(
            manifest["status"].as_str(),
            Some("planned" | "prepared" | "completed" | "falsified")
        ));
        let seeds: Vec<u64> = manifest["seeds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|seed| seed.as_integer().unwrap() as u64)
            .collect();
        assert_eq!(seeds, params::SEEDS);
    }

    #[test]
    fn the_mutation_plan_lists_the_hard_counters_and_every_one_is_expected() {
        let plan = experiment_file("tests/mutations.toml");
        assert_eq!(strings(&plan, "hard_counters"), metrics::HARD_COUNTERS);
        assert_eq!(plan["subcommand"].as_str(), Some(params::BENCHMARK));
        let mutations = plan["mutation"].as_array().unwrap();
        for counter in metrics::HARD_COUNTERS {
            assert!(
                mutations.iter().any(|m| m["expect"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|e| e.as_str() == Some(counter))),
                "{counter} is expected by no mutation"
            );
        }
        let seed = plan["seed"].as_integer().unwrap() as u64;
        assert!(!params::SEEDS.contains(&seed) && !params::PILOT_SEEDS.contains(&seed));
    }

    #[test]
    fn a_pilot_seed_at_the_pinned_scale_reports_the_procedure_and_separates_the_controls() {
        let json = &pilot_run().json;
        let path = |p: &str| resolve(json, p).cloned();
        assert_eq!(path("benchmark"), Some(Json::str("fast-weight-memory")));
        assert_eq!(path("scale"), Some(Json::str("pinned")));
        assert_eq!(path("procedure.seed_is_declared"), Some(Json::Bool(false)));
        assert_eq!(path("procedure.seed_is_pilot"), Some(Json::Bool(true)));
        assert_eq!(path("procedure.shape_is_pinned"), Some(Json::Bool(true)));
        assert_eq!(
            path("procedure.pool_sha256"),
            Some(Json::str(params::POOL_SHA256))
        );
        assert_eq!(
            path("procedure.vectors_sha256"),
            Some(Json::str(params::VECTORS_SHA256))
        );
        for control in ["oracle_cheat", "random", "stale_first"] {
            assert_eq!(
                path(&format!("controls.{control}.required_rule_tripped")),
                Some(Json::Bool(true)),
                "{control}"
            );
        }
        assert_eq!(path("queries"), Some(Json::Int(400)));
        assert_eq!(path("written_facts"), Some(Json::Int(600)));
        assert_eq!(path("harness_gaps"), Some(Json::Arr(vec![])));
        // The abstention contract: a threshold the pilot set is applied unchanged.
        assert_eq!(path("oracle_label_mismatches"), Some(Json::Int(0)));
        // The harness's own contract checks hold on an unmutated product.
        for counter in [
            "readback_failures",
            "decode_contract_violations",
            "stale_residual_violations",
            "projection_mismatch_accepted",
            "product_panics",
            "product_errors",
            "stale_in_live_index",
            "token_budget_violations",
            "window_budget_violations",
        ] {
            assert_eq!(path(counter), Some(Json::Int(0)), "{counter}");
        }
    }

    #[test]
    fn every_criteria_path_resolves_to_an_integer_and_every_rule_can_fail() {
        let json = &pilot_run().json;
        for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
            let tokens: Vec<&str> = rule.split(' ').collect();
            for token in [tokens[0], tokens[2]] {
                if token.parse::<i64>().is_ok() {
                    continue;
                }
                assert!(
                    matches!(resolve(json, token), Some(Json::Int(_))),
                    "{rule}: {token} is {:?}",
                    resolve(json, token)
                );
            }
            // Break the rule through its left counter and see it fail.
            let (op, right) = (tokens[1], tokens[2]);
            let bound = right
                .parse::<i64>()
                .ok()
                .or_else(|| match resolve(json, right) {
                    Some(Json::Int(value)) => Some(*value),
                    _ => None,
                })
                .expect("a bound");
            let violating = if op == ">=" { bound - 1 } else { bound + 1 };
            let broken = with(json, tokens[0], Json::Int(violating));
            assert_eq!(evaluate(&broken, rule), Ok(false), "{rule}");
        }
    }

    #[test]
    fn every_manifest_metric_is_printed_and_the_judgements_are_the_harnesss_own() {
        let report = pilot_run();
        for name in metrics::METRIC_NAMES {
            assert!(resolve(&report.json, name).is_some(), "{name}");
        }
        assert_eq!(report.coverage.len(), metrics::coverage().len());
        assert_eq!(report.falsifiers.len(), metrics::falsifiers().len());
        assert!(report.gaps.is_empty());
        let hard: i64 = metrics::HARD_COUNTERS
            .iter()
            .map(|name| match resolve(&report.json, name) {
                Some(Json::Int(value)) => *value,
                other => panic!("{name}: {other:?}"),
            })
            .sum();
        assert_eq!(
            resolve(&report.json, "hard_failures"),
            Some(&Json::Int(hard))
        );
        assert!(report.line.starts_with('{') && !report.line.contains('\n'));
    }

    fn without_timing(json: &Json) -> String {
        let Json::Obj(pairs) = json else {
            panic!("object")
        };
        let kept: Vec<(String, Json)> = pairs
            .iter()
            .filter(|(key, _)| !matches!(key.as_str(), "timing_ns" | "latency_ns_per_query"))
            .cloned()
            .collect();
        Json::Obj(kept).render()
    }

    #[test]
    fn a_reduced_run_is_deterministic_and_marked_custom() {
        let first = execute(2001, Scale::Custom(60)).unwrap();
        let second = execute(2001, Scale::Custom(60)).unwrap();
        assert_eq!(without_timing(&first.json), without_timing(&second.json));
        assert_eq!(resolve(&first.json, "scale"), Some(&Json::str("custom")));
        assert_eq!(resolve(&first.json, "queries"), Some(&Json::Int(60)));
        // A reduced session misses the coverage floors: it is no evidence.
        assert!(first.coverage_unmet() > 0);
    }

    #[test]
    fn arguments_are_strict() {
        let args = |items: &[&str]| items.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            parse_args(&args(&["17"])),
            Ok(Request::Run(17, Scale::Pinned))
        );
        assert_eq!(
            parse_args(&args(&["400", "2001"])),
            Ok(Request::Run(2001, Scale::Custom(400)))
        );
        assert_eq!(
            parse_args(&args(&["calibrate", "1001", "1002"])),
            Ok(Request::Calibrate(vec![1001, 1002]))
        );
        for bad in [
            &[][..],
            &["x"],
            &["1", "y"],
            &["calibrate"],
            &["calibrate", "z"],
            &["1", "2", "3"],
        ] {
            assert!(parse_args(&args(bad)).is_err(), "{bad:?}");
        }
        assert!(execute(1, Scale::Custom(5)).is_err());
    }
}
