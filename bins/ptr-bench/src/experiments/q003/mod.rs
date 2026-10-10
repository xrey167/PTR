//! Experiment Q003, PostgreSQL generation-joined hybrid retrieval, as a harness:
//! `ptr-bench postgres-hybrid <seed>` drives a seeded live-lifecycle corpus through `ptr-pg`
//! (its projector, `upsert_document` and `search`) and prints exactly one JSON object on stdout.
//!
//! One run is one seed's evidence, judged mechanically by the experiment's `aggregate.py` under
//! `criteria.toml`; a run at another scale (the unit tests, or the mutation checker's
//! `postgres-hybrid <queries> <seed>`) or without the strong arm is no evidence. The code under
//! test is `ptr_pg`: the lifecycle join of `search`, the live check of `upsert_document` and the
//! recall of its hybrid retrieval. The baselines are not under test: `rag_reference` (BM25 +
//! cosine + RRF over MiniLM vectors) and `strong_retrieval` (BM25 + bge-m3 dense + RRF +
//! bge-reranker-v2-m3), both run as Python drivers over a JSON-lines bridge, rebuilt at each
//! checkpoint from the harness's own live table.
//!
//! The object holds, in this order: the identity of the run, every metric the manifest names (a
//! JSON integer or an object of integers), `hard_failures`, the timings (never read by a rule),
//! the procedure, the negative control, the notes, the counters that belong to no metric
//! (`internal`), and the two judgements (`coverage` and `falsifiers`). The exit status is 1 when a
//! coverage prerequisite is unmet, a falsifier is tripped or a metric has no value and no
//! declared reason, and 2 when the harness itself cannot run: bad arguments, an input whose
//! SHA-256 differs from its pin, a driver that fails, the no-live-join control that returns no
//! planted row, a search that does not repeat. Everything the product does wrong, a panic
//! included, is a counter and a falsifier.

pub mod bridge;
pub mod data;
pub mod metrics;
pub mod oracle;
pub mod params;
pub mod pgarm;
pub mod preflight;
pub mod resume;
pub mod runner;
pub mod workload;

use std::path::PathBuf;
use std::time::Instant;

use super::layered_kv::emit::{count, judge, Check, Gaps, Json, Status, Tally};
use data::Data;
use params::{Scale, Shape};
use runner::{Options, Run, StrongMode};
use workload::Workload;

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

fn world_json(workload: &Workload) -> Json {
    let fate = |wanted: workload::Fate| {
        workload
            .fates
            .iter()
            .filter(|fate| **fate == wanted)
            .count() as i64
    };
    Json::obj([
        ("capsules", Json::Int(workload.shape.capsules as i64)),
        ("stay", Json::Int(fate(workload::Fate::Stay))),
        ("superseded_once", Json::Int(fate(workload::Fate::Once))),
        ("superseded_twice", Json::Int(fate(workload::Fate::Twice))),
        ("revoked_at_1", Json::Int(fate(workload::Fate::Revoked(1)))),
        ("revoked_at_2", Json::Int(fate(workload::Fate::Revoked(2)))),
        ("revoked_at_3", Json::Int(fate(workload::Fate::Revoked(3)))),
        ("queries", Json::Int(workload.queries.len() as i64)),
        (
            "strong_queries",
            Json::Int(workload.strong.iter().filter(|picked| **picked).count() as i64),
        ),
        ("plants", Json::Int(workload.plants.len() as i64)),
        ("late_writes", Json::Int(workload.late_writes.len() as i64)),
        (
            "stale_probes",
            Json::Int(workload.stale_probes.len() as i64),
        ),
        ("live_probes", Json::Int(workload.live_probes.len() as i64)),
    ])
}

/// Run the workload of `seed` at `scale` through every arm and assemble the one JSON object. A run
/// the harness cannot make is an `Err` with the reason.
pub fn execute(seed: u64, scale: Scale, options: &Options) -> Result<Report, String> {
    let started = Instant::now();
    let shape = scale.shape()?;
    let workload = Workload::generate(seed, shape)?;
    let digest = workload.digest(seed);
    let pinned = if shape == Shape::PINNED {
        params::pinned_digest(seed)
    } else {
        None
    };
    let mut tally = Tally::new(metrics::LIST);
    tally.set("workload_digest_mismatches", 0);
    if let Some(pinned) = pinned {
        if pinned != digest {
            tally.add("workload_digest_mismatches", 1);
        }
    }
    let data = Data::load(&data::repo_root(), options.strong != StrongMode::Off)?;
    let mut run = Run::open(seed, &workload, &digest, &data, options, tally)?;
    let outcome = run.go();
    let server = run.server();
    let hellos = std::mem::take(&mut run.hellos);
    let (ingest_ns, register_ns, strong_ns) = (run.ingest_ns, run.register_ns, run.strong_ns);
    let tally = std::mem::replace(&mut run.tally, Tally::new(metrics::LIST));
    run.close();
    outcome?;

    // The negative control: the no-live-join queries must have returned the planted rows.
    if tally.get("control_planted_returned") < 1 {
        return Err("the no-live-join control returned no planted row".to_string());
    }
    let hard = metrics::hard_failures(&tally);
    let (printed, gaps) = tally.metrics();
    let unsupported_names = tally.unsupported_names();
    let strong_ran = options.strong != StrongMode::Off;
    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str(params::BENCHMARK)),
        ("experiment".to_string(), Json::str(params::EXPERIMENT_ID)),
        (
            "status_note".to_string(),
            note("one seed of the Q003 harness: evidence only when run at the pinned scale with the full strong arm from the frozen commit and judged by aggregate.py under criteria.toml; the exit status and the coverage and falsifiers lists below are the harness's own reading of the same rules"),
        ),
        ("seed".to_string(), Json::Int(seed as i64)),
        ("scale".to_string(), Json::str(scale.name())),
    ];
    pairs.extend(printed);
    pairs.push(("hard_failures".to_string(), Json::Int(hard)));
    pairs.push((
        "timing_ns".to_string(),
        Json::obj([
            ("elapsed_ns", Json::Int(started.elapsed().as_nanos() as i64)),
            ("register_space_ns", Json::Int(register_ns as i64)),
            ("pg_ingest_ns", Json::Int(ingest_ns as i64)),
            ("strong_arm_ns", Json::Int(strong_ns as i64)),
        ]),
    ));
    pairs.push((
        "procedure".to_string(),
        Json::obj([
            (
                "seed_is_declared",
                Json::Bool(params::SEEDS.contains(&seed)),
            ),
            ("shape_is_pinned", Json::Bool(scale == Scale::Pinned)),
            ("workload_digest", Json::str(digest.clone())),
            (
                "pinned_workload_digest",
                pinned.map_or(Json::Null, Json::str),
            ),
            (
                "strong_arm",
                Json::obj([
                    ("mode", Json::str(options.strong.name())),
                    ("ran", Json::Bool(strong_ran)),
                    ("reranked", Json::Bool(options.strong == StrongMode::Full)),
                    (
                        "resumed_queries",
                        Json::Int(tally.get("strong_results_resumed")),
                    ),
                ]),
            ),
            (
                "inputs",
                Json::obj([
                    ("pool_sha256", Json::str(data.pool_sha256.clone())),
                    ("vectors_sha256", Json::str(data.vectors_sha256.clone())),
                    (
                        "strong_vectors_sha256",
                        data.strong_vectors_sha256
                            .clone()
                            .map_or(Json::Null, Json::str),
                    ),
                    ("embedding_model", Json::str(params::EMBEDDING_MODEL)),
                    ("embedding_revision", Json::str(params::EMBEDDING_REVISION)),
                ]),
            ),
            ("server", Json::str(server)),
            ("drivers", Json::Obj(hellos)),
            ("world", world_json(&workload)),
            (
                "recall_margin_permille",
                Json::Int(params::RECALL_MARGIN_PERMILLE),
            ),
        ]),
    ));
    pairs.push((
        "controls".to_string(),
        Json::obj([(
            "no_live_join",
            Json::obj([
                (
                    "required_rule",
                    Json::str(format!(
                        "control_planted_returned >= {}",
                        params::MIN_CONTROL_PLANTED_RETURNED
                    )),
                ),
                (
                    "required_rule_tripped",
                    Json::Bool(
                        tally.get("control_planted_returned")
                            >= params::MIN_CONTROL_PLANTED_RETURNED,
                    ),
                ),
                (
                    "lexical_returned",
                    Json::Int(tally.get("control_lexical_returned")),
                ),
                (
                    "vector_returned",
                    Json::Int(tally.get("control_vector_returned")),
                ),
                ("stale_probes", Json::Int(tally.get("stale_probe_queries"))),
            ]),
        )]),
    ));
    pairs.push(("unsupported".to_string(), tally.unsupported_json()));
    pairs.push((
        "measurement_notes".to_string(),
        Json::obj([
            (
                "oracle",
                note("the live table is the harness's own record built from the workload's events (oracle.rs); every hit of every list (lexical, vector, fused) of every product answer, probes included, is held against it, and the product's is_admissible is compared with it for every capsule and generation at both checkpoints (lifecycle_projection_mismatches)"),
            ),
            (
                "planted",
                note("stale rows are inserted by raw SQL into the derived table, bypassing upsert_document, to simulate index lag or a failed delete: superseded generations and revoked generations (whose live_generation row still names them), planted before the queries of each checkpoint; a planted row is a near duplicate of the live document and matches the same queries"),
            ),
            (
                "late_writes",
                note("upsert_document of a non-live generation (superseded, revoked, ahead of the live one, or of a capsule never committed) must be refused with NotLive and store no row; any other outcome is late_write_accepted or late_write_wrong_refusal"),
            ),
            (
                "probes",
                note("the main queries are natural-language questions, which plainto_tsquery ANDs term by term, so the lexical mode rarely matches them; the stale probes (a planted row's own body as the query) and the live probes (a live body) are what exercise the lexical lifecycle join and show the lexical mode returns hits; probes are audited for stale hits and are never part of recall"),
            ),
            (
                "control",
                note("the no-live-join control runs the same two queries without the lifecycle join on the stale probes; the exact lexical mode must return the planted row of every probe (else exit 2) and control_planted_returned counts both modes"),
            ),
            (
                "baselines",
                note("rag_reference and strong_retrieval are rebuilt from the live table at each checkpoint, so they hold no stale rows by construction; a baseline hit is never audited for staleness; the baselines are not under test and any failure of a driver is exit 2"),
            ),
            (
                "recall",
                note("recall@10 and nDCG@10 are over the answerable queries, the relevant document being the live generation of the target capsule; revoked targets are unanswerable and excluded; PG is the fused list of the product's search (limit 40 per mode, equal weights, k 60); descriptive: the PG lexical and vector lists alone and the BM25 list alone"),
            ),
            (
                "wall_time",
                note("timing_ns and latency_ns are recorded and never judged; strong_arm_ns is the wall time the strong driver took (loading and queries); the reranker runs in fp32 on shared CPU"),
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

/// The parsed command line after the subcommand.
#[derive(Debug, PartialEq)]
pub struct Parsed {
    pub seed: u64,
    pub scale: Scale,
    pub options: Options,
    pub digest_only: bool,
    /// `--preflight`: check the environment on a tiny world instead of running a seed.
    pub preflight: bool,
}

/// `<seed>` (the pinned scale) or `<queries> <seed>` (the mutation checker's form), with the
/// development flags `--no-strong`, `--strong-python <path>`, `--models-root <dir>`,
/// `--strong-threads <n>` and `--print-workload-digest`. Strict, so a typo stops the run.
pub fn parse_args(
    args: &[String],
    environment: &dyn Fn(&str) -> Option<String>,
) -> Result<Parsed, String> {
    let number = |text: &str, what: &str| {
        text.parse::<u64>()
            .map_err(|_| format!("{what} must be a non-negative integer, got {text:?}"))
    };
    let mut positional = Vec::new();
    let mut no_strong = false;
    let mut digest_only = false;
    let mut preflight = false;
    let mut strong_python = environment("PTR_Q003_STRONG_PYTHON").filter(|value| !value.is_empty());
    let mut models_root = environment("PTR_Q003_MODELS_ROOT")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);
    let mut strong_threads = None;
    let mut resume_dir = environment("PTR_Q003_STRONG_RESUME_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);
    let mut iterator = args.iter();
    while let Some(argument) = iterator.next() {
        let mut value_of = |flag: &str| {
            iterator
                .next()
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match argument.as_str() {
            "--no-strong" => no_strong = true,
            "--print-workload-digest" => digest_only = true,
            "--preflight" => preflight = true,
            "--strong-python" => strong_python = Some(value_of("--strong-python")?),
            "--strong-resume-dir" => {
                resume_dir = Some(PathBuf::from(value_of("--strong-resume-dir")?))
            }
            "--models-root" => models_root = Some(PathBuf::from(value_of("--models-root")?)),
            "--strong-threads" => {
                strong_threads = Some(
                    u32::try_from(number(&value_of("--strong-threads")?, "--strong-threads")?)
                        .map_err(|_| "--strong-threads is too large".to_string())?,
                );
            }
            flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}")),
            other => positional.push(other.to_string()),
        }
    }
    let usage = "usage: postgres-hybrid <seed> | postgres-hybrid <queries> <seed> [--no-strong] [--strong-python PATH] [--models-root DIR] [--strong-threads N] [--strong-resume-dir DIR] [--print-workload-digest] | postgres-hybrid --preflight";
    let (seed, scale) = match positional.as_slice() {
        [] if preflight => (0, Scale::Pinned),
        [seed] => (number(seed, "seed")?, Scale::Pinned),
        [queries, seed] => (
            number(seed, "seed")?,
            Scale::Custom(number(queries, "queries")? as usize),
        ),
        _ => return Err(usage.to_string()),
    };
    let strong = if no_strong {
        StrongMode::Off
    } else if scale == Scale::Pinned {
        StrongMode::Full
    } else {
        StrongMode::Off
    };
    Ok(Parsed {
        seed,
        scale,
        options: Options {
            strong,
            strong_python,
            models_root,
            strong_threads,
            resume_dir,
        },
        digest_only,
        preflight,
    })
}

/// Run the experiment's harness and print its one JSON object. Exits 1 when the run is not clean
/// and 2 when the harness cannot run.
pub fn run(args: &[String]) {
    let parsed = match parse_args(args, &|name| std::env::var(name).ok()) {
        Ok(parsed) => parsed,
        Err(reason) => {
            eprintln!("{}: {reason}", params::BENCHMARK);
            std::process::exit(2);
        }
    };
    if parsed.preflight {
        match preflight::run(&parsed.options, &|name| std::env::var(name).ok()) {
            Ok(()) => return,
            Err(reason) => {
                eprintln!("{} preflight failed: {reason}", params::BENCHMARK);
                std::process::exit(2);
            }
        }
    }
    if parsed.digest_only {
        let result = parsed
            .scale
            .shape()
            .and_then(|shape| Workload::generate(parsed.seed, shape))
            .map(|workload| workload.digest(parsed.seed));
        match result {
            Ok(digest) => println!("{{\"workload_digest\":\"{digest}\"}}"),
            Err(reason) => {
                eprintln!("{}: {reason}", params::BENCHMARK);
                std::process::exit(2);
            }
        }
        return;
    }
    if !params::SEEDS.contains(&parsed.seed) {
        eprintln!(
            "seed {} is not one of the manifest's seeds {:?}; this is a dry run",
            parsed.seed,
            params::SEEDS
        );
    }
    print_and_exit(execute(parsed.seed, parsed.scale, &parsed.options));
}

#[cfg(test)]
mod tests;
