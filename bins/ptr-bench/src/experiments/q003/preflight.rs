//! `postgres-hybrid --preflight`: the check the runner makes before it reserves a seed.
//!
//! A confirmatory seed costs hours and cannot be run twice, so everything the run needs from its
//! environment is exercised first, on a tiny world: the server answers and has PostgreSQL 16+ and
//! pgvector 0.8+, the strong arm's interpreter is an absolute path that imports `torch` and
//! `transformers`, the models directory is absolute and holds the pinned reranker (the driver
//! checks its revision when it starts), one strong-arm query is answered end to end through the
//! driver with the reranker, the reference and BM25 `python3` drivers (started from `PATH`
//! exactly as a seed starts them) answer one query each, and one `ptr-pg` hybrid query is answered
//! on a temporary schema that is dropped afterwards. Nothing is measured and nothing is printed but progress and the reason of
//! a failure; no value of a variable is ever printed (the connection string is scrubbed from every
//! message that could echo it). A failed preflight can leave `q003_0_*` schemas behind when the
//! product arm itself exits (`die`/`expect` in `PgArm`); they may be dropped by hand and cannot
//! affect a seed (a seed's prefix carries its own seed number, never 0). Exit 0 when everything holds, 2
//! otherwise.

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use super::bridge::{Bridge, DocRef};
use super::data::{repo_root, Data};
use super::params;
use super::pgarm::{search_document, Call, PgArm};
use super::runner::{spawn_bm25, spawn_reference, Options};
use super::workload::{capsule_id, Event, EventKind};
use crate::experiments::layered_kv::emit::{resolve, Json};
use crate::experiments::pg::DSN_VARIABLE;

/// The documents of the tiny world (cues of the pool, generation 1) and the cue the query asks.
const CUES: [usize; 5] = [0, 1, 2, 3, 4];
const IMPORT_TIMEOUT: Duration = Duration::from_secs(300);
const REPLY_TIMEOUT: Duration = Duration::from_secs(1800);

/// `text` with every secret removed, so an error that echoes a connection string leaks none.
pub fn scrub(text: &str, secrets: &[&str]) -> String {
    let mut out = text.to_string();
    for secret in secrets {
        if secret.len() >= 4 {
            out = out.replace(secret, "<redacted>");
        }
    }
    out
}

/// Why `value` cannot be a declared variable's value: empty, or with whitespace at either end.
pub fn unusable(value: &str) -> Option<&'static str> {
    if value.is_empty() {
        Some("is empty")
    } else if value != value.trim() {
        Some("has leading or trailing whitespace")
    } else {
        None
    }
}

/// An absolute path variable's problem, if any.
pub fn absolute_problem(value: &str, wants_file: bool) -> Option<&'static str> {
    let path = Path::new(value);
    if !path.is_absolute() {
        Some("is not an absolute path")
    } else if wants_file && !path.is_file() {
        Some("is not a file")
    } else if !wants_file && !path.is_dir() {
        Some("is not a directory")
    } else {
        None
    }
}

/// `0.8.1` is at least `0.8`: the numeric components compared in order.
pub fn at_least(version: &str, minimum: &[u32]) -> bool {
    let parts: Vec<u32> = version
        .split('.')
        .map(|part| {
            part.chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
                .parse()
                .unwrap_or(0)
        })
        .collect();
    for (index, wanted) in minimum.iter().enumerate() {
        let have = parts.get(index).copied().unwrap_or(0);
        if have != *wanted {
            return have > *wanted;
        }
    }
    true
}

fn step(number: usize, what: &str) {
    eprintln!("postgres-hybrid preflight {number}/7: {what}");
}

fn server_checks(dsn: &str) -> Result<(), String> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("tokio runtime: {error}"))?;
    runtime.block_on(async {
        let connect = tokio_postgres::connect(dsn, tokio_postgres::NoTls);
        let (client, connection) = tokio::time::timeout(Duration::from_secs(30), connect)
            .await
            .map_err(|_| "the server did not answer within 30 s".to_string())?
            .map_err(|error| {
                format!(
                    "cannot connect to the server named by {DSN_VARIABLE}: {}",
                    scrub(&error.to_string(), &[dsn])
                )
            })?;
        tokio::spawn(async move {
            let _ = connection.await;
        });
        let number: String = client
            .query_one("SHOW server_version_num", &[])
            .await
            .map_err(|error| format!("cannot read the server version: {error}"))?
            .get(0);
        let number: u32 = number.parse().unwrap_or(0);
        if number < 160_000 {
            return Err(format!(
                "the server is version number {number}; PostgreSQL 16+ is required"
            ));
        }
        let available: Option<String> = client
            .query_opt(
                "SELECT default_version FROM pg_available_extensions WHERE name = 'vector'",
                &[],
            )
            .await
            .map_err(|error| format!("cannot read the available extensions: {error}"))?
            .map(|row| row.get(0));
        match available {
            Some(version) if at_least(&version, &[0, 8]) => Ok(()),
            Some(version) => Err(format!("pgvector {version} is available; 0.8+ is required")),
            None => Err("pgvector is not available on the server".to_string()),
        }
    })
}

fn imports_work(python: &str) -> Result<(), String> {
    let mut child = Command::new(python)
        .args(["-c", "import torch, transformers"])
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("cannot start the strong interpreter: {error}"))?;
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => return Ok(()),
            Ok(Some(_)) => {
                return Err("the strong interpreter cannot import torch and transformers".into())
            }
            Ok(None) if started.elapsed() > IMPORT_TIMEOUT => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("importing torch and transformers took over 300 s".to_string());
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(200)),
            Err(error) => return Err(format!("cannot wait for the strong interpreter: {error}")),
        }
    }
}

fn strong_query(
    data: &Data,
    python: &str,
    models: &str,
    threads: Option<u32>,
) -> Result<(), String> {
    let root = repo_root();
    let mut args = vec![
        "research/baselines/strong_retrieval/driver.py".to_string(),
        "--reranker".to_string(),
        "bge".to_string(),
        "--models-root".to_string(),
        models.to_string(),
    ];
    if let Some(threads) = threads {
        args.push("--threads".to_string());
        args.push(threads.to_string());
    }
    let mut bridge =
        Bridge::spawn("strong", python, &args, &root, REPLY_TIMEOUT).map_err(String::from)?;
    let outcome = (|| {
        let hello = bridge.hello()?;
        if resolve(&hello, "reranker.active") != Some(&Json::Bool(true)) {
            return Err("the strong driver's reranker is not active".to_string());
        }
        let ids: Vec<String> = CUES.iter().map(|cue| format!("preflight-{cue}")).collect();
        let rows: Vec<usize> = CUES.iter().map(|cue| data.fact(*cue, 1)).collect();
        let docs: Vec<DocRef<'_>> = ids
            .iter()
            .zip(&rows)
            .map(|(id, row)| {
                Ok(DocRef {
                    id,
                    text: data.text(*row),
                    embedding: data
                        .strong_embedding(*row)
                        .ok_or("the strong vectors are not loaded")?,
                })
            })
            .collect::<Result<_, String>>()?;
        bridge.load(&docs)?;
        let query = data.query(CUES[0], 0);
        let strong = data
            .strong_embedding(query)
            .ok_or("the strong vectors are not loaded")?;
        let reply = bridge.search(data.text(query), strong, 3)?;
        if reply.reranked != Some(true) {
            return Err(format!(
                "the strong query was not reranked: {:?}",
                reply.reranked
            ));
        }
        if reply.hits.is_empty() || reply.hits.iter().any(|hit| !ids.contains(hit)) {
            return Err("the strong query returned no hit or an unknown document".to_string());
        }
        Ok(())
    })();
    let closed = bridge.close().map_err(String::from);
    outcome.and(closed)
}

/// One tiny load and one query through an already started reference or BM25 driver; `embedded`
/// says whether the driver is given the MiniLM vectors (the reference) or none (BM25).
fn bridge_round(data: &Data, bridge: &mut Bridge, embedded: bool) -> Result<(), String> {
    let ids: Vec<String> = CUES.iter().map(|cue| format!("preflight-{cue}")).collect();
    let rows: Vec<usize> = CUES.iter().map(|cue| data.fact(*cue, 1)).collect();
    let docs: Vec<DocRef<'_>> = ids
        .iter()
        .zip(&rows)
        .map(|(id, row)| DocRef {
            id,
            text: data.text(*row),
            embedding: if embedded { data.embedding(*row) } else { &[] },
        })
        .collect();
    bridge.load(&docs)?;
    if bridge.size()? != CUES.len() as i64 {
        return Err("the driver does not hold the five documents just loaded".to_string());
    }
    let query = data.query(CUES[0], 0);
    let embedding = if embedded { data.embedding(query) } else { &[] };
    let reply = bridge.search(data.text(query), embedding, 3)?;
    if reply.hits.is_empty() || reply.hits.iter().any(|hit| !ids.contains(hit)) {
        return Err("the query returned no hit or an unknown document".to_string());
    }
    Ok(())
}

/// The reference and BM25 drivers, started as `Run::open` starts them (`python3` from `PATH`,
/// same arguments), asked one tiny question each, and ended on every path.
fn reference_and_bm25(data: &Data) -> Result<(), String> {
    let root = repo_root();
    let mut reference = spawn_reference(&root)?;
    let outcome = (|| {
        let hello = reference.hello()?;
        if resolve(&hello, "embedding_model") != Some(&Json::str(params::EMBEDDING_MODEL)) {
            return Err("the reference driver pins another embedding model than Q003".to_string());
        }
        bridge_round(data, &mut reference, true)
            .map_err(|error| format!("reference driver: {error}"))
    })();
    let closed = reference.close().map_err(String::from);
    outcome?;
    closed?;
    let mut bm25 = spawn_bm25(&root)?;
    let outcome = (|| {
        bm25.hello()?;
        bridge_round(data, &mut bm25, false).map_err(|error| format!("BM25 driver: {error}"))
    })();
    let closed = bm25.close().map_err(String::from);
    outcome.and(closed)
}

fn product_query(data: &Data) -> Result<(), String> {
    let (mut arm, _) = PgArm::open(0);
    let outcome = (|| {
        for cue in CUES {
            let commit = Event {
                kind: EventKind::Commit,
                cue,
            };
            if !matches!(arm.apply(&commit), Call::Done(())) {
                return Err("the projector refused a commit".to_string());
            }
            let row = data.fact(cue, 1);
            let document =
                search_document(&capsule_id(cue), 1, data.text(row), data.embedding(row));
            if !matches!(arm.upsert(&document), Call::Done(())) {
                return Err("upsert_document refused a live generation".to_string());
            }
        }
        let query = data.query(CUES[0], 0);
        match arm.search(data.text(query), data.embedding(query)) {
            Call::Done(results) if !results.fused.is_empty() => Ok(()),
            Call::Done(_) => Err("the hybrid query returned an empty fused list".to_string()),
            _ => Err("the hybrid query failed".to_string()),
        }
    })();
    arm.finish();
    outcome
}

/// Run every check; the first failure is the error.
pub fn run(options: &Options, environment: &dyn Fn(&str) -> Option<String>) -> Result<(), String> {
    let dsn = environment(DSN_VARIABLE).unwrap_or_default();
    if let Some(problem) = unusable(&dsn) {
        return Err(format!("{DSN_VARIABLE} {problem}"));
    }
    let python = options.strong_python.clone().unwrap_or_default();
    let models = options
        .models_root
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_default();
    step(1, "the server answers, PostgreSQL 16+, pgvector 0.8+");
    server_checks(&dsn)?;
    step(
        2,
        "the strong interpreter is an absolute file that imports torch and transformers",
    );
    if let Some(problem) = unusable(&python).or_else(|| absolute_problem(&python, true)) {
        return Err(format!("PTR_Q003_STRONG_PYTHON {problem}"));
    }
    imports_work(&python)?;
    step(
        3,
        "the models directory is absolute and holds the pinned reranker",
    );
    if let Some(problem) = unusable(&models).or_else(|| absolute_problem(&models, false)) {
        return Err(format!("PTR_Q003_MODELS_ROOT {problem}"));
    }
    if !Path::new(&models).join("bge-reranker-v2-m3").is_dir() {
        return Err("PTR_Q003_MODELS_ROOT holds no bge-reranker-v2-m3 directory".to_string());
    }
    step(4, "the frozen inputs match their pins");
    let data = Data::load(&repo_root(), true)?;
    step(
        5,
        "one strong-arm query end to end through the driver, reranker included",
    );
    strong_query(&data, &python, &models, options.strong_threads)?;
    step(
        6,
        "the reference and BM25 drivers start from PATH and answer one query each",
    );
    reference_and_bm25(&data)?;
    step(7, "one ptr-pg hybrid query on a temporary schema");
    product_query(&data)?;
    eprintln!("postgres-hybrid preflight: ok");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_compare_numerically() {
        assert!(at_least("0.8.0", &[0, 8]));
        assert!(at_least("0.10.1", &[0, 8]));
        assert!(at_least("1.0", &[0, 8]));
        assert!(!at_least("0.7.4", &[0, 8]));
        assert!(!at_least("garbage", &[0, 8]));
    }

    #[test]
    fn values_are_checked_without_being_shown() {
        assert_eq!(unusable(""), Some("is empty"));
        assert_eq!(unusable(" x"), Some("has leading or trailing whitespace"));
        assert_eq!(unusable("x\n"), Some("has leading or trailing whitespace"));
        assert_eq!(unusable("x"), None);
        assert_eq!(
            absolute_problem("relative/python", true),
            Some("is not an absolute path")
        );
        assert_eq!(
            absolute_problem("/nonexistent-q003/python", true),
            Some("is not a file")
        );
        assert_eq!(
            absolute_problem("/nonexistent-q003", false),
            Some("is not a directory")
        );
        assert_eq!(absolute_problem("/", false), None);
    }

    #[test]
    fn the_bm25_and_reference_drivers_answer_the_tiny_round() {
        let data = Data::load(&repo_root(), false).expect("the frozen inputs load");
        let root = repo_root();
        let mut bm25 = spawn_bm25(&root).expect("python3 starts the BM25 driver");
        let outcome = bridge_round(&data, &mut bm25, false);
        bm25.close().unwrap();
        assert_eq!(outcome, Ok(()));
        let mut reference = spawn_reference(&root).expect("python3 starts the reference driver");
        let outcome = bridge_round(&data, &mut reference, true);
        reference.close().unwrap();
        assert_eq!(outcome, Ok(()));
    }

    #[test]
    fn a_message_that_echoes_the_connection_string_is_scrubbed() {
        let dsn = "host=h user=u password=s3cretpw";
        let said = scrub(&format!("bad string {dsn} here"), &[dsn]);
        assert_eq!(said, "bad string <redacted> here");
        assert!(!said.contains("s3cretpw"));
    }
}
