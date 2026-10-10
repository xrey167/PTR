//! Resume protection for the strong arm: its cross-encoder costs hours a seed, so each query's
//! answer is appended to an on-disk file as soon as it exists, and a run that starts again with the
//! same workload, vectors and model revisions reuses them.
//!
//! The file is bounded (at most [`MAX_FILE_BYTES`], one line per query, at most `k` ids a line),
//! its first line is a header that must equal the one this run computes (seed, workload digest,
//! vectors digest, the dense and reranker revisions the driver reports, mode, `k`), and a line for a
//! query that is not one of the strong arm's is refused. Anything else is an error, never a silent
//! restart: the operator deletes a stale file. A resumed answer is exactly the list an earlier run
//! of the same driver returned; it is recorded as resumed (`strong_results_resumed`), and the
//! cross-encoder's fp32 scores are not bit-reproducible across machines or thread counts, so a
//! resumed run mixes answers computed under possibly different conditions.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use super::bridge::parse_json;
use super::params;
use crate::experiments::layered_kv::emit::{resolve, Json};

pub const MAX_FILE_BYTES: u64 = 1 << 20;
const MAX_LINE_BYTES: usize = 4096;

pub struct Resume {
    pub path: PathBuf,
    entries: HashMap<usize, Vec<String>>,
    file: File,
}

/// The header line of a run: everything an answer depends on besides the query itself.
pub fn header(
    seed: u64,
    workload_digest: &str,
    strong_vectors_sha256: &str,
    hello: &Json,
    mode: &str,
    provenance: &[(String, String)],
) -> Result<String, String> {
    let text = |path: &str| match resolve(hello, path) {
        Some(Json::Str(value)) => Ok(value.clone()),
        other => Err(format!(
            "the strong driver's hello has no {path}: {other:?}"
        )),
    };
    let mut pairs = vec![
        ("format".to_string(), Json::Int(2)),
        ("seed".to_string(), Json::Int(seed as i64)),
        ("workload_digest".to_string(), Json::str(workload_digest)),
        (
            "strong_vectors_sha256".to_string(),
            Json::str(strong_vectors_sha256),
        ),
        (
            "dense_revision".to_string(),
            Json::str(text("dense.revision")?),
        ),
        (
            "reranker_revision".to_string(),
            Json::str(text("reranker.revision")?),
        ),
        ("mode".to_string(), Json::str(mode)),
        ("k".to_string(), Json::Int(params::K as i64)),
    ];
    // What produced the answers beyond the models: the baseline directories' fingerprints and
    // the checkout (commit and whether the harness, crates or baselines were modified).
    for (key, value) in provenance {
        pairs.push((key.clone(), Json::str(value.clone())));
    }
    Ok(Json::Obj(pairs).render())
}

impl Resume {
    /// Open (creating it with its header if absent) the file of `seed` and `workload_digest` in
    /// `dir`. `allowed` is the set of query indexes the strong arm answers.
    pub fn open(
        dir: &Path,
        seed: u64,
        workload_digest: &str,
        header: &str,
        allowed: &HashSet<usize>,
    ) -> Result<Self, String> {
        std::fs::create_dir_all(dir)
            .map_err(|error| format!("cannot create {}: {error}", dir.display()))?;
        let path = dir.join(format!(
            "q003-strong-seed{seed}-{}.jsonl",
            &workload_digest[..16]
        ));
        let mut entries = HashMap::new();
        if path.exists() {
            repair_partial_last_line(&path)?;
            let size = std::fs::metadata(&path)
                .map_err(|error| format!("{}: {error}", path.display()))?
                .len();
            if size > MAX_FILE_BYTES {
                return Err(format!(
                    "{} is {size} bytes, more than {MAX_FILE_BYTES}",
                    path.display()
                ));
            }
            let text = std::fs::read_to_string(&path)
                .map_err(|error| format!("{}: {error}", path.display()))?;
            let mut lines = text.lines();
            if lines.next() != Some(header) {
                return Err(format!(
                    "{} was written for another run (header differs); delete it to start over",
                    path.display()
                ));
            }
            for line in lines {
                if line.len() > MAX_LINE_BYTES {
                    return Err(format!(
                        "{}: a line is longer than {MAX_LINE_BYTES} bytes",
                        path.display()
                    ));
                }
                let (index, ids) =
                    parse_line(line).map_err(|error| format!("{}: {error}", path.display()))?;
                if !allowed.contains(&index) {
                    return Err(format!(
                        "{}: query {index} is not a strong-arm query",
                        path.display()
                    ));
                }
                entries.insert(index, ids);
            }
        }
        let fresh = !path.exists();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        if fresh {
            writeln!(file, "{header}").map_err(|error| format!("{}: {error}", path.display()))?;
        }
        Ok(Self {
            path,
            entries,
            file,
        })
    }

    pub fn get(&self, index: usize) -> Option<&Vec<String>> {
        self.entries.get(&index)
    }

    pub fn record(&mut self, index: usize, ids: &[String]) -> Result<(), String> {
        let line = Json::obj([("i", Json::Int(index as i64)), ("ids", Json::strs(ids))]).render();
        writeln!(self.file, "{line}")
            .and_then(|()| self.file.flush())
            .map_err(|error| format!("{}: {error}", self.path.display()))?;
        self.entries.insert(index, ids.to_vec());
        Ok(())
    }
}

/// A run killed between the line and its newline leaves a last line without one (or a torn
/// one); the next append would run into it. Cut the file back to its last newline.
fn repair_partial_last_line(path: &Path) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.is_empty() || bytes.last() == Some(&b'\n') {
        return Ok(());
    }
    let keep = bytes
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(0, |at| at + 1);
    let file = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    file.set_len(keep as u64)
        .map_err(|error| format!("{}: {error}", path.display()))
}

fn parse_line(line: &str) -> Result<(usize, Vec<String>), String> {
    let parsed = parse_json(line)?;
    let index = match resolve(&parsed, "i") {
        Some(Json::Int(index)) if *index >= 0 => *index as usize,
        other => return Err(format!("bad query index {other:?}")),
    };
    let Some(Json::Arr(items)) = resolve(&parsed, "ids") else {
        return Err("no ids".to_string());
    };
    if items.len() > params::K {
        return Err(format!("{} ids, more than k", items.len()));
    }
    let mut ids = Vec::new();
    for item in items {
        match item {
            Json::Str(id) if id.len() <= 64 && id.contains('@') => ids.push(id.clone()),
            other => return Err(format!("bad id {other:?}")),
        }
    }
    Ok((index, ids))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hello() -> Json {
        parse_json(r#"{"dense":{"revision":"d1"},"reranker":{"revision":"r1","active":true}}"#)
            .unwrap()
    }

    fn prov() -> Vec<(String, String)> {
        vec![("harness_commit".to_string(), "abc".to_string())]
    }

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("q003-resume-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn answers_survive_a_restart_and_a_different_run_is_refused() {
        let dir = temp("a");
        let digest = "0123456789abcdef0123456789abcdef";
        let head = header(17, digest, "vv", &hello(), "full", &prov()).unwrap();
        let allowed: HashSet<usize> = [3, 5, 9].into_iter().collect();
        let mut first = Resume::open(&dir, 17, digest, &head, &allowed).unwrap();
        assert!(first.get(3).is_none());
        first.record(3, &["cap-0001@2".to_string()]).unwrap();
        first.record(5, &[]).unwrap();
        drop(first);
        let again = Resume::open(&dir, 17, digest, &head, &allowed).unwrap();
        assert_eq!(again.get(3), Some(&vec!["cap-0001@2".to_string()]));
        assert_eq!(again.get(5), Some(&Vec::new()));
        assert!(again.get(9).is_none());
        drop(again);
        // Another revision, another vectors digest or another mode is another header.
        let other = header(
            17,
            digest,
            "vv",
            &parse_json(r#"{"dense":{"revision":"d2"},"reranker":{"revision":"r1"}}"#).unwrap(),
            "full",
            &prov(),
        )
        .unwrap();
        assert!(Resume::open(&dir, 17, digest, &other, &allowed).is_err());
        let other = header(17, digest, "ww", &hello(), "full", &prov()).unwrap();
        assert!(Resume::open(&dir, 17, digest, &other, &allowed).is_err());
        // Another harness commit or baseline fingerprint is another header.
        let moved = header(
            17,
            digest,
            "vv",
            &hello(),
            "full",
            &[("harness_commit".to_string(), "def".to_string())],
        )
        .unwrap();
        assert!(Resume::open(&dir, 17, digest, &moved, &allowed).is_err());
        // A query that is not the strong arm's is refused.
        let narrow: HashSet<usize> = [3].into_iter().collect();
        assert!(Resume::open(&dir, 17, digest, &head, &narrow).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_malformed_or_oversized_file_is_refused() {
        let digest = "fedcba9876543210fedcba9876543210";
        let head = header(1, digest, "vv", &hello(), "full", &prov()).unwrap();
        let allowed: HashSet<usize> = [1].into_iter().collect();
        for (name, bad) in [
            ("neg", "{\"i\":-1,\"ids\":[]}".to_string()),
            ("noat", "{\"i\":1,\"ids\":[\"x\"]}".to_string()),
            ("junk", "not json".to_string()),
            (
                "many",
                format!("{{\"i\":1,\"ids\":[{}]}}", ["\"a@1\""; 11].join(",")),
            ),
        ] {
            let dir = temp(name);
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join(format!("q003-strong-seed1-{}.jsonl", &digest[..16]));
            std::fs::write(&path, format!("{head}\n{bad}\n")).unwrap();
            assert!(
                Resume::open(&dir, 1, digest, &head, &allowed).is_err(),
                "{name}"
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
        let dir = temp("big");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("q003-strong-seed1-{}.jsonl", &digest[..16]));
        std::fs::write(&path, vec![b'x'; (MAX_FILE_BYTES + 1) as usize]).unwrap();
        assert!(Resume::open(&dir, 1, digest, &head, &allowed).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_torn_last_line_is_cut_back_and_the_next_answer_is_appended_cleanly() {
        let dir = temp("torn");
        let digest = "00112233445566778899aabbccddeeff";
        let head = header(5, digest, "vv", &hello(), "full", &prov()).unwrap();
        let allowed: HashSet<usize> = [1, 2].into_iter().collect();
        let mut first = Resume::open(&dir, 5, digest, &head, &allowed).unwrap();
        first.record(1, &["cap-0001@1".to_string()]).unwrap();
        let path = first.path.clone();
        drop(first);
        // A kill between the line and its newline, and then a torn half line.
        let mut text = std::fs::read_to_string(&path).unwrap();
        text.pop();
        std::fs::write(&path, &text).unwrap();
        let mut again = Resume::open(&dir, 5, digest, &head, &allowed).unwrap();
        assert!(
            again.get(1).is_none(),
            "the line without its newline is not trusted"
        );
        again.record(2, &["cap-0002@1".to_string()]).unwrap();
        drop(again);
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"{\"i\":1,\"ids\":[\"ca")
            .unwrap();
        let last = Resume::open(&dir, 5, digest, &head, &allowed).unwrap();
        assert_eq!(last.get(2), Some(&vec!["cap-0002@1".to_string()]));
        assert!(last.get(1).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
