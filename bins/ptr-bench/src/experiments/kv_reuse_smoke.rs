//! KV-reuse smoke run on the CPU reference backend (doc 36, P3 skeleton).
//!
//! Runs `requests` prompts that share a prefix, each once with prefix reuse and once cold,
//! and reports whether every pair is bitwise equal (generated tokens and the digest of the
//! logits each token was chosen from). This is a plumbing and exactness smoke test on a
//! seeded random tiny model. It is NOT an R005 result: it uses none of R005's
//! preregistered parameters, writes no result record and says nothing about hit rate or
//! latency of a real model.

use super::json::json_string;
use ptr_runtime::{ReferenceKvError, ReferenceKvLimits, ReferenceKvReuseBackend, ReferenceKvScope};
use ptr_types::{PrincipalId, ProjectId, RequestId, Revision, SemanticContext};
use std::time::Instant;

/// Tokens every prompt shares (four chunks).
pub const SHARED_PREFIX_TOKENS: usize = 64;
/// Distinct tokens after the shared prefix.
pub const UNIQUE_SUFFIX_TOKENS: usize = 12;
/// Most requests one smoke run accepts.
pub const MAX_REQUESTS: usize = 256;

#[derive(Clone, Debug, PartialEq)]
pub struct SmokeReport {
    pub requests: usize,
    pub hit_tokens_total: usize,
    pub prompt_tokens_total: usize,
    pub cold_ms: f64,
    pub warm_ms: f64,
    pub bitwise_equal: bool,
}

impl SmokeReport {
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"benchmark":"kv-reuse-smoke","requests":{},"hit_tokens_total":{},"prompt_tokens_total":{},"cold_ms":{:.3},"warm_ms":{:.3},"bitwise_equal":{},"evidence":{}}}"#,
            self.requests,
            self.hit_tokens_total,
            self.prompt_tokens_total,
            self.cold_ms,
            self.warm_ms,
            self.bitwise_equal,
            json_string("smoke on seeded random tiny model; not an R005 result"),
        )
    }
}

/// `count` ASCII bytes from a fixed formula in `seed`; the shared prefix uses seed 0.
fn text(count: usize, seed: usize) -> String {
    (0..count)
        .map(|i| char::from(b'a' + ((i * 7 + seed * 13 + i / 16 * 5) % 26) as u8))
        .collect()
}

fn context() -> SemanticContext {
    SemanticContext {
        revision: Revision(1),
        entries: Vec::new(),
    }
}

pub fn measure(requests: usize) -> Result<SmokeReport, ReferenceKvError> {
    let scope = ReferenceKvScope {
        principal: PrincipalId::from("bench"),
        project: ProjectId::from("kv-reuse-smoke"),
    };
    let backend = ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), scope.clone())?;
    let prefix = text(SHARED_PREFIX_TOKENS, 0);
    let prompts: Vec<String> = (0..requests)
        .map(|index| format!("{prefix}{}", text(UNIQUE_SUFFIX_TOKENS, index + 1)))
        .collect();
    let id = |kind: &str, index: usize| RequestId::from(format!("{kind}-{index}").as_str());

    let started = Instant::now();
    let mut warm = Vec::with_capacity(requests);
    for (index, prompt) in prompts.iter().enumerate() {
        warm.push(backend.infer_text(&scope, id("warm", index), context(), prompt)?);
    }
    let warm_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    let mut cold = Vec::with_capacity(requests);
    for (index, prompt) in prompts.iter().enumerate() {
        cold.push(backend.infer_text_cold(&scope, id("cold", index), context(), prompt)?);
    }
    let cold_ms = started.elapsed().as_secs_f64() * 1000.0;

    let bitwise_equal = warm
        .iter()
        .zip(&cold)
        .all(|(w, c)| w.generated == c.generated && w.logits_digest == c.logits_digest);
    Ok(SmokeReport {
        requests,
        hit_tokens_total: warm.iter().map(|outcome| outcome.hit_tokens).sum(),
        prompt_tokens_total: warm.iter().map(|outcome| outcome.prompt_tokens).sum(),
        cold_ms,
        warm_ms,
        bitwise_equal,
    })
}

/// Print the JSON line; exit 1 when a warm run differs from its cold twin, 2 on a
/// refused parameter.
pub fn run(requests: usize) {
    if requests == 0 || requests > MAX_REQUESTS {
        eprintln!("requests must be in 1..={MAX_REQUESTS}, got {requests}");
        std::process::exit(2);
    }
    match measure(requests) {
        Ok(report) => {
            println!("{}", report.to_json());
            if !report.bitwise_equal {
                eprintln!("a reuse run differed from its cold twin");
                std::process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("kv-reuse-smoke failed: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warm_runs_reuse_the_shared_prefix_and_equal_their_cold_twins() {
        let report = measure(4).unwrap();
        assert!(report.bitwise_equal);
        assert_eq!(report.requests, 4);
        // The first request is cold; each later one reuses all four shared chunks.
        assert_eq!(report.hit_tokens_total, 3 * SHARED_PREFIX_TOKENS);
        assert!(report.to_json().contains(r#""bitwise_equal":true"#));
    }
}
