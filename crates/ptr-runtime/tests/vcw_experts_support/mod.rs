//! Minimal local verifiers for the first expert tasks. They exist for the expert tests and the
//! `vcw_qwen` example only; the production verifiers are a separate package.

#![allow(dead_code)]

use ptr_runtime::vcw::experts::TextPayload;
use ptr_types::{Probability, VerificationLevel};
use ptr_verifier::{Finding, VerificationReport, VerificationStatus, Verifier};

pub const NOT_A_NUMBER: &str = "not_a_number";
pub const WRONG_ANSWER: &str = "wrong_answer";
pub const INVALID_JSON: &str = "invalid_json";
pub const MISSING_FIELD: &str = "missing_field";
pub const WRONG_TYPE: &str = "wrong_type";

pub fn report(findings: &[&str]) -> VerificationReport {
    let clean = findings.is_empty();
    VerificationReport {
        status: if clean {
            VerificationStatus::Pass
        } else {
            VerificationStatus::Fail
        },
        level: VerificationLevel::Deterministic,
        score: Probability::new(if clean { 1.0 } else { 0.0 }).unwrap(),
        findings: findings
            .iter()
            .map(|code| Finding {
                code: (*code).to_owned(),
                message: String::new(),
                hard: true,
            })
            .collect(),
    }
}

/// The answer must be exactly the integer `expected`.
pub struct ArithmeticVerifier {
    pub expected: i64,
}

impl Verifier<TextPayload> for ArithmeticVerifier {
    fn verify(&self, candidate: &TextPayload) -> VerificationReport {
        match candidate.text().trim().parse::<i64>() {
            Err(_) => report(&[NOT_A_NUMBER]),
            Ok(value) if value == self.expected => report(&[]),
            Ok(_) => report(&[WRONG_ANSWER]),
        }
    }
}

/// The JSON kind a field must have.
#[derive(Clone, Copy)]
pub enum Kind {
    String,
    Integer,
}

/// The candidate must be a JSON object with each `fields` entry present and of its kind.
pub struct JsonObjectVerifier {
    pub fields: Vec<(&'static str, Kind)>,
}

impl Verifier<TextPayload> for JsonObjectVerifier {
    fn verify(&self, candidate: &TextPayload) -> VerificationReport {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(candidate.text()) else {
            return report(&[INVALID_JSON]);
        };
        let Some(object) = value.as_object() else {
            return report(&[INVALID_JSON]);
        };
        let mut findings = Vec::new();
        for (name, kind) in &self.fields {
            match object.get(*name) {
                None => push_once(&mut findings, MISSING_FIELD),
                Some(found) => {
                    let ok = match kind {
                        Kind::String => found.is_string(),
                        Kind::Integer => found.is_i64() || found.is_u64(),
                    };
                    if !ok {
                        push_once(&mut findings, WRONG_TYPE);
                    }
                }
            }
        }
        report(&findings)
    }
}

fn push_once(findings: &mut Vec<&'static str>, code: &'static str) {
    if !findings.contains(&code) {
        findings.push(code);
    }
}

/// The one-line descriptions a repairer is shown, per code, for both task kinds.
pub const ARITHMETIC_CODES: [(&str, &str); 2] = [
    (NOT_A_NUMBER, "the answer is not a single integer"),
    (WRONG_ANSWER, "the integer is not the correct result"),
];

pub const JSON_CODES: [(&str, &str); 3] = [
    (INVALID_JSON, "the text is not one valid JSON object"),
    (MISSING_FIELD, "a required field is missing"),
    (WRONG_TYPE, "a field has the wrong JSON type"),
];
