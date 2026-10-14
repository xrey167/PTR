//! Serializable trajectory records for training and GEPA, as one JSON line per run.
//!
//! A [`TrajectoryRecord`] holds only strings, numbers and booleans. It is built from a finished
//! [`RepairRun`] with [`TrajectoryRecord::from_run`], which renders each candidate through
//! [`CandidateText`] and redacts every text field ([`crate::redact`]) before anything is
//! stored. The line format is versioned ([`SCHEMA_VERSION`]); the hand-written decoder accepts
//! exactly that schema and rejects unknown fields, unknown enumerated values and oversized
//! lines.

use std::error::Error;
use std::fmt;

use ptr_types::VerificationLevel;
use ptr_verifier::VerificationStatus;

use crate::candidate::Candidate;
use crate::json::{self, Json, JsonError};
use crate::redact::redact;
use crate::repair::{
    Decision, ExhaustionReason, Origin, Outcome, RepairRun, RewriteReason, Step, Usage,
};
use crate::text::{truncate_chars, CandidateText};

/// Version written to, and required of, every line.
pub const SCHEMA_VERSION: u32 = 1;
/// Longest encoded line, in bytes, newline excluded.
pub const MAX_LINE_BYTES: usize = 2 * 1024 * 1024;
/// Most steps in one record.
pub const MAX_STEPS: usize = 256;
/// Most characters kept of a candidate text; the rest is cut and flagged.
pub const MAX_TEXT_CHARS: usize = 8192;
/// Most characters of an id, code or run id.
pub const MAX_ID_CHARS: usize = 256;
/// Most issues listed for one step.
pub const MAX_ISSUES: usize = 64;

const ORIGINS: [&str; 3] = ["initial", "repaired", "rewritten"];
const REWRITE_REASONS: [&str; 2] = ["not_repairable", "declined"];
const DECISIONS: [&str; 4] = ["accepted", "repair", "rewrite", "stop"];
const STOP_REASONS: [&str; 2] = ["verifier_calls", "rounds"];
const STATUSES: [&str; 5] = ["pass", "fail", "disputed", "unknown", "unverified"];
const LEVELS: [&str; 6] = [
    "none",
    "unverified",
    "latent_agreement",
    "sample_verified",
    "full_semantic",
    "deterministic",
];

/// Why a record could not be built, encoded or decoded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecordError {
    /// The line is not valid JSON (byte offset of the first error).
    Syntax { offset: usize },
    /// The JSON nests deeper than the schema allows.
    TooDeep,
    /// The line is not valid UTF-8.
    NotUtf8,
    /// The line, or the record to encode, exceeds [`MAX_LINE_BYTES`].
    LineTooLong,
    /// A required field is absent.
    MissingField(&'static str),
    /// A field the schema does not define is present.
    UnexpectedField,
    /// A field has the wrong JSON type.
    WrongType(&'static str),
    /// A field has a value outside its allowed set or range.
    InvalidValue(&'static str),
    /// The line's schema version is not [`SCHEMA_VERSION`].
    UnsupportedVersion(u32),
    /// A field exceeds its length or count bound.
    TooLarge(&'static str),
    /// A score is NaN or infinite.
    NonFiniteScore,
}

impl fmt::Display for RecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax { offset } => write!(f, "invalid JSON at byte {offset}"),
            Self::TooDeep => f.write_str("JSON nests too deeply"),
            Self::NotUtf8 => f.write_str("the line is not UTF-8"),
            Self::LineTooLong => f.write_str("the line exceeds the size bound"),
            Self::MissingField(name) => write!(f, "missing field `{name}`"),
            Self::UnexpectedField => f.write_str("unexpected field"),
            Self::WrongType(name) => write!(f, "field `{name}` has the wrong type"),
            Self::InvalidValue(name) => write!(f, "field `{name}` has an invalid value"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported schema version {v}"),
            Self::TooLarge(name) => write!(f, "field `{name}` is too large"),
            Self::NonFiniteScore => f.write_str("a score is not finite"),
        }
    }
}

impl Error for RecordError {}

/// How a run ended.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OutcomeRecord {
    Verified {
        candidate: String,
    },
    /// `reason` is `verifier_calls` or `rounds`.
    Exhausted {
        reason: String,
    },
}

/// What a run spent.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UsageRecord {
    pub verifier_calls: u32,
    pub rounds: u32,
    pub generated: u32,
    pub repaired: u32,
    pub declined: u32,
}

/// One verified candidate of a run.
#[derive(Clone, Debug, PartialEq)]
pub struct StepRecord {
    pub candidate: String,
    /// `initial`, `repaired` or `rewritten`.
    pub origin: String,
    /// The candidate this one replaced.
    pub from: Option<String>,
    /// For `rewritten`: `not_repairable` or `declined`.
    pub rewrite_reason: Option<String>,
    /// The candidate text, redacted and cut to [`MAX_TEXT_CHARS`].
    pub text: String,
    pub text_truncated: bool,
    /// `pass`, `fail`, `disputed`, `unknown` or `unverified` (no report).
    pub status: String,
    /// `none` (no report), `unverified`, `latent_agreement`, `sample_verified`,
    /// `full_semantic` or `deterministic`.
    pub level: String,
    pub score: f64,
    /// Codes of the hard findings.
    pub issues: Vec<String>,
    /// `None` for the accepted candidate, which has no critique.
    pub repairable: Option<bool>,
    /// `accepted`, `repair`, `rewrite` or `stop`.
    pub decision: String,
    /// For `stop`: `verifier_calls` or `rounds`.
    pub stop_reason: Option<String>,
}

/// One run, ready to be written as a line.
#[derive(Clone, Debug, PartialEq)]
pub struct TrajectoryRecord {
    pub run_id: String,
    pub outcome: OutcomeRecord,
    pub usage: UsageRecord,
    pub steps: Vec<StepRecord>,
}

impl TrajectoryRecord {
    /// Record `run` under `run_id`, rendering each value with [`CandidateText`] and redacting
    /// every string.
    ///
    /// # Errors
    /// [`RecordError::TooLarge`] when the run has more than [`MAX_STEPS`] steps.
    pub fn from_run<T: CandidateText>(
        run_id: &str,
        run: &RepairRun<T>,
    ) -> Result<Self, RecordError> {
        if run.trajectory.steps.len() > MAX_STEPS {
            return Err(RecordError::TooLarge("steps"));
        }
        let outcome = match &run.outcome {
            Outcome::Verified(candidate) => OutcomeRecord::Verified {
                candidate: id_text(&candidate.id.to_string()),
            },
            Outcome::Exhausted { reason } => OutcomeRecord::Exhausted {
                reason: exhaustion_name(*reason).to_owned(),
            },
        };
        let Usage {
            verifier_calls,
            rounds,
            generated,
            repaired,
            declined,
        } = run.usage;
        Ok(Self {
            run_id: id_text(run_id),
            outcome,
            usage: UsageRecord {
                verifier_calls,
                rounds,
                generated,
                repaired,
                declined,
            },
            steps: run.trajectory.steps.iter().map(step_record).collect(),
        })
    }

    /// Encode as one JSON line (no trailing newline). Every string is redacted again, so a
    /// hand-built record cannot smuggle a secret through.
    ///
    /// # Errors
    /// Returns [`RecordError`] when the record is invalid, has a non-finite score or encodes
    /// to more than [`MAX_LINE_BYTES`].
    pub fn encode(&self) -> Result<String, RecordError> {
        validate_record(self)?;
        let mut out = String::with_capacity(512);
        out.push_str("{\"v\":");
        out.push_str(&SCHEMA_VERSION.to_string());
        out.push_str(",\"run_id\":");
        put(&mut out, &self.run_id);
        out.push_str(",\"outcome\":");
        match &self.outcome {
            OutcomeRecord::Verified { candidate } => {
                out.push_str("{\"kind\":\"verified\",\"candidate\":");
                put(&mut out, candidate);
            }
            OutcomeRecord::Exhausted { reason } => {
                out.push_str("{\"kind\":\"exhausted\",\"reason\":");
                put(&mut out, reason);
            }
        }
        out.push('}');
        let u = &self.usage;
        out.push_str(&format!(
            ",\"usage\":{{\"verifier_calls\":{},\"rounds\":{},\"generated\":{},\"repaired\":{},\"declined\":{}}}",
            u.verifier_calls, u.rounds, u.generated, u.repaired, u.declined
        ));
        out.push_str(",\"steps\":[");
        for (index, step) in self.steps.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            encode_step(&mut out, step);
        }
        out.push_str("]}");
        if out.len() > MAX_LINE_BYTES {
            return Err(RecordError::LineTooLong);
        }
        Ok(out)
    }

    /// Decode one line produced by [`TrajectoryRecord::encode`].
    ///
    /// # Errors
    /// Returns [`RecordError`] for any deviation from the schema.
    pub fn decode(line: &str) -> Result<Self, RecordError> {
        if line.len() > MAX_LINE_BYTES {
            return Err(RecordError::LineTooLong);
        }
        let value = json::parse(line).map_err(|error| match error {
            JsonError::Syntax { offset } => RecordError::Syntax { offset },
            JsonError::TooDeep => RecordError::TooDeep,
        })?;
        let mut root = Fields::new(value, "record")?;
        let version = root.take_u32("v")?;
        if version != SCHEMA_VERSION {
            return Err(RecordError::UnsupportedVersion(version));
        }
        let run_id = root.take_string("run_id")?;
        let mut outcome = Fields::new(root.take("outcome")?, "outcome")?;
        let kind = outcome.take_string("kind")?;
        let outcome_record = match kind.as_str() {
            "verified" => OutcomeRecord::Verified {
                candidate: outcome.take_string("candidate")?,
            },
            "exhausted" => OutcomeRecord::Exhausted {
                reason: outcome.take_string("reason")?,
            },
            _ => return Err(RecordError::InvalidValue("outcome.kind")),
        };
        outcome.finish()?;
        let mut usage = Fields::new(root.take("usage")?, "usage")?;
        let usage_record = UsageRecord {
            verifier_calls: usage.take_u32("verifier_calls")?,
            rounds: usage.take_u32("rounds")?,
            generated: usage.take_u32("generated")?,
            repaired: usage.take_u32("repaired")?,
            declined: usage.take_u32("declined")?,
        };
        usage.finish()?;
        let Json::Arr(items) = root.take("steps")? else {
            return Err(RecordError::WrongType("steps"));
        };
        if items.len() > MAX_STEPS {
            return Err(RecordError::TooLarge("steps"));
        }
        let steps = items
            .into_iter()
            .map(decode_step)
            .collect::<Result<Vec<_>, _>>()?;
        root.finish()?;
        let record = Self {
            run_id,
            outcome: outcome_record,
            usage: usage_record,
            steps,
        };
        validate_record(&record)?;
        Ok(record)
    }
}

fn put(out: &mut String, text: &str) {
    json::write_string(out, &redact(text));
}

fn put_opt(out: &mut String, value: &Option<String>) {
    match value {
        Some(text) => put(out, text),
        None => out.push_str("null"),
    }
}

fn encode_step(out: &mut String, step: &StepRecord) {
    out.push_str("{\"candidate\":");
    put(out, &step.candidate);
    out.push_str(",\"origin\":");
    put(out, &step.origin);
    out.push_str(",\"from\":");
    put_opt(out, &step.from);
    out.push_str(",\"rewrite_reason\":");
    put_opt(out, &step.rewrite_reason);
    out.push_str(",\"text\":");
    put(out, &step.text);
    out.push_str(",\"text_truncated\":");
    out.push_str(if step.text_truncated { "true" } else { "false" });
    out.push_str(",\"status\":");
    put(out, &step.status);
    out.push_str(",\"level\":");
    put(out, &step.level);
    out.push_str(",\"score\":");
    out.push_str(&step.score.to_string());
    out.push_str(",\"issues\":[");
    for (index, issue) in step.issues.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        put(out, issue);
    }
    out.push_str("],\"repairable\":");
    out.push_str(match step.repairable {
        Some(true) => "true",
        Some(false) => "false",
        None => "null",
    });
    out.push_str(",\"decision\":");
    put(out, &step.decision);
    out.push_str(",\"stop_reason\":");
    put_opt(out, &step.stop_reason);
    out.push('}');
}

fn decode_step(value: Json) -> Result<StepRecord, RecordError> {
    let mut f = Fields::new(value, "step")?;
    let step = StepRecord {
        candidate: f.take_string("candidate")?,
        origin: f.take_string("origin")?,
        from: f.take_opt_string("from")?,
        rewrite_reason: f.take_opt_string("rewrite_reason")?,
        text: f.take_string("text")?,
        text_truncated: f.take_bool("text_truncated")?,
        status: f.take_string("status")?,
        level: f.take_string("level")?,
        score: f.take_f64("score")?,
        issues: f.take_string_list("issues")?,
        repairable: match f.take("repairable")? {
            Json::Null => None,
            Json::Bool(b) => Some(b),
            _ => return Err(RecordError::WrongType("repairable")),
        },
        decision: f.take_string("decision")?,
        stop_reason: f.take_opt_string("stop_reason")?,
    };
    f.finish()?;
    Ok(step)
}

fn in_set(value: &str, set: &[&str], field: &'static str) -> Result<(), RecordError> {
    if set.contains(&value) {
        Ok(())
    } else {
        Err(RecordError::InvalidValue(field))
    }
}

fn bounded(value: &str, max_chars: usize, field: &'static str) -> Result<(), RecordError> {
    if value.chars().count() > max_chars {
        Err(RecordError::TooLarge(field))
    } else {
        Ok(())
    }
}

fn validate_record(record: &TrajectoryRecord) -> Result<(), RecordError> {
    bounded(&record.run_id, MAX_ID_CHARS, "run_id")?;
    match &record.outcome {
        OutcomeRecord::Verified { candidate } => bounded(candidate, MAX_ID_CHARS, "outcome")?,
        OutcomeRecord::Exhausted { reason } => in_set(reason, &STOP_REASONS, "outcome.reason")?,
    }
    if record.steps.len() > MAX_STEPS {
        return Err(RecordError::TooLarge("steps"));
    }
    for step in &record.steps {
        bounded(&step.candidate, MAX_ID_CHARS, "candidate")?;
        in_set(&step.origin, &ORIGINS, "origin")?;
        if let Some(from) = &step.from {
            bounded(from, MAX_ID_CHARS, "from")?;
        }
        if let Some(reason) = &step.rewrite_reason {
            in_set(reason, &REWRITE_REASONS, "rewrite_reason")?;
        }
        bounded(&step.text, MAX_TEXT_CHARS, "text")?;
        in_set(&step.status, &STATUSES, "status")?;
        in_set(&step.level, &LEVELS, "level")?;
        if !step.score.is_finite() {
            return Err(RecordError::NonFiniteScore);
        }
        if step.issues.len() > MAX_ISSUES {
            return Err(RecordError::TooLarge("issues"));
        }
        for issue in &step.issues {
            bounded(issue, MAX_ID_CHARS, "issues")?;
        }
        in_set(&step.decision, &DECISIONS, "decision")?;
        if let Some(reason) = &step.stop_reason {
            in_set(reason, &STOP_REASONS, "stop_reason")?;
        }
    }
    Ok(())
}

fn id_text(text: &str) -> String {
    redact(truncate_chars(text, MAX_ID_CHARS))
}

fn step_record<T: CandidateText>(step: &Step<T>) -> StepRecord {
    let rendered = step.candidate.value.candidate_text();
    let kept = truncate_chars(&rendered, MAX_TEXT_CHARS);
    let (status, level, score) = report_fields(&step.candidate);
    let (from, rewrite_reason) = match &step.origin {
        Origin::Initial => (None, None),
        Origin::Repaired { from } => (Some(id_text(&from.to_string())), None),
        Origin::Rewritten { from, reason } => (
            Some(id_text(&from.to_string())),
            Some(rewrite_name(*reason).to_owned()),
        ),
    };
    let (decision, stop_reason) = match step.decision {
        Decision::Accepted => ("accepted", None),
        Decision::Repair => ("repair", None),
        Decision::Rewrite(_) => ("rewrite", None),
        Decision::Stop(reason) => ("stop", Some(exhaustion_name(reason).to_owned())),
    };
    StepRecord {
        candidate: id_text(&step.candidate.id.to_string()),
        origin: match step.origin {
            Origin::Initial => "initial",
            Origin::Repaired { .. } => "repaired",
            Origin::Rewritten { .. } => "rewritten",
        }
        .to_owned(),
        from,
        rewrite_reason,
        text: redact(kept),
        text_truncated: kept.len() < rendered.len(),
        status: status.to_owned(),
        level: level.to_owned(),
        score,
        issues: step.critique.as_ref().map_or_else(Vec::new, |critique| {
            critique
                .issues
                .iter()
                .take(MAX_ISSUES)
                .map(|issue| id_text(issue))
                .collect()
        }),
        repairable: step.critique.as_ref().map(|critique| critique.repairable),
        decision: decision.to_owned(),
        stop_reason,
    }
}

fn report_fields<T>(candidate: &Candidate<T>) -> (&'static str, &'static str, f64) {
    match &candidate.verification {
        None => ("unverified", "none", 0.0),
        Some(report) => (
            match report.status {
                VerificationStatus::Pass => "pass",
                VerificationStatus::Fail => "fail",
                VerificationStatus::Disputed => "disputed",
                VerificationStatus::Unknown => "unknown",
            },
            match report.level {
                VerificationLevel::Unverified => "unverified",
                VerificationLevel::LatentAgreement => "latent_agreement",
                VerificationLevel::SampleVerified => "sample_verified",
                VerificationLevel::FullSemantic => "full_semantic",
                VerificationLevel::Deterministic => "deterministic",
            },
            f64::from(report.score.get()),
        ),
    }
}

fn rewrite_name(reason: RewriteReason) -> &'static str {
    match reason {
        RewriteReason::NotRepairable => "not_repairable",
        RewriteReason::Declined => "declined",
    }
}

fn exhaustion_name(reason: ExhaustionReason) -> &'static str {
    match reason {
        ExhaustionReason::VerifierCalls => "verifier_calls",
        ExhaustionReason::Rounds => "rounds",
    }
}

/// The fields of one decoded JSON object, taken by name; leftovers are an error.
struct Fields {
    fields: Vec<(String, Json)>,
}

impl Fields {
    fn new(value: Json, context: &'static str) -> Result<Self, RecordError> {
        match value {
            Json::Obj(fields) => Ok(Self { fields }),
            _ => Err(RecordError::WrongType(context)),
        }
    }

    fn take(&mut self, name: &'static str) -> Result<Json, RecordError> {
        let position = self
            .fields
            .iter()
            .position(|(key, _)| key == name)
            .ok_or(RecordError::MissingField(name))?;
        Ok(self.fields.swap_remove(position).1)
    }

    fn take_string(&mut self, name: &'static str) -> Result<String, RecordError> {
        match self.take(name)? {
            Json::Str(text) => Ok(text),
            _ => Err(RecordError::WrongType(name)),
        }
    }

    fn take_opt_string(&mut self, name: &'static str) -> Result<Option<String>, RecordError> {
        match self.take(name)? {
            Json::Null => Ok(None),
            Json::Str(text) => Ok(Some(text)),
            _ => Err(RecordError::WrongType(name)),
        }
    }

    fn take_bool(&mut self, name: &'static str) -> Result<bool, RecordError> {
        match self.take(name)? {
            Json::Bool(b) => Ok(b),
            _ => Err(RecordError::WrongType(name)),
        }
    }

    fn take_f64(&mut self, name: &'static str) -> Result<f64, RecordError> {
        match self.take(name)? {
            Json::Num(n) => Ok(n),
            _ => Err(RecordError::WrongType(name)),
        }
    }

    fn take_u32(&mut self, name: &'static str) -> Result<u32, RecordError> {
        let number = self.take_f64(name)?;
        if number.fract() == 0.0 && (0.0..=f64::from(u32::MAX)).contains(&number) {
            Ok(number as u32)
        } else {
            Err(RecordError::InvalidValue(name))
        }
    }

    fn take_string_list(&mut self, name: &'static str) -> Result<Vec<String>, RecordError> {
        let Json::Arr(items) = self.take(name)? else {
            return Err(RecordError::WrongType(name));
        };
        if items.len() > MAX_ISSUES {
            return Err(RecordError::TooLarge(name));
        }
        items
            .into_iter()
            .map(|item| match item {
                Json::Str(text) => Ok(text),
                _ => Err(RecordError::WrongType(name)),
            })
            .collect()
    }

    fn finish(self) -> Result<(), RecordError> {
        if self.fields.is_empty() {
            Ok(())
        } else {
            Err(RecordError::UnexpectedField)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(text: &str) -> StepRecord {
        StepRecord {
            candidate: "c0".into(),
            origin: "initial".into(),
            from: None,
            rewrite_reason: None,
            text: text.into(),
            text_truncated: false,
            status: "fail".into(),
            level: "deterministic".into(),
            score: 0.25,
            issues: vec!["target".into()],
            repairable: Some(true),
            decision: "repair".into(),
            stop_reason: None,
        }
    }

    fn record(text: &str) -> TrajectoryRecord {
        TrajectoryRecord {
            run_id: "run-1".into(),
            outcome: OutcomeRecord::Exhausted {
                reason: "rounds".into(),
            },
            usage: UsageRecord {
                verifier_calls: 2,
                rounds: 1,
                generated: 1,
                repaired: 1,
                declined: 0,
            },
            steps: vec![step(text)],
        }
    }

    #[test]
    fn a_record_round_trips_including_awkward_text() {
        let original = record("line1\nline2 \"quoted\" \\ \t é 😀 \u{1}");
        let line = original.encode().unwrap();
        assert!(!line.contains('\n'));
        assert_eq!(TrajectoryRecord::decode(&line).unwrap(), original);
    }

    #[test]
    fn encoding_redacts_secrets_in_every_string_field() {
        let mut r = record("set password=hunter2 now");
        r.run_id = "run token=abc".into();
        r.steps[0].issues = vec!["api_key=zzz".into()];
        let line = r.encode().unwrap();
        for leaked in ["hunter2", "abc\"", "zzz"] {
            assert!(!line.contains(leaked), "{leaked} leaked in {line}");
        }
        assert!(line.contains("[REDACTED]"));
    }

    #[test]
    fn decoding_rejects_every_kind_of_deviation() {
        let good = record("x").encode().unwrap();
        assert!(TrajectoryRecord::decode(&good).is_ok());
        let cases: Vec<(String, RecordError)> = vec![
            (
                good.replace("\"v\":1", "\"v\":2"),
                RecordError::UnsupportedVersion(2),
            ),
            (good.replace("\"v\":1,", ""), RecordError::MissingField("v")),
            (
                good.replace("}]}", ",\"extra\":1}]}"),
                RecordError::UnexpectedField,
            ),
            (
                good.replace("\"rounds\":1,", "\"rounds\":-1,"),
                RecordError::InvalidValue("rounds"),
            ),
            (
                good.replace("\"rounds\":1,", "\"rounds\":1.5,"),
                RecordError::InvalidValue("rounds"),
            ),
            (
                good.replace("\"initial\"", "\"mystery\""),
                RecordError::InvalidValue("origin"),
            ),
            (
                good.replace("\"fail\"", "\"maybe\""),
                RecordError::InvalidValue("status"),
            ),
            (
                good.replace("\"score\":0.25", "\"score\":\"x\""),
                RecordError::WrongType("score"),
            ),
            (
                good.replace("\"text_truncated\":false", "\"text_truncated\":1"),
                RecordError::WrongType("text_truncated"),
            ),
            ("[]".into(), RecordError::WrongType("record")),
        ];
        let truncated = &good[..good.len() - 5];
        assert!(matches!(
            TrajectoryRecord::decode(truncated),
            Err(RecordError::Syntax { .. })
        ));
        for (line, expected) in cases {
            assert_eq!(TrajectoryRecord::decode(&line), Err(expected), "{line}");
        }
    }

    #[test]
    fn size_bounds_are_enforced_on_both_sides() {
        let mut r = record("x");
        r.steps[0].text = "a".repeat(MAX_TEXT_CHARS + 1);
        assert_eq!(r.encode(), Err(RecordError::TooLarge("text")));
        let mut r = record("x");
        r.steps = vec![step("x"); MAX_STEPS + 1];
        assert_eq!(r.encode(), Err(RecordError::TooLarge("steps")));
        let mut r = record("x");
        r.steps[0].score = f64::NAN;
        assert_eq!(r.encode(), Err(RecordError::NonFiniteScore));
        let long = "a".repeat(MAX_LINE_BYTES + 1);
        assert_eq!(
            TrajectoryRecord::decode(&long),
            Err(RecordError::LineTooLong)
        );
        let mut r = record("x");
        r.steps[0].origin = "sideways".into();
        assert_eq!(r.encode(), Err(RecordError::InvalidValue("origin")));
    }
}
