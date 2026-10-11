//! The first task kinds, their prompts and the extraction of a candidate from model text.

use ptr_types::{Digest, TypeId};
use sha2::{Digest as _, Sha256};

use crate::vcw::VcwPayload;

/// Payload type of an arithmetic answer.
pub const ARITHMETIC_ANSWER_TYPE: &str = "vcw.answer.arithmetic";
/// Payload type of a JSON object answer.
pub const JSON_OBJECT_ANSWER_TYPE: &str = "vcw.answer.json_object";

/// What kind of answer the task asks for.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum TaskKind {
    /// A single integer.
    Arithmetic,
    /// A single JSON object.
    JsonObject,
}

impl TaskKind {
    #[must_use]
    pub fn payload_type(self) -> TypeId {
        TypeId::from(match self {
            Self::Arithmetic => ARITHMETIC_ANSWER_TYPE,
            Self::JsonObject => JSON_OBJECT_ANSWER_TYPE,
        })
    }

    fn format_rule(self) -> &'static str {
        match self {
            Self::Arithmetic => "Reply with only the final integer, nothing else.",
            Self::JsonObject => "Reply with only one JSON object, no prose and no code fences.",
        }
    }

    /// A default system prompt for a proposer of this kind.
    #[must_use]
    pub fn proposer_system_prompt(self) -> &'static str {
        match self {
            Self::Arithmetic => "You are a helpful assistant that solves arithmetic problems.",
            Self::JsonObject => "You are a helpful assistant that writes JSON.",
        }
    }

    /// A default system prompt for a repairer of this kind.
    #[must_use]
    pub fn repairer_system_prompt(self) -> &'static str {
        match self {
            Self::Arithmetic => {
                "You are a helpful assistant that corrects wrong arithmetic answers."
            }
            Self::JsonObject => "You are a helpful assistant that corrects JSON.",
        }
    }

    /// The candidate in `model_text`. When the text holds no such answer, the trimmed text itself
    /// is the candidate, so the verifier, not this function, reports what is wrong with it.
    #[must_use]
    pub fn extract(self, model_text: &str) -> String {
        let trimmed = model_text.trim();
        let found = match self {
            Self::Arithmetic => first_integer(trimmed),
            Self::JsonObject => first_json_object(trimmed),
        };
        found.unwrap_or_else(|| trimmed.to_owned())
    }
}

/// The request: what is asked and what kind of answer is expected.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertTask {
    kind: TaskKind,
    statement: String,
}

impl ExpertTask {
    #[must_use]
    pub fn new(kind: TaskKind, statement: impl Into<String>) -> Self {
        Self {
            kind,
            statement: statement.into(),
        }
    }

    #[must_use]
    pub fn kind(&self) -> TaskKind {
        self.kind
    }

    #[must_use]
    pub fn statement(&self) -> &str {
        &self.statement
    }

    /// The user message a proposer receives in `round`.
    ///
    /// Round 0 is the task and the format rule. A later proposal (the loop asked for a fresh
    /// one) appends the attempt number, which carries no finding, report or score: a greedy
    /// model asked the identical question returns the identical text, and this perturbation may
    /// make a regenerated proposal differ. It is not a guarantee of a different answer.
    #[must_use]
    pub fn proposer_message(&self, round: u32) -> String {
        let base = format!("{}\n{}", self.statement, self.kind.format_rule());
        if round == 0 {
            base
        } else {
            format!("{base}\nAttempt {}.", round.saturating_add(1))
        }
    }

    /// The user message a repairer receives: the task, the previous candidate and exactly one
    /// finding, described by `description`.
    #[must_use]
    pub fn repairer_message(&self, previous: &str, code: &str, description: &str) -> String {
        format!(
            "{}\nA previous answer was: {}\nA checker reported one problem ({code}): {description}.\n\
             Give the corrected answer, changing only what that problem concerns. {}",
            self.statement,
            previous,
            self.kind.format_rule()
        )
    }
}

/// A candidate made of text, typed by the task kind. The loop never reads the text; the host's
/// verifier does.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextPayload {
    type_id: TypeId,
    text: String,
}

impl TextPayload {
    #[must_use]
    pub fn new(type_id: TypeId, text: impl Into<String>) -> Self {
        Self {
            type_id,
            text: text.into(),
        }
    }

    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    #[must_use]
    pub fn type_id(&self) -> &TypeId {
        &self.type_id
    }
}

impl VcwPayload for TextPayload {
    fn payload_type(&self) -> TypeId {
        self.type_id.clone()
    }

    fn payload_digest(&self) -> Digest {
        let mut hasher = Sha256::new();
        hasher.update(self.type_id.0.as_bytes());
        hasher.update([0]);
        hasher.update(self.text.as_bytes());
        hasher.finalize().into()
    }
}

/// The first (optionally signed) run of ASCII digits, as the integer text. A hyphen is a sign
/// only at the start or after a character that is not a letter or a digit: `-3` and `(-3)` give
/// `-3`, while `a-3` and `5-3` read as `3` and `5`.
fn first_integer(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index].is_ascii_digit() {
            let signed = index > 0
                && bytes[index - 1] == b'-'
                && (index == 1 || !bytes[index - 2].is_ascii_alphanumeric());
            let start = if signed { index - 1 } else { index };
            let mut end = index;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            return Some(text[start..end].to_owned());
        }
        index += 1;
    }
    None
}

/// The first balanced `{ ... }` (string and escape aware), as written.
fn first_json_object(text: &str) -> Option<String> {
    let start = text.find('{')?;
    let mut depth = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, ch) in text[start..].char_indices() {
        if in_string {
            match (escaped, ch) {
                (true, _) => escaped = false,
                (false, '\\') => escaped = true,
                (false, '"') => in_string = false,
                _ => {}
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..=start + offset].to_owned());
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_integer_is_found_with_its_sign_and_prose_is_ignored() {
        let kind = TaskKind::Arithmetic;
        assert_eq!(kind.extract("42"), "42");
        assert_eq!(kind.extract("The answer is 5445."), "5445");
        assert_eq!(kind.extract("  -17 "), "-17");
        assert_eq!(kind.extract("x - 3"), "3");
        assert_eq!(kind.extract("a-3"), "3");
        assert_eq!(kind.extract("5-3"), "5");
        assert_eq!(kind.extract("(-3)"), "-3");
        assert_eq!(kind.extract("x = -3"), "-3");
        assert_eq!(kind.extract("no number here"), "no number here");
    }

    #[test]
    fn a_json_object_is_cut_out_of_fences_and_prose() {
        let kind = TaskKind::JsonObject;
        let fenced = "```json\n{\n  \"name\": \"Ada\", \"note\": \"a } b\"\n}\n```";
        assert_eq!(
            kind.extract(fenced),
            "{\n  \"name\": \"Ada\", \"note\": \"a } b\"\n}"
        );
        assert_eq!(
            kind.extract("Sure: {\"a\": {\"b\": 1}} done"),
            "{\"a\": {\"b\": 1}}"
        );
        // An unbalanced object is returned as is, for the verifier to reject.
        assert_eq!(kind.extract("{\"a\": 1"), "{\"a\": 1");
    }

    #[test]
    fn the_digest_binds_type_and_text() {
        let a = TextPayload::new(TypeId::from("t"), "1");
        let b = TextPayload::new(TypeId::from("t"), "1");
        let c = TextPayload::new(TypeId::from("u"), "1");
        let d = TextPayload::new(TypeId::from("t"), "2");
        assert_eq!(a.payload_digest(), b.payload_digest());
        assert_ne!(a.payload_digest(), c.payload_digest());
        assert_ne!(a.payload_digest(), d.payload_digest());
    }

    #[test]
    fn only_a_later_proposal_carries_an_attempt_number_and_never_a_finding() {
        let task = ExpertTask::new(TaskKind::Arithmetic, "What is 2 + 2?");
        let first = task.proposer_message(0);
        assert_eq!(
            first,
            "What is 2 + 2?\nReply with only the final integer, nothing else."
        );
        let later = task.proposer_message(2);
        assert!(later.starts_with(&first));
        assert!(later.ends_with("Attempt 3."));
        assert_ne!(later, task.proposer_message(1));
    }

    #[test]
    fn a_repair_message_names_one_code_and_not_the_report() {
        let task = ExpertTask::new(TaskKind::Arithmetic, "What is 2 + 2?");
        let message = task.repairer_message("5", "wrong_answer", "the value is not correct");
        assert!(message.contains("wrong_answer"));
        assert!(message.contains("A previous answer was: 5"));
        assert!(message.contains("What is 2 + 2?"));
    }
}
