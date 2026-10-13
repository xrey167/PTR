//! The request and response types of `POST /v1/vcw/tasks`, and its typed errors.

use axum::{
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use ptr_runtime::RuntimeError;
use serde::{Deserialize, Serialize};

/// One task. Unknown fields are refused.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VcwTaskRequest {
    /// The request id; it names the committed evidence, so it can be used once.
    pub id: String,
    pub task: VcwTask,
}

/// What is asked. `reference` is the ground truth the **simulated** backend draws its clean
/// candidates from; it never reaches a verifier and never influences acceptance. The `qwen-cpu`
/// backend refuses a request that carries one.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum VcwTask {
    /// An integer or rational result of `expression` (`+ - * /`, unary signs, parentheses).
    Arithmetic {
        expression: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reference: Option<String>,
    },
    /// One JSON object that satisfies `schema` (the pinned JSON Schema subset).
    JsonObject {
        instruction: String,
        schema: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reference: Option<serde_json::Value>,
    },
    /// A program in the verifier's small language that passes `tests`.
    Code {
        instruction: String,
        entry: String,
        tests: Vec<VcwCodeTest>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reference: Option<String>,
    },
    /// Claims checked against the runtime's committed semantic state. `claims` is what the
    /// simulated backend asserts when it is right; the verifier decides.
    Fact {
        question: String,
        claims: Vec<VcwClaim>,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VcwCodeTest {
    pub id: String,
    pub args: Vec<i64>,
    pub expected: i64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VcwClaim {
    pub subject: String,
    pub attribute: String,
    pub value: String,
}

/// A Pod whose output is in the answer.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct VcwAuthor {
    pub pod: String,
    pub generation: u64,
}

/// Where in the ledger this request's commits are: events `first_index..first_index + count`.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct VcwCommitted {
    pub first_index: u64,
    pub count: u64,
}

/// What the request spent.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct VcwSpent {
    pub rounds: u32,
    pub verifier_calls: u32,
    pub pod_calls: u32,
}

/// An answer the verifier accepted and the runtime admitted.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct VcwVerified {
    pub id: String,
    pub answer: String,
    pub answer_type: String,
    pub authors: Vec<VcwAuthor>,
    #[serde(flatten)]
    pub spent: VcwSpent,
    /// Hex SHA-256 of the run's trace: a function of the seeds, the Pods' answers and the
    /// verifier's reports.
    pub trace_digest: String,
    /// Which experts produced the answer: `simulated` (the candidate came from the client's own
    /// `reference`, not from a model) or `qwen-cpu`. The same marker is in the manifest origins
    /// of the committed admission and in the capabilities of the serving Pods.
    pub backend: String,
    pub committed: VcwCommitted,
}

/// A request that ended without an answer. No candidate is returned.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct VcwUnanswered {
    pub id: String,
    /// The loop's reason (`Rounds`, `VerifierCalls`, `NoProposer`, ...) or `Deadline` when the
    /// per-request wall-clock deadline ended the request.
    pub reason: String,
    #[serde(flatten)]
    pub spent: VcwSpent,
    pub trace_digest: String,
    /// The backend of the experts, as in [`VcwVerified::backend`].
    pub backend: String,
}

/// The typed outcome. There is no variant that carries a rejected candidate.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum VcwResponse {
    Verified(VcwVerified),
    Exhausted(VcwUnanswered),
    Escalated(VcwUnanswered),
}

impl VcwResponse {
    #[must_use]
    pub fn status(&self) -> StatusCode {
        match self {
            Self::Verified(_) => StatusCode::OK,
            // A well-formed request the system declined to answer: 422, so a client that looks
            // only at the status can not mistake it for an answer.
            Self::Exhausted(_) | Self::Escalated(_) => StatusCode::UNPROCESSABLE_ENTITY,
        }
    }
}

impl IntoResponse for VcwResponse {
    fn into_response(self) -> Response {
        (self.status(), Json(self)).into_response()
    }
}

/// Why a request got no typed outcome.
#[derive(Debug)]
#[non_exhaustive]
pub enum VcwApiError {
    /// The input is invalid (400).
    BadRequest(String),
    /// A field exceeds its bound (413).
    TooLarge(String),
    /// The request id was used before (409).
    Conflict(String),
    /// The runtime refused (422, as the existing API does).
    Runtime(RuntimeError),
    /// The verified answer was not admitted; nothing is returned in its place (422).
    NotAdmitted(String),
    /// The request mailbox is full (503 with `Retry-After`).
    Saturated,
    Internal(String),
}

impl IntoResponse for VcwApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::Saturated => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    [(header::RETRY_AFTER, "1")],
                    Json(serde_json::json!({ "error": "request mailbox saturated" })),
                )
                    .into_response();
            }
            Self::BadRequest(message) => (StatusCode::BAD_REQUEST, message),
            Self::TooLarge(message) => (StatusCode::PAYLOAD_TOO_LARGE, message),
            Self::Conflict(message) => (StatusCode::CONFLICT, message),
            Self::Runtime(error) => (StatusCode::UNPROCESSABLE_ENTITY, format!("{error:?}")),
            Self::NotAdmitted(message) => (StatusCode::UNPROCESSABLE_ENTITY, message),
            Self::Internal(message) => (StatusCode::INTERNAL_SERVER_ERROR, message),
        };
        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_fields_are_refused_at_every_level() {
        for text in [
            r#"{"id":"a","task":{"kind":"arithmetic","expression":"1+1"},"extra":1}"#,
            r#"{"id":"a","task":{"kind":"arithmetic","expression":"1+1","extra":1}}"#,
            r#"{"id":"a","task":{"kind":"nope","expression":"1+1"}}"#,
            r#"{"id":"a","task":{"kind":"code","instruction":"i","entry":"f",
                "tests":[{"id":"t","args":[1],"expected":1,"extra":0}]}}"#,
        ] {
            assert!(
                serde_json::from_str::<VcwTaskRequest>(text).is_err(),
                "{text}"
            );
        }
        let ok: VcwTaskRequest =
            serde_json::from_str(r#"{"id":"a","task":{"kind":"arithmetic","expression":"1+1"}}"#)
                .unwrap();
        assert!(matches!(
            ok.task,
            VcwTask::Arithmetic {
                reference: None,
                ..
            }
        ));
    }

    #[test]
    fn the_response_is_tagged_by_outcome_and_unanswered_is_422() {
        let response = VcwResponse::Exhausted(VcwUnanswered {
            id: "a".into(),
            reason: "Rounds".into(),
            spent: VcwSpent {
                rounds: 1,
                verifier_calls: 2,
                pod_calls: 3,
            },
            trace_digest: "00".into(),
            backend: "simulated".into(),
        });
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["outcome"], "exhausted");
        assert_eq!(json["rounds"], 1);
        assert!(json.get("answer").is_none());
        let back: VcwResponse = serde_json::from_value(json).unwrap();
        assert_eq!(back, response);
    }
}
