//! One task as the server understands it: the text it commits as evidence, the verifier that
//! judges candidates for it, and the description the simulated backend plants defects from.

use ptr_config::{VcwBackend, VcwTaskKind};
use ptr_protocol::TypedPayload;
use ptr_runtime::vcw::experts::{ARITHMETIC_ANSWER_TYPE, JSON_OBJECT_ANSWER_TYPE};
use ptr_runtime::PtrRuntime;
use ptr_types::{Digest, Probability, TypeId, VerificationLevel};
use ptr_verifier::{
    arithmetic, code_test, fact, json_schema, ArithmeticVerifier, CodeTask, CodeTest,
    CodeTestVerifier, FactClaim, FactVerifier, Finding, JsonSchemaVerifier, VerificationReport,
    VerificationStatus, Verifier,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use super::wire::{VcwApiError, VcwClaim, VcwTask};

/// Payload type of a program.
pub const CODE_ANSWER_TYPE: &str = "vcw.answer.code";
/// Payload type of a list of claims (a JSON array of `{subject, attribute, value}`).
pub const FACT_ANSWER_TYPE: &str = "vcw.answer.fact_claims";

/// Longest instruction, question or statement, in bytes.
pub const MAX_INSTRUCTION_BYTES: usize = 16 * 1024;
/// Longest reference answer, in bytes.
pub const MAX_REFERENCE_BYTES: usize = 64 * 1024;
/// Longest function name, in bytes.
pub const MAX_ENTRY_BYTES: usize = 64;
/// Most arguments of one test.
pub const MAX_TEST_ARGS: usize = 8;

/// The payload type id of `kind`.
#[must_use]
pub fn answer_type(kind: VcwTaskKind) -> TypeId {
    TypeId::from(match kind {
        VcwTaskKind::Arithmetic => ARITHMETIC_ANSWER_TYPE,
        VcwTaskKind::JsonObject => JSON_OBJECT_ANSWER_TYPE,
        VcwTaskKind::Code => CODE_ANSWER_TYPE,
        VcwTaskKind::Fact => FACT_ANSWER_TYPE,
    })
}

/// What the simulated backend knows about a task: the clean answer and the shape of the code.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub(super) struct SimJob {
    pub kind: Option<VcwTaskKind>,
    pub reference: String,
    pub entry: String,
    pub arity: usize,
}

enum Spec {
    Arithmetic { expression: String },
    Json { schema: String },
    Code { task: CodeTask },
    Fact,
}

/// A validated task.
pub(super) struct Job {
    pub kind: VcwTaskKind,
    /// The text committed as raw evidence and shown to a model.
    pub statement: String,
    /// A canonical digest of what the verifier judges against (expression, schema, entry and
    /// tests, claims). The request digest commits to it, so two requests with the same
    /// statement but different criteria leave different evidence, while hidden tests stay out
    /// of the statement a model sees.
    pub criteria: Digest,
    pub sim: SimJob,
    spec: Spec,
}

/// Length-prefixed SHA-256 over `parts`, tagged with the task kind.
fn criteria_digest(kind: VcwTaskKind, parts: &[&[u8]]) -> Digest {
    let mut hasher = Sha256::new();
    for part in std::iter::once(kind.as_str().as_bytes()).chain(parts.iter().copied()) {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    hasher.finalize().into()
}

/// The canonical bytes of one code test: id, argument count, arguments and expected value.
fn code_test_bytes(id: &str, args: &[i64], expected: i64) -> Vec<u8> {
    let mut out = Vec::with_capacity(id.len() + 24 + args.len() * 8);
    out.extend_from_slice(&(id.len() as u64).to_le_bytes());
    out.extend_from_slice(id.as_bytes());
    out.extend_from_slice(&(args.len() as u64).to_le_bytes());
    for arg in args {
        out.extend_from_slice(&arg.to_le_bytes());
    }
    out.extend_from_slice(&expected.to_le_bytes());
    out
}

fn too_large(what: &str, limit: usize) -> VcwApiError {
    VcwApiError::TooLarge(format!("{what} is longer than {limit} bytes"))
}

fn non_blank(what: &str, value: &str, limit: usize) -> Result<(), VcwApiError> {
    if value.trim().is_empty() {
        return Err(VcwApiError::BadRequest(format!("{what} must not be blank")));
    }
    if value.len() > limit {
        return Err(too_large(what, limit));
    }
    Ok(())
}

fn reference_for(
    backend: VcwBackend,
    given: Option<String>,
    limit: usize,
) -> Result<String, VcwApiError> {
    match (backend, given) {
        (VcwBackend::Simulated, Some(reference)) => {
            if reference.len() > limit {
                return Err(too_large("reference", limit));
            }
            Ok(reference)
        }
        (VcwBackend::Simulated, None) => Err(VcwApiError::BadRequest(
            "the simulated backend needs a `reference` answer to draw clean candidates from".into(),
        )),
        (VcwBackend::QwenCpu, Some(_)) => Err(VcwApiError::BadRequest(
            "`reference` is for the simulated backend; the qwen-cpu backend refuses it".into(),
        )),
        (VcwBackend::QwenCpu, None) => Ok(String::new()),
    }
}

impl Job {
    pub(super) fn from_request(task: &VcwTask, backend: VcwBackend) -> Result<Self, VcwApiError> {
        match task {
            VcwTask::Arithmetic {
                expression,
                reference,
            } => {
                non_blank("expression", expression, arithmetic::MAX_TASK_BYTES)?;
                let reference = reference_for(backend, reference.clone(), MAX_REFERENCE_BYTES)?;
                Ok(Self {
                    kind: VcwTaskKind::Arithmetic,
                    statement: format!("What is {expression}?"),
                    criteria: criteria_digest(VcwTaskKind::Arithmetic, &[expression.as_bytes()]),
                    sim: SimJob {
                        kind: Some(VcwTaskKind::Arithmetic),
                        reference,
                        entry: String::new(),
                        arity: 0,
                    },
                    spec: Spec::Arithmetic {
                        expression: expression.clone(),
                    },
                })
            }
            VcwTask::JsonObject {
                instruction,
                schema,
                reference,
            } => {
                non_blank("instruction", instruction, MAX_INSTRUCTION_BYTES)?;
                let schema_text = serde_json::to_string(schema)
                    .map_err(|error| VcwApiError::BadRequest(format!("schema: {error}")))?;
                if schema_text.len() > json_schema::MAX_SCHEMA_BYTES {
                    return Err(too_large("schema", json_schema::MAX_SCHEMA_BYTES));
                }
                let reference = match reference {
                    Some(value) => {
                        if !value.is_object() {
                            return Err(VcwApiError::BadRequest(
                                "reference must be a JSON object".into(),
                            ));
                        }
                        let text = serde_json::to_string(value).map_err(|error| {
                            VcwApiError::BadRequest(format!("reference: {error}"))
                        })?;
                        reference_for(backend, Some(text), MAX_REFERENCE_BYTES)?
                    }
                    None => reference_for(backend, None, MAX_REFERENCE_BYTES)?,
                };
                Ok(Self {
                    kind: VcwTaskKind::JsonObject,
                    statement: format!(
                        "{instruction}\nThe JSON object must satisfy this JSON Schema: {schema_text}"
                    ),
                    criteria: criteria_digest(VcwTaskKind::JsonObject, &[schema_text.as_bytes()]),
                    sim: SimJob {
                        kind: Some(VcwTaskKind::JsonObject),
                        reference,
                        entry: String::new(),
                        arity: 0,
                    },
                    spec: Spec::Json {
                        schema: schema_text,
                    },
                })
            }
            VcwTask::Code {
                instruction,
                entry,
                tests,
                reference,
            } => {
                non_blank("instruction", instruction, MAX_INSTRUCTION_BYTES)?;
                non_blank("entry", entry, MAX_ENTRY_BYTES)?;
                if tests.len() > code_test::MAX_TESTS {
                    return Err(too_large("tests", code_test::MAX_TESTS));
                }
                let mut parsed = Vec::with_capacity(tests.len());
                let mut encoded = Vec::with_capacity(tests.len());
                for test in tests {
                    non_blank("test id", &test.id, code_test::MAX_TEST_ID_BYTES)?;
                    if test.args.len() > MAX_TEST_ARGS {
                        return Err(too_large("test arguments", MAX_TEST_ARGS));
                    }
                    parsed.push(CodeTest::new(&test.id, &test.args, test.expected));
                    encoded.push(code_test_bytes(&test.id, &test.args, test.expected));
                }
                let reference =
                    reference_for(backend, reference.clone(), code_test::MAX_SOURCE_BYTES)?;
                let arity = tests.first().map_or(0, |test| test.args.len());
                let mut parts: Vec<&[u8]> = vec![entry.as_bytes()];
                parts.extend(encoded.iter().map(Vec::as_slice));
                let criteria = criteria_digest(VcwTaskKind::Code, &parts);
                Ok(Self {
                    kind: VcwTaskKind::Code,
                    statement: format!(
                        "{instruction}\nWrite the function `{entry}` with {arity} integer argument(s)."
                    ),
                    criteria,
                    sim: SimJob {
                        kind: Some(VcwTaskKind::Code),
                        reference,
                        entry: entry.clone(),
                        arity,
                    },
                    spec: Spec::Code {
                        task: CodeTask {
                            entry: entry.clone(),
                            tests: parsed,
                        },
                    },
                })
            }
            VcwTask::Fact { question, claims } => {
                non_blank("question", question, MAX_INSTRUCTION_BYTES)?;
                if claims.len() > fact::MAX_CLAIMS {
                    return Err(too_large("claims", fact::MAX_CLAIMS));
                }
                for claim in claims {
                    for field in [&claim.subject, &claim.attribute, &claim.value] {
                        if field.len() > fact::MAX_FIELD_BYTES {
                            return Err(too_large("claim field", fact::MAX_FIELD_BYTES));
                        }
                    }
                }
                let reference = claims_text(claims);
                Ok(Self {
                    kind: VcwTaskKind::Fact,
                    statement: question.clone(),
                    criteria: criteria_digest(VcwTaskKind::Fact, &[reference.as_bytes()]),
                    sim: SimJob {
                        kind: Some(VcwTaskKind::Fact),
                        reference,
                        entry: String::new(),
                        arity: 0,
                    },
                    spec: Spec::Fact,
                })
            }
        }
    }

    /// The verifier for this task over the runtime's state now. Facts are judged against the
    /// runtime's committed semantic snapshot at its current revision, so a verifier built before
    /// a commit is stale and says `Unknown`; build a new one after.
    pub(super) fn judge(&self, runtime: &PtrRuntime) -> Judge {
        let inner = match &self.spec {
            Spec::Arithmetic { expression } => {
                Inner::Arithmetic(ArithmeticVerifier::new(expression))
            }
            Spec::Json { schema } => Inner::Json(JsonSchemaVerifier::new(schema)),
            Spec::Code { task } => Inner::Code(CodeTestVerifier::new(task.clone())),
            Spec::Fact => Inner::Fact(FactVerifier::new(runtime.snapshot(), runtime.revision())),
        };
        Judge {
            answer_type: answer_type(self.kind),
            inner,
        }
    }
}

/// The text form of claims: a JSON array of `{subject, attribute, value}`.
pub(super) fn claims_text(claims: &[VcwClaim]) -> String {
    serde_json::to_string(claims).unwrap_or_else(|_| "[]".to_owned())
}

enum Inner {
    Arithmetic(ArithmeticVerifier),
    Json(JsonSchemaVerifier),
    Code(CodeTestVerifier),
    Fact(FactVerifier),
}

/// The real deterministic verifier of a task, over the typed payload the loop carries.
pub(super) struct Judge {
    answer_type: TypeId,
    inner: Inner,
}

fn refusal(code: &str, message: &str) -> VerificationReport {
    VerificationReport {
        status: VerificationStatus::Fail,
        level: VerificationLevel::Deterministic,
        score: Probability::new(0.0).expect("0 is a probability"),
        findings: vec![Finding {
            code: code.to_owned(),
            message: message.to_owned(),
            hard: true,
        }],
    }
}

impl Verifier<TypedPayload> for Judge {
    fn verify(&self, payload: &TypedPayload) -> VerificationReport {
        if payload.type_id != self.answer_type {
            return refusal(
                "vcw.answer_type",
                "the payload is not of the task's answer type",
            );
        }
        let Ok(text) = String::from_utf8(payload.bytes.clone()) else {
            return refusal("vcw.utf8", "the answer is not UTF-8 text");
        };
        match &self.inner {
            Inner::Arithmetic(verifier) => verifier.verify(&text),
            Inner::Json(verifier) => verifier.verify(&text),
            Inner::Code(verifier) => verifier.verify(&text),
            Inner::Fact(verifier) => match serde_json::from_str::<Vec<VcwClaim>>(&text) {
                Ok(claims) => {
                    let claims: Vec<FactClaim> = claims
                        .iter()
                        .map(|c| FactClaim::new(&c.subject, &c.attribute, &c.value))
                        .collect();
                    verifier.verify(&claims)
                }
                // `fact.malformed` is a hard code of the table, routed to the format role.
                Err(_) => refusal(
                    "fact.malformed",
                    "the claims are not a JSON array of subject, attribute and value",
                ),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_config::PtrConfig;

    fn typed(kind: VcwTaskKind, text: &str) -> TypedPayload {
        TypedPayload {
            type_id: answer_type(kind),
            bytes: text.as_bytes().to_vec(),
        }
    }

    fn code_job(expected: i64) -> Job {
        let task: VcwTask = serde_json::from_value(serde_json::json!({
            "kind": "code",
            "instruction": "Add one.",
            "entry": "inc",
            "tests": [{"id": "t1", "args": [1], "expected": expected}],
            "reference": "def inc(x):\n    return x + 1\n",
        }))
        .unwrap();
        Job::from_request(&task, VcwBackend::Simulated).unwrap()
    }

    #[test]
    fn the_criteria_digest_binds_the_code_tests_but_the_statement_hides_them() {
        let (a, b) = (code_job(2), code_job(3));
        assert_eq!(a.statement, b.statement);
        assert!(!a.statement.contains("t1"));
        assert_ne!(a.criteria, b.criteria);
        assert_eq!(a.criteria, code_job(2).criteria);
    }

    #[test]
    fn the_judge_is_the_real_verifier_and_refuses_a_foreign_payload_type() {
        let runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        let job = Job::from_request(
            &VcwTask::Arithmetic {
                expression: "(3 + 4) * 5".into(),
                reference: Some("35".into()),
            },
            VcwBackend::Simulated,
        )
        .unwrap();
        let judge = job.judge(&runtime);
        let pass = judge.verify(&typed(VcwTaskKind::Arithmetic, "35"));
        assert_eq!(pass.status, VerificationStatus::Pass);
        let fail = judge.verify(&typed(VcwTaskKind::Arithmetic, "36"));
        assert_eq!(fail.findings[0].code, "arith.value");
        let foreign = judge.verify(&typed(VcwTaskKind::Code, "35"));
        assert_eq!(foreign.findings[0].code, "vcw.answer_type");
    }

    #[test]
    fn references_follow_the_backend() {
        let task = |reference: Option<&str>| VcwTask::Arithmetic {
            expression: "1 + 1".into(),
            reference: reference.map(str::to_owned),
        };
        assert!(Job::from_request(&task(None), VcwBackend::Simulated).is_err());
        assert!(Job::from_request(&task(Some("2")), VcwBackend::QwenCpu).is_err());
        assert!(Job::from_request(&task(None), VcwBackend::QwenCpu).is_ok());
        assert!(Job::from_request(&task(Some("2")), VcwBackend::Simulated).is_ok());
    }

    #[test]
    fn oversize_and_blank_fields_are_typed_errors() {
        let big = "9".repeat(arithmetic::MAX_TASK_BYTES + 1);
        let error = Job::from_request(
            &VcwTask::Arithmetic {
                expression: big,
                reference: Some("1".into()),
            },
            VcwBackend::Simulated,
        )
        .err()
        .unwrap();
        assert!(matches!(error, VcwApiError::TooLarge(_)));
        let error = Job::from_request(
            &VcwTask::Arithmetic {
                expression: "  ".into(),
                reference: Some("1".into()),
            },
            VcwBackend::Simulated,
        )
        .err()
        .unwrap();
        assert!(matches!(error, VcwApiError::BadRequest(_)));
    }
}
