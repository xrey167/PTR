//! Read-only access to the deterministic verifiers of `ptr-verifier`.
//!
//! Python can run a verifier on a task and a candidate and read the report. It cannot
//! build a report (there is no constructor), so a report that reaches the router is always
//! one a verifier produced. No function here takes a threshold, a level to lower or a
//! verifier to plug in.

use ptr_semdb::{SemanticDelta, SemanticHost};
use ptr_types::Revision;
use ptr_verifier::{
    codes_for_role, fact_key, repair_role_for, ArithmeticVerifier, CodeTask, CodeTest,
    CodeTestVerifier, FactClaim, FactVerifier, Finding, JsonSchemaVerifier, RepairRole,
    VerificationReport, VerificationStatus, Verifier, CODE_TABLE,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::convert::{level_name, PtrVerifierError};

pub(crate) fn status_name(status: VerificationStatus) -> &'static str {
    match status {
        VerificationStatus::Pass => "pass",
        VerificationStatus::Fail => "fail",
        VerificationStatus::Disputed => "disputed",
        VerificationStatus::Unknown => "unknown",
    }
}

/// One finding of a verification report.
///
/// `severity` is `"hard"` for a deterministic contradiction (the verifier's `Fail`,
/// repairable by a role Pod) and `"soft"` otherwise. Read-only.
#[pyclass(frozen, skip_from_py_object, name = "Finding", module = "ptr_py")]
#[derive(Clone)]
pub(crate) struct PyFinding {
    #[pyo3(get)]
    code: String,
    #[pyo3(get)]
    message: String,
    #[pyo3(get)]
    hard: bool,
}

#[pymethods]
impl PyFinding {
    #[getter]
    fn severity(&self) -> &'static str {
        if self.hard {
            "hard"
        } else {
            "soft"
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Finding(code={:?}, severity={:?}, message={:?})",
            self.code,
            self.severity(),
            self.message
        )
    }
}

impl From<&Finding> for PyFinding {
    fn from(finding: &Finding) -> Self {
        Self {
            code: finding.code.clone(),
            message: finding.message.clone(),
            hard: finding.hard,
        }
    }
}

/// The report of one deterministic verifier on one candidate.
///
/// `status` is pass, fail, disputed or unknown; `level` the verification level reached;
/// `score` a probability that is recorded and never read by any decision; `findings` the
/// findings in report order. Only a verifier function of this module builds one: the class
/// has no constructor, so a report cannot be forged from Python.
#[pyclass(frozen, name = "VerificationReport", module = "ptr_py")]
pub(crate) struct PyReport {
    pub(crate) inner: VerificationReport,
    /// The task this report was verified against. Rust-only: the router keeps it so that a
    /// later candidate is verified against the same task inside Rust.
    pub(crate) spec: TaskSpec,
}

#[pymethods]
impl PyReport {
    #[getter]
    fn status(&self) -> &'static str {
        status_name(self.inner.status)
    }

    #[getter]
    fn level(&self) -> &'static str {
        level_name(self.inner.level)
    }

    #[getter]
    fn score(&self) -> f64 {
        f64::from(self.inner.score.get())
    }

    #[getter]
    fn findings(&self) -> Vec<PyFinding> {
        self.inner.findings.iter().map(PyFinding::from).collect()
    }

    /// The distinct hard finding codes, in report order.
    fn hard_codes(&self) -> Vec<String> {
        let mut codes: Vec<String> = Vec::new();
        for finding in self.inner.findings.iter().filter(|finding| finding.hard) {
            if !codes.contains(&finding.code) {
                codes.push(finding.code.clone());
            }
        }
        codes
    }

    /// `{"status", "level", "score", "findings": [{"code", "severity", "message"}]}`.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        report_dict(py, &self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "VerificationReport(status={:?}, level={:?}, score={}, findings={})",
            self.status(),
            self.level(),
            self.score(),
            self.inner.findings.len()
        )
    }
}

pub(crate) fn report_dict<'py>(
    py: Python<'py>,
    report: &VerificationReport,
) -> PyResult<Bound<'py, PyDict>> {
    let findings = PyList::empty(py);
    for finding in &report.findings {
        let item = PyDict::new(py);
        item.set_item("code", &finding.code)?;
        item.set_item("severity", if finding.hard { "hard" } else { "soft" })?;
        item.set_item("message", &finding.message)?;
        findings.append(item)?;
    }
    let out = PyDict::new(py);
    out.set_item("status", status_name(report.status))?;
    out.set_item("level", level_name(report.level))?;
    out.set_item("score", f64::from(report.score.get()))?;
    out.set_item("findings", findings)?;
    Ok(out)
}

/// The task a report was verified against: the verifier kind and everything it needs.
#[derive(Clone)]
pub(crate) enum TaskSpec {
    Arithmetic(String),
    Json(String),
    Code {
        entry: String,
        tests: Vec<CodeTest>,
    },
    Facts {
        facts: Vec<(String, String, String)>,
        live_revision: Option<u64>,
    },
}

/// A candidate to verify: text for the first three kinds, claims for facts.
pub(crate) enum Candidate {
    Text(String),
    Claims(Vec<(String, String, String)>),
}

impl Candidate {
    /// Canonical bytes of the candidate: the text itself, or each claim field as a
    /// little-endian u64 length followed by its bytes. Used for the logged digest.
    pub(crate) fn canonical_bytes(&self) -> Vec<u8> {
        match self {
            Self::Text(text) => text.as_bytes().to_vec(),
            Self::Claims(claims) => {
                let mut out = Vec::new();
                for (subject, attribute, value) in claims {
                    for field in [subject, attribute, value] {
                        out.extend_from_slice(&(field.len() as u64).to_le_bytes());
                        out.extend_from_slice(field.as_bytes());
                    }
                }
                out
            }
        }
    }
}

impl TaskSpec {
    /// Verify `candidate` against this task, in Rust. A candidate of the wrong shape for
    /// the kind is a PtrVerifierError.
    pub(crate) fn verify(&self, candidate: &Candidate) -> PyResult<VerificationReport> {
        match (self, candidate) {
            (Self::Arithmetic(task), Candidate::Text(text)) => {
                Ok(ArithmeticVerifier::new(task).verify(text))
            }
            (Self::Json(schema), Candidate::Text(text)) => {
                Ok(JsonSchemaVerifier::new(schema).verify(text))
            }
            (Self::Code { entry, tests }, Candidate::Text(text)) => {
                Ok(CodeTestVerifier::new(CodeTask {
                    entry: entry.clone(),
                    tests: tests.clone(),
                })
                .verify(text))
            }
            (
                Self::Facts {
                    facts,
                    live_revision,
                },
                Candidate::Claims(claims),
            ) => {
                let mut host = SemanticHost::default();
                if !facts.is_empty() {
                    let mut delta = SemanticDelta::default();
                    for (subject, attribute, value) in facts {
                        delta
                            .upserts
                            .insert(fact_key(subject, attribute), value.as_str().into());
                    }
                    host.apply_delta(delta)
                        .map_err(|error| PtrVerifierError::new_err(error.to_string()))?;
                }
                let live = live_revision.map_or_else(|| host.revision(), Revision);
                let verifier = FactVerifier::new(host.snapshot(), live);
                let claims: Vec<FactClaim> = claims
                    .iter()
                    .map(|(subject, attribute, value)| FactClaim::new(subject, attribute, value))
                    .collect();
                Ok(verifier.verify(&claims))
            }
            _ => Err(PtrVerifierError::new_err(
                "the candidate has the wrong shape for this task: text for arithmetic, \
                 json_schema and code, a list of (subject, attribute, value) claims for facts",
            )),
        }
    }

    /// Extract a candidate of the shape this task needs from a Python value.
    pub(crate) fn candidate(&self, value: &Bound<'_, PyAny>) -> PyResult<Candidate> {
        let wrong = |error: PyErr| {
            PtrVerifierError::new_err(format!("candidate has the wrong type: {error}"))
        };
        match self {
            Self::Facts { .. } => value.extract().map(Candidate::Claims).map_err(wrong),
            _ => value.extract().map(Candidate::Text).map_err(wrong),
        }
    }
}

fn wrap(inner: VerificationReport, spec: TaskSpec) -> PyReport {
    PyReport { inner, spec }
}

/// Verify a claimed result against one arithmetic task (exact rational arithmetic).
///
/// `task` is an expression of at most 512 bytes and `candidate` the claimed result.
/// Returns a `VerificationReport`. A task the verifier cannot evaluate, or an
/// oversized input, is reported as a finding, not raised. Read-only.
#[pyfunction]
pub(crate) fn verify_arithmetic(task: &str, candidate: String) -> PyReport {
    let spec = TaskSpec::Arithmetic(task.to_owned());
    let inner = ArithmeticVerifier::new(task).verify(&candidate);
    wrap(inner, spec)
}

/// Verify a JSON document (the candidate text) against a schema of the pinned subset.
///
/// Returns a `VerificationReport`. A schema outside the subset is reported as an
/// `unknown` report, not raised. Read-only.
#[pyfunction]
pub(crate) fn verify_json_schema(schema: &str, candidate: String) -> PyReport {
    let spec = TaskSpec::Json(schema.to_owned());
    let inner = JsonSchemaVerifier::new(schema).verify(&candidate);
    wrap(inner, spec)
}

fn required_item<'py>(test: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    test.get_item(key)?
        .ok_or_else(|| PtrVerifierError::new_err(format!("a code test needs the key {key:?}")))
}

fn optional_item<'py>(test: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(test.get_item(key)?.filter(|value| !value.is_none()))
}

fn bad_key(key: &str, error: PyErr) -> PyErr {
    PtrVerifierError::new_err(format!("code test key {key:?}: {error}"))
}

/// Verify a candidate program against the tests of a task, in the embedded sandboxed
/// interpreter (integer-only, bounded fuel, no I/O).
///
/// `entry` is the function the tests call. `tests` is a list of dicts with keys `id`
/// (str), `args` (list[int]), `expected` (int) and the optional `expected_output` (str)
/// and `visible` (bool, default False: a hidden test's finding names only its id).
/// A task with no tests, too many tests or a bad id is an `unknown` report with the
/// `code.task` finding. Returns a `VerificationReport`.
///
/// Raises PtrVerifierError when a test dict is malformed.
#[pyfunction]
pub(crate) fn verify_code(
    py: Python<'_>,
    entry: &str,
    tests: Vec<Bound<'_, PyDict>>,
    candidate: String,
) -> PyResult<PyReport> {
    let mut built = Vec::with_capacity(tests.len());
    for test in &tests {
        let id: String = required_item(test, "id")?
            .extract()
            .map_err(|error| bad_key("id", error))?;
        let args: Vec<i64> = required_item(test, "args")?
            .extract()
            .map_err(|error| bad_key("args", error))?;
        let expected: i64 = required_item(test, "expected")?
            .extract()
            .map_err(|error| bad_key("expected", error))?;
        let mut item = CodeTest::new(&id, &args, expected);
        if let Some(output) = optional_item(test, "expected_output")? {
            item.expected_output = Some(
                output
                    .extract::<String>()
                    .map_err(|error| bad_key("expected_output", error))?,
            );
        }
        if let Some(visible) = optional_item(test, "visible")? {
            if visible
                .extract::<bool>()
                .map_err(|error| bad_key("visible", error))?
            {
                item = item.visible();
            }
        }
        built.push(item);
    }
    let spec = TaskSpec::Code {
        entry: entry.to_owned(),
        tests: built,
    };
    let report = py.detach(|| spec.verify(&Candidate::Text(candidate)))?;
    Ok(wrap(report, spec))
}

/// Verify claims against a fact snapshot built from `facts`.
///
/// `facts` and `claims` are lists of `(subject, attribute, value)`. The snapshot is built
/// in a fresh SemDB host and verified read-only. `live_revision` is the host's current
/// revision; by default it is the snapshot's own. A different value models a stale
/// snapshot: every report is then `unknown` with the soft `fact.stale_snapshot` finding.
/// Returns a `VerificationReport`.
///
/// Raises PtrVerifierError when SemDB refuses the facts (for example a `/` in a key part).
#[pyfunction]
#[pyo3(signature = (facts, claims, *, live_revision=None))]
pub(crate) fn verify_facts(
    facts: Vec<(String, String, String)>,
    claims: Vec<(String, String, String)>,
    live_revision: Option<u64>,
) -> PyResult<PyReport> {
    let spec = TaskSpec::Facts {
        facts,
        live_revision,
    };
    let report = spec.verify(&Candidate::Claims(claims))?;
    Ok(wrap(report, spec))
}

fn role(name: &str) -> PyResult<RepairRole> {
    RepairRole::ALL
        .into_iter()
        .find(|role| role.as_str() == name)
        .ok_or_else(|| {
            let known: Vec<&str> = RepairRole::ALL.iter().map(|role| role.as_str()).collect();
            PtrVerifierError::new_err(format!(
                "unknown repair role {name:?}; expected one of {}",
                known.join(", ")
            ))
        })
}

/// The finding code table, in table order: `[{"code", "hard", "role"}]`. `role` is the
/// repair role of a hard code and None for a soft one. Read-only.
#[pyfunction]
pub(crate) fn finding_code_table(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
    let rows = PyList::empty(py);
    for spec in CODE_TABLE {
        let row = PyDict::new(py);
        row.set_item("code", spec.code)?;
        row.set_item("hard", spec.hard)?;
        row.set_item("role", spec.role.map(RepairRole::as_str))?;
        rows.append(row)?;
    }
    Ok(rows)
}

/// The names of the repair roles, in their fixed order.
#[pyfunction]
pub(crate) fn repair_roles() -> Vec<&'static str> {
    RepairRole::ALL.iter().map(|role| role.as_str()).collect()
}

/// The hard finding codes routed to the repair role `role` (for example
/// `"format-repair"`), in table order.
///
/// Raises PtrVerifierError for an unknown role name.
#[pyfunction(name = "codes_for_role")]
pub(crate) fn py_codes_for_role(role_name: &str) -> PyResult<Vec<&'static str>> {
    Ok(codes_for_role(role(role_name)?))
}

/// The repair role a hard finding code is routed to, or None for a soft or unknown code.
#[pyfunction(name = "repair_role_for")]
pub(crate) fn py_repair_role_for(code: &str) -> Option<&'static str> {
    repair_role_for(code).map(RepairRole::as_str)
}
