//! The tasks of F007 and the real verifiers that judge them.
//!
//! A task is arithmetic (judged by `ptr_verifier::ArithmeticVerifier`), a JSON object under a
//! schema (`JsonSchemaVerifier`) or a small program with tests (`CodeTestVerifier`). A candidate
//! is the task's clean answer corrupted by a set of planted defect codes; its text is what the
//! real verifier judges, so which code the verifier reports is the verifier's decision, not the
//! simulation's. The harness never trusts the verifier alone: it keeps the planted defect set and
//! the clean text as an independent oracle (`VerifyEvent::clean`, [`Judge::independent_pass`]).

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::sync::Arc;

use ptr_ledger::integrity::sha256;
use ptr_runtime::vcw::experts::{DefectPayload, ARITHMETIC_ANSWER_TYPE, JSON_OBJECT_ANSWER_TYPE};
use ptr_runtime::vcw::VcwPayload;
use ptr_types::{Digest, VerificationLevel};
use ptr_verifier::{
    ArithmeticVerifier, CodeTask, CodeTest, CodeTestVerifier, JsonSchemaVerifier,
    VerificationReport, VerificationStatus, Verifier,
};

use super::super::rng::Rng;

/// Payload type of a program.
pub const CODE_ANSWER_TYPE: &str = "vcw.answer.code";

/// The task kinds, one lane of the loop each.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Kind {
    Arithmetic,
    Json,
    Code,
}

pub const KINDS: [Kind; 3] = [Kind::Arithmetic, Kind::Json, Kind::Code];

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Arithmetic => "arithmetic",
            Self::Json => "json",
            Self::Code => "code",
        }
    }

    pub fn index(self) -> usize {
        match self {
            Self::Arithmetic => 0,
            Self::Json => 1,
            Self::Code => 2,
        }
    }

    /// The hard finding codes this kind plants and the real verifier can report.
    pub fn codes(self) -> &'static [&'static str] {
        match self {
            Self::Arithmetic => &["arith.format", "arith.value", "arith.bounds"],
            Self::Json => &["json.parse", "json.type", "json.required", "json.extra"],
            Self::Code => &["code.syntax", "code.test_failed", "code.timeout"],
        }
    }

    pub fn answer_type(self) -> &'static str {
        match self {
            Self::Arithmetic => ARITHMETIC_ANSWER_TYPE,
            Self::Json => JSON_OBJECT_ANSWER_TYPE,
            Self::Code => CODE_ANSWER_TYPE,
        }
    }
}

/// Every plantable code, in kind order.
pub fn all_codes() -> Vec<&'static str> {
    KINDS
        .iter()
        .flat_map(|kind| kind.codes().iter().copied())
        .collect()
}

#[derive(Clone, Debug)]
enum Spec {
    Arithmetic {
        expression: String,
    },
    Json {
        schema: String,
        /// `(key, value text)` in document order.
        fields: Vec<(String, String)>,
    },
    Code {
        entry: String,
        arity: usize,
        source: String,
        tests: Vec<(String, Vec<i64>, i64)>,
    },
}

/// One task with its clean answer.
#[derive(Clone, Debug)]
pub struct Task {
    pub kind: Kind,
    pub index: u64,
    /// The text of the clean answer: what a defect-free candidate renders to.
    pub reference: String,
    spec: Spec,
}

const FIELD_POOL: [(&str, &str); 7] = [
    ("name", "string"),
    ("city", "string"),
    ("tag", "string"),
    ("age", "integer"),
    ("score", "integer"),
    ("id", "integer"),
    ("active", "boolean"),
];

impl Task {
    /// The task of request `index` of the stream `seed`, of the given kind. Pure.
    pub fn generate(seed: u64, index: u64, kind: Kind) -> Self {
        let mut rng = Rng::new(seed ^ 0x7F4A_7C15_F007_0001).fork(index);
        match kind {
            Kind::Arithmetic => Self::arithmetic(index, &mut rng),
            Kind::Json => Self::json(index, &mut rng),
            Kind::Code => Self::code(index, &mut rng),
        }
    }

    fn arithmetic(index: u64, rng: &mut Rng) -> Self {
        let operands = 3 + rng.index(2);
        let first = rng.range(2, 60) as i128;
        let mut expression = first.to_string();
        let mut value = first;
        for _ in 1..operands {
            let operand = rng.range(2, 60) as i128;
            let (op, next) = match rng.index(3) {
                0 => ('+', value + operand),
                1 => ('-', value - operand),
                _ => ('*', value * operand),
            };
            expression = format!("({expression}) {op} {operand}");
            value = next;
        }
        Self {
            kind: Kind::Arithmetic,
            index,
            reference: value.to_string(),
            spec: Spec::Arithmetic { expression },
        }
    }

    fn json(index: u64, rng: &mut Rng) -> Self {
        let mut pool: Vec<(&str, &str)> = FIELD_POOL.to_vec();
        let count = 3 + rng.index(3);
        let mut chosen = Vec::new();
        for _ in 0..count {
            chosen.push(pool.remove(rng.index(pool.len())));
        }
        let properties: Vec<String> = chosen
            .iter()
            .map(|(key, ty)| format!("\"{key}\":{{\"type\":\"{ty}\"}}"))
            .collect();
        let required: Vec<String> = chosen.iter().map(|(key, _)| format!("\"{key}\"")).collect();
        let schema = format!(
            "{{\"type\":\"object\",\"properties\":{{{}}},\"required\":[{}],\"additionalProperties\":false}}",
            properties.join(","),
            required.join(",")
        );
        let fields: Vec<(String, String)> = chosen
            .iter()
            .map(|(key, ty)| {
                let value = match *ty {
                    "string" => format!("\"v{}\"", rng.range(0, 99)),
                    "integer" => rng.range(0, 999).to_string(),
                    _ => if rng.chance(0.5) { "true" } else { "false" }.to_string(),
                };
                ((*key).to_string(), value)
            })
            .collect();
        let reference = render_object(&fields);
        Self {
            kind: Kind::Json,
            index,
            reference,
            spec: Spec::Json { schema, fields },
        }
    }

    fn code(index: u64, rng: &mut Rng) -> Self {
        let template = rng.index(4);
        let constant = rng.range(1, 9) as i64;
        let (arity, source): (usize, String) = match template {
            0 => (1, format!("fn f(a0) {{ return a0 * a0 + {constant}; }}")),
            1 => (2, "fn f(a0, a1) { return a0 * a1 - a0; }".to_string()),
            2 => (
                1,
                "fn f(a0) { let s = 0; let i = 0; while i < a0 { s = s + i; i = i + 1; } return s; }"
                    .to_string(),
            ),
            _ => (
                2,
                "fn f(a0, a1) { if a0 > a1 { return a0 - a1; } return a1 - a0; }".to_string(),
            ),
        };
        let expected = |args: &[i64]| -> i64 {
            match template {
                0 => args[0] * args[0] + constant,
                1 => args[0] * args[1] - args[0],
                2 => args[0] * (args[0] - 1) / 2,
                _ => (args[0] - args[1]).abs(),
            }
        };
        let tests: Vec<(String, Vec<i64>, i64)> = (0..4)
            .map(|n| {
                let args: Vec<i64> = (0..arity).map(|_| rng.range(1, 12) as i64).collect();
                let want = expected(&args);
                (format!("t{n}"), args, want)
            })
            .collect();
        Self {
            kind: Kind::Code,
            index,
            reference: source.clone(),
            spec: Spec::Code {
                entry: "f".to_string(),
                arity,
                source,
                tests,
            },
        }
    }

    /// The text of a candidate that carries `defects`. When several defects are planted, the
    /// first in the kind's masking order shows and the others are revealed after it is fixed.
    pub fn render(&self, defects: &BTreeSet<String>) -> String {
        let has = |code: &str| defects.contains(code);
        match &self.spec {
            Spec::Arithmetic { .. } => {
                if has("arith.format") {
                    "n/a".to_string()
                } else if has("arith.bounds") {
                    "9".repeat(40)
                } else if has("arith.value") {
                    self.reference
                        .parse::<i128>()
                        .map_or_else(|_| format!("{}1", self.reference), |n| (n + 1).to_string())
                } else {
                    self.reference.clone()
                }
            }
            Spec::Json { fields, .. } => {
                let mut fields = fields.clone();
                if has("json.required") {
                    fields.remove(0);
                } else if has("json.type") {
                    if let Some(last) = fields.last_mut() {
                        last.1 = "null".to_string();
                    }
                }
                if has("json.extra") {
                    fields.push(("zz_unexpected".to_string(), "0".to_string()));
                }
                let mut text = render_object(&fields);
                if has("json.parse") {
                    text.pop();
                }
                text
            }
            Spec::Code {
                entry,
                arity,
                source,
                ..
            } => {
                let params: Vec<String> = (0..*arity).map(|i| format!("a{i}")).collect();
                let params = params.join(", ");
                if has("code.syntax") {
                    let mut broken = source.clone();
                    broken.pop();
                    broken
                } else if has("code.timeout") {
                    format!("fn {entry}({params}) {{ while 1 {{ }} return 0; }}")
                } else if has("code.test_failed") {
                    format!("fn {entry}({params}) {{ return 1234567891; }}")
                } else {
                    source.clone()
                }
            }
        }
    }

    fn verifier(&self) -> Inner {
        match &self.spec {
            Spec::Arithmetic { expression } => {
                Inner::Arithmetic(ArithmeticVerifier::new(expression))
            }
            Spec::Json { schema, .. } => Inner::Json(JsonSchemaVerifier::new(schema)),
            Spec::Code { entry, tests, .. } => Inner::Code(CodeTestVerifier::new(CodeTask {
                entry: entry.clone(),
                tests: tests
                    .iter()
                    .map(|(id, args, expected)| CodeTest::new(id, args, *expected))
                    .collect(),
            })),
        }
    }

    /// A short digest of the task, for the tests' determinism checks.
    #[cfg(test)]
    pub fn digest(&self) -> Digest {
        sha256(format!("{:?}|{}|{}", self.kind, self.index, self.reference).as_bytes())
    }
}

fn render_object(fields: &[(String, String)]) -> String {
    let body: Vec<String> = fields
        .iter()
        .map(|(key, value)| format!("\"{key}\":{value}"))
        .collect();
    format!("{{{}}}", body.join(","))
}

thread_local! {
    static CURRENT: RefCell<Option<Arc<Task>>> = const { RefCell::new(None) };
}

/// Run `f` with `task` as the task the simulated experts draw clean candidates from. The
/// experts are built once per lane and are handed the task through this scope, because
/// [`DefectPayload::clean`] receives only a string.
pub fn with_task<R>(task: &Arc<Task>, f: impl FnOnce() -> R) -> R {
    CURRENT.with(|current| *current.borrow_mut() = Some(Arc::clone(task)));
    let result = f();
    CURRENT.with(|current| *current.borrow_mut() = None);
    result
}

/// A candidate: a task and the defect codes it currently carries.
#[derive(Clone, Debug)]
pub struct Served {
    task: Arc<Task>,
    defects: BTreeSet<String>,
}

impl Served {
    /// The text the real verifier judges.
    pub fn render(&self) -> String {
        self.task.render(&self.defects)
    }

    /// The independent oracle: no defect planted and the text is the clean answer.
    pub fn is_clean(&self) -> bool {
        self.defects.is_empty() && self.render() == self.task.reference
    }
}

impl VcwPayload for Served {
    fn payload_type(&self) -> ptr_types::TypeId {
        ptr_types::TypeId::from(self.task.kind.answer_type())
    }

    fn payload_digest(&self) -> Digest {
        let mut bytes = self.task.kind.answer_type().as_bytes().to_vec();
        bytes.push(0);
        bytes.extend_from_slice(self.render().as_bytes());
        sha256(&bytes)
    }

    fn confined_to(&self, edited: &Self, named: &[String]) -> bool {
        let rest = |doc: &Self| -> BTreeSet<String> {
            doc.defects
                .iter()
                .filter(|code| !named.contains(code))
                .cloned()
                .collect()
        };
        self.task.index == edited.task.index && rest(self) == rest(edited)
    }
}

impl DefectPayload for Served {
    /// The task comes from [`with_task`]; the string is only a label.
    fn clean(_task: &str) -> Self {
        let task = CURRENT.with(|current| current.borrow().clone());
        Self {
            task: task.expect("a request runs inside with_task"),
            defects: BTreeSet::new(),
        }
    }

    fn defects(&self) -> BTreeSet<String> {
        self.defects.clone()
    }

    fn with_defect(&self, code: &str) -> Self {
        let mut next = self.clone();
        next.defects.insert(code.to_owned());
        next
    }

    fn without_defect(&self, code: &str) -> Self {
        let mut next = self.clone();
        next.defects.remove(code);
        next
    }
}

enum Inner {
    Arithmetic(ArithmeticVerifier),
    Json(JsonSchemaVerifier),
    Code(CodeTestVerifier),
}

/// What the boundary around the verifier saw of one call.
#[derive(Clone, Debug)]
pub struct VerifyEvent {
    pub digest: Digest,
    /// The independent oracle's verdict: the candidate is the clean answer.
    pub clean: bool,
    pub status: VerificationStatus,
    pub level: VerificationLevel,
    pub hard_codes: Vec<String>,
    pub any_hard: bool,
    pub planted: BTreeSet<String>,
}

impl VerifyEvent {
    /// The harness's own restatement of acceptance: Pass, no hard finding, Deterministic level.
    pub fn accepts(&self) -> bool {
        self.status == VerificationStatus::Pass
            && !self.any_hard
            && self.level == VerificationLevel::Deterministic
    }
}

/// The real verifier of one task, with a log of every call it answered.
pub struct Judge {
    inner: Inner,
    events: RefCell<Vec<VerifyEvent>>,
}

impl Judge {
    pub fn new(task: &Task) -> Self {
        Self {
            inner: task.verifier(),
            events: RefCell::new(Vec::new()),
        }
    }

    fn report_for(&self, text: &str) -> VerificationReport {
        let owned = text.to_string();
        match &self.inner {
            Inner::Arithmetic(verifier) => verifier.verify(&owned),
            Inner::Json(verifier) => verifier.verify(&owned),
            Inner::Code(verifier) => verifier.verify(&owned),
        }
    }

    /// The calls answered since the last take, in order.
    pub fn take_events(&self) -> Vec<VerifyEvent> {
        std::mem::take(&mut *self.events.borrow_mut())
    }

    /// A second, fresh verifier instance and the oracle agree that `served` is correct.
    pub fn independent_pass(task: &Task, served: &Served) -> bool {
        let fresh = Judge::new(task);
        let report = fresh.report_for(&served.render());
        served.is_clean()
            && report.status == VerificationStatus::Pass
            && !report.findings.iter().any(|finding| finding.hard)
            && report.level == VerificationLevel::Deterministic
    }
}

impl Verifier<Served> for Judge {
    fn verify(&self, candidate: &Served) -> VerificationReport {
        let report = self.report_for(&candidate.render());
        self.events.borrow_mut().push(VerifyEvent {
            digest: candidate.payload_digest(),
            clean: candidate.is_clean(),
            status: report.status,
            level: report.level,
            hard_codes: report
                .findings
                .iter()
                .filter(|finding| finding.hard)
                .map(|finding| finding.code.clone())
                .collect(),
            any_hard: report.findings.iter().any(|finding| finding.hard),
            planted: candidate.defects.clone(),
        });
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn served(task: &Arc<Task>, defects: &[&str]) -> Served {
        Served {
            task: Arc::clone(task),
            defects: defects.iter().map(|code| (*code).to_string()).collect(),
        }
    }

    #[test]
    fn a_clean_candidate_passes_the_real_verifier_for_every_kind_and_many_tasks() {
        for kind in KINDS {
            for index in 0..60 {
                let task = Arc::new(Task::generate(1001, index, kind));
                let judge = Judge::new(&task);
                let clean = served(&task, &[]);
                let report = judge.verify(&clean);
                assert_eq!(report.status, VerificationStatus::Pass, "{kind:?} {index}");
                assert!(Judge::independent_pass(&task, &clean));
            }
        }
    }

    #[test]
    fn each_single_defect_is_reported_as_exactly_that_hard_code() {
        for kind in KINDS {
            for code in kind.codes() {
                for index in 0..20 {
                    let task = Arc::new(Task::generate(7, index, kind));
                    let judge = Judge::new(&task);
                    let defective = served(&task, &[code]);
                    let report = judge.verify(&defective);
                    assert_eq!(report.status, VerificationStatus::Fail, "{code} {index}");
                    let events = judge.take_events();
                    let mut codes = events[0].hard_codes.clone();
                    codes.dedup();
                    assert_eq!(codes, vec![(*code).to_string()], "{code} {index}");
                    assert!(!events[0].clean);
                }
            }
        }
    }

    #[test]
    fn reported_codes_are_always_a_nonempty_subset_of_the_planted_ones() {
        for kind in KINDS {
            let codes = kind.codes();
            for mask in 1..(1_u32 << codes.len()) {
                let planted: Vec<&str> = codes
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask & (1 << i) != 0)
                    .map(|(_, c)| *c)
                    .collect();
                let task = Arc::new(Task::generate(3, u64::from(mask), kind));
                let judge = Judge::new(&task);
                judge.verify(&served(&task, &planted));
                let event = &judge.take_events()[0];
                assert!(!event.hard_codes.is_empty(), "{planted:?}");
                for code in &event.hard_codes {
                    assert!(planted.contains(&code.as_str()), "{planted:?} -> {code}");
                }
            }
        }
    }

    #[test]
    fn tasks_are_a_pure_function_of_seed_index_and_kind() {
        for kind in KINDS {
            let a = Task::generate(17, 5, kind);
            let b = Task::generate(17, 5, kind);
            assert_eq!(a.digest(), b.digest());
            assert_ne!(a.digest(), Task::generate(17, 6, kind).digest());
            assert_ne!(a.digest(), Task::generate(29, 5, kind).digest());
        }
    }

    #[test]
    fn the_defect_payload_clears_and_plants_and_is_confined_to_the_named_codes() {
        let task = Arc::new(Task::generate(5, 1, Kind::Json));
        let doc = served(&task, &["json.type", "json.extra"]);
        let fixed = doc.without_defect("json.type");
        assert!(doc.confined_to(&fixed, &["json.type".to_string()]));
        assert!(!doc.confined_to(&fixed, &["json.extra".to_string()]));
        assert_ne!(doc.payload_digest(), fixed.payload_digest());
        let again = with_task(&task, || Served::clean("label"));
        assert!(again.is_clean());
    }
}
