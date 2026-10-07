//! The cases C1 to C9 of the draft preregistration, each driving the PUBLIC runtime API on a
//! [`fixture::Rt`] (a `PtrRuntime` over a durable `FileLedger` in a temporary directory), and
//! the counting they share.
//!
//! Counting rules:
//!
//! * An **attempt** counts only after the taint record is persisted (a taint record is in the
//!   ledger: [`fixture::Rt::persisted`]) and before any cold exact recomputation of the same
//!   recipe has completed in the case (`Cx::recomputed`): the attempt definition of the
//!   draft. An attempt made outside that window is executed but counted under
//!   `attempts_outside_window` (internal), never as an attempt.
//! * A **violation** counter (`*_admitted`, `*_promotions`, `taint_lost`, ...) counts what the
//!   runtime did wrong; the sub-case that produced it is recorded under `subcases`, together
//!   with whether that sub-case measures behaviour WP4b of the runtime is expected to change.
//! * A refusal that is not the expected typed error, or a fixture that does not build, is a
//!   harness error (`Err`, exit status 2), never a pass.

pub mod action;
pub mod clearing;
pub mod fixture;
pub mod raw;
pub mod semdb;

use super::params;
use super::stub::{self, Reuse, StubRun};
use crate::experiments::layered_kv::corpus;
use crate::experiments::layered_kv::emit::{Json, Tally};
use crate::experiments::layered_kv::lab::{self, fail, Res};
use crate::experiments::layered_kv::{pins, Scale};
use fixture::{Probe, Rt};
use ptr_model_api::{ByteTokenizer, PromptSegment, SegmentedModelRequest, Tokenizer};
use ptr_runtime::{
    DerivedRun, DerivedRunRequest, PtrRuntime, ReferenceKvReuseBackend, RuntimeError,
};
use ptr_types::{ActionIr, RequestId, Revision, SemanticContext};
use std::collections::BTreeMap;

/// The counters that count what the runtime did wrong (the falsifier counters).
pub const VIOLATIONS: [&str; 8] = [
    "approximate_run_actions_admitted",
    "approximate_semdb_promotions",
    "approximate_candidate_exact_admissions",
    "taint_lost",
    "approximate_pod_result_promotions_accepted",
    "misclassified_raw_admissions",
    "laundered_exact_admissions",
    "taint_cleared_by_echo",
];

/// Sub-cases whose outcome depends on behaviour WP4b of the runtime adds (the Content subject
/// of a plain Approximate run, taint through a Pod output's bindings, taint through a reused
/// request id). They measure the runtime as it is; they are listed so a reader can tell which
/// counters a change of the runtime moves.
pub const WP4B_SUBCASES: [&str; 5] = [
    "C1 request id reuse",
    "C4 plain-run text host write",
    "C5 binding to a tainted stored candidate",
    "C8 plain-run output as Raw",
    "C8 plain-run output as Raw (server path)",
];

/// The state of a run: the backend, the tally, the corpus and the case being counted.
pub struct Cx {
    pub backend: ReferenceKvReuseBackend,
    pub tally: Tally,
    pub seed: u64,
    pub scale: Scale,
    prompts: Vec<String>,
    prompt_order: Vec<usize>,
    case: String,
    case_index: usize,
    counts: BTreeMap<String, i64>,
    sub: String,
    subs: BTreeMap<String, (bool, i64)>,
    /// A cold exact recomputation of the case's recipe has completed: later attempts are not
    /// counted.
    pub recomputed: bool,
    pub cases: Vec<(String, Json)>,
    next_id: u64,
}

/// The corpus text of sequence `index`: the tokens mapped to the 94 printable ASCII
/// characters `!`..`~` (so a prompt is valid UTF-8 and its byte tokens are its characters).
pub fn prompt_text(index: u64) -> String {
    corpus::sequence(
        params::CORPUS_GENERATOR_SEED,
        index,
        params::PROMPT_TOKENS as usize,
    )
    .iter()
    .map(|token| char::from(b'!' + (token % 94) as u8))
    .collect()
}

/// The sequences of the corpus selection, in stream order (what the selection digest covers).
pub fn selection_sequences() -> Vec<Vec<u32>> {
    (0..params::SELECTION_SEQUENCES as u64)
        .map(|index| {
            corpus::sequence(
                params::CORPUS_GENERATOR_SEED,
                index,
                params::PROMPT_TOKENS as usize,
            )
        })
        .collect()
}

pub fn selection_digest() -> String {
    corpus::selection_digest(params::SELECTION_DOMAIN, &selection_sequences())
}

impl Cx {
    pub fn new(seed: u64, scale: Scale) -> Res<Self> {
        let mut tally = Tally::new(super::metrics::LIST);
        for name in super::ZERO_BASED {
            tally.add(name, 0);
        }
        Ok(Self {
            backend: lab::backend()?,
            tally,
            seed,
            scale,
            prompts: (0..params::SELECTION_SEQUENCES as u64)
                .map(prompt_text)
                .collect(),
            prompt_order: corpus::order(
                seed,
                params::PROMPT_ORDER_LABEL,
                params::SELECTION_SEQUENCES,
            ),
            case: String::new(),
            case_index: 0,
            counts: BTreeMap::new(),
            sub: String::new(),
            subs: BTreeMap::new(),
            recomputed: false,
            cases: Vec::new(),
            next_id: 0,
        })
    }

    /// Hops of the candidate chain: the pinned count, or fewer at the reduced scale.
    pub fn hops(&self) -> usize {
        match self.scale {
            Scale::Pinned => params::HOPS_PER_CHAIN as usize,
            Scale::Reduced => params::MIN_CANDIDATE_TAINT_HOPS as usize,
        }
    }

    pub fn reduced(&self) -> bool {
        self.scale == Scale::Reduced
    }

    pub fn begin(&mut self, case: &str, index: usize) {
        self.case = case.to_string();
        self.case_index = index;
        self.counts.clear();
        self.recomputed = false;
        self.sub = case.to_string();
    }

    /// Name the sub-case whose violations the next counts belong to.
    pub fn sub(&mut self, label: &str) {
        self.sub = label.to_string();
        self.subs
            .entry(label.to_string())
            .or_insert((WP4B_SUBCASES.contains(&label), 0));
    }

    pub fn end(&mut self) {
        let counts = std::mem::take(&mut self.counts);
        self.cases.push((
            self.case.clone(),
            Json::Obj(
                counts
                    .into_iter()
                    .map(|(name, value)| (name, Json::Int(value)))
                    .collect(),
            ),
        ));
    }

    /// Add `amount` to `name` in the tally and in the current case.
    pub fn count(&mut self, name: &str, amount: i64) {
        self.tally.add(name, amount);
        *self.counts.entry(name.to_string()).or_insert(0) += amount;
        if amount > 0 && VIOLATIONS.contains(&name) {
            let sub = self.sub.clone();
            self.subs
                .entry(sub.clone())
                .or_insert((WP4B_SUBCASES.contains(&sub.as_str()), 0))
                .1 += amount;
        }
    }

    /// One internal counter (not a metric), shown under `internal` and in the case.
    pub fn note(&mut self, name: &str) {
        self.count(name, 1);
    }

    /// Count one attempt of metric `name`, if the attempt definition holds.
    pub fn attempt(&mut self, name: &str, rt: &Rt) {
        if rt.persisted() && !self.recomputed {
            self.count(name, 1);
        } else {
            self.note("attempts_outside_window");
        }
    }

    /// A fresh request id of the current case.
    pub fn rid(&mut self, tag: &str) -> RequestId {
        self.next_id += 1;
        RequestId::from(format!("r012-{}-{tag}-{}", self.case, self.next_id).as_str())
    }

    /// A fresh idempotency key.
    pub fn key(&mut self) -> String {
        self.next_id += 1;
        format!("r012-key-{}", self.next_id)
    }

    /// The prompt of slot `slot` of the current case: a corpus entry the seed assigns to it.
    pub fn prompt(&self, slot: usize) -> String {
        assert!(slot < params::PROMPT_SLOTS_PER_CASE, "prompt slot {slot}");
        let at = (self.case_index * params::PROMPT_SLOTS_PER_CASE + slot) % self.prompts.len();
        self.prompts[self.prompt_order[at]].clone()
    }

    /// The corpus indexes of the case's prompt slots (recorded for the reader).
    pub fn prompt_indexes(&self, case_index: usize) -> Vec<i64> {
        (0..params::PROMPT_SLOTS_PER_CASE)
            .map(|slot| {
                let at = (case_index * params::PROMPT_SLOTS_PER_CASE + slot) % self.prompts.len();
                self.prompt_order[at] as i64
            })
            .collect()
    }

    /// A pinned procedure parameter observed in this case differs from the pin.
    pub fn procedure_mismatch(&mut self, observed_ok: bool) {
        if !observed_ok {
            self.count("procedure_mismatches", 1);
        }
    }

    /// The shape of the backend's stack against the pinned values (checked once per case).
    pub fn check_shape(&mut self) {
        let config = self.backend.stack_config().clone();
        let ok = config.num_layers as u64 == params::TEST_MODEL_LAYERS
            && config.layer.hidden_size as u64 == params::HIDDEN_SIZE
            && config.layer.num_attention_heads as u64 == params::NUM_HEADS
            && config.layer.num_key_value_heads as u64 == params::NUM_KV_HEADS
            && (config.layer.hidden_size / config.layer.num_attention_heads) as u64
                == params::HEAD_DIM
            && config.vocab_size as u64 == params::VOCAB_SIZE
            && config.layer.intermediate_size as u64 == params::INTERMEDIATE_SIZE;
        self.procedure_mismatch(ok);
    }

    /// A request the harness builds from the corpus has the pinned prompt size.
    pub fn check_prompt(&mut self, request: &SegmentedModelRequest) {
        self.procedure_mismatch(request.total_tokens() as u64 == params::PROMPT_TOKENS);
    }

    /// A stub run generated the pinned number of tokens.
    pub fn check_run(&mut self, run: &StubRun) {
        self.procedure_mismatch(run.outcome.generated.len() as u64 == params::MAX_NEW_TOKENS);
    }

    /// The violations of each sub-case, for the result.
    pub fn subcases_json(&self) -> Json {
        Json::Obj(
            self.subs
                .iter()
                .map(|(label, (wp4b, violations))| {
                    (
                        label.clone(),
                        Json::obj([
                            ("wp4b_dependent", Json::Bool(*wp4b)),
                            ("violations", Json::Int(*violations)),
                        ]),
                    )
                })
                .collect(),
        )
    }
}

// ---- requests, runs and refusals --------------------------------------------------------------

pub fn raw_segment(text: &str) -> Res<PromptSegment> {
    PromptSegment::raw_from_text(&ByteTokenizer::new(), text).map_err(fail("a Raw segment"))
}

pub fn segmented(id: &RequestId, segments: Vec<PromptSegment>) -> Res<SegmentedModelRequest> {
    SegmentedModelRequest::new(
        id.clone(),
        SemanticContext {
            revision: Revision(0),
            entries: Vec::new(),
        },
        ByteTokenizer::new().identity().clone(),
        segments,
    )
    .map_err(fail("a segmented request"))
}

/// Ingest `text` as the request's own ingress record and return the request of that one Raw
/// segment.
pub fn own_request(
    runtime: &mut PtrRuntime,
    id: &RequestId,
    text: &str,
) -> Res<SegmentedModelRequest> {
    runtime
        .ingest_text(id.clone(), text)
        .map_err(fail("ingesting request text"))?;
    segmented(id, vec![raw_segment(text)?])
}

/// A request of the request's own Raw text (the first `64 - derived` characters of `prompt`)
/// followed by `derived`, so every request the harness builds has the pinned 64 tokens.
pub fn hop_request(
    runtime: &mut PtrRuntime,
    id: &RequestId,
    prompt: &str,
    derived: &PromptSegment,
) -> Res<SegmentedModelRequest> {
    let own_len = (params::PROMPT_TOKENS as usize).saturating_sub(derived.tokens().len());
    if own_len == 0 {
        return Err("a derived segment fills the whole prompt".to_string());
    }
    let own = &prompt[..own_len];
    runtime
        .ingest_text(id.clone(), own)
        .map_err(fail("ingesting request text"))?;
    segmented(id, vec![raw_segment(own)?, derived.clone()])
}

/// One derivation hop under the stub's identity and parameters.
pub fn derive(
    runtime: &mut PtrRuntime,
    request: &SegmentedModelRequest,
    run: &ptr_model_api::ModelRun,
) -> Result<DerivedRun, RuntimeError> {
    derive_as(runtime, request, run, &stub::identity(), stub::parameters())
}

pub fn derive_as(
    runtime: &mut PtrRuntime,
    request: &SegmentedModelRequest,
    run: &ptr_model_api::ModelRun,
    producer: &str,
    parameters: &[u8],
) -> Result<DerivedRun, RuntimeError> {
    runtime.ingest_derived_run(
        &DerivedRunRequest {
            request_id: request.request_id(),
            request,
            producer,
            generation_parameters: parameters,
        },
        run,
    )
}

/// The outcome of an attempt that the runtime may only refuse with one typed error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Verdict {
    /// Refused with the expected typed error.
    Refused,
    /// The runtime let it through: a violation the caller counts.
    Admitted,
}

/// `Admitted` for an `Ok`, `Refused` for the expected error, a harness error for any other.
pub fn classify<T: std::fmt::Debug>(
    what: &str,
    result: Result<T, RuntimeError>,
    expected: impl Fn(&RuntimeError) -> bool,
) -> Res<Verdict> {
    match result {
        Ok(_) => Ok(Verdict::Admitted),
        Err(error) if expected(&error) => Ok(Verdict::Refused),
        Err(error) => Err(format!(
            "{what}: refused with an unexpected error: {error:?}"
        )),
    }
}

pub fn is_effect_boundary(error: &RuntimeError) -> bool {
    matches!(error, RuntimeError::ApproximateStateAtEffectBoundary { .. })
}

/// A tainted request and what it produced: the base of most cases.
pub struct Base {
    pub request: SegmentedModelRequest,
    pub prompt: String,
    pub run: StubRun,
    pub derived: DerivedRun,
}

impl Base {
    pub fn action(&self) -> &ActionIr {
        &self.run.action
    }

    pub fn text(&self) -> &str {
        &self.run.text
    }
}

/// An Approximate stub run over the case's prompt `slot`, ingested as one derivation hop (the
/// Derivation, Content, Action and Request subjects are recorded before it returns).
pub fn tainted_base(cx: &mut Cx, rt: &mut Rt, slot: usize) -> Res<Base> {
    let prompt = cx.prompt(slot);
    let id = cx.rid("base");
    let request = own_request(&mut rt.runtime, &id, &prompt)?;
    cx.check_prompt(&request);
    let run = stub::produce(&cx.backend, &request, Reuse::Approximate)?;
    cx.check_run(&run);
    let derived = derive(&mut rt.runtime, &request, &run.run)
        .map_err(fail("ingesting the Approximate stub run"))?;
    Ok(Base {
        request,
        prompt,
        run,
        derived,
    })
}

/// The probe and the session of the effect boundary of `rt`, registered now.
pub fn session(rt: &mut Rt) -> Res<(ptr_runtime::execution::ExecutionSession, Probe)> {
    let probe = Probe::default();
    let session = fixture::exec_session(&mut rt.runtime, &probe)?;
    Ok((session, probe))
}

/// Run every case once, in the seed's order.
pub fn run_all(cx: &mut Cx) -> Res<Vec<Json>> {
    type Case = (&'static str, fn(&mut Cx) -> Res<()>);
    let cases: [Case; 9] = [
        ("C1", action::c1_action_admission),
        ("C2", action::c2_restart_and_compaction),
        ("C3", action::c3_candidate_path),
        ("C4", semdb::c4_semdb_promotion),
        ("C5", semdb::c5_every_pod_kind),
        ("C6", clearing::c6_exact_echo),
        ("C7", clearing::c7_fresh_cold_recompute),
        ("C8", raw::c8_raw_laundering),
        ("C9", raw::c9_misclassified_raw),
    ];
    let mut order = Vec::new();
    for index in corpus::order(cx.seed, params::CASE_ORDER_LABEL, cases.len()) {
        let (name, case) = cases[index];
        cx.begin(name, index);
        cx.check_shape();
        case(cx).map_err(|error| format!("{name}: {error}"))?;
        cx.end();
        order.push(Json::str(name));
    }
    Ok(order)
}

/// The pin check that only a stub can fail: it reports Approximate, and its content is the
/// independent cold run's (not the backend's own, which `pins::base_checks` covers).
pub fn stub_pin_checks(checks: &mut pins::PinChecks, backend: &ReferenceKvReuseBackend) -> Res<()> {
    let tokens = corpus::sequence(
        params::CORPUS_GENERATOR_SEED,
        9_997,
        params::PROMPT_TOKENS as usize,
    );
    let request = corpus::raw_request("pin-stub", &tokens);
    let run = stub::produce(backend, &request, Reuse::Approximate)?;
    let stack = pins::reference_stack()?;
    let (generated, logits) = pins::independent_cold_run(&stack, None, &tokens)?;
    let observed: Vec<u32> = run.outcome.generated.iter().map(|token| token.0).collect();
    checks.check(
        "stub_content_is_the_independent_cold_run",
        observed == generated && run.outcome.logits_digest == logits,
        Json::Null,
    );
    checks.check(
        "stub_reports_approximate",
        run.run.kv_report.map(|report| report.mode())
            == Some(ptr_model_api::KvReuseMode::Approximate)
            && run.run.taint().is_approximate(),
        Json::str(stub::identity()),
    );
    checks.check(
        "stub_identity",
        stub::identity() == format!("{}/{}", params::STUB_NAME, params::STUB_VERSION),
        Json::str(stub::identity()),
    );
    Ok(())
}
