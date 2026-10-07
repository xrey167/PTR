//! Derivation hops, segment admission (the Raw rule) and the clearing rule of the approximate-KV
//! taint: Approximate taint follows model output across hops, survives a reopen, is not shed
//! by an echo, a warm recompute or a recompute of another recipe, and is cleared only by a cold
//! exact recomputation of the same recipe from clean inputs. Nothing is appended on a refusal.

use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_ledger::{KvTaintChange, LedgerEvent};
use ptr_model_api::{
    ByteTokenizer, KvReuseMode, KvReusePolicy, KvReuseReport, KvReuseTaint, ModelEvent, ModelRun,
    PromptSegment, SegmentSource, SegmentedModelRequest, Tokenizer as _,
};
use ptr_runtime::{DerivedRun, DerivedRunRequest, PtrRuntime, RuntimeError};
use ptr_types::{
    CapabilityId, CapsuleId, Effect, EvidenceId, Generation, ProjectId, RequestId, Revision,
    SemanticContext, TypeId,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

const PRODUCER: &str = "stub-approximate-producer/1";
const PARAMS: &[u8] = b"max_new_tokens=8";

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ptr-kv-derivation-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        Self(dir)
    }
    fn log(&self) -> PathBuf {
        self.0.join("ledger")
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn action(payload: &[u8]) -> ActionIr {
    ActionIr {
        operation: "write".into(),
        target: "artifact:a".into(),
        capability: CapabilityId::from("file.write"),
        effect: Effect::Mutation,
        input_type: TypeId::from("Bytes"),
        generation: Generation(1),
        revision: Revision(0),
        payload: payload.to_vec(),
    }
}

/// The action as the runtime would stamp it now: revision-independent taint identity, but the
/// hard boundary wants the current revision.
fn stamped(runtime: &PtrRuntime, action: &ActionIr) -> ActionIr {
    ActionIr {
        revision: runtime.revision(),
        ..action.clone()
    }
}

fn grant(runtime: &mut PtrRuntime) {
    runtime
        .permissions_mut()
        .capabilities
        .insert(CapabilityId::from("file.write"));
    runtime.permissions_mut().allow_mutation = true;
}

fn new_runtime(temp: Option<&Temp>) -> PtrRuntime {
    let mut runtime = match temp {
        Some(temp) => PtrRuntime::open_durable(PtrConfig::default(), temp.log()).unwrap(),
        None => PtrRuntime::new(PtrConfig::default()).unwrap(),
    };
    if runtime.committed_events().is_empty() {
        runtime
            .commit(LedgerEvent::CapsuleCommitted {
                project: ProjectId::from("p"),
                capsule: CapsuleId::from("artifact:a"),
                generation: Generation(1),
            })
            .unwrap();
    }
    grant(&mut runtime);
    runtime
}

fn is_denied(runtime: &PtrRuntime, action: &ActionIr) -> bool {
    matches!(
        runtime.authorize_action(&stamped(runtime, action)),
        Err(RuntimeError::ApproximateStateAtEffectBoundary { .. })
    )
}

fn is_authorized(runtime: &PtrRuntime, action: &ActionIr) -> bool {
    runtime.authorize_action(&stamped(runtime, action)).is_ok()
}

fn raw(text: &str) -> PromptSegment {
    PromptSegment::raw_from_text(&ByteTokenizer::new(), text).unwrap()
}

fn request(id: &str, segments: Vec<PromptSegment>) -> SegmentedModelRequest {
    SegmentedModelRequest::new(
        RequestId::from(id),
        SemanticContext {
            revision: Revision(0),
            entries: Vec::new(),
        },
        ByteTokenizer::new().identity().clone(),
        segments,
    )
    .unwrap()
}

/// Ingest `text` as the request's raw text and return the request of that one Raw segment.
fn raw_request(runtime: &mut PtrRuntime, id: &str, text: &str) -> SegmentedModelRequest {
    runtime.ingest_text(RequestId::from(id), text).unwrap();
    request(id, vec![raw(text)])
}

fn run(text: &str, action: Option<&ActionIr>, report: Option<KvReuseMode>) -> ModelRun {
    let mut events = vec![ModelEvent::Token(text.to_owned())];
    events.extend(action.map(|action| ModelEvent::ActionReady(action.clone())));
    events.push(ModelEvent::Finished);
    ModelRun {
        events,
        kv_report: report.map(|mode| {
            let policy = match mode {
                KvReuseMode::Exact => KvReusePolicy::ExactOnly,
                KvReuseMode::Approximate => KvReusePolicy::AllowApproximate,
            };
            KvReuseReport::new(16, mode, 64, policy).unwrap()
        }),
    }
}

fn derive(
    runtime: &mut PtrRuntime,
    request: &SegmentedModelRequest,
    producer: &str,
    parameters: &[u8],
    run: &ModelRun,
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

fn hop(
    runtime: &mut PtrRuntime,
    request: &SegmentedModelRequest,
    run: &ModelRun,
) -> Result<DerivedRun, RuntimeError> {
    derive(runtime, request, PRODUCER, PARAMS, run)
}

fn len(runtime: &PtrRuntime) -> usize {
    runtime.committed_events().len()
}

fn clears(runtime: &PtrRuntime) -> usize {
    runtime
        .committed_events()
        .iter()
        .filter(|committed| {
            matches!(
                committed.event,
                LedgerEvent::KvTaintCommitted {
                    change: KvTaintChange::Clear { .. },
                    ..
                }
            )
        })
        .count()
}

/// A runtime holding one tainted derivation: an Approximate run over `raw("alpha")` that
/// generated "T" and asked for `action(b"a")`.
fn tainted_base() -> (PtrRuntime, ActionIr, DerivedRun) {
    let mut runtime = new_runtime(None);
    let tainted = action(b"a");
    let first = raw_request(&mut runtime, "r1", "alpha");
    let derived = hop(
        &mut runtime,
        &first,
        &run("T", Some(&tainted), Some(KvReuseMode::Approximate)),
    )
    .unwrap();
    assert!(derived.taint.is_approximate());
    assert!(is_denied(&runtime, &tainted));
    (runtime, tainted, derived)
}

fn forged_id(byte: u8) -> String {
    let hex: String = (0..32).map(|_| format!("{byte:02x}")).collect();
    format!("kv-derived:{hex}")
}

#[test]
fn taint_follows_three_hops_and_survives_reopening_between_them() {
    let temp = Temp::new();
    let (a1, a2, a3) = (action(b"a1"), action(b"a2"), action(b"a3"));
    let d2 = {
        let mut runtime = new_runtime(Some(&temp));
        let first = raw_request(&mut runtime, "r1", "alpha");
        // Hop 0: the Approximate run itself.
        let d1 = hop(
            &mut runtime,
            &first,
            &run("T1", Some(&a1), Some(KvReuseMode::Approximate)),
        )
        .unwrap();
        assert!(d1.taint.is_approximate());
        assert!(is_denied(&runtime, &a1));
        // Hop 1: an otherwise Exact, cold run over D1.
        let second = request("r2", vec![raw("beta"), d1.segment.clone()]);
        runtime.ingest_text(RequestId::from("r2"), "beta").unwrap();
        assert!(matches!(
            runtime.admit_segments(second.request_id(), &second),
            Ok(admitted) if admitted.taint.is_approximate()
                && !admitted.segments[0].is_approximate()
                && admitted.segments[1].is_approximate()
        ));
        let d2 = hop(&mut runtime, &second, &run("T2", Some(&a2), None)).unwrap();
        assert!(d2.taint.is_approximate());
        assert_eq!(d2.resumable.kv_taint, d2.taint);
        assert!(is_denied(&runtime, &a2));
        assert_ne!(d1.recipe_digest, d2.recipe_digest);
        d2
    };

    // Restart: the derived segments held by the caller are recognised from the registry.
    let d3 = {
        let mut runtime = new_runtime(Some(&temp));
        assert!(is_denied(&runtime, &a1));
        assert!(is_denied(&runtime, &a2));
        let third = {
            runtime.ingest_text(RequestId::from("r3"), "gamma").unwrap();
            request("r3", vec![raw("gamma"), d2.segment.clone()])
        };
        let d3 = hop(&mut runtime, &third, &run("T3", Some(&a3), None)).unwrap();
        assert!(d3.taint.is_approximate());
        assert!(is_denied(&runtime, &a3));
        d3
    };

    // And once more after a second restart; the earlier hops are still refused.
    let mut runtime = new_runtime(Some(&temp));
    for tainted in [&a1, &a2, &a3] {
        assert!(is_denied(&runtime, tainted));
    }
    runtime.ingest_text(RequestId::from("r4"), "delta").unwrap();
    let fourth = request("r4", vec![raw("delta"), d3.segment.clone()]);
    let a4 = action(b"a4");
    let d4 = hop(&mut runtime, &fourth, &run("T4", Some(&a4), None)).unwrap();
    assert!(d4.taint.is_approximate());
    assert!(is_denied(&runtime, &a4));
}

#[test]
fn a_clean_derivation_is_recognised_in_memory_and_refused_after_a_restart() {
    let temp = Temp::new();
    let clean = action(b"clean");
    let derived = {
        let mut runtime = new_runtime(Some(&temp));
        let first = raw_request(&mut runtime, "r1", "alpha");
        let before = len(&runtime);
        let derived = hop(&mut runtime, &first, &run("T", Some(&clean), None)).unwrap();
        assert_eq!(derived.taint, KvReuseTaint::Exact);
        assert_eq!(len(&runtime), before, "an Exact cold run appends nothing");
        assert!(is_authorized(&runtime, &clean));
        let next = request("r2", vec![derived.segment.clone()]);
        let admitted = runtime.admit_segments(next.request_id(), &next).unwrap();
        assert_eq!(admitted.taint, KvReuseTaint::Exact);
        derived
    };
    let runtime = new_runtime(Some(&temp));
    let next = request("r2", vec![derived.segment]);
    assert_eq!(
        runtime.admit_segments(next.request_id(), &next),
        Err(RuntimeError::UnknownDerivedSegment { segment: 0 })
    );
}

#[test]
fn a_cold_exact_recompute_of_the_same_recipe_clears_and_authorizes_the_action() {
    let (mut runtime, tainted, derived) = tainted_base();
    let other = raw_request(&mut runtime, "r1-recompute", "alpha");
    let before = len(&runtime);
    let recomputed = hop(&mut runtime, &other, &run("T", Some(&tainted), None)).unwrap();
    assert_eq!(recomputed.taint, KvReuseTaint::Exact);
    assert_eq!(recomputed.recipe_digest, derived.recipe_digest);
    assert_eq!(recomputed.segment, derived.segment, "byte-identical output");
    assert_eq!(len(&runtime), before + 1, "exactly one Clear record");
    assert_eq!(clears(&runtime), 1);
    assert!(is_authorized(&runtime, &tainted));
    assert_eq!(runtime.action_kv_taint(&tainted), KvReuseTaint::Exact);
    // The tainted derived segment is clean now, and so is its text.
    let next = request("r3", vec![derived.segment.clone()]);
    assert_eq!(
        runtime
            .admit_segments(next.request_id(), &next)
            .unwrap()
            .taint,
        KvReuseTaint::Exact
    );
    runtime.ingest_text(RequestId::from("r3"), "T").unwrap();
    let text = request("r3", vec![raw("T")]);
    assert_eq!(
        runtime
            .admit_segments(text.request_id(), &text)
            .unwrap()
            .taint,
        KvReuseTaint::Exact
    );
}

#[test]
fn a_recompute_with_other_bytes_clears_the_derivation_but_not_the_content() {
    let (mut runtime, tainted, derived) = tainted_base();
    let other = raw_request(&mut runtime, "r1-recompute", "alpha");
    // Same recipe, another generated text: the Derivation subject is cleared, the Action subject
    // (same action bytes) is cleared, but the Content subject of the old text is not.
    let recomputed = hop(&mut runtime, &other, &run("U", Some(&tainted), None)).unwrap();
    assert_eq!(recomputed.recipe_digest, derived.recipe_digest);
    assert_eq!(clears(&runtime), 1);
    assert!(is_authorized(&runtime, &tainted));
    runtime.ingest_text(RequestId::from("r3"), "T").unwrap();
    let old_text = request("r3", vec![raw("T")]);
    assert!(runtime
        .admit_segments(old_text.request_id(), &old_text)
        .unwrap()
        .taint
        .is_approximate());
}

#[test]
fn a_recompute_with_other_bytes_for_the_action_leaves_the_action_tainted() {
    let (mut runtime, tainted, _) = tainted_base();
    let other = raw_request(&mut runtime, "r1-recompute", "alpha");
    let different = action(b"a-different");
    hop(&mut runtime, &other, &run("T", Some(&different), None)).unwrap();
    assert_eq!(clears(&runtime), 1);
    assert!(
        is_denied(&runtime, &tainted),
        "the recorded action bytes differ"
    );
    assert!(is_authorized(&runtime, &different));
}

#[test]
fn a_recompute_of_another_recipe_does_not_clear() {
    let (mut runtime, tainted, _) = tainted_base();
    // Different input.
    let other_input = raw_request(&mut runtime, "r-other-input", "alpha, changed");
    let before = len(&runtime);
    let derived = hop(&mut runtime, &other_input, &run("T", Some(&tainted), None)).unwrap();
    assert_eq!(derived.taint, KvReuseTaint::Exact);
    // Different producer.
    let same_input = raw_request(&mut runtime, "r-other-producer", "alpha");
    let before_producer = len(&runtime);
    derive(
        &mut runtime,
        &same_input,
        "another-producer/1",
        PARAMS,
        &run("T", Some(&tainted), None),
    )
    .unwrap();
    assert_eq!(len(&runtime), before_producer);
    // Different generation parameters.
    let same_again = raw_request(&mut runtime, "r-other-params", "alpha");
    let before_params = len(&runtime);
    derive(
        &mut runtime,
        &same_again,
        PRODUCER,
        b"max_new_tokens=16",
        &run("T", Some(&tainted), None),
    )
    .unwrap();
    assert_eq!(len(&runtime), before_params);
    assert!(before_params >= before);
    assert_eq!(clears(&runtime), 0);
    assert!(is_denied(&runtime, &tainted));
}

#[test]
fn a_warm_exact_recompute_does_not_clear() {
    let (mut runtime, tainted, derived) = tainted_base();
    let other = raw_request(&mut runtime, "r1-recompute", "alpha");
    let before = len(&runtime);
    let warm = hop(
        &mut runtime,
        &other,
        &run("T", Some(&tainted), Some(KvReuseMode::Exact)),
    )
    .unwrap();
    assert_eq!(warm.taint, KvReuseTaint::Exact);
    assert_eq!(warm.recipe_digest, derived.recipe_digest);
    assert_eq!(len(&runtime), before);
    assert_eq!(clears(&runtime), 0);
    assert!(is_denied(&runtime, &tainted));
}

#[test]
fn an_exact_echo_of_the_tainted_segment_does_not_clear() {
    let (mut runtime, tainted, derived) = tainted_base();
    runtime
        .ingest_text(RequestId::from("r-echo"), "beta")
        .unwrap();
    let echo_request = request("r-echo", vec![raw("beta"), derived.segment.clone()]);
    // A cold Exact pass that reproduces the same text and the same action.
    let echo = hop(&mut runtime, &echo_request, &run("T", Some(&tainted), None)).unwrap();
    assert!(
        echo.taint.is_approximate(),
        "its input taint is Approximate"
    );
    assert_ne!(echo.recipe_digest, derived.recipe_digest);
    assert_eq!(clears(&runtime), 0, "taint_cleared_by_echo == 0");
    assert!(is_denied(&runtime, &tainted));
    // The echo's own derivation is tainted too.
    let again = request("r-echo2", vec![echo.segment.clone()]);
    assert!(runtime
        .admit_segments(again.request_id(), &again)
        .unwrap()
        .taint
        .is_approximate());
}

#[test]
fn raw_is_admitted_only_as_the_requests_own_ingress_text() {
    let mut runtime = new_runtime(None);
    runtime.ingest_text(RequestId::from("r1"), "alpha").unwrap();
    runtime.ingest_text(RequestId::from("r2"), "beta").unwrap();
    let before = len(&runtime);

    let own = request("r1", vec![raw("alpha")]);
    let admitted = runtime.admit_segments(own.request_id(), &own).unwrap();
    assert_eq!(admitted.taint, KvReuseTaint::Exact);
    assert_eq!(admitted.segments, vec![KvReuseTaint::Exact]);

    // Text that is no ingress record at all.
    let invented = request("r1", vec![raw("alpha"), raw("something else")]);
    assert!(matches!(
        runtime.admit_segments(invented.request_id(), &invented),
        Err(RuntimeError::MisclassifiedRawSegment { segment: 1, .. })
    ));
    // Text that is another request's ingress record, presented as Raw.
    let foreign = request("r1", vec![raw("beta")]);
    assert!(matches!(
        runtime.admit_segments(foreign.request_id(), &foreign),
        Err(RuntimeError::MisclassifiedRawSegment { segment: 0, .. })
    ));
    // A request that was never ingested has no ingress record to equal.
    let unknown = request("never-ingested", vec![raw("alpha")]);
    assert!(matches!(
        runtime.admit_segments(unknown.request_id(), &unknown),
        Err(RuntimeError::MisclassifiedRawSegment { segment: 0, .. })
    ));
    // The same refusal reaches ingest_derived_run, with nothing appended.
    assert!(matches!(
        hop(&mut runtime, &foreign, &run("T", None, None)),
        Err(RuntimeError::MisclassifiedRawSegment { segment: 0, .. })
    ));
    assert_eq!(len(&runtime), before);
}

#[test]
fn tainted_output_resubmitted_as_the_requests_own_raw_text_keeps_its_taint() {
    let (mut runtime, tainted, derived) = tainted_base();
    // The tainted text "T" arrives as a new request's own raw text.
    let laundered = raw_request(&mut runtime, "r-launder", "T");
    let admitted = runtime
        .admit_segments(laundered.request_id(), &laundered)
        .unwrap();
    assert!(admitted.taint.is_approximate(), "admitted, but tainted");
    let laundered_action = action(b"laundered");
    let hopped = hop(
        &mut runtime,
        &laundered,
        &run("L", Some(&laundered_action), None),
    )
    .unwrap();
    assert!(hopped.taint.is_approximate());
    assert!(is_denied(&runtime, &laundered_action));
    // A second hop on top of the laundered one stays tainted.
    runtime
        .ingest_text(RequestId::from("r-launder2"), "x")
        .unwrap();
    let again = request("r-launder2", vec![raw("x"), hopped.segment.clone()]);
    let a2 = action(b"laundered-2");
    assert!(hop(&mut runtime, &again, &run("M", Some(&a2), None))
        .unwrap()
        .taint
        .is_approximate());
    assert!(is_denied(&runtime, &a2));
    // The original stays refused, and a text with no taint match is not tainted.
    assert!(is_denied(&runtime, &tainted));
    let clean = raw_request(&mut runtime, "r-clean", "T ");
    assert_eq!(
        runtime
            .admit_segments(clean.request_id(), &clean)
            .unwrap()
            .taint,
        KvReuseTaint::Exact,
        "whitespace edit evades the exact digest: stated limit"
    );
    let _ = derived;
}

#[test]
fn the_server_path_keeps_laundering_protection_without_a_server_change() {
    let (mut runtime, _, _) = tainted_base();
    let launder = action(b"via-resumable");
    let resumable = runtime
        .run_model_run_resumable(RequestId::from("r-resumable"), "T", |context| {
            let stamped = ActionIr {
                revision: context.revision,
                ..launder.clone()
            };
            Ok(run("whatever", Some(&stamped), None))
        })
        .unwrap();
    assert!(resumable.kv_taint.is_approximate());
    assert!(is_denied(&runtime, &launder));
    // Untainted request text stays Exact and authorized.
    let honest = action(b"honest");
    let resumable = runtime
        .run_model_run_resumable(RequestId::from("r-honest"), "plain text", |context| {
            let stamped = ActionIr {
                revision: context.revision,
                ..honest.clone()
            };
            Ok(run("ok", Some(&stamped), None))
        })
        .unwrap();
    assert_eq!(resumable.kv_taint, KvReuseTaint::Exact);
    assert!(is_authorized(&runtime, &honest));
}

#[test]
fn a_forged_or_foreign_derived_segment_is_refused() {
    let (mut runtime, _, derived) = tainted_base();
    let tokenizer = ByteTokenizer::new();
    let forge = |id: String, generation: u64, text: &str| {
        PromptSegment::from_text(
            &tokenizer,
            SegmentSource::Derived {
                id: EvidenceId(id),
                generation: Generation(generation),
            },
            text,
        )
        .unwrap()
    };
    runtime.ingest_text(RequestId::from("r2"), "beta").unwrap();
    let before = len(&runtime);
    for forged in [
        forge(forged_id(0x11), 0, "T"),
        forge("kv-derived:short".into(), 0, "T"),
        forge("evidence-1".into(), 0, "T"),
    ] {
        let forged_request = request("r2", vec![raw("beta"), forged]);
        assert_eq!(
            runtime.admit_segments(forged_request.request_id(), &forged_request),
            Err(RuntimeError::UnknownDerivedSegment { segment: 1 })
        );
        assert_eq!(
            hop(&mut runtime, &forged_request, &run("X", None, None)),
            Err(RuntimeError::UnknownDerivedSegment { segment: 1 })
        );
    }
    assert_eq!(len(&runtime), before);

    // A clean minted id carrying other content than was minted under it is forged too.
    let clean_runtime = &mut new_runtime(None);
    let first = raw_request(clean_runtime, "r1", "alpha");
    let clean = hop(clean_runtime, &first, &run("T", None, None)).unwrap();
    let SegmentSource::Derived { id, .. } = clean.segment.source() else {
        panic!("a derived run mints a Derived segment");
    };
    let swapped = forge(id.0.clone(), 0, "not what was minted");
    let swapped_request = request("r1", vec![raw("alpha"), swapped]);
    assert_eq!(
        clean_runtime.admit_segments(swapped_request.request_id(), &swapped_request),
        Err(RuntimeError::UnknownDerivedSegment { segment: 1 })
    );
    // Another generation than the minted one is refused as well.
    let regen = forge(id.0.clone(), 7, "T");
    let regen_request = request("r1", vec![raw("alpha"), regen]);
    assert_eq!(
        clean_runtime.admit_segments(regen_request.request_id(), &regen_request),
        Err(RuntimeError::UnknownDerivedSegment { segment: 1 })
    );
    let _ = derived;
}

#[test]
fn a_run_that_generated_no_text_mints_nothing_and_appends_nothing() {
    let mut runtime = new_runtime(None);
    let first = raw_request(&mut runtime, "r1", "alpha");
    let before = len(&runtime);
    let silent = ModelRun {
        events: vec![
            ModelEvent::ActionReady(action(b"silent")),
            ModelEvent::Finished,
        ],
        kv_report: Some(
            KvReuseReport::new(
                16,
                KvReuseMode::Approximate,
                64,
                KvReusePolicy::AllowApproximate,
            )
            .unwrap(),
        ),
    };
    assert!(matches!(
        hop(&mut runtime, &first, &silent),
        Err(RuntimeError::DerivedRunUnavailable { .. })
    ));
    assert_eq!(len(&runtime), before);
    assert!(is_authorized(&runtime, &action(b"silent")));
}

#[test]
fn a_run_with_two_control_events_is_refused_before_anything_is_recorded() {
    let mut runtime = new_runtime(None);
    let first = raw_request(&mut runtime, "r1", "alpha");
    let before = len(&runtime);
    let mut doubled = run("T", Some(&action(b"one")), Some(KvReuseMode::Approximate));
    doubled
        .events
        .insert(1, ModelEvent::ActionReady(action(b"two")));
    assert!(matches!(
        hop(&mut runtime, &first, &doubled),
        Err(RuntimeError::MultipleControlEvents { count: 2 })
    ));
    assert_eq!(len(&runtime), before);
}
