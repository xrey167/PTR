//! A refused egress is committed history: it replays, projects, and holds no payload.
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_runtime::egress::EgressRefusalRecord;
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{
    EgressRefusalReason, Generation, NamespaceId, PodAddress, PodId, PodRevisionAddress, ProjectId,
    TraceId,
};

fn source() -> PodRevisionAddress {
    PodRevisionAddress {
        address: PodAddress::new(
            ProjectId::from("project"),
            NamespaceId::from("ns"),
            PodId::from("caller"),
        )
        .unwrap(),
        semantic_revision: [5; 32],
        generation: Generation(2),
    }
}

fn record(target: &str, reason: EgressRefusalReason) -> EgressRefusalRecord {
    EgressRefusalRecord {
        trace_id: TraceId::from("trace-egress"),
        source: source(),
        target: target.into(),
        reason,
    }
}

#[test]
fn a_refusal_is_committed_projected_and_replayed() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let index = runtime
        .record_egress_refusal(record(
            "broker.example:1883",
            EgressRefusalReason::HostNotAllowed,
        ))
        .unwrap();

    let committed = runtime
        .committed_events()
        .iter()
        .find(|committed| committed.index == index)
        .unwrap();
    assert!(matches!(
        &committed.event,
        LedgerEvent::EgressRefused { target, reason, .. }
            if target == "broker.example:1883" && *reason == EgressRefusalReason::HostNotAllowed
    ));

    let key = ptr_state::egress_refusal_key(&source().address);
    let expected = ptr_state::egress_refusal_entry(
        index.0,
        EgressRefusalReason::HostNotAllowed,
        "broker.example:1883",
    );
    assert_eq!(
        runtime.materialized_state().values.get(&key),
        Some(&expected),
        "the refusal appears in the projection"
    );

    let replayed = PtrRuntime::replay(PtrConfig::default(), runtime.committed_events()).unwrap();
    assert_eq!(
        replayed.materialized_state().values.get(&key),
        Some(&expected),
        "and survives a replay of the committed history"
    );
}

#[test]
fn the_projection_keeps_the_latest_refusal_per_pod_so_a_flood_cannot_grow_it() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .record_egress_refusal(record("a.example:1", EgressRefusalReason::HostNotAllowed))
        .unwrap();
    let before = runtime.materialized_state().values.len();
    let mut last = None;
    for attempt in 0..50 {
        last = Some(
            runtime
                .record_egress_refusal(record(
                    &format!("h{attempt}.example:1"),
                    EgressRefusalReason::RateLimited,
                ))
                .unwrap(),
        );
    }
    assert_eq!(runtime.materialized_state().values.len(), before);
    assert_eq!(
        runtime
            .materialized_state()
            .values
            .get(&ptr_state::egress_refusal_key(&source().address)),
        Some(&ptr_state::egress_refusal_entry(
            last.unwrap().0,
            EgressRefusalReason::RateLimited,
            "h49.example:1"
        ))
    );
}

#[test]
fn a_record_that_could_carry_a_payload_is_refused_before_anything_is_appended() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let before = runtime.committed_events().len();
    for target in ["", "host: 80", "line\nbreak", &"x".repeat(257)] {
        let refused =
            runtime.record_egress_refusal(record(target, EgressRefusalReason::InvalidTopic));
        assert!(
            matches!(refused, Err(RuntimeError::Ledger(_))),
            "{target:?}: {refused:?}"
        );
    }
    let mut no_trace = record("host:1", EgressRefusalReason::HostNotAllowed);
    no_trace.trace_id = TraceId::from("");
    assert!(matches!(
        runtime.record_egress_refusal(no_trace),
        Err(RuntimeError::Ledger(_))
    ));
    assert_eq!(runtime.committed_events().len(), before);
}

#[test]
fn a_generic_commit_cannot_forge_a_refusal() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let forged = LedgerEvent::EgressRefused {
        trace_id: TraceId::from("trace-egress"),
        source: source(),
        target: "host:1".into(),
        reason: EgressRefusalReason::HostNotAllowed,
    };
    assert!(matches!(
        runtime.commit(forged),
        Err(RuntimeError::StructuralRecordOutsideAdmissionPath)
    ));
}
