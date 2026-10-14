//! A refused egress is committed history: counted, bounded, replayed, projected, and holding
//! no payload.
use ptr_config::PtrConfig;
use ptr_ledger::{EgressRefusalSource, LedgerEvent};
use ptr_pods::{NativeProtocolResponse, ProtocolBinding, ProtocolError};
use ptr_runtime::egress::{
    EgressRefusalOutcome, EgressRefusalRecord, EGRESS_WINDOW_MILLIS,
    MAX_EGRESS_EVENTS_PER_POD_WINDOW, MAX_EGRESS_KEYS_PER_POD, OVERFLOW_TARGET,
};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{
    EgressRefusalReason, Generation, NamespaceId, PodAddress, PodId, PodRevisionAddress, ProjectId,
    RequestId, Timestamp, TraceId, TypeId,
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

fn record(target: &str, reason: EgressRefusalReason, at: u64) -> EgressRefusalRecord {
    EgressRefusalRecord {
        trace_id: TraceId::from("trace-egress"),
        source: source(),
        target: target.into(),
        reason,
        observed_at: Timestamp(at),
    }
}

fn refusals(runtime: &PtrRuntime) -> Vec<(String, EgressRefusalReason, u32)> {
    runtime
        .committed_events()
        .iter()
        .filter_map(|committed| match &committed.event {
            LedgerEvent::EgressRefused {
                target,
                reason,
                count,
                ..
            } => Some((target.clone(), *reason, *count)),
            _ => None,
        })
        .collect()
}

#[test]
fn a_refusal_is_counted_flushed_projected_and_replayed() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    assert_eq!(
        runtime
            .record_egress_refusal(record(
                "host:0123456789abcdef:14:1883",
                EgressRefusalReason::HostNotAllowed,
                10,
            ))
            .unwrap(),
        EgressRefusalOutcome::Counted
    );
    assert!(
        refusals(&runtime).is_empty(),
        "nothing is appended before the window closes"
    );
    runtime
        .record_egress_refusal(record(
            "host:0123456789abcdef:14:1883",
            EgressRefusalReason::HostNotAllowed,
            20,
        ))
        .unwrap();
    let indexes = runtime.flush_egress_refusals().unwrap();
    assert_eq!(indexes.len(), 1);
    assert_eq!(
        refusals(&runtime),
        [(
            "host:0123456789abcdef:14:1883".to_owned(),
            EgressRefusalReason::HostNotAllowed,
            2
        )]
    );

    let key = ptr_state::egress_refusal_key(&source().address);
    let expected = ptr_state::egress_refusal_entry(
        indexes[0].0,
        EgressRefusalReason::HostNotAllowed,
        2,
        "host:0123456789abcdef:14:1883",
    );
    assert_eq!(
        runtime.materialized_state().values.get(&key),
        Some(&expected)
    );
    let replayed = PtrRuntime::replay(PtrConfig::default(), runtime.committed_events()).unwrap();
    assert_eq!(
        replayed.materialized_state().values.get(&key),
        Some(&expected)
    );
}

#[test]
fn a_thousand_refusals_become_a_bounded_number_of_events_whose_counts_add_up() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let reasons = EgressRefusalReason::ALL;
    for attempt in 0..1000u64 {
        runtime
            .record_egress_refusal(record(
                &format!("host:{:016x}:1:80", attempt % 50),
                reasons[(attempt as usize) % reasons.len()],
                attempt,
            ))
            .unwrap();
    }
    runtime.flush_egress_refusals().unwrap();
    let events = refusals(&runtime);
    assert!(
        events.len() <= MAX_EGRESS_EVENTS_PER_POD_WINDOW,
        "{} events",
        events.len()
    );
    assert_eq!(
        events
            .iter()
            .map(|(_, _, count)| u64::from(*count))
            .sum::<u64>(),
        1000
    );
    assert!(
        events
            .iter()
            .any(|(target, _, _)| target == OVERFLOW_TARGET),
        "distinct pairs beyond the bound are folded under one target"
    );
    assert!(
        events
            .iter()
            .filter(|(target, _, _)| target != OVERFLOW_TARGET)
            .count()
            <= MAX_EGRESS_KEYS_PER_POD
    );
}

#[test]
fn a_window_that_closes_appends_its_counters_and_the_next_one_starts_fresh() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    for at in [1, 2, 3] {
        runtime
            .record_egress_refusal(record(
                "host:aaaaaaaaaaaaaaaa:9:1",
                EgressRefusalReason::RateLimited,
                at,
            ))
            .unwrap();
    }
    let outcome = runtime
        .record_egress_refusal(record(
            "host:aaaaaaaaaaaaaaaa:9:1",
            EgressRefusalReason::RateLimited,
            EGRESS_WINDOW_MILLIS + 1,
        ))
        .unwrap();
    assert!(matches!(outcome, EgressRefusalOutcome::Committed(ref indexes) if indexes.len() == 1));
    assert_eq!(
        refusals(&runtime),
        [(
            "host:aaaaaaaaaaaaaaaa:9:1".to_owned(),
            EgressRefusalReason::RateLimited,
            3
        )]
    );
    runtime.flush_egress_refusals().unwrap();
    assert_eq!(refusals(&runtime).len(), 2);
}

#[test]
fn a_record_that_could_carry_a_payload_is_refused_before_anything_is_counted() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    for target in [
        "",
        "host: 80",
        "line\nbreak",
        &"x".repeat(257),
        "broker.example:1883",
        "host:ABCDEF0123456789:1:1",
        "host:0123456789abcdef:1:99999",
        "topic:0123456789abcdef",
    ] {
        let refused =
            runtime.record_egress_refusal(record(target, EgressRefusalReason::InvalidTopic, 1));
        assert!(
            matches!(refused, Err(RuntimeError::Ledger(_))),
            "{target:?}: {refused:?}"
        );
    }
    let mut no_trace = record(
        "host:0123456789abcdef:1:1",
        EgressRefusalReason::HostNotAllowed,
        1,
    );
    no_trace.trace_id = TraceId::from("");
    assert!(matches!(
        runtime.record_egress_refusal(no_trace),
        Err(RuntimeError::Ledger(_))
    ));
    let mut long_trace = record(
        "host:0123456789abcdef:1:1",
        EgressRefusalReason::HostNotAllowed,
        1,
    );
    long_trace.trace_id = TraceId::from("t".repeat(257).as_str());
    assert!(matches!(
        runtime.record_egress_refusal(long_trace),
        Err(RuntimeError::Ledger(_))
    ));
    assert!(runtime.flush_egress_refusals().unwrap().is_empty());
    assert!(refusals(&runtime).is_empty());
}

#[test]
fn a_generic_commit_cannot_forge_a_refusal() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let forged = LedgerEvent::EgressRefused {
        trace_id: TraceId::from("trace-egress"),
        source: EgressRefusalSource::Pod(source()),
        target: "host:0123456789abcdef:1:1".into(),
        count: 1,
        reason: EgressRefusalReason::HostNotAllowed,
    };
    assert!(matches!(
        runtime.commit(forged),
        Err(RuntimeError::StructuralRecordOutsideAdmissionPath)
    ));
}

#[test]
fn a_refused_native_call_reaches_the_ledger_through_the_runtime_dispatch_hook() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let refused: Result<NativeProtocolResponse, ProtocolError> =
        Err(ProtocolError::EgressRefused {
            target: "host:fedcba9876543210:12:9".into(),
            reason: EgressRefusalReason::HostNotAllowed,
        });
    let counted = runtime
        .record_native_protocol_outcome(
            source(),
            TraceId::from("trace-native"),
            Timestamp(5),
            &refused,
        )
        .unwrap();
    assert_eq!(counted, Some(EgressRefusalOutcome::Counted));

    // Outcomes that are not egress refusals are not recorded.
    let answered: Result<NativeProtocolResponse, ProtocolError> = Ok(NativeProtocolResponse {
        output: ptr_protocol::TypedPayload {
            type_id: TypeId::from("Response"),
            bytes: vec![],
        },
        request_id: RequestId::from("r"),
        protocol: ProtocolBinding::Tcp,
    });
    for other in [
        answered,
        Err(ProtocolError::Deadline),
        Err(ProtocolError::InvalidEndpoint),
    ] {
        assert_eq!(
            runtime
                .record_native_protocol_outcome(
                    source(),
                    TraceId::from("trace-native"),
                    Timestamp(6),
                    &other,
                )
                .unwrap(),
            None
        );
    }
    runtime.flush_egress_refusals().unwrap();
    assert_eq!(
        refusals(&runtime),
        [(
            "host:fedcba9876543210:12:9".to_owned(),
            EgressRefusalReason::HostNotAllowed,
            1
        )]
    );
}
