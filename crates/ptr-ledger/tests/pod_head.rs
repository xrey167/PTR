use ptr_ledger::{FileLedger, LedgerEvent};
use ptr_types::{
    Generation, NamespaceId, PodAddress, PodId, PodIdentity, PodRevisionAddress, ProjectId,
};
use std::time::{SystemTime, UNIX_EPOCH};

fn address(generation: u64) -> PodRevisionAddress {
    PodRevisionAddress {
        address: PodAddress::new(
            ProjectId::from("project"),
            NamespaceId::from("ns"),
            PodId::from("reader"),
        )
        .unwrap(),
        semantic_revision: [4; 32],
        generation: Generation(generation),
    }
}

fn temp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "ptr-{name}-{}.ledger",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

#[test]
fn pod_head_moves_round_trip_through_the_durable_ledger() {
    let path = temp_path("pod-head");
    let first = LedgerEvent::PodHeadAdvanced {
        project: ProjectId::from("project"),
        identity: PodIdentity::from("pod:stable"),
        old: None,
        new: address(1),
    };
    let second = LedgerEvent::PodHeadAdvanced {
        project: ProjectId::from("project"),
        identity: PodIdentity::from("pod:stable"),
        old: Some(address(1)),
        new: address(2),
    };
    {
        let mut ledger = FileLedger::open(&path).unwrap();
        ledger.append_durable(first.clone()).unwrap();
        ledger.append_durable(second.clone()).unwrap();
    }
    let ledger = FileLedger::open(&path).unwrap();
    let events: Vec<_> = ledger.events().iter().map(|e| e.event.clone()).collect();
    assert_eq!(events, vec![first, second]);
    drop(ledger);
    let _ = std::fs::remove_file(path);
}

#[test]
fn a_non_rising_head_move_is_refused_before_any_byte_is_appended() {
    let path = temp_path("pod-head-refused");
    let stale = LedgerEvent::PodHeadAdvanced {
        project: ProjectId::from("project"),
        identity: PodIdentity::from("pod:stable"),
        old: Some(address(2)),
        new: address(2),
    };
    let mut ledger = FileLedger::open(&path).unwrap();
    assert!(ledger.append_durable(stale).is_err());
    assert!(ledger.events().is_empty());
    drop(ledger);
    let _ = std::fs::remove_file(path);
}

#[test]
fn egress_refusals_and_pod_head_moves_round_trip_side_by_side_in_one_log() {
    use ptr_ledger::EgressRefusalSource;
    use ptr_types::{EgressRefusalReason, TraceId};

    let path = temp_path("tags-37-38");
    let refusal = LedgerEvent::EgressRefused {
        trace_id: TraceId::from("trace"),
        source: EgressRefusalSource::Pod(address(1)),
        target: "*".into(),
        count: 2,
        saturated: false,
        reason: EgressRefusalReason::HostNotAllowed,
    };
    let head = LedgerEvent::PodHeadAdvanced {
        project: ProjectId::from("project"),
        identity: PodIdentity::from("pod:stable"),
        old: None,
        new: address(1),
    };
    let overflow = LedgerEvent::EgressRefused {
        trace_id: TraceId::from("*"),
        source: EgressRefusalSource::Overflow,
        target: "*".into(),
        count: 2,
        saturated: false,
        reason: EgressRefusalReason::RateLimited,
    };
    // The ledger alone does not judge whether a head move names the current head, so the same
    // first move can appear twice; the runtime is what refuses that.
    let sequence = vec![refusal, head.clone(), overflow, head];
    {
        let mut ledger = FileLedger::open(&path).unwrap();
        for event in &sequence {
            ledger.append_durable(event.clone()).unwrap();
        }
    }
    let ledger = FileLedger::open(&path).unwrap();
    let events: Vec<_> = ledger.events().iter().map(|e| e.event.clone()).collect();
    assert_eq!(events, sequence);
    drop(ledger);
    let _ = std::fs::remove_file(path);
}

#[test]
fn the_same_identity_in_two_projects_round_trips_as_two_independent_records() {
    let path = temp_path("pod-head-projects");
    let in_project = |project: &str| {
        let mut new = address(1);
        new.address.project = ProjectId::from(project);
        LedgerEvent::PodHeadAdvanced {
            project: ProjectId::from(project),
            identity: PodIdentity::from("pod:stable"),
            old: None,
            new,
        }
    };
    let (alpha, beta) = (in_project("alpha"), in_project("beta"));
    {
        let mut ledger = FileLedger::open(&path).unwrap();
        ledger.append_durable(alpha.clone()).unwrap();
        ledger.append_durable(beta.clone()).unwrap();
        // A record whose project differs from the project of its address is refused.
        let mut crossed = in_project("alpha");
        if let LedgerEvent::PodHeadAdvanced { project, .. } = &mut crossed {
            *project = ProjectId::from("beta");
        }
        assert!(ledger.append_durable(crossed).is_err());
        assert_eq!(ledger.events().len(), 2);
    }
    let ledger = FileLedger::open(&path).unwrap();
    let events: Vec<_> = ledger.events().iter().map(|e| e.event.clone()).collect();
    assert_eq!(events, vec![alpha, beta]);
    drop(ledger);
    let _ = std::fs::remove_file(path);
}
