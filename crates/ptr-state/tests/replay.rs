use std::collections::BTreeSet;

use ptr_ledger::{CommittedEvent, LedgerEvent};
use ptr_state::{classify_next, projection_entries, ApplyOutcome, MaterializedState};
use ptr_types::{CommitIndex, Generation, Revision};

fn event(index: u64, key: &str) -> CommittedEvent {
    CommittedEvent {
        index: CommitIndex(index),
        event: LedgerEvent::HardConstraintCommitted {
            key: key.into(),
            generation: Generation(index),
        },
    }
}

#[test]
fn duplicate_and_out_of_order_events_do_not_rewind_state() {
    let mut state = MaterializedState::default();
    assert_eq!(state.try_apply(&event(1, "a")), ApplyOutcome::Applied);
    assert_eq!(state.try_apply(&event(1, "a")), ApplyOutcome::Duplicate);
    assert_eq!(state.try_apply(&event(3, "c")), ApplyOutcome::Gap);
    assert_eq!(state.last_applied, 1);
    assert_eq!(state.try_apply(&event(2, "b")), ApplyOutcome::Applied);
    assert_eq!(state.try_apply(&event(1, "a")), ApplyOutcome::OutOfOrder);
    assert_eq!(state.last_applied, 2);
}

#[test]
fn commit_classification_does_not_wrap_at_the_u64_boundary() {
    assert_eq!(classify_next(u64::MAX - 1, u64::MAX), None);
    assert_eq!(
        classify_next(u64::MAX - 2, u64::MAX),
        Some(ApplyOutcome::Gap)
    );
    assert_eq!(classify_next(u64::MAX, 0), Some(ApplyOutcome::OutOfOrder));
    assert_eq!(classify_next(0, 0), Some(ApplyOutcome::Duplicate));
}

#[test]
fn rejected_events_do_not_leak_projected_values_and_can_be_retried_in_order() {
    let mut state = MaterializedState::default();
    state.try_apply(&event(1, "first"));
    let before = state.values.clone();
    for (index, outcome) in [
        (1, ApplyOutcome::Duplicate),
        (0, ApplyOutcome::OutOfOrder),
        (3, ApplyOutcome::Gap),
    ] {
        assert_eq!(state.try_apply(&event(index, "must-not-appear")), outcome);
        assert_eq!(state.values, before);
        assert_eq!(state.last_applied, 1);
    }
    assert_eq!(state.try_apply(&event(2, "next")), ApplyOutcome::Applied);
    assert_eq!(
        state.try_apply(&event(3, "must-not-appear")),
        ApplyOutcome::Applied
    );
    assert_eq!(
        state
            .values
            .get("constraint:must-not-appear")
            .map(String::as_str),
        Some("3")
    );
}

#[test]
fn semantic_projection_contains_only_the_revision_and_never_payload_bytes() {
    let committed = CommittedEvent {
        index: CommitIndex(1),
        event: LedgerEvent::SemanticDeltaCommitted {
            base_revision: Revision(40),
            revision: Revision(41),
            encoded_delta: b"private payload".to_vec(),
            origin: ptr_ledger::SemanticOrigin::Legacy,
        },
    };
    assert_eq!(
        projection_entries(&committed),
        vec![("semdb:revision".into(), "41".into())]
    );
    let mut state = MaterializedState::default();
    assert_eq!(state.try_apply(&committed), ApplyOutcome::Applied);
    assert_eq!(state.values.len(), 1);
    assert_eq!(
        state.values.get("semdb:revision").map(String::as_str),
        Some("41")
    );
}

#[test]
fn an_attested_record_projects_the_marker() {
    let attested = [
        ptr_ledger::SemanticOrigin::Request {
            request: "r1".into(),
        },
        ptr_ledger::SemanticOrigin::PodOutput {
            request: "r1".into(),
            pod: "echo".into(),
            level: ptr_types::VerificationLevel::SampleVerified,
        },
        ptr_ledger::SemanticOrigin::Host {
            principal: "operator".into(),
            verification: ptr_ledger::Attestation {
                required: ptr_types::VerificationLevel::Deterministic,
                level: ptr_types::VerificationLevel::Deterministic,
                verifiers: vec!["schema".into()],
                findings: Vec::new(),
            },
        },
    ];
    for origin in attested {
        let committed = CommittedEvent {
            index: CommitIndex(1),
            event: LedgerEvent::SemanticDeltaCommitted {
                base_revision: Revision(40),
                revision: Revision(41),
                encoded_delta: b"private payload".to_vec(),
                origin,
            },
        };
        // The revision, and the marker that a record with an origin exists;
        // still never the payload.
        assert_eq!(
            projection_entries(&committed),
            vec![
                ("semdb:revision".into(), "41".into()),
                (ptr_state::ATTESTED_MARKER.into(), "1".into()),
            ]
        );
        let mut state = MaterializedState::default();
        assert_eq!(state.try_apply(&committed), ApplyOutcome::Applied);
        assert_eq!(
            state
                .values
                .get(ptr_state::ATTESTED_MARKER)
                .map(String::as_str),
            Some("1")
        );
    }
}

fn merge_of(branch: &str, index: u64, plan: [u8; 32]) -> CommittedEvent {
    CommittedEvent {
        index: CommitIndex(index),
        event: LedgerEvent::SemanticDeltaCommitted {
            base_revision: Revision(40),
            revision: Revision(41),
            encoded_delta: b"private payload".to_vec(),
            origin: ptr_ledger::SemanticOrigin::Merge(ptr_ledger::MergeRecord {
                branch: branch.into(),
                author: "agent-1".into(),
                seal: [1; 32],
                plan,
                dependencies: [3; 32],
                rebased: Default::default(),
                verification: ptr_ledger::Attestation {
                    required: ptr_types::VerificationLevel::Deterministic,
                    level: ptr_types::VerificationLevel::Deterministic,
                    verifiers: vec!["schema".into()],
                    findings: Vec::new(),
                },
                authority: ptr_ledger::MergeAuthorityRecord::Reviewed {
                    reviewer: "reviewer-1".into(),
                },
            }),
        },
    }
}

#[test]
fn a_merge_projects_its_branch_key_with_its_index_and_plan() {
    let mut plan = [0; 32];
    for (n, byte) in plan.iter_mut().enumerate() {
        *byte = n as u8 * 8 + 1;
    }
    let committed = merge_of("b1", 7, plan);
    let entry = "7:0109111921293139414951596169717981899199a1a9b1b9c1c9d1d9e1e9f1f9";
    // The revision, the marker, and the branch's merge key: never the
    // payload, the author or the verifiers.
    assert_eq!(
        projection_entries(&committed),
        vec![
            ("semdb:revision".into(), "41".into()),
            (ptr_state::ATTESTED_MARKER.into(), "1".into()),
            ("branch-merge:2:b1".into(), entry.into()),
        ]
    );
    assert_eq!(ptr_state::merged_branch_entry(7, &plan), entry);
    assert_eq!(ptr_state::parse_merged_branch_entry(entry), Some((7, plan)));
    let mut state = MaterializedState::default();
    for index in 1..7 {
        assert_eq!(state.try_apply(&event(index, "c")), ApplyOutcome::Applied);
    }
    assert_eq!(state.try_apply(&committed), ApplyOutcome::Applied);
    assert_eq!(
        state
            .values
            .get(&ptr_state::merged_branch_key("b1"))
            .map(String::as_str),
        Some(entry)
    );
}

#[test]
fn branch_merge_keys_are_length_delimited() {
    // Ids that would share a key if it only concatenated prefix and id, or
    // counted characters rather than bytes.
    let ids = [
        "b1", "b", "1:b1", "b1:", "2:b1", "", ":", "\u{e9}", "ee", "b1 ",
    ];
    let keys: BTreeSet<String> = ids
        .iter()
        .map(|id| ptr_state::merged_branch_key(id))
        .collect();
    assert_eq!(keys.len(), ids.len());
    assert_eq!(ptr_state::merged_branch_key("b1"), "branch-merge:2:b1");
    assert_eq!(
        ptr_state::merged_branch_key("\u{e9}"),
        "branch-merge:2:\u{e9}"
    );
    for id in ids {
        let key = ptr_state::merged_branch_key(id);
        assert!(key.starts_with(ptr_state::MERGED_BRANCH_PREFIX));
        assert_eq!(ptr_state::merged_branch_of(&key), Some(id), "{key}");
    }
    // A key under the prefix whose length does not match its id, or is not
    // written as a merge writes it, names no branch.
    for key in [
        "branch-merge:3:b1",
        "branch-merge:1:b1",
        "branch-merge:02:b1",
        "branch-merge:+2:b1",
        "branch-merge::b1",
        "branch-merge:b1",
        "branch-merge:2",
        "branch-merge2:b1",
        "semdb:attested",
    ] {
        assert_eq!(ptr_state::merged_branch_of(key), None, "{key}");
    }
    // Nor does a value that is not an index and a 64-digit lowercase
    // hexadecimal plan digest.
    let hex = "0".repeat(64);
    for entry in [
        format!("07:{hex}"),
        format!("+7:{hex}"),
        format!(":{hex}"),
        format!("7:{}", "0".repeat(63)),
        format!("7:{}", "0".repeat(65)),
        format!("7:{}", "A".repeat(64)),
        format!("7:{}g", "0".repeat(63)),
        format!("18446744073709551616:{hex}"),
        "7".to_owned(),
    ] {
        assert_eq!(
            ptr_state::parse_merged_branch_entry(&entry),
            None,
            "{entry}"
        );
    }
    assert_eq!(
        ptr_state::parse_merged_branch_entry(&format!("0:{hex}")),
        Some((0, [0; 32]))
    );
    // Two merges of ids that share a character prefix project two keys.
    let mut state = MaterializedState::default();
    assert_eq!(
        state.try_apply(&merge_of("b1", 1, [1; 32])),
        ApplyOutcome::Applied
    );
    assert_eq!(
        state.try_apply(&merge_of("b1:", 2, [2; 32])),
        ApplyOutcome::Applied
    );
    let merged: Vec<&String> = state
        .values
        .keys()
        .filter(|key| key.starts_with(ptr_state::MERGED_BRANCH_PREFIX))
        .collect();
    assert_eq!(merged, ["branch-merge:2:b1", "branch-merge:3:b1:"]);
}
