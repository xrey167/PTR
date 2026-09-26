//! A sealed branch exists only through `Branch::seal` and
//! `SealedBranch::from_parts`, which refuse every declaration sealing an open
//! branch could not have produced: parts edited by hand or read back from a
//! tampered store never become a branch certification would plan.

use std::collections::BTreeMap;

use ptr_branch::{
    certify, counter_value, Branch, BranchError, BranchId, BranchOp, InputsDigest, SealedBranch,
    SealedBranchParts, ValueDigest, RESERVED_PREFIXES,
};
use ptr_semdb::{SemanticDelta, SemanticHost, SemanticSnapshot, SemanticValue};
use ptr_types::{Generation, PrincipalId, Validity};

fn text(value: &str) -> SemanticValue {
    SemanticValue::Text(value.into())
}

fn snapshot_with(entries: &[(&str, SemanticValue)]) -> SemanticSnapshot {
    let mut host = SemanticHost::default();
    let mut delta = SemanticDelta::default();
    for (key, value) in entries {
        delta.upserts.insert((*key).into(), value.clone());
    }
    host.apply_delta(delta).unwrap();
    host.snapshot()
}

fn no_lifecycle(_: &str, _: Generation) -> Option<Validity> {
    None
}

/// Parts for `ops` on `snapshot` that satisfy every invariant but the one a
/// test breaks: each operated key is read at its current value, and its
/// current value and input set are recorded as touched. This is everything a
/// submitter who can read the store can supply.
fn parts_for(snapshot: &SemanticSnapshot, ops: Vec<BranchOp>) -> SealedBranchParts {
    let mut reads = BTreeMap::new();
    let mut touched_base = BTreeMap::new();
    let mut touched_inputs = BTreeMap::new();
    for op in &ops {
        let key = op.key();
        let digest = ValueDigest::of(key, snapshot.value(key)).unwrap();
        reads.insert(key.to_owned(), digest);
        touched_base.insert(key.to_owned(), digest);
        touched_inputs.insert(key.to_owned(), InputsDigest::of(key, snapshot.inputs(key)));
    }
    SealedBranchParts {
        id: BranchId::from("hand-built"),
        author: PrincipalId::from("agent-1"),
        base_revision: snapshot.revision,
        reads,
        scans: BTreeMap::new(),
        relied: BTreeMap::new(),
        touched_base,
        touched_inputs,
        ops,
    }
}

#[test]
fn from_parts_rebuilds_exactly_the_branch_sealing_produced() {
    let snapshot = snapshot_with(&[("a", text("1")), ("c", counter_value(3))]);
    let mut work = Branch::open(
        BranchId::from("b1"),
        PrincipalId::from("agent-1"),
        snapshot.clone(),
    );
    work.read("a").unwrap();
    work.read("c").unwrap();
    work.put("a", text("2")).unwrap();
    work.stage_commutative(BranchOp::Add {
        key: "c".into(),
        amount: 1,
    })
    .unwrap();
    work.scan_prefix("a").unwrap();
    work.rely_on("capsule:x", Generation(2)).unwrap();
    let sealed = work.seal().unwrap();
    assert_eq!(sealed.recheck(), Ok(()));
    let rebuilt = SealedBranch::from_parts(sealed.clone().into_parts()).unwrap();
    assert_eq!(rebuilt, sealed);
}

#[test]
fn a_hand_built_branch_that_writes_a_reserved_namespace_is_refused_by_the_constructor() {
    // The submitter read the ingress-owned key and supplies its current
    // digests, so the declaration matches the store exactly; the operation
    // is still one no open branch could stage.
    let snapshot = snapshot_with(&[
        ("request:r1:raw", text("original")),
        ("pod-output:p1", counter_value(1)),
    ]);
    for op in [
        BranchOp::Put {
            key: "request:r1:raw".into(),
            value: text("rewritten"),
        },
        BranchOp::Remove {
            key: "request:r1:raw".into(),
        },
        BranchOp::Add {
            key: "pod-output:p1".into(),
            amount: 1,
        },
        BranchOp::SetInsert {
            key: "pod-output:tags".into(),
            member: "m".into(),
        },
        BranchOp::SetRemove {
            key: "request:r2:raw".into(),
            member: "m".into(),
        },
    ] {
        let key = op.key().to_owned();
        assert!(RESERVED_PREFIXES
            .iter()
            .any(|prefix| key.starts_with(prefix)));
        let refused = SealedBranch::from_parts(parts_for(&snapshot, vec![op])).unwrap_err();
        assert_eq!(refused, BranchError::ReservedNamespace { key });
        assert_eq!(refused.code(), "PTR_BRANCH_RESERVED_NAMESPACE");
    }
    // Reading a reserved key is allowed: only writing it is refused.
    let mut parts = parts_for(&snapshot, Vec::new());
    parts.reads.insert(
        "request:r1:raw".into(),
        ValueDigest::of("request:r1:raw", snapshot.value("request:r1:raw")).unwrap(),
    );
    let reader = SealedBranch::from_parts(parts).unwrap();
    assert!(certify(&reader, &snapshot, no_lifecycle)
        .unwrap()
        .plan()
        .is_noop());
}

#[test]
fn a_hand_built_put_or_remove_of_an_unread_key_is_refused_by_the_constructor() {
    let snapshot = snapshot_with(&[("k", text("1"))]);
    for op in [
        BranchOp::Put {
            key: "k".into(),
            value: text("2"),
        },
        BranchOp::Remove { key: "k".into() },
    ] {
        let mut parts = parts_for(&snapshot, vec![op]);
        parts.reads.clear();
        let refused = SealedBranch::from_parts(parts).unwrap_err();
        assert_eq!(refused, BranchError::UnreadTarget { key: "k".into() });
        assert_eq!(refused.code(), "PTR_BRANCH_UNREAD_TARGET");
    }
    // A commutative operation needs no read of its key.
    let mut parts = parts_for(
        &snapshot_with(&[("c", counter_value(1))]),
        vec![BranchOp::Add {
            key: "c".into(),
            amount: 1,
        }],
    );
    parts.reads.clear();
    assert!(SealedBranch::from_parts(parts).is_ok());
}

#[test]
fn a_hand_built_set_operation_with_an_empty_member_is_refused_by_the_constructor() {
    let snapshot = snapshot_with(&[]);
    for op in [
        BranchOp::SetInsert {
            key: "tags".into(),
            member: String::new(),
        },
        BranchOp::SetRemove {
            key: "tags".into(),
            member: String::new(),
        },
    ] {
        assert_eq!(
            SealedBranch::from_parts(parts_for(&snapshot, vec![op])).unwrap_err(),
            BranchError::InvalidMember { key: "tags".into() }
        );
    }
}

#[test]
fn an_operated_key_without_a_recorded_base_value_or_input_set_is_refused_by_the_constructor() {
    let snapshot = snapshot_with(&[("a", text("1")), ("c", counter_value(1))]);
    let ops = || {
        vec![
            BranchOp::Put {
                key: "a".into(),
                value: text("2"),
            },
            BranchOp::Add {
                key: "c".into(),
                amount: 1,
            },
        ]
    };
    for key in ["a", "c"] {
        let mut without_base = parts_for(&snapshot, ops());
        without_base.touched_base.remove(key);
        let mut without_inputs = parts_for(&snapshot, ops());
        without_inputs.touched_inputs.remove(key);
        for (parts, reason) in [
            (
                without_base,
                "a key an operation touches has no recorded base value",
            ),
            (
                without_inputs,
                "a key an operation touches has no recorded input set",
            ),
        ] {
            let refused = SealedBranch::from_parts(parts).unwrap_err();
            assert_eq!(
                refused,
                BranchError::MalformedSeal {
                    key: key.into(),
                    reason
                }
            );
            assert_eq!(refused.code(), "PTR_BRANCH_MALFORMED_SEAL");
        }
    }
}

#[test]
fn a_base_value_or_input_set_for_a_key_no_operation_touches_is_refused_by_the_constructor() {
    let snapshot = snapshot_with(&[("a", text("1")), ("b", text("1"))]);
    let put = || {
        vec![BranchOp::Put {
            key: "a".into(),
            value: text("2"),
        }]
    };
    let mut stray_base = parts_for(&snapshot, put());
    stray_base.touched_base.insert(
        "b".into(),
        ValueDigest::of("b", snapshot.value("b")).unwrap(),
    );
    let mut stray_inputs = parts_for(&snapshot, put());
    stray_inputs
        .touched_inputs
        .insert("b".into(), InputsDigest::of("b", []));
    for parts in [stray_base, stray_inputs] {
        assert_eq!(
            SealedBranch::from_parts(parts).unwrap_err(),
            BranchError::MalformedSeal {
                key: "b".into(),
                reason: "a base value or input set is recorded for a key no operation touches",
            }
        );
    }
}

#[test]
fn a_touched_base_value_that_differs_from_the_read_of_the_same_key_is_refused_by_the_constructor() {
    // Both digests are of the key's base value, so sealing never records two
    // different ones. A forged base value could only turn a Rebased
    // certification into a Clean one or the reverse.
    let snapshot = snapshot_with(&[("c", counter_value(1))]);
    let mut parts = parts_for(
        &snapshot,
        vec![BranchOp::Add {
            key: "c".into(),
            amount: 1,
        }],
    );
    parts.touched_base.insert(
        "c".into(),
        ValueDigest::of("c", Some(&counter_value(7))).unwrap(),
    );
    assert_eq!(
        SealedBranch::from_parts(parts).unwrap_err(),
        BranchError::MalformedSeal {
            key: "c".into(),
            reason: "the base value recorded for a touched key differs from the value read",
        }
    );
}

#[test]
fn a_constructor_refusal_names_the_first_broken_invariant_in_operation_order() {
    // The reserved write comes after an unread overwrite, so the unread
    // overwrite is what is reported; nothing about the second op is checked
    // before the first passes.
    let snapshot = snapshot_with(&[("k", text("1")), ("request:r1:raw", text("x"))]);
    let mut parts = parts_for(
        &snapshot,
        vec![
            BranchOp::Put {
                key: "k".into(),
                value: text("2"),
            },
            BranchOp::Remove {
                key: "request:r1:raw".into(),
            },
        ],
    );
    parts.reads.remove("k");
    assert_eq!(
        SealedBranch::from_parts(parts).unwrap_err(),
        BranchError::UnreadTarget { key: "k".into() }
    );
}
