//! A sealed branch exists only through `Branch::seal` and
//! `SealedBranch::from_parts`, which refuse every declaration sealing an open
//! branch could not have produced: parts edited by hand or read back from a
//! tampered store never become a branch certification would plan.

use std::collections::BTreeMap;

use ptr_branch::{
    certify, counter_value, set_value, Branch, BranchError, BranchId, BranchOp, InputsDigest,
    RangeDigest, SealedBranch, SealedBranchParts, ValueDigest, RESERVED_PREFIXES,
};
use ptr_semdb::{
    canonical_input_bytes, SemanticDelta, SemanticHost, SemanticPayload, SemanticSnapshot,
    SemanticValue, MAX_DELTA_BYTES,
};
use ptr_types::{Generation, PrincipalId, Revision, TypeId, Validity};
use sha2::{Digest, Sha256};

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
            in_base: false,
        },
        BranchOp::SetRemove {
            key: "request:r2:raw".into(),
            member: "m".into(),
            in_base: false,
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
            in_base: false,
        },
        BranchOp::SetRemove {
            key: "tags".into(),
            member: String::new(),
            in_base: false,
        },
    ] {
        assert_eq!(
            SealedBranch::from_parts(parts_for(&snapshot, vec![op])).unwrap_err(),
            BranchError::InvalidMember { key: "tags".into() }
        );
    }
}

#[test]
fn a_put_of_a_value_the_journal_cannot_encode_is_refused_when_put_and_by_the_constructor() {
    let snapshot = snapshot_with(&[]);
    let payload = |type_id: &str, source: &str| {
        SemanticValue::Payload(SemanticPayload {
            type_id: TypeId::from(type_id),
            source: source.into(),
            bytes: vec![1, 2, 3],
        })
    };
    let refused = BranchError::InvalidValue { key: "doc".into() };
    for value in [
        payload("", "unit-test"),
        payload("ptr.test.bytes", ""),
        text(&"x".repeat(MAX_DELTA_BYTES + 1)),
    ] {
        assert!(canonical_input_bytes("doc", &value).is_err());
        // Parts holding one are no sealed branch, so nothing can merge,
        // digest or store it.
        let parts = parts_for(
            &snapshot,
            vec![BranchOp::Put {
                key: "doc".into(),
                value: value.clone(),
            }],
        );
        assert_eq!(SealedBranch::from_parts(parts).unwrap_err(), refused);
        // An open branch refuses the put and records nothing.
        let mut work = Branch::open(
            BranchId::from("b"),
            PrincipalId::from("agent-1"),
            snapshot.clone(),
        );
        work.read("doc").unwrap();
        assert_eq!(work.put("doc", value).unwrap_err(), refused);
        let sealed = work.seal().unwrap();
        assert!(sealed.ops().is_empty());
        assert!(sealed.seal_digest().is_ok());
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

#[test]
fn a_hand_built_remove_of_a_derived_key_is_refused_by_the_constructor() {
    // Staging refuses the removal (DerivedRemoval); parts that record it
    // with the key's non-empty input set are refused the same way.
    let snapshot = snapshot_with(&[("k", text("1"))]);
    let mut parts = parts_for(&snapshot, vec![BranchOp::Remove { key: "k".into() }]);
    parts
        .touched_inputs
        .insert("k".into(), InputsDigest::of("k", ["input"]));
    let refused = SealedBranch::from_parts(parts.clone()).unwrap_err();
    assert_eq!(refused, BranchError::DerivedRemoval { key: "k".into() });
    assert_eq!(refused.code(), "PTR_BRANCH_DERIVED_REMOVAL");
    // With the empty input set it is recorded with, the removal is rebuilt.
    parts
        .touched_inputs
        .insert("k".into(), InputsDigest::of("k", []));
    assert!(SealedBranch::from_parts(parts).is_ok());
}

#[test]
fn hand_built_set_operations_that_misstate_their_member_at_the_base_are_refused_by_the_constructor()
{
    let op = |insert: bool, in_base: bool| {
        if insert {
            BranchOp::SetInsert {
                key: "tags".into(),
                member: "m".into(),
                in_base,
            }
        } else {
            BranchOp::SetRemove {
                key: "tags".into(),
                member: "m".into(),
                in_base,
            }
        }
    };
    let with_set = snapshot_with(&[("tags", set_value(&["m".to_owned()].into()).unwrap())]);
    // Two operations on one member describe one base.
    let refused =
        SealedBranch::from_parts(parts_for(&with_set, vec![op(true, true), op(false, false)]))
            .unwrap_err();
    assert_eq!(
        refused,
        BranchError::MalformedSeal {
            key: "tags".into(),
            reason: "two set operations on one member record different base presences",
        }
    );
    // A base without the key holds no member.
    let without = snapshot_with(&[]);
    assert_eq!(
        SealedBranch::from_parts(parts_for(&without, vec![op(false, true)])).unwrap_err(),
        BranchError::MalformedSeal {
            key: "tags".into(),
            reason: "a set operation records its member present in a base without its key",
        }
    );
    // What staging records is rebuilt.
    assert!(
        SealedBranch::from_parts(parts_for(&with_set, vec![op(true, true), op(false, true)]))
            .is_ok()
    );
    assert!(SealedBranch::from_parts(parts_for(&without, vec![op(true, false)])).is_ok());
}

#[test]
fn a_derived_removal_is_reported_only_when_the_parts_break_no_other_invariant() {
    // Sealing once produced a Remove of a derived key and broke nothing else,
    // so a store tells such a branch, to be re-run, from parts no sealing
    // produced by whether the removal is the only refusal: it is reported
    // only once every other invariant holds.
    let snapshot = snapshot_with(&[("j", text("1")), ("k", text("2"))]);
    let remove = |key: &str| BranchOp::Remove { key: key.into() };
    let derived = |parts: &mut SealedBranchParts, key: &str| {
        parts
            .touched_inputs
            .insert(key.into(), InputsDigest::of(key, ["input"]));
    };
    // A set operation recording its member present in a base without its
    // key is refused whichever operation comes first.
    let present_without_key = BranchOp::SetRemove {
        key: "s".into(),
        member: "m".into(),
        in_base: true,
    };
    for ops in [
        vec![remove("k"), present_without_key.clone()],
        vec![present_without_key.clone(), remove("k")],
    ] {
        let mut parts = parts_for(&snapshot, ops.clone());
        derived(&mut parts, "k");
        assert_eq!(
            SealedBranch::from_parts(parts).unwrap_err(),
            BranchError::MalformedSeal {
                key: "s".into(),
                reason: "a set operation records its member present in a base without its key",
            },
            "{ops:?}"
        );
    }
    // So is a digest checked after every operation.
    let mut parts = parts_for(&snapshot, vec![remove("k")]);
    derived(&mut parts, "k");
    parts
        .touched_inputs
        .insert("stray".into(), InputsDigest::of("stray", []));
    assert_eq!(
        SealedBranch::from_parts(parts).unwrap_err(),
        BranchError::MalformedSeal {
            key: "stray".into(),
            reason: "a base value or input set is recorded for a key no operation touches",
        }
    );
    // With nothing else broken, the first derived removal in operation
    // order is reported.
    let mut parts = parts_for(&snapshot, vec![remove("k"), remove("j")]);
    derived(&mut parts, "j");
    derived(&mut parts, "k");
    assert_eq!(
        SealedBranch::from_parts(parts).unwrap_err(),
        BranchError::DerivedRemoval { key: "k".into() }
    );
}

/// Parts with an entry in every map and one operation of every kind, which
/// satisfy every sealing invariant: `a` is put and `d` removed, both read;
/// `c` is added to and `s` has a member inserted and one removed.
fn every_part() -> SealedBranchParts {
    let snapshot = snapshot_with(&[
        ("a", text("1")),
        ("c", counter_value(3)),
        ("d", text("gone")),
        ("s", set_value(&["m".to_owned()].into()).unwrap()),
    ]);
    let digest = |key: &str| ValueDigest::of(key, snapshot.value(key)).unwrap();
    let inputs = |key: &str| InputsDigest::of(key, snapshot.inputs(key));
    SealedBranchParts {
        id: BranchId::from("b1"),
        author: PrincipalId::from("agent-1"),
        base_revision: snapshot.revision,
        reads: BTreeMap::from([("a".to_owned(), digest("a")), ("d".to_owned(), digest("d"))]),
        scans: BTreeMap::from([("a".to_owned(), RangeDigest::from_bytes([7; 32]))]),
        relied: BTreeMap::from([("capsule:x".to_owned(), Generation(2))]),
        touched_base: ["a", "c", "d", "s"]
            .into_iter()
            .map(|key| (key.to_owned(), digest(key)))
            .collect(),
        touched_inputs: ["a", "c", "d", "s"]
            .into_iter()
            .map(|key| (key.to_owned(), inputs(key)))
            .collect(),
        ops: vec![
            BranchOp::Put {
                key: "a".into(),
                value: text("2"),
            },
            BranchOp::Remove { key: "d".into() },
            BranchOp::Add {
                key: "c".into(),
                amount: 1,
            },
            BranchOp::SetInsert {
                key: "s".into(),
                member: "n".into(),
                in_base: false,
            },
            BranchOp::SetRemove {
                key: "s".into(),
                member: "m".into(),
                in_base: true,
            },
        ],
    }
}

fn seal(parts: SealedBranchParts) -> [u8; 32] {
    SealedBranch::from_parts(parts)
        .unwrap()
        .seal_digest()
        .unwrap()
}

#[test]
fn seal_digest_changes_with_every_part_and_op() {
    let base = seal(every_part());
    // Rebuilding the branch from its parts keeps its digest.
    let rebuilt = SealedBranch::from_parts(every_part()).unwrap();
    assert_eq!(
        seal(rebuilt.clone().into_parts()),
        rebuilt.seal_digest().unwrap()
    );
    assert_eq!(rebuilt.seal_digest().unwrap(), base);

    type Edit = fn(&mut SealedBranchParts);
    let edits: [(&str, Edit); 15] = [
        ("id", |p| p.id = BranchId::from("b2")),
        ("author", |p| p.author = PrincipalId::from("agent-2")),
        ("base revision", |p| {
            p.base_revision = Revision(p.base_revision.0 + 1)
        }),
        ("a read", |p| {
            p.reads.insert("x".into(), ValueDigest::from_bytes([1; 32]));
        }),
        ("a scan", |p| {
            p.scans.insert("a".into(), RangeDigest::from_bytes([8; 32]));
        }),
        ("a relied generation", |p| {
            p.relied.insert("capsule:x".into(), Generation(3));
        }),
        ("a touched base value", |p| {
            p.touched_base
                .insert("c".into(), ValueDigest::from_bytes([2; 32]));
        }),
        ("a touched input set", |p| {
            p.touched_inputs
                .insert("c".into(), InputsDigest::of("c", ["i"]));
        }),
        ("a put value", |p| {
            p.ops[0] = BranchOp::Put {
                key: "a".into(),
                value: text("3"),
            }
        }),
        ("an added amount", |p| {
            p.ops[2] = BranchOp::Add {
                key: "c".into(),
                amount: 2,
            }
        }),
        ("an inserted member", |p| {
            p.ops[3] = BranchOp::SetInsert {
                key: "s".into(),
                member: "o".into(),
                in_base: false,
            }
        }),
        ("an in_base", |p| {
            p.ops[3] = BranchOp::SetInsert {
                key: "s".into(),
                member: "n".into(),
                in_base: true,
            }
        }),
        ("an insert turned into a removal", |p| {
            p.ops[3] = BranchOp::SetRemove {
                key: "s".into(),
                member: "n".into(),
                in_base: false,
            }
        }),
        ("the operation order", |p| p.ops.swap(1, 2)),
        ("an operation dropped", |p| {
            p.ops.remove(1);
            p.reads.remove("d");
            p.touched_base.remove("d");
            p.touched_inputs.remove("d");
        }),
    ];
    let mut seen = std::collections::BTreeSet::from([base]);
    for (name, edit) in edits {
        let mut parts = every_part();
        edit(&mut parts);
        let digest = seal(parts);
        assert!(seen.insert(digest), "{name} did not change the digest");
    }

    // The same 32 bytes as the last read or as the only scan are different
    // branches, although the entries alone would hash the same bytes in the
    // same order: the sections are counted.
    let mut as_read = every_part();
    as_read.scans.clear();
    as_read
        .reads
        .insert("x".into(), ValueDigest::from_bytes([9; 32]));
    let mut as_scan = every_part();
    as_scan.scans.clear();
    as_scan
        .scans
        .insert("x".into(), RangeDigest::from_bytes([9; 32]));
    assert_ne!(seal(as_read), seal(as_scan));
}

#[test]
fn seal_digest_is_the_documented_layout() {
    // Written out byte by byte: storage binds a stored branch to this
    // digest, so its layout may not move.
    fn text(hasher: &mut Sha256, value: &str) {
        hasher.update((value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    let parts = every_part();
    let mut hasher = Sha256::new();
    hasher.update(b"ptr-branch/sealed-branch/v1");
    text(&mut hasher, "b1");
    text(&mut hasher, "agent-1");
    hasher.update(parts.base_revision.0.to_le_bytes());
    for map in [
        parts
            .reads
            .iter()
            .map(|(key, digest)| (key, *digest.as_bytes()))
            .collect::<Vec<_>>(),
        parts
            .scans
            .iter()
            .map(|(key, digest)| (key, *digest.as_bytes()))
            .collect(),
    ] {
        hasher.update((map.len() as u64).to_le_bytes());
        for (key, digest) in map {
            text(&mut hasher, key);
            hasher.update(digest);
        }
    }
    hasher.update(1u64.to_le_bytes());
    text(&mut hasher, "capsule:x");
    hasher.update(2u64.to_le_bytes());
    for map in [
        parts
            .touched_base
            .iter()
            .map(|(key, digest)| (key, *digest.as_bytes()))
            .collect::<Vec<_>>(),
        parts
            .touched_inputs
            .iter()
            .map(|(key, digest)| (key, *digest.as_bytes()))
            .collect(),
    ] {
        hasher.update((map.len() as u64).to_le_bytes());
        for (key, digest) in map {
            text(&mut hasher, key);
            hasher.update(digest);
        }
    }
    hasher.update(5u64.to_le_bytes());
    let put = canonical_input_bytes("a", &text_value("2")).unwrap();
    hasher.update([1]);
    text(&mut hasher, "a");
    hasher.update((put.len() as u64).to_le_bytes());
    hasher.update(&put);
    hasher.update([2]);
    text(&mut hasher, "d");
    hasher.update([3]);
    text(&mut hasher, "c");
    hasher.update(1i64.to_le_bytes());
    hasher.update([4]);
    text(&mut hasher, "s");
    text(&mut hasher, "n");
    hasher.update([0]);
    hasher.update([5]);
    text(&mut hasher, "s");
    text(&mut hasher, "m");
    hasher.update([1]);
    assert_eq!(seal(parts), <[u8; 32]>::from(hasher.finalize()));
}

fn text_value(value: &str) -> SemanticValue {
    SemanticValue::Text(value.into())
}
