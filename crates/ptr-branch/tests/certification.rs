use std::collections::BTreeSet;

use ptr_branch::{
    certify, counter_value, read_counter_value, read_set_value, set_value, Branch, BranchError,
    BranchId, BranchOp, Certification,
};
use ptr_semdb::{SemanticDelta, SemanticHost, SemanticValue};
use ptr_types::{Generation, PrincipalId, Validity};

fn text(value: &str) -> SemanticValue {
    SemanticValue::Text(value.into())
}

#[test]
fn a_read_of_an_absent_key_conflicts_with_a_concurrent_insert() {
    let mut host = SemanticHost::default();
    let mut work = branch(&host, "absent-read");
    assert_eq!(work.read("reservation"), Ok(None));
    work.put("reservation", text("mine")).unwrap();
    let sealed = work.seal().unwrap();
    change(&mut host, "reservation", text("theirs"));
    assert_eq!(
        certify(&sealed, &host.snapshot(), no_lifecycle).unwrap_err(),
        BranchError::Conflict {
            keys: BTreeSet::from(["reservation".into()])
        }
    );
    assert_eq!(host.snapshot().get("reservation"), Some("theirs"));
}

#[test]
fn prefix_scans_detect_deletions_and_changes_to_existing_values() {
    for remove in [false, true] {
        let mut host = host_with(&[("order:1", text("open"))]);
        let mut work = branch(&host, "scan");
        work.scan_prefix("order:").unwrap();
        let sealed = work.seal().unwrap();
        let mut delta = SemanticDelta::default();
        if remove {
            delta.removals.insert("order:1".into());
        } else {
            delta.upserts.insert("order:1".into(), text("closed"));
        }
        host.apply_delta(delta).unwrap();
        assert_eq!(
            certify(&sealed, &host.snapshot(), no_lifecycle).unwrap_err(),
            BranchError::Conflict {
                keys: BTreeSet::from(["order:*".into()])
            }
        );
    }
}

#[test]
fn reading_a_counter_prevents_rebasing_an_addition_over_a_changed_value() {
    let mut host = host_with(&[("stock", counter_value(5))]);
    let mut work = branch(&host, "stock-check");
    work.read("stock").unwrap();
    work.stage_commutative(BranchOp::Add {
        key: "stock".into(),
        amount: -2,
    })
    .unwrap();
    let sealed = work.seal().unwrap();
    change(&mut host, "stock", counter_value(1));
    assert_eq!(
        certify(&sealed, &host.snapshot(), no_lifecycle).unwrap_err(),
        BranchError::Conflict {
            keys: BTreeSet::from(["stock".into()])
        }
    );
}

#[test]
fn concurrent_set_insertions_preserve_both_members_in_either_commit_order() {
    for reverse in [false, true] {
        let mut host = host_with(&[("tags", set_value(&BTreeSet::from(["base".into()])))]);
        let sealed: Vec<_> = ["red", "blue"]
            .into_iter()
            .map(|member| {
                let mut work = branch(&host, member);
                work.stage_commutative(BranchOp::SetInsert {
                    key: "tags".into(),
                    member: member.into(),
                })
                .unwrap();
                work.seal().unwrap()
            })
            .collect();
        let order = if reverse { [1, 0] } else { [0, 1] };
        for (position, index) in order.into_iter().enumerate() {
            let certification = certify(&sealed[index], &host.snapshot(), no_lifecycle).unwrap();
            assert_eq!(
                matches!(certification, Certification::Rebased(_)),
                position == 1
            );
            commit(&mut host, certification);
        }
        assert_eq!(
            read_set_value(host.snapshot().value("tags").unwrap()),
            Some(BTreeSet::from(["base".into(), "blue".into(), "red".into()]))
        );
    }
}

#[test]
fn counter_overflow_at_rebase_is_refused_without_changing_the_host() {
    let mut host = host_with(&[("count", counter_value(0))]);
    let mut work = branch(&host, "increment");
    work.stage_commutative(BranchOp::Add {
        key: "count".into(),
        amount: 1,
    })
    .unwrap();
    let sealed = work.seal().unwrap();
    change(&mut host, "count", counter_value(i64::MAX));
    let revision = host.revision();
    assert_eq!(
        certify(&sealed, &host.snapshot(), no_lifecycle).unwrap_err(),
        BranchError::CounterOverflow {
            key: "count".into()
        }
    );
    assert_eq!(host.revision(), revision);
    assert_eq!(
        read_counter_value(host.snapshot().value("count").unwrap()),
        Some(i64::MAX)
    );
}

#[test]
fn operations_that_cancel_produce_a_noop_plan_and_preserve_author_identity() {
    let host = host_with(&[("count", counter_value(5))]);
    let mut work = branch(&host, "cancelled");
    for amount in [3, -3] {
        work.stage_commutative(BranchOp::Add {
            key: "count".into(),
            amount,
        })
        .unwrap();
    }
    let sealed = work.seal().unwrap();
    assert_eq!(sealed.author, PrincipalId::from("agent-1"));
    let certification = certify(&sealed, &host.snapshot(), no_lifecycle).unwrap();
    assert!(certification.plan().is_noop());
    assert_eq!(certification.plan().expected, host.revision());
}

fn host_with(entries: &[(&str, SemanticValue)]) -> SemanticHost {
    let mut host = SemanticHost::default();
    let mut delta = SemanticDelta::default();
    for (key, value) in entries {
        delta.upserts.insert((*key).into(), value.clone());
    }
    host.apply_delta(delta).unwrap();
    host
}

fn branch(host: &SemanticHost, id: &str) -> Branch {
    Branch::open(
        BranchId::from(id),
        PrincipalId::from("agent-1"),
        host.snapshot(),
    )
}

/// A lifecycle authority where nothing is relied on.
fn no_lifecycle(_: &str, _: Generation) -> Option<Validity> {
    None
}

fn commit(host: &mut SemanticHost, certification: Certification) {
    let (expected, delta) = certification.plan().clone().into_parts();
    assert_eq!(expected, host.revision());
    host.apply_delta(delta).unwrap();
}

fn change(host: &mut SemanticHost, key: &str, value: SemanticValue) {
    let mut delta = SemanticDelta::default();
    delta.upserts.insert(key.into(), value);
    host.apply_delta(delta).unwrap();
}

#[test]
fn an_unchanged_read_set_certifies_clean_and_merges_as_one_delta() {
    let mut host = host_with(&[("price:sku-1", text("10")), ("stock:sku-1", text("4"))]);
    let mut work = branch(&host, "b1");
    assert_eq!(work.read("price:sku-1").unwrap(), Some(text("10")));
    work.put("price:sku-1", text("12")).unwrap();
    let sealed = work.seal().unwrap();

    let certification = certify(&sealed, &host.snapshot(), no_lifecycle).unwrap();
    assert!(matches!(certification, Certification::Clean(_)));
    commit(&mut host, certification);
    assert_eq!(host.snapshot().get("price:sku-1"), Some("12"));
    assert_eq!(host.snapshot().get("stock:sku-1"), Some("4"));
}

#[test]
fn a_value_the_branch_read_that_changed_is_a_conflict_and_nothing_merges() {
    let mut host = host_with(&[("price:sku-1", text("10")), ("discount:sku-1", text("0"))]);
    let mut work = branch(&host, "b1");
    work.read("discount:sku-1").unwrap();
    work.read("price:sku-1").unwrap();
    work.put("price:sku-1", text("9")).unwrap();
    let sealed = work.seal().unwrap();

    change(&mut host, "discount:sku-1", text("20"));

    assert_eq!(
        certify(&sealed, &host.snapshot(), no_lifecycle).unwrap_err(),
        BranchError::Conflict {
            keys: BTreeSet::from(["discount:sku-1".to_owned()])
        }
    );
}

#[test]
fn a_key_inserted_under_a_scanned_prefix_refuses_certification() {
    let mut host = host_with(&[("order:1", text("open")), ("order:2", text("open"))]);
    let mut work = branch(&host, "b1");
    let open_orders = work.scan_prefix("order:").unwrap();
    assert_eq!(open_orders.len(), 2);
    work.stage_commutative(BranchOp::Add {
        key: "orders:reviewed".into(),
        amount: 2,
    })
    .unwrap();
    let sealed = work.seal().unwrap();

    // A phantom: no point read covers it, the scan does.
    change(&mut host, "order:3", text("open"));

    assert_eq!(
        certify(&sealed, &host.snapshot(), no_lifecycle).unwrap_err(),
        BranchError::Conflict {
            keys: BTreeSet::from(["order:*".to_owned()])
        }
    );
}

#[test]
fn a_change_outside_every_declared_dependency_does_not_conflict() {
    let mut host = host_with(&[("a", text("1")), ("b", text("1"))]);
    let mut work = branch(&host, "b1");
    work.read("a").unwrap();
    work.put("a", text("2")).unwrap();
    let sealed = work.seal().unwrap();

    change(&mut host, "b", text("5"));

    let certification = certify(&sealed, &host.snapshot(), no_lifecycle).unwrap();
    commit(&mut host, certification);
    assert_eq!(host.snapshot().get("a"), Some("2"));
    assert_eq!(host.snapshot().get("b"), Some("5"));
}

#[test]
fn a_revoked_or_superseded_relied_on_generation_refuses_certification() {
    let host = host_with(&[("a", text("1"))]);
    let mut work = branch(&host, "b1");
    work.rely_on("capsule:policy", Generation(3));
    work.read("a").unwrap();
    work.put("a", text("2")).unwrap();
    let sealed = work.seal().unwrap();

    for now in [Some(Validity::Revoked), Some(Validity::Superseded), None] {
        let lifecycle = |target: &str, generation: Generation| {
            (target == "capsule:policy" && generation == Generation(3))
                .then_some(now)
                .flatten()
        };
        assert_eq!(
            certify(&sealed, &host.snapshot(), lifecycle).unwrap_err(),
            BranchError::LifecycleChanged {
                targets: BTreeSet::from(["capsule:policy".to_owned()])
            }
        );
    }
    let live = |_: &str, _: Generation| Some(Validity::Live);
    assert!(certify(&sealed, &host.snapshot(), live).is_ok());
}

#[test]
fn two_concurrent_counter_additions_both_survive() {
    let mut host = host_with(&[("reserved:sku-1", counter_value(3))]);
    let add = |amount| BranchOp::Add {
        key: "reserved:sku-1".into(),
        amount,
    };
    let mut first = branch(&host, "b1");
    first.stage_commutative(add(2)).unwrap();
    let mut second = branch(&host, "b2");
    second.stage_commutative(add(5)).unwrap();
    let (first, second) = (first.seal().unwrap(), second.seal().unwrap());

    let certification = certify(&first, &host.snapshot(), no_lifecycle).unwrap();
    assert!(matches!(certification, Certification::Clean(_)));
    commit(&mut host, certification);

    let certification = certify(&second, &host.snapshot(), no_lifecycle).unwrap();
    match &certification {
        Certification::Rebased(plan) => assert!(plan.rebased.contains("reserved:sku-1")),
        Certification::Clean(_) => panic!("the second branch must report its rebase"),
    }
    commit(&mut host, certification);
    let value = host.snapshot().value("reserved:sku-1").cloned().unwrap();
    assert_eq!(read_counter_value(&value), Some(10));
}

#[test]
fn overwriting_a_key_that_was_never_read_is_refused_when_staged() {
    let host = host_with(&[("a", text("1"))]);
    let mut work = branch(&host, "b1");
    assert_eq!(
        work.put("a", text("2")).unwrap_err(),
        BranchError::UnreadTarget { key: "a".into() }
    );
    assert_eq!(
        work.remove("a").unwrap_err(),
        BranchError::UnreadTarget { key: "a".into() }
    );
}

#[test]
fn raw_request_text_and_pod_outputs_cannot_be_written_by_a_branch() {
    let host = host_with(&[("request:r1:raw", text("original"))]);
    let mut work = branch(&host, "b1");
    work.read("request:r1:raw").unwrap();
    assert_eq!(
        work.put("request:r1:raw", text("rewritten")).unwrap_err(),
        BranchError::ReservedNamespace {
            key: "request:r1:raw".into()
        }
    );
    assert!(matches!(
        work.stage_commutative(BranchOp::Add {
            key: "pod-output:x".into(),
            amount: 1
        }),
        Err(BranchError::ReservedNamespace { .. })
    ));
}

#[test]
fn writing_a_derived_key_reads_its_declared_inputs() {
    let mut host = SemanticHost::default();
    let mut setup = SemanticDelta::default();
    setup.upserts.insert("input:rate".into(), text("0.19"));
    setup.upserts.insert("derived:tax".into(), text("19"));
    setup.dependencies.insert(
        "derived:tax".into(),
        BTreeSet::from(["input:rate".to_owned()]),
    );
    host.apply_delta(setup).unwrap();

    let mut work = branch(&host, "b1");
    work.read("derived:tax").unwrap();
    work.put("derived:tax", text("20")).unwrap();
    let sealed = work.seal().unwrap();
    assert!(sealed.reads.contains_key("input:rate"));

    // The input changes after the branch computed its value; the stale
    // derivation must not merge.
    change(&mut host, "input:rate", text("0.07"));
    assert!(matches!(
        certify(&sealed, &host.snapshot(), no_lifecycle),
        Err(BranchError::Conflict { .. })
    ));
}

#[test]
fn a_branch_reads_its_own_staged_writes() {
    let host = host_with(&[("tags", text("x"))]);
    let mut work = branch(&host, "b1");
    work.read("tags").unwrap();
    work.put("tags", text("y")).unwrap();
    assert_eq!(work.read("tags").unwrap(), Some(text("y")));
}

#[test]
fn certifying_against_a_snapshot_older_than_the_base_is_refused() {
    let mut host = host_with(&[("a", text("1"))]);
    let old = host.snapshot();
    change(&mut host, "a", text("2"));

    let mut work = branch(&host, "b1");
    work.read("a").unwrap();
    let sealed = work.seal().unwrap();
    assert!(matches!(
        certify(&sealed, &old, no_lifecycle),
        Err(BranchError::SnapshotBehindBase { .. })
    ));
}

#[test]
fn the_plan_digest_changes_with_the_delta_and_with_the_dependencies() {
    let host = host_with(&[("a", text("1")), ("b", text("1"))]);
    let plan_for = |value: &str, extra_read: bool| {
        let mut work = branch(&host, "b1");
        work.read("a").unwrap();
        if extra_read {
            work.read("b").unwrap();
        }
        work.put("a", text(value)).unwrap();
        certify(&work.seal().unwrap(), &host.snapshot(), no_lifecycle)
            .unwrap()
            .plan()
            .digest()
            .unwrap()
    };
    let base = plan_for("2", false);
    assert_eq!(base, plan_for("2", false));
    assert_ne!(base, plan_for("3", false));
    assert_ne!(base, plan_for("2", true));
}

#[test]
fn the_plan_digest_binds_the_branch_so_an_approval_cannot_be_replayed_for_another() {
    let host = host_with(&[("a", text("1"))]);
    let plan_for = |id: &str| {
        let mut work = branch(&host, id);
        work.read("a").unwrap();
        work.put("a", text("2")).unwrap();
        certify(&work.seal().unwrap(), &host.snapshot(), no_lifecycle)
            .unwrap()
            .plan()
            .clone()
    };
    let first = plan_for("b1");
    let second = plan_for("b2");
    // Everything but the branch identity is the same.
    assert_eq!(first.expected, second.expected);
    assert_eq!(first.delta, second.delta);
    assert_eq!(first.dependencies, second.dependencies);
    assert_ne!(first.digest().unwrap(), second.digest().unwrap());
}
