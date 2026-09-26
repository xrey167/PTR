use std::collections::BTreeSet;

use ptr_branch::{
    certify, counter_value, read_counter_value, read_set_value, set_value, Branch, BranchError,
    BranchId, BranchOp, Certification, SealedBranch,
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
        let mut host = host_with(&[("tags", set_value(&BTreeSet::from(["base".into()])).unwrap())]);
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
    assert_eq!(sealed.author(), &PrincipalId::from("agent-1"));
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
    work.rely_on("capsule:policy", Generation(3)).unwrap();
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
    assert!(sealed.reads().contains_key("input:rate"));

    // The input changes after the branch computed its value; the stale
    // derivation must not merge.
    change(&mut host, "input:rate", text("0.07"));
    assert!(matches!(
        certify(&sealed, &host.snapshot(), no_lifecycle),
        Err(BranchError::Conflict { .. })
    ));
}

#[test]
fn a_touched_key_whose_input_set_changed_conflicts_even_when_every_value_it_read_is_unchanged() {
    // Each case rewires `derived:d` concurrently while recomputing it to the
    // value it already had: from `input:a` to `input:b`, from `input:a` to
    // no inputs at all, and for a key with no inputs, to `input:a`. The
    // branch computed its value against the base's inputs, and a merge would
    // keep the target's set, so its value would stand under inputs it was
    // never computed from.
    let a = || BTreeSet::from(["input:a".to_owned()]);
    let b = || BTreeSet::from(["input:b".to_owned()]);
    for (before, after) in [(a(), b()), (a(), BTreeSet::new()), (BTreeSet::new(), a())] {
        let mut host = SemanticHost::default();
        let mut setup = SemanticDelta::default();
        setup.upserts.insert("input:a".into(), text("1"));
        setup.upserts.insert("input:b".into(), text("1"));
        setup.upserts.insert("derived:d".into(), text("2"));
        setup
            .dependencies
            .insert("derived:d".into(), before.clone());
        host.apply_delta(setup).unwrap();

        let mut work = branch(&host, "b1");
        work.read("derived:d").unwrap();
        work.put("derived:d", text("3")).unwrap();
        let sealed = work.seal().unwrap();

        let mut rewire = SemanticDelta::default();
        rewire.upserts.insert("derived:d".into(), text("2"));
        rewire
            .dependencies
            .insert("derived:d".into(), after.clone());
        host.apply_delta(rewire).unwrap();
        let target = host.snapshot();
        assert!(target.revision > sealed.base_revision());
        for (key, value) in [("input:a", "1"), ("input:b", "1"), ("derived:d", "2")] {
            assert_eq!(target.get(key), Some(value));
        }

        assert_eq!(
            certify(&sealed, &target, no_lifecycle).unwrap_err(),
            BranchError::Conflict {
                keys: BTreeSet::from(["derived:d".into()])
            },
            "{before:?} -> {after:?}"
        );
    }
}

#[test]
fn a_touched_key_without_a_recorded_input_set_is_refused_before_certification() {
    // A sealed branch rebuilt from storage without the input set of a touched
    // key cannot show that the key's dependencies are unchanged, so it is
    // never rebuilt at all.
    let host = host_with(&[("a", text("1"))]);
    let mut work = branch(&host, "b1");
    work.read("a").unwrap();
    work.put("a", text("2")).unwrap();
    let sealed = work.seal().unwrap();
    assert!(certify(&sealed, &host.snapshot(), no_lifecycle).is_ok());
    let mut parts = sealed.into_parts();
    parts.touched_inputs.clear();
    let refused = SealedBranch::from_parts(parts).unwrap_err();
    assert!(
        matches!(refused, BranchError::MalformedSeal { ref key, .. } if key == "a"),
        "{refused:?}"
    );
    assert_eq!(refused.code(), "PTR_BRANCH_MALFORMED_SEAL");
}

#[test]
fn a_counter_addition_to_a_key_that_became_derived_is_a_conflict_not_a_rebase() {
    let mut host = host_with(&[("input:a", text("1")), ("count", counter_value(3))]);
    let mut work = branch(&host, "b1");
    work.stage_commutative(BranchOp::Add {
        key: "count".into(),
        amount: 1,
    })
    .unwrap();
    let sealed = work.seal().unwrap();

    let mut rewire = SemanticDelta::default();
    rewire.upserts.insert("count".into(), counter_value(3));
    rewire
        .dependencies
        .insert("count".into(), BTreeSet::from(["input:a".to_owned()]));
    host.apply_delta(rewire).unwrap();

    assert_eq!(
        certify(&sealed, &host.snapshot(), no_lifecycle).unwrap_err(),
        BranchError::Conflict {
            keys: BTreeSet::from(["count".into()])
        }
    );
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
fn the_plan_digest_changes_when_a_touched_key_was_computed_against_another_input_set() {
    // Two hosts at the same revision with the same values; only the input set
    // declared for `derived:d` differs. The branch reads every key on both,
    // so its value digests and its delta are identical.
    let plan_for = |input: &str| {
        let mut host = SemanticHost::default();
        let mut setup = SemanticDelta::default();
        setup.upserts.insert("input:a".into(), text("1"));
        setup.upserts.insert("input:b".into(), text("1"));
        setup.upserts.insert("derived:d".into(), text("2"));
        setup
            .dependencies
            .insert("derived:d".into(), BTreeSet::from([input.to_owned()]));
        host.apply_delta(setup).unwrap();
        let mut work = branch(&host, "b1");
        for key in ["input:a", "input:b", "derived:d"] {
            work.read(key).unwrap();
        }
        work.put("derived:d", text("3")).unwrap();
        let sealed = work.seal().unwrap();
        certify(&sealed, &host.snapshot(), no_lifecycle)
            .unwrap()
            .plan()
            .clone()
    };
    let from_a = plan_for("input:a");
    let from_b = plan_for("input:b");
    assert_eq!(from_a.expected, from_b.expected);
    assert_eq!(from_a.delta, from_b.delta);
    assert_ne!(from_a.dependencies, from_b.dependencies);
    assert_ne!(from_a.digest().unwrap(), from_b.digest().unwrap());
    assert_eq!(
        from_a.digest().unwrap(),
        plan_for("input:a").digest().unwrap()
    );
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

/// A host where `derived` holds `value` and depends on `input:i`.
fn derived_host(value: SemanticValue) -> SemanticHost {
    let mut host = SemanticHost::default();
    let mut setup = SemanticDelta::default();
    setup.upserts.insert("input:i".into(), text("1"));
    setup.upserts.insert("derived".into(), value);
    setup
        .dependencies
        .insert("derived".into(), BTreeSet::from(["input:i".to_owned()]));
    host.apply_delta(setup).unwrap();
    host
}

#[test]
fn a_refused_commutative_operation_leaves_no_read_of_its_inputs_behind() {
    let oversized = "m".repeat(ptr_semdb::MAX_DELTA_BYTES);
    for (value, op, refused) in [
        (
            text("not a counter"),
            BranchOp::Add {
                key: "derived".into(),
                amount: 1,
            },
            BranchError::NotACounter {
                key: "derived".into(),
            },
        ),
        (
            counter_value(i64::MAX),
            BranchOp::Add {
                key: "derived".into(),
                amount: 1,
            },
            BranchError::CounterOverflow {
                key: "derived".into(),
            },
        ),
        (
            set_value(&BTreeSet::new()).unwrap(),
            BranchOp::SetInsert {
                key: "derived".into(),
                member: oversized.clone(),
            },
            BranchError::InvalidValue {
                key: "derived".into(),
            },
        ),
    ] {
        let mut host = derived_host(value);
        let mut work = branch(&host, "b1");
        assert_eq!(work.stage_commutative(op), Err(refused.clone()));
        let sealed = work.seal().unwrap();
        assert!(sealed.ops().is_empty(), "{refused:?}");
        assert!(
            sealed.reads().is_empty(),
            "{refused:?}: {:?}",
            sealed.reads()
        );
        // A change to the input the refused operation would have depended on
        // does not refuse a branch that does nothing with it.
        change(&mut host, "input:i", text("2"));
        let certification = certify(&sealed, &host.snapshot(), no_lifecycle).unwrap();
        assert!(certification.plan().is_noop(), "{refused:?}");
    }
}

#[test]
fn a_second_generation_of_a_relied_on_target_is_refused_when_declared() {
    let host = host_with(&[("a", text("1"))]);
    let mut work = branch(&host, "b1");
    work.rely_on("capsule:x", Generation(1)).unwrap();
    // The same declaration again changes nothing.
    work.rely_on("capsule:x", Generation(1)).unwrap();
    let refused = work.rely_on("capsule:x", Generation(2)).unwrap_err();
    assert_eq!(
        refused,
        BranchError::ConflictingReliance {
            target: "capsule:x".into(),
            relied: Generation(1),
            declared: Generation(2),
        }
    );
    assert_eq!(refused.code(), "PTR_BRANCH_CONFLICTING_RELIANCE");
    let sealed = work.seal().unwrap();
    assert_eq!(
        *sealed.relied(),
        [("capsule:x".to_owned(), Generation(1))]
            .into_iter()
            .collect()
    );
    // What the branch concluded from generation 1 is not certified merely
    // because generation 2 is live.
    let upgraded = |_: &str, generation: Generation| {
        Some(if generation == Generation(2) {
            Validity::Live
        } else {
            Validity::Superseded
        })
    };
    assert_eq!(
        certify(&sealed, &host.snapshot(), upgraded).unwrap_err(),
        BranchError::LifecycleChanged {
            targets: BTreeSet::from(["capsule:x".to_owned()])
        }
    );
}

#[test]
fn a_sealed_branch_that_breaks_what_staging_guarantees_is_refused_when_rebuilt_or_certified() {
    // A Put of a key the branch no longer records as read would merge as a
    // blind overwrite of a concurrent change: no such branch is rebuilt.
    let host = host_with(&[("k", text("a"))]);
    let mut work = branch(&host, "b1");
    work.read("k").unwrap();
    work.put("k", text("mine")).unwrap();
    let mut parts = work.seal().unwrap().into_parts();
    parts.reads.clear();
    assert_eq!(
        SealedBranch::from_parts(parts).unwrap_err(),
        BranchError::UnreadTarget { key: "k".into() }
    );

    // Without the base value of a key it adds to, the branch cannot tell a
    // rebase from a clean merge: no such branch is rebuilt either.
    let mut host = host_with(&[("c", counter_value(10))]);
    let mut work = branch(&host, "b2");
    work.stage_commutative(BranchOp::Add {
        key: "c".into(),
        amount: 1,
    })
    .unwrap();
    let sealed = work.seal().unwrap();
    change(&mut host, "c", counter_value(20));
    assert!(matches!(
        certify(&sealed, &host.snapshot(), no_lifecycle),
        Ok(Certification::Rebased(plan)) if plan.rebased == BTreeSet::from(["c".to_owned()])
    ));
    let mut parts = sealed.into_parts();
    parts.touched_base.clear();
    assert!(matches!(
        SealedBranch::from_parts(parts),
        Err(BranchError::MalformedSeal { key, .. }) if key == "c"
    ));

    // Staging reads every input of the key an operation touches; a branch
    // without that read never declared the dependency. The input set hides
    // behind its digest, so the branch is rebuilt, and certification, which
    // knows the inputs from the target, refuses it.
    let host = derived_host(text("2"));
    let mut work = branch(&host, "b3");
    work.read("derived").unwrap();
    work.put("derived", text("3")).unwrap();
    let sealed = work.seal().unwrap();
    assert!(certify(&sealed, &host.snapshot(), no_lifecycle).is_ok());
    let mut parts = sealed.into_parts();
    parts.reads.remove("input:i");
    let sealed = SealedBranch::from_parts(parts).unwrap();
    assert_eq!(
        certify(&sealed, &host.snapshot(), no_lifecycle).unwrap_err(),
        BranchError::Conflict {
            keys: BTreeSet::from(["input:i".into()])
        }
    );
}

#[test]
fn a_set_the_journal_cannot_carry_is_refused_rather_than_truncated() {
    // A member as long as a whole delta may be: the set's encoding, with its
    // count and length fields, is longer than any delta the journal accepts.
    let oversized = "m".repeat(ptr_semdb::MAX_DELTA_BYTES);
    assert_eq!(set_value(&BTreeSet::from([oversized.clone()])), None);
    // An empty member has no canonical encoding either.
    assert_eq!(set_value(&BTreeSet::from([String::new()])), None);
    // Whatever set_value encodes reads back exactly.
    let members = BTreeSet::from(["a".to_owned(), "b\nc".to_owned()]);
    assert_eq!(
        read_set_value(&set_value(&members).unwrap()),
        Some(members.clone())
    );

    let host = host_with(&[("tags", set_value(&members).unwrap())]);
    let mut work = branch(&host, "b1");
    assert_eq!(
        work.stage_commutative(BranchOp::SetInsert {
            key: "tags".into(),
            member: oversized,
        }),
        Err(BranchError::InvalidValue { key: "tags".into() })
    );
    assert!(work.seal().unwrap().ops().is_empty());
}

#[test]
fn a_merge_delta_the_journal_cannot_carry_is_refused_at_approval_and_commit_not_truncated() {
    // The set's own encoding is exactly MAX_DELTA_BYTES (count, then length
    // and bytes of "a" and of the new member), so staging accepts it; the
    // delta that carries it also holds the key, the type, the source and
    // their framing, which no journal delta has room for.
    let host = host_with(&[(
        "tags",
        set_value(&BTreeSet::from(["a".to_owned()])).unwrap(),
    )]);
    let mut work = branch(&host, "b1");
    work.stage_commutative(BranchOp::SetInsert {
        key: "tags".into(),
        member: "m".repeat(ptr_semdb::MAX_DELTA_BYTES - 13),
    })
    .unwrap();
    let certification = certify(&work.seal().unwrap(), &host.snapshot(), no_lifecycle).unwrap();
    let plan = certification.plan();
    assert_eq!(
        plan.digest(),
        Err(BranchError::InvalidValue {
            key: "<merge delta>".into()
        })
    );
    let mut host = host;
    let before = host.revision();
    assert_eq!(
        host.apply_delta(plan.delta.clone()),
        Err(ptr_semdb::SemanticError::LimitExceeded)
    );
    assert_eq!(host.revision(), before);
}
