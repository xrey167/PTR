use std::collections::{BTreeMap, BTreeSet};

use ptr_branch::{
    certify, counter_value, read_counter_value, read_set_value, set_value, Branch, BranchError,
    BranchId, BranchOp, Certification, InputsDigest, SealedBranch, SealedBranchParts, ValueDigest,
    COUNTER_TYPE,
};
use ptr_semdb::{SemanticDelta, SemanticHost, SemanticPayload, SemanticValue};
use ptr_types::{Generation, PrincipalId, TypeId, Validity};

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
                    in_base: false,
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
    assert_eq!(certification.plan().expected(), host.revision());
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

/// Commit a plan to a bare host, which has no lifecycle authority: only for
/// plans whose branch relied on no generation.
fn commit(host: &mut SemanticHost, certification: Certification) {
    let (expected, delta, relied) = certification.plan().clone().into_parts();
    assert!(relied.is_empty());
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
        Certification::Rebased(plan) => assert!(plan.rebased().contains("reserved:sku-1")),
        Certification::Clean(_) => panic!("the second branch must report its rebase"),
    }
    commit(&mut host, certification);
    let value = host.snapshot().value("reserved:sku-1").cloned().unwrap();
    assert_eq!(read_counter_value(&value), Some(10));
}

#[test]
fn certify_labels_a_plan_clean_exactly_when_it_rebased_nothing_and_the_plan_keeps_the_record() {
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

    let clean = certify(&first, &host.snapshot(), no_lifecycle).unwrap();
    assert!(matches!(&clean, Certification::Clean(plan) if plan.rebased().is_empty()));
    commit(&mut host, clean);
    let Certification::Rebased(plan) = certify(&second, &host.snapshot(), no_lifecycle).unwrap()
    else {
        panic!("the second branch rebased onto the first one's addition");
    };
    assert_eq!(
        plan.rebased(),
        &BTreeSet::from(["reserved:sku-1".to_owned()])
    );
    // The variants are public, so a certification built elsewhere can call
    // the plan clean. The label is all that changes: the plan's rebased keys,
    // which only certify sets, and the digest an approval binds are the
    // same.
    let relabelled = Certification::Clean(plan.clone());
    assert_eq!(relabelled.plan().rebased(), plan.rebased());
    assert_eq!(relabelled.plan().digest(), plan.digest());
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
    assert_eq!(from_a.expected(), from_b.expected());
    assert_eq!(from_a.delta(), from_b.delta());
    assert_ne!(from_a.dependencies(), from_b.dependencies());
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
    assert_eq!(first.expected(), second.expected());
    assert_eq!(first.delta(), second.delta());
    assert_eq!(first.dependencies(), second.dependencies());
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
                in_base: false,
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
        Ok(Certification::Rebased(plan)) if *plan.rebased() == BTreeSet::from(["c".to_owned()])
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
            in_base: false,
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
        in_base: false,
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
        host.apply_delta(plan.delta().clone()),
        Err(ptr_semdb::SemanticError::LimitExceeded)
    );
    assert_eq!(host.revision(), before);
}

/// A host whose `tags` set holds `members`.
fn tags_host(members: &[&str]) -> SemanticHost {
    let members: BTreeSet<String> = members.iter().map(|m| (*m).to_owned()).collect();
    host_with(&[("tags", set_value(&members).unwrap())])
}

fn tags(host: &SemanticHost) -> BTreeSet<String> {
    read_set_value(host.snapshot().value("tags").unwrap()).unwrap()
}

/// A branch over `host` that stages `ops` on `tags` alone.
fn set_branch(host: &SemanticHost, id: &str, ops: Vec<BranchOp>) -> SealedBranch {
    let mut work = branch(host, id);
    for op in ops {
        work.stage_commutative(op).unwrap();
    }
    work.seal().unwrap()
}

fn insert(member: &str) -> BranchOp {
    BranchOp::SetInsert {
        key: "tags".into(),
        member: member.into(),
        in_base: false,
    }
}

fn remove(member: &str) -> BranchOp {
    BranchOp::SetRemove {
        key: "tags".into(),
        member: member.into(),
        in_base: false,
    }
}

#[test]
fn staging_records_whether_a_set_member_was_in_the_base_whatever_the_operation_carried() {
    let host = tags_host(&["a"]);
    let mut work = branch(&host, "b1");
    for (member, carried) in [("a", false), ("x", true)] {
        work.stage_commutative(BranchOp::SetRemove {
            key: "tags".into(),
            member: member.into(),
            in_base: carried,
        })
        .unwrap();
    }
    assert_eq!(
        work.seal().unwrap().ops(),
        [
            BranchOp::SetRemove {
                key: "tags".into(),
                member: "a".into(),
                in_base: true,
            },
            BranchOp::SetRemove {
                key: "tags".into(),
                member: "x".into(),
                in_base: false,
            },
        ]
    );
}

#[test]
fn a_set_operation_that_would_undo_a_concurrent_change_of_its_member_is_a_conflict() {
    // Each case: the base's members, the branch that commits first, and the
    // branch certified after it. The second leaves `x` as its base had it,
    // so merging it would undo the first: a removal would delete the
    // concurrent insert, and an insert would resurrect the concurrent
    // removal. Staging an insert and then a removal nets to the base too.
    for (base, first, second) in [
        (vec![], vec![insert("x")], vec![remove("x")]),
        (vec!["x"], vec![remove("x")], vec![insert("x")]),
        (vec![], vec![insert("x")], vec![insert("x"), remove("x")]),
    ] {
        let mut host = tags_host(&base);
        let first = set_branch(&host, "first", first);
        let second = set_branch(&host, "second", second);
        let certification = certify(&first, &host.snapshot(), no_lifecycle).unwrap();
        commit(&mut host, certification);
        let after_first = tags(&host);
        let revision = host.revision();
        assert_eq!(
            certify(&second, &host.snapshot(), no_lifecycle).unwrap_err(),
            BranchError::Conflict {
                keys: BTreeSet::from(["tags".into()])
            },
            "base {base:?}"
        );
        assert_eq!(host.revision(), revision);
        assert_eq!(tags(&host), after_first);
    }
}

#[test]
fn concurrent_set_operations_that_move_a_member_the_same_way_both_merge() {
    // Two inserts of a member the base lacked, and two removals of one it
    // held: the second merges as a rebase and the member ends where both
    // branches moved it. A member the target holds as the base did merges
    // as staged.
    for (base, op, expected) in [
        (vec!["a"], insert as fn(&str) -> BranchOp, vec!["a", "x"]),
        (vec!["a", "x"], remove, vec!["a"]),
    ] {
        let mut host = tags_host(&base);
        let first = set_branch(&host, "first", vec![op("x")]);
        let second = set_branch(&host, "second", vec![op("x"), insert("y")]);
        let certification = certify(&first, &host.snapshot(), no_lifecycle).unwrap();
        commit(&mut host, certification);
        let certification = certify(&second, &host.snapshot(), no_lifecycle).unwrap();
        assert!(matches!(certification, Certification::Rebased(_)));
        commit(&mut host, certification);
        let mut expected: BTreeSet<String> = expected.iter().map(|m| (*m).to_owned()).collect();
        expected.insert("y".into());
        assert_eq!(tags(&host), expected);
    }
}

#[test]
fn an_unchanged_write_of_a_derived_key_whose_input_the_plan_changes_is_published_not_evicted() {
    // A commit evicts a derived key whose input it changes unless it writes
    // the key. The branch recomputed `d` from the new `a` and got the value
    // it already had, so the plan must still write it.
    let host_for = || {
        let mut host = SemanticHost::default();
        let mut setup = SemanticDelta::default();
        setup.upserts.insert("a".into(), text("1"));
        setup.upserts.insert("d".into(), text("x"));
        setup.upserts.insert(
            "s".into(),
            set_value(&BTreeSet::from(["m".to_owned()])).unwrap(),
        );
        setup
            .dependencies
            .insert("d".into(), BTreeSet::from(["a".to_owned()]));
        setup
            .dependencies
            .insert("s".into(), BTreeSet::from(["d".to_owned()]));
        host.apply_delta(setup).unwrap();
        host
    };
    let mut host = host_for();
    let mut work = branch(&host, "b1");
    work.read("a").unwrap();
    work.read("d").unwrap();
    work.put("a", text("2")).unwrap();
    work.put("d", text("x")).unwrap();
    // `s` is derived from `d`, so from `a` transitively, and the commit
    // evicts it even though `d` keeps its value. A set operation would build
    // on the evicted value and is refused; the branch writes the value it
    // recomputed instead, which is the value `s` already holds.
    assert_eq!(
        work.stage_commutative(BranchOp::SetInsert {
            key: "s".into(),
            member: "m".into(),
            in_base: true,
        }),
        Err(BranchError::EvictedOperand {
            key: "s".into(),
            input: Some("a".into()),
        })
    );
    assert_eq!(work.read("s").unwrap(), None);
    work.put("s", set_value(&BTreeSet::from(["m".to_owned()])).unwrap())
        .unwrap();
    let certification = certify(&work.seal().unwrap(), &host.snapshot(), no_lifecycle).unwrap();
    let delta = certification.plan().delta();
    assert_eq!(delta.upserts.get("d"), Some(&text("x")));
    assert!(delta.upserts.contains_key("s"));
    commit(&mut host, certification);
    let snapshot = host.snapshot();
    assert_eq!(snapshot.get("a"), Some("2"));
    assert_eq!(snapshot.get("d"), Some("x"));
    assert_eq!(
        read_set_value(snapshot.value("s").unwrap()),
        Some(BTreeSet::from(["m".to_owned()]))
    );

    // Where nothing the key derives from changes, an unchanged write is
    // left out and the plan stays a no-op.
    let host = host_for();
    let mut work = branch(&host, "b2");
    work.read("d").unwrap();
    work.put("d", text("x")).unwrap();
    let certification = certify(&work.seal().unwrap(), &host.snapshot(), no_lifecycle).unwrap();
    assert!(certification.plan().is_noop());
}

#[test]
fn a_branch_read_shows_a_derived_key_its_own_write_to_an_input_evicts() {
    // `total` is derived from `tax`, which is derived from `rate`.
    let mut host = SemanticHost::default();
    let mut setup = SemanticDelta::default();
    setup.upserts.insert("input:rate".into(), text("0.19"));
    setup.upserts.insert("derived:tax".into(), text("19"));
    setup.upserts.insert("derived:total".into(), text("119"));
    setup.dependencies.insert(
        "derived:tax".into(),
        BTreeSet::from(["input:rate".to_owned()]),
    );
    setup.dependencies.insert(
        "derived:total".into(),
        BTreeSet::from(["derived:tax".to_owned()]),
    );
    host.apply_delta(setup).unwrap();

    // Writing the value an input already holds changes nothing.
    let mut work = branch(&host, "b1");
    work.read("input:rate").unwrap();
    work.put("input:rate", text("0.19")).unwrap();
    assert_eq!(work.read("derived:tax").unwrap(), Some(text("19")));

    // A changed input evicts every key derived from it, directly or not.
    work.put("input:rate", text("0.07")).unwrap();
    assert_eq!(work.read("derived:tax").unwrap(), None);
    assert_eq!(work.read("derived:total").unwrap(), None);
    // A derived key the branch writes shows what it wrote.
    work.put("derived:tax", text("7")).unwrap();
    assert_eq!(work.read("derived:tax").unwrap(), Some(text("7")));
    assert_eq!(work.read("derived:total").unwrap(), None);
    let sealed = work.seal().unwrap();
    // The recorded reads are of the base values, whatever was shown.
    assert_eq!(
        sealed.reads().get("derived:total"),
        Some(&ValueDigest::of("derived:total", Some(&text("119"))).unwrap())
    );

    // What the branch read is what the merge publishes.
    let certification = certify(&sealed, &host.snapshot(), no_lifecycle).unwrap();
    commit(&mut host, certification);
    let snapshot = host.snapshot();
    assert_eq!(snapshot.get("input:rate"), Some("0.07"));
    assert_eq!(snapshot.get("derived:tax"), Some("7"));
    assert_eq!(snapshot.get("derived:total"), None);
}

#[test]
fn a_prefix_scan_shows_the_branch_s_own_work_while_its_range_digest_stays_the_base() {
    // Under `order:`, `order:4` is derived from `input:rate` and `order:8`
    // from `order:4`; `orders` is outside the prefix.
    let host_for = || {
        let mut host = SemanticHost::default();
        let mut setup = SemanticDelta::default();
        for (key, value) in [
            ("input:rate", text("0.19")),
            ("order:1", text("open")),
            ("order:2", text("open")),
            ("order:4", text("19")),
            ("order:5", counter_value(1)),
            (
                "order:6",
                set_value(&BTreeSet::from(["a".to_owned()])).unwrap(),
            ),
            ("order:8", text("119")),
            ("orders", text("outside the prefix")),
        ] {
            setup.upserts.insert(key.into(), value);
        }
        setup
            .dependencies
            .insert("order:4".into(), BTreeSet::from(["input:rate".to_owned()]));
        setup
            .dependencies
            .insert("order:8".into(), BTreeSet::from(["order:4".to_owned()]));
        host.apply_delta(setup).unwrap();
        host
    };
    let mut host = host_for();
    let entry = |key: &str, value: SemanticValue| (key.to_owned(), value);

    // With nothing staged, a scan shows the base.
    let mut pristine = branch(&host, "pristine");
    assert_eq!(
        pristine.scan_prefix("order:").unwrap(),
        vec![
            entry("order:1", text("open")),
            entry("order:2", text("open")),
            entry("order:4", text("19")),
            entry("order:5", counter_value(1)),
            entry(
                "order:6",
                set_value(&BTreeSet::from(["a".to_owned()])).unwrap()
            ),
            entry("order:8", text("119")),
        ]
    );
    let base_scans = pristine.seal().unwrap().scans().clone();

    // The branch updates, removes and inserts keys under the prefix, adds to
    // a counter and to a set, creates a counter, and changes an input of a
    // derived key, writing that key's recomputed value.
    let mut work = branch(&host, "b1");
    for key in ["order:1", "order:2", "order:3", "order:4", "input:rate"] {
        work.read(key).unwrap();
    }
    work.put("order:1", text("closed")).unwrap();
    work.remove("order:2").unwrap();
    work.put("order:3", text("open")).unwrap();
    work.put("input:rate", text("0.07")).unwrap();
    work.put("order:4", text("7")).unwrap();
    work.stage_commutative(BranchOp::Add {
        key: "order:5".into(),
        amount: 2,
    })
    .unwrap();
    work.stage_commutative(BranchOp::SetInsert {
        key: "order:6".into(),
        member: "b".into(),
        in_base: false,
    })
    .unwrap();
    work.stage_commutative(BranchOp::Add {
        key: "order:7".into(),
        amount: 4,
    })
    .unwrap();

    // The scan shows what merging the branch onto its unchanged base
    // leaves: the removed key and `order:8`, which the changed `order:4`
    // evicts, are gone.
    let scanned = work.scan_prefix("order:").unwrap();
    assert_eq!(
        scanned,
        vec![
            entry("order:1", text("closed")),
            entry("order:3", text("open")),
            entry("order:4", text("7")),
            entry("order:5", counter_value(3)),
            entry(
                "order:6",
                set_value(&BTreeSet::from(["a".to_owned(), "b".to_owned()])).unwrap()
            ),
            entry("order:7", counter_value(4)),
        ]
    );
    // Every key under the prefix reads as the scan shows it.
    for key in (1..=8).map(|n| format!("order:{n}")) {
        let shown = scanned
            .iter()
            .find(|(scanned, _)| *scanned == key)
            .map(|(_, value)| value.clone());
        assert_eq!(work.read(&key).unwrap(), shown, "{key}");
    }
    let sealed = work.seal().unwrap();
    // The scan is recorded as the base's range, whatever it showed.
    assert_eq!(sealed.scans(), &base_scans);

    // So a key a concurrent commit inserts under the prefix, which no read
    // covers, is still a phantom that refuses the branch.
    let mut moved = host_for();
    change(&mut moved, "order:9", text("open"));
    assert_eq!(
        certify(&sealed, &moved.snapshot(), no_lifecycle).unwrap_err(),
        BranchError::Conflict {
            keys: BTreeSet::from(["order:*".to_owned()])
        }
    );

    // Against the unchanged base, what the scan showed is what the merge
    // leaves.
    let certification = certify(&sealed, &host.snapshot(), no_lifecycle).unwrap();
    commit(&mut host, certification);
    assert_eq!(
        branch(&host, "after").scan_prefix("order:").unwrap(),
        scanned
    );
}

/// A host where `derived:d` holds `value` and is derived from `input:rate`,
/// and `derived:total` holds a counter derived from `derived:d`.
fn rate_host(value: SemanticValue) -> SemanticHost {
    let mut host = SemanticHost::default();
    let mut setup = SemanticDelta::default();
    setup.upserts.insert("input:rate".into(), text("0.19"));
    setup.upserts.insert("derived:d".into(), value);
    setup
        .upserts
        .insert("derived:total".into(), counter_value(1));
    setup.dependencies.insert(
        "derived:d".into(),
        BTreeSet::from(["input:rate".to_owned()]),
    );
    setup.dependencies.insert(
        "derived:total".into(),
        BTreeSet::from(["derived:d".to_owned()]),
    );
    host.apply_delta(setup).unwrap();
    host
}

fn evicted(key: &str, input: &str) -> BranchError {
    BranchError::EvictedOperand {
        key: key.into(),
        input: Some(input.into()),
    }
}

#[test]
fn a_commutative_operation_never_builds_on_a_derived_value_the_branch_s_own_change_evicts() {
    let add = |key: &str| BranchOp::Add {
        key: key.into(),
        amount: 5,
    };
    let insert_m = |key: &str| BranchOp::SetInsert {
        key: key.into(),
        member: "m".into(),
        in_base: false,
    };
    // The branch changes the input first. The keys derived from it read as
    // absent, since the merge evicts them, and an operation that would build
    // on the value one held is refused, whatever that value is: a counter, a
    // set, text a set operation could not apply to, or a key derived from
    // the input through another derived key.
    let ab = BTreeSet::from(["a".to_owned(), "b".to_owned()]);
    for (value, op) in [
        (counter_value(100), add("derived:d")),
        (set_value(&ab).unwrap(), insert_m("derived:d")),
        (text("not a set"), insert_m("derived:d")),
        (counter_value(100), add("derived:total")),
    ] {
        let mut host = rate_host(value);
        let mut work = branch(&host, "b1");
        work.read("input:rate").unwrap();
        work.put("input:rate", text("0.07")).unwrap();
        let key = op.key().to_owned();
        assert_eq!(work.read(&key).unwrap(), None, "{op:?}");
        let refused = work.stage_commutative(op.clone()).unwrap_err();
        assert_eq!(refused, evicted(&key, "input:rate"), "{op:?}");
        assert_eq!(refused.code(), "PTR_BRANCH_EVICTED_OPERAND");
        // Nothing was recorded, not even a read of the key's inputs, and the
        // key still reads as the merge leaves it.
        assert_eq!(work.read(&key).unwrap(), None, "{op:?}");
        let sealed = work.seal().unwrap();
        assert_eq!(sealed.ops().len(), 1, "{op:?}");
        assert_eq!(
            sealed.reads().keys().collect::<Vec<_>>(),
            [&key, "input:rate"],
            "{op:?}"
        );
        let certification = certify(&sealed, &host.snapshot(), no_lifecycle).unwrap();
        commit(&mut host, certification);
        assert_eq!(host.snapshot().value(&key), None, "{op:?}");
    }

    // The operation first: a change to the input it builds on is refused,
    // and neither key moves.
    let mut host = rate_host(counter_value(100));
    let mut work = branch(&host, "b2");
    work.stage_commutative(add("derived:d")).unwrap();
    work.read("input:rate").unwrap();
    assert_eq!(
        work.put("input:rate", text("0.07")),
        Err(evicted("derived:d", "input:rate"))
    );
    assert_eq!(work.read("input:rate").unwrap(), Some(text("0.19")));
    assert_eq!(work.read("derived:d").unwrap(), Some(counter_value(105)));
    // Writing the value the input already holds changes nothing.
    work.put("input:rate", text("0.19")).unwrap();
    // A Put of the value recomputed from the new input is what the branch
    // stages instead: operations after it apply to it, and the input change
    // is accepted. A key derived from that changed value is evicted in turn.
    work.put("derived:d", counter_value(7)).unwrap();
    work.put("input:rate", text("0.07")).unwrap();
    work.stage_commutative(add("derived:d")).unwrap();
    assert_eq!(work.read("derived:d").unwrap(), Some(counter_value(12)));
    assert_eq!(
        work.stage_commutative(add("derived:total")),
        Err(evicted("derived:total", "derived:d"))
    );
    assert_eq!(work.read("derived:total").unwrap(), None);

    // What the branch read is what the merge publishes.
    let certification = certify(&work.seal().unwrap(), &host.snapshot(), no_lifecycle).unwrap();
    commit(&mut host, certification);
    let snapshot = host.snapshot();
    assert_eq!(snapshot.get("input:rate"), Some("0.07"));
    assert_eq!(snapshot.value("derived:d"), Some(&counter_value(12)));
    assert_eq!(snapshot.value("derived:total"), None);
}

#[test]
fn certification_refuses_a_commutative_operation_on_a_key_its_own_plan_evicts() {
    // A branch that adds to `derived:d` and changes `input:rate`, which
    // staging refuses in either order, rebuilt from parts (by hand, or from
    // a store written before staging refused it). Every digest matches, so
    // only the dependency graph shows that the addition would be published
    // onto the value the plan's own change evicts.
    let host = rate_host(counter_value(100));
    let target = host.snapshot();
    let digest = |key: &str| ValueDigest::of(key, target.value(key)).unwrap();
    let inputs = |key: &str| InputsDigest::of(key, target.inputs(key));
    let parts = SealedBranchParts {
        id: BranchId::from("b1"),
        author: PrincipalId::from("agent-1"),
        base_revision: target.revision,
        reads: BTreeMap::from([("input:rate".to_owned(), digest("input:rate"))]),
        scans: BTreeMap::new(),
        relied: BTreeMap::new(),
        touched_base: BTreeMap::from([
            ("derived:d".to_owned(), digest("derived:d")),
            ("input:rate".to_owned(), digest("input:rate")),
        ]),
        touched_inputs: BTreeMap::from([
            ("derived:d".to_owned(), inputs("derived:d")),
            ("input:rate".to_owned(), inputs("input:rate")),
        ]),
        ops: vec![
            BranchOp::Add {
                key: "derived:d".into(),
                amount: 5,
            },
            BranchOp::Put {
                key: "input:rate".into(),
                value: text("0.07"),
            },
        ],
    };
    let sealed = SealedBranch::from_parts(parts.clone()).unwrap();
    assert_eq!(
        certify(&sealed, &target, no_lifecycle).unwrap_err(),
        evicted("derived:d", "input:rate")
    );
    // The same parts with the input written back to its value certify: the
    // addition then builds on a value nothing evicts.
    let mut unchanged = parts;
    unchanged.ops[1] = BranchOp::Put {
        key: "input:rate".into(),
        value: text("0.19"),
    };
    let certification = certify(
        &SealedBranch::from_parts(unchanged).unwrap(),
        &target,
        no_lifecycle,
    )
    .unwrap();
    assert_eq!(
        certification.plan().delta().upserts.get("derived:d"),
        Some(&counter_value(105))
    );
}

#[test]
fn a_derived_key_cannot_be_removed_so_a_merge_never_drops_its_dependency_set() {
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
    let refused = work.remove("derived:tax").unwrap_err();
    assert_eq!(
        refused,
        BranchError::DerivedRemoval {
            key: "derived:tax".into()
        }
    );
    assert_eq!(refused.code(), "PTR_BRANCH_DERIVED_REMOVAL");
    // A refused removal records nothing, not even a read of the inputs.
    let sealed = work.seal().unwrap();
    assert!(sealed.ops().is_empty());
    assert!(!sealed.reads().contains_key("input:rate"));

    // A key with no inputs is still removed, and the derivation stands: a
    // change to the input evicts the derived key.
    let mut work = branch(&host, "b2");
    work.read("input:rate").unwrap();
    work.remove("input:rate").unwrap();
    let certification = certify(&work.seal().unwrap(), &host.snapshot(), no_lifecycle).unwrap();
    commit(&mut host, certification);
    let snapshot = host.snapshot();
    assert_eq!(snapshot.get("derived:tax"), None);
    assert_eq!(
        snapshot.inputs("derived:tax").collect::<Vec<_>>(),
        ["input:rate"]
    );
}

#[test]
fn the_plan_digest_changes_with_which_keys_were_rebased() {
    // Two branches with the same id, base, reads, operation and input sets
    // that differ only in the base value recorded for the counter they add
    // to: certified against one target they plan the same delta with the
    // same dependency digest, one clean and one rebased.
    let host = host_with(&[("c", counter_value(5))]);
    let target = host.snapshot();
    let parts = |base: i64| SealedBranchParts {
        id: BranchId::from("b1"),
        author: PrincipalId::from("agent-1"),
        base_revision: target.revision,
        reads: BTreeMap::new(),
        scans: BTreeMap::new(),
        relied: BTreeMap::new(),
        touched_base: BTreeMap::from([(
            "c".to_owned(),
            ValueDigest::of("c", Some(&counter_value(base))).unwrap(),
        )]),
        touched_inputs: BTreeMap::from([("c".to_owned(), InputsDigest::of("c", []))]),
        ops: vec![BranchOp::Add {
            key: "c".into(),
            amount: 1,
        }],
    };
    let clean = certify(
        &SealedBranch::from_parts(parts(5)).unwrap(),
        &target,
        no_lifecycle,
    )
    .unwrap();
    let rebased = certify(
        &SealedBranch::from_parts(parts(3)).unwrap(),
        &target,
        no_lifecycle,
    )
    .unwrap();
    assert!(matches!(clean, Certification::Clean(_)));
    assert!(matches!(rebased, Certification::Rebased(_)));
    let (clean, rebased) = (clean.plan(), rebased.plan());
    assert_eq!(clean.branch(), rebased.branch());
    assert_eq!(clean.expected(), rebased.expected());
    assert_eq!(clean.delta(), rebased.delta());
    assert_eq!(clean.dependencies(), rebased.dependencies());
    assert_ne!(clean.rebased(), rebased.rebased());
    assert_ne!(clean.digest().unwrap(), rebased.digest().unwrap());
}

#[test]
fn a_plan_carries_the_generations_its_branch_relied_on() {
    let host = host_with(&[("a", text("1"))]);
    let mut work = branch(&host, "b1");
    work.rely_on("capsule:policy", Generation(3)).unwrap();
    work.read("a").unwrap();
    work.put("a", text("2")).unwrap();
    let sealed = work.seal().unwrap();
    let live = |_: &str, _: Generation| Some(Validity::Live);
    let plan = certify(&sealed, &host.snapshot(), live)
        .unwrap()
        .plan()
        .clone();
    assert_eq!(plan.relied(), sealed.relied());
    let (expected, delta, relied) = plan.clone().into_parts();
    assert_eq!(expected, plan.expected());
    assert_eq!(&delta, plan.delta());
    assert_eq!(
        relied,
        BTreeMap::from([("capsule:policy".to_owned(), Generation(3))])
    );
}

/// The refusal of a commutative operation on `key`, which is derived and
/// holds no value in the snapshot the operation is checked against: an
/// earlier change to its inputs evicted it, or it was never computed.
fn absent_operand(key: &str) -> BranchError {
    BranchError::EvictedOperand {
        key: key.into(),
        input: None,
    }
}

/// A host where the counter `d` and the set `s` are derived from `m`, which
/// is derived from `n`.
fn chain_host() -> SemanticHost {
    let mut host = SemanticHost::default();
    let mut setup = SemanticDelta::default();
    setup.upserts.insert("n".into(), text("a"));
    setup.upserts.insert("m".into(), text("1"));
    setup.upserts.insert("d".into(), counter_value(10));
    setup.upserts.insert(
        "s".into(),
        set_value(&BTreeSet::from(["x".to_owned()])).unwrap(),
    );
    setup
        .dependencies
        .insert("m".into(), BTreeSet::from(["n".to_owned()]));
    for key in ["d", "s"] {
        setup
            .dependencies
            .insert(key.into(), BTreeSet::from(["m".to_owned()]));
    }
    host.apply_delta(setup).unwrap();
    host
}

#[test]
fn a_commutative_operation_never_builds_on_a_derived_value_a_concurrent_commit_evicts() {
    let add = BranchOp::Add {
        key: "d".into(),
        amount: 5,
    };
    let insert = BranchOp::SetInsert {
        key: "s".into(),
        member: "y".into(),
        in_base: false,
    };
    // Concurrent commits that evict `d` and `s` and leave every value the
    // branch read and every input set as the base had them: `m` changes and
    // changes back, or `n` changes and `m` is recomputed to the value it had.
    let changed_and_back = |host: &mut SemanticHost| {
        change(host, "m", text("2"));
        change(host, "m", text("1"));
    };
    let upstream = |host: &mut SemanticHost| {
        let mut delta = SemanticDelta::default();
        delta.upserts.insert("n".into(), text("b"));
        delta.upserts.insert("m".into(), text("1"));
        host.apply_delta(delta).unwrap();
    };
    for (concurrent, why) in [
        (
            changed_and_back as fn(&mut SemanticHost),
            "m changed and back",
        ),
        (upstream, "n changed and m recomputed to its value"),
    ] {
        for op in [add.clone(), insert.clone()] {
            let mut host = chain_host();
            let mut work = branch(&host, "b1");
            work.stage_commutative(op.clone()).unwrap();
            let sealed = work.seal().unwrap();
            // Staging read the key's input, as it always does.
            assert_eq!(sealed.reads().keys().collect::<Vec<_>>(), ["m"], "{why}");
            concurrent(&mut host);
            let target = host.snapshot();
            let key = op.key();
            assert_eq!(target.value(key), None, "{why}: {op:?}");
            assert_eq!(target.inputs(key).collect::<Vec<_>>(), ["m"], "{why}");
            assert_eq!(target.get("m"), Some("1"), "{why}");
            // Applied to the evicted key, the operation would publish 5, or
            // {y}, as derived from `m`: a value never computed from it.
            assert_eq!(
                certify(&sealed, &target, no_lifecycle),
                Err(absent_operand(key)),
                "{why}: {op:?}"
            );
        }
    }

    // A Put of the value recomputed from the inputs is the way through, and
    // an operation after it applies to it.
    let mut host = chain_host();
    changed_and_back(&mut host);
    let mut work = branch(&host, "b2");
    assert_eq!(work.read("d").unwrap(), None);
    assert_eq!(
        work.stage_commutative(add.clone()),
        Err(absent_operand("d"))
    );
    work.put("d", counter_value(10)).unwrap();
    work.stage_commutative(add).unwrap();
    assert_eq!(work.read("d").unwrap(), Some(counter_value(15)));
    let certification = certify(&work.seal().unwrap(), &host.snapshot(), no_lifecycle).unwrap();
    commit(&mut host, certification);
    let snapshot = host.snapshot();
    assert_eq!(snapshot.value("d"), Some(&counter_value(15)));
    assert_eq!(snapshot.inputs("d").collect::<Vec<_>>(), ["m"]);
}

#[test]
fn a_commutative_operation_on_a_derived_key_evicted_before_the_base_is_refused() {
    // `m` changed before the branch opened, evicting `d` and `s`: each reads
    // as absent, as a key never written does, but a value computed from
    // absence would be published as derived from `m`.
    let mut host = chain_host();
    change(&mut host, "m", text("2"));
    let base = host.snapshot();
    for op in [
        BranchOp::Add {
            key: "d".into(),
            amount: 5,
        },
        BranchOp::SetInsert {
            key: "s".into(),
            member: "y".into(),
            in_base: false,
        },
    ] {
        let key = op.key().to_owned();
        assert_eq!(base.value(&key), None);
        let mut work = branch(&host, "b1");
        assert_eq!(work.read(&key).unwrap(), None, "{op:?}");
        let refused = work.stage_commutative(op.clone()).unwrap_err();
        assert_eq!(refused, absent_operand(&key), "{op:?}");
        assert_eq!(refused.code(), "PTR_BRANCH_EVICTED_OPERAND");
        assert!(refused.to_string().contains("holds no value"), "{refused}");
        // Nothing was recorded, not even a read of `m`.
        let sealed = work.seal().unwrap();
        assert!(sealed.ops().is_empty(), "{op:?}");
        assert_eq!(sealed.reads().keys().collect::<Vec<_>>(), [&key], "{op:?}");

        // Parts that hold the operation anyway are rebuilt, since no sealing
        // invariant covers the base's values, and certification refuses
        // them against a target that holds no value for the key.
        let mut parts = sealed.into_parts();
        parts
            .reads
            .insert("m".into(), ValueDigest::of("m", base.value("m")).unwrap());
        parts
            .touched_base
            .insert(key.clone(), ValueDigest::of(&key, None).unwrap());
        parts
            .touched_inputs
            .insert(key.clone(), InputsDigest::of(&key, base.inputs(&key)));
        parts.ops.push(op.clone());
        let rebuilt = SealedBranch::from_parts(parts).unwrap();
        assert_eq!(
            certify(&rebuilt, &base, no_lifecycle),
            Err(absent_operand(&key)),
            "{op:?}"
        );
    }

    // A key with no value and no inputs is a counter or set not yet
    // written: an operation on it is staged and merged from zero.
    let mut work = branch(&host, "b2");
    work.stage_commutative(BranchOp::Add {
        key: "fresh".into(),
        amount: 5,
    })
    .unwrap();
    // A Put of the value recomputed from `m` is the way through, and an
    // operation after it applies to it.
    work.read("d").unwrap();
    work.put("d", counter_value(20)).unwrap();
    work.stage_commutative(BranchOp::Add {
        key: "d".into(),
        amount: 5,
    })
    .unwrap();
    assert_eq!(work.read("d").unwrap(), Some(counter_value(25)));
    let certification = certify(&work.seal().unwrap(), &host.snapshot(), no_lifecycle).unwrap();
    assert!(matches!(certification, Certification::Clean(_)));
    commit(&mut host, certification);
    let snapshot = host.snapshot();
    assert_eq!(snapshot.value("d"), Some(&counter_value(25)));
    assert_eq!(snapshot.value("fresh"), Some(&counter_value(5)));
}

#[test]
fn certification_refuses_a_staged_operation_whose_input_changes_only_against_the_target() {
    // `d` is derived from `m`, which is derived from the counter `n`. The
    // branch adds one to `n` and takes it away again, which leaves `n` as
    // its base had it, so staging accepts an addition to `d`.
    let mut host = SemanticHost::default();
    let mut setup = SemanticDelta::default();
    setup.upserts.insert("n".into(), counter_value(5));
    setup.upserts.insert("m".into(), counter_value(10));
    setup.upserts.insert("d".into(), counter_value(20));
    setup
        .dependencies
        .insert("m".into(), BTreeSet::from(["n".to_owned()]));
    setup
        .dependencies
        .insert("d".into(), BTreeSet::from(["m".to_owned()]));
    host.apply_delta(setup).unwrap();
    let mut work = branch(&host, "b1");
    for amount in [1, -1] {
        work.stage_commutative(BranchOp::Add {
            key: "n".into(),
            amount,
        })
        .unwrap();
    }
    work.stage_commutative(BranchOp::Add {
        key: "d".into(),
        amount: 5,
    })
    .unwrap();
    assert_eq!(work.read("d").unwrap(), Some(counter_value(25)));
    let sealed = work.seal().unwrap();
    assert!(matches!(
        certify(&sealed, &host.snapshot(), no_lifecycle),
        Ok(Certification::Clean(_))
    ));

    // A concurrent commit writes the same count to `n` under another source
    // and recomputes `m` and `d` to the values they had: no value the branch
    // read, no input set and no link of the graph changed.
    let ingress = SemanticValue::Payload(SemanticPayload {
        type_id: TypeId::from(COUNTER_TYPE),
        source: "ingress".into(),
        bytes: 5i64.to_le_bytes().to_vec(),
    });
    let mut delta = SemanticDelta::default();
    delta.upserts.insert("n".into(), ingress);
    delta.upserts.insert("m".into(), counter_value(10));
    delta.upserts.insert("d".into(), counter_value(20));
    host.apply_delta(delta).unwrap();
    let target = host.snapshot();
    assert_eq!(target.value("d"), Some(&counter_value(20)));
    assert_eq!(target.inputs("d").collect::<Vec<_>>(), ["m"]);
    assert_eq!(target.inputs("m").collect::<Vec<_>>(), ["n"]);
    // Applied to the target's `n`, the branch's additions keep the count
    // but give it the merge's source, which changes `n`: the plan would
    // evict `d`, so the addition staging accepted is refused.
    assert_eq!(
        certify(&sealed, &target, no_lifecycle),
        Err(evicted("d", "n"))
    );
}
