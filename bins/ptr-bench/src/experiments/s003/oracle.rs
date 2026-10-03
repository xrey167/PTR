//! The oracle: what a branch's declaration and the state it is merged into
//! imply, decided from the reference model alone.
//!
//! It never calls `certify` or a digest. It reads the branch's declared
//! footprint (`program::Footprint`), the model at the position the branch
//! opened at (the footprint holds what the branch saw there) and the model at
//! the position it is merged at, and answers two questions: which hazards the
//! merge would run into if it were committed, and exactly what the runtime
//! must do, following the precedence certification documents.

use std::collections::{BTreeMap, BTreeSet};

use super::model::{Delta, Model, Net, Op, OpRefusal, Refusal, Val, Validity};
use super::program::Footprint;

/// The hazards a merge would commit if certification did not refuse it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Hazards {
    /// A key the branch overwrites holds another value than it read.
    pub lost_update: bool,
    /// A key the branch read and did not overwrite holds another value.
    pub stale_read: bool,
    /// A prefix the branch scanned holds other keys.
    pub phantom: bool,
    /// A prefix holds the same keys with another value under one.
    pub stale_scan: bool,
    /// A key the branch touches declares another input set.
    pub stale_input: bool,
    /// A generation the branch relied on is not live.
    pub stale_reliance: bool,
}

impl Hazards {
    /// The names of the hazards that hold, in a fixed order.
    pub fn names(&self) -> Vec<&'static str> {
        [
            (self.lost_update, "lost_update"),
            (self.stale_read, "stale_read"),
            (self.phantom, "phantom"),
            (self.stale_scan, "stale_scan"),
            (self.stale_input, "stale_input"),
            (self.stale_reliance, "stale_reliance"),
        ]
        .into_iter()
        .filter_map(|(holds, name)| holds.then_some(name))
        .collect()
    }
}

/// What the runtime must do with a branch merged now.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Predicted {
    /// Refused: a generation the branch relied on is not live.
    LifecycleChanged(BTreeSet<String>),
    /// Refused: what the branch declared no longer holds, at these keys (a
    /// scanned prefix as `<prefix>*`).
    Conflict(BTreeSet<String>),
    /// Certified, but applying its operations to the state is refused.
    OperationRefused(OpRefusal),
    /// Certified, but the merged delta does not apply.
    DeltaRefused(Refusal),
    /// The state the merge would publish holds a negative counter: held.
    VerificationRejected { rebased: BTreeSet<String> },
    /// Certified and admitted, but it changes nothing.
    NoChange { rebased: BTreeSet<String> },
    /// Certified and admitted: it commits `delta`, with these keys rebased.
    Merge {
        rebased: BTreeSet<String>,
        delta: Delta,
    },
}

/// The oracle's whole judgement of a branch against a target state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Judgement {
    pub hazards: Hazards,
    pub predicted: Predicted,
    /// A set member's last operation leaves it as the base had it while the
    /// target holds it the other way: certification refuses that as a
    /// conflict on the set's key, although merging would not lose an update.
    pub set_member_undo: bool,
}

fn holds_member(value: Option<&Val>, member: &str) -> bool {
    value
        .and_then(Val::as_set)
        .is_some_and(|members| members.contains(member))
}

/// Whether any valid counter under `ctr:` in `model` is negative.
pub fn negative_counter(model: &Model) -> bool {
    model.negative_counter()
}

/// The hazards `footprint` runs into merged into `target`.
pub fn hazards(footprint: &Footprint, target: &Model) -> Hazards {
    let mut hazards = Hazards::default();
    for (key, seen) in &footprint.reads {
        if target.value(key) != seen.as_ref() {
            if footprint.written.contains(key) {
                hazards.lost_update = true;
            } else {
                hazards.stale_read = true;
            }
        }
    }
    for (prefix, entries) in &footprint.scans {
        let now = target.entries_under(prefix);
        let keys_then: Vec<&str> = entries.iter().map(|(key, _)| key.as_str()).collect();
        let keys_now: Vec<&str> = now.iter().map(|(key, _)| key.as_str()).collect();
        if keys_then != keys_now {
            hazards.phantom = true;
        } else if *entries != now {
            hazards.stale_scan = true;
        }
    }
    for (key, touched) in &footprint.touched {
        if target.inputs(key) != touched.inputs {
            hazards.stale_input = true;
        }
    }
    hazards.stale_reliance = !stale_reliance(footprint, target).is_empty();
    hazards
}

/// The relied targets whose generation is not live in `target`.
fn stale_reliance(footprint: &Footprint, target: &Model) -> BTreeSet<String> {
    footprint
        .relied
        .iter()
        .filter(|(subject, generation)| {
            target.lifecycle.validity(subject, **generation) != Some(Validity::Live)
        })
        .map(|(subject, _)| subject.clone())
        .collect()
}

/// The last operation on each set member: whether the member is in the set
/// afterwards and whether it was in the base's.
fn last_set_operations(footprint: &Footprint) -> BTreeMap<(&str, &str), (bool, bool)> {
    let mut members = BTreeMap::new();
    for op in &footprint.ops {
        if let Some((member, after, in_base)) = op.set_member() {
            members.insert((op.key(), member), (after, in_base));
        }
    }
    members
}

/// The keys a branch conflicts on in `target`, following certification's
/// documented rules: changed reads, `<prefix>*` for a changed scan, a touched
/// key whose input set changed, an input of a touched key with an unchanged
/// input set that the branch did not read, and a set whose member the
/// branch's last operation leaves as the base had it while the target does
/// not.
fn conflicts(footprint: &Footprint, target: &Model) -> (BTreeSet<String>, bool) {
    let mut keys = BTreeSet::new();
    for (key, seen) in &footprint.reads {
        if target.value(key) != seen.as_ref() {
            keys.insert(key.clone());
        }
    }
    for (prefix, entries) in &footprint.scans {
        if target.entries_under(prefix) != *entries {
            keys.insert(format!("{prefix}*"));
        }
    }
    for (key, touched) in &footprint.touched {
        let inputs = target.inputs(key);
        if inputs != touched.inputs {
            keys.insert(key.clone());
        } else {
            keys.extend(
                inputs
                    .into_iter()
                    .filter(|input| !footprint.reads.contains_key(input)),
            );
        }
    }
    let mut set_member_undo = false;
    for ((key, member), (after, in_base)) in last_set_operations(footprint) {
        if after == in_base && holds_member(target.value(key), member) != in_base {
            keys.insert(key.to_string());
            set_member_undo = true;
        }
    }
    (keys, set_member_undo)
}

/// The delta a merge of `ops` publishes onto `target`, built as certification
/// documents: each key's operations folded over the target's value, and a key
/// left out when its value is unchanged unless the merge changes a key it is
/// derived from, directly or through other derived keys of the target's
/// graph. Propagates the first type, empty-member or overflow
/// refusal without changing `target`.
pub fn merge_delta(ops: &[Op], target: &Model) -> Result<Delta, OpRefusal> {
    let mut finals: BTreeMap<&str, Option<Val>> = BTreeMap::new();
    for op in ops {
        let current = match finals.remove(op.key()) {
            Some(value) => value,
            None => target.value(op.key()).cloned(),
        };
        let next = op.apply(current.as_ref())?;
        finals.insert(op.key(), next);
    }
    let changed: BTreeSet<&str> = finals
        .iter()
        .filter(|(key, value)| value.as_ref() != target.value(key))
        .map(|(key, _)| *key)
        .collect();
    let derived_from_changed = |key: &str| {
        let mut pending: Vec<String> = target.inputs(key).into_iter().collect();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        while let Some(input) = pending.pop() {
            if changed.contains(input.as_str()) {
                return true;
            }
            if seen.insert(input.clone()) {
                pending.extend(target.inputs(&input));
            }
        }
        false
    };
    let mut delta = Delta::default();
    for (key, value) in &finals {
        match (value, target.value(key)) {
            (Some(next), Some(current)) if next == current && !derived_from_changed(key) => {}
            (Some(next), _) => {
                delta.upserts.insert((*key).to_string(), next.clone());
            }
            (None, Some(_)) => {
                delta.removals.insert((*key).to_string());
            }
            (None, None) => {}
        }
    }
    Ok(delta)
}

/// The commutative keys of `footprint` whose value in `target` is not the one
/// the branch's base held: what a merge rebases.
pub fn rebased_keys(footprint: &Footprint, target: &Model) -> BTreeSet<String> {
    footprint
        .ops
        .iter()
        .filter(|op| op.commutes())
        .filter(|op| {
            footprint
                .touched
                .get(op.key())
                .is_some_and(|touched| target.value(op.key()) != touched.base.as_ref())
        })
        .map(|op| op.key().to_string())
        .collect()
}

/// The plan a certified merge of `footprint` into `target` would carry: the
/// rebased keys and the delta, or `None` when an operation is refused. With
/// the target's revision it is what a plan digest covers, so two calls that
/// return equal values and revisions stand for the same plan.
pub fn plan(footprint: &Footprint, target: &Model) -> Option<(BTreeSet<String>, Delta)> {
    let delta = merge_delta(&footprint.ops, target).ok()?;
    Some((rebased_keys(footprint, target), delta))
}

/// Judge `footprint` merged into `target`.
pub fn judge(footprint: &Footprint, target: &Model) -> Judgement {
    let hazards = hazards(footprint, target);
    let (conflict_keys, set_member_undo) = conflicts(footprint, target);
    let stale = stale_reliance(footprint, target);
    let predicted = if !stale.is_empty() {
        Predicted::LifecycleChanged(stale)
    } else if !conflict_keys.is_empty() {
        Predicted::Conflict(conflict_keys)
    } else {
        let rebased = rebased_keys(footprint, target);
        match merge_delta(&footprint.ops, target) {
            Err(refusal) => Predicted::OperationRefused(refusal),
            Ok(delta) => match target.plan(&delta) {
                Err(refusal) => Predicted::DeltaRefused(refusal),
                Ok(plan) if target.negative_counter_after(&plan.net) => {
                    Predicted::VerificationRejected { rebased }
                }
                Ok(plan) if !plan.moved() => Predicted::NoChange { rebased },
                Ok(_) => Predicted::Merge { rebased, delta },
            },
        }
    };
    Judgement {
        hazards,
        predicted,
        set_member_undo,
    }
}

/// Whether a commutative key of the branch holds, after the merge, other than
/// what its operations make of what the target held before it: an increment
/// or a set change was lost. `before` and `after` say what a key held.
/// Also returns true when an operation cannot be applied to its operand.
pub fn lost_increments_with(
    footprint: &Footprint,
    before: impl Fn(&str) -> Option<Val>,
    after: impl Fn(&str) -> Option<Val>,
) -> bool {
    footprint.commutative.iter().any(|key| {
        let mut expected = before(key);
        for op in footprint.ops.iter().filter(|op| op.key() == key) {
            match op.apply(expected.as_ref()) {
                Ok(next) => expected = next,
                Err(_) => return true,
            }
        }
        after(key) != expected
    })
}

/// [`lost_increments_with`] over the state before and the state after.
pub fn lost_increments(footprint: &Footprint, before: &Model, after: &Model) -> bool {
    lost_increments_with(
        footprint,
        |key| before.value(key).cloned(),
        |key| after.value(key).cloned(),
    )
}

/// [`lost_increments_with`] over the state before and what a merge changed.
pub fn lost_increments_net(footprint: &Footprint, before: &Model, net: &Net) -> bool {
    lost_increments_with(
        footprint,
        |key| before.value(key).cloned(),
        |key| net.value_after(key, before.value(key)),
    )
}

#[cfg(test)]
mod tests {
    use super::super::program::Touched;
    use super::*;

    fn set(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|item| (*item).to_string()).collect()
    }

    fn model(values: &[(&str, &str)]) -> Model {
        let mut delta = Delta::default();
        for (key, value) in values {
            delta.upserts.insert((*key).to_string(), Val::text(*value));
        }
        let mut model = Model::default();
        model.apply(&delta).expect("applies");
        model
    }

    fn footprint_put(key: &str, seen: &str, put: &str) -> Footprint {
        Footprint {
            reads: [(key.to_string(), Some(Val::text(seen)))].into(),
            written: set(&[key]),
            touched: [(
                key.to_string(),
                Touched {
                    base: Some(Val::text(seen)),
                    inputs: BTreeSet::new(),
                },
            )]
            .into(),
            ops: vec![Op::Put {
                key: key.to_string(),
                value: Val::text(put),
            }],
            ..Footprint::default()
        }
    }

    #[test]
    fn nothing_changed_since_the_base_is_a_merge_of_the_operations() {
        let target = model(&[("k", "1")]);
        let judgement = judge(&footprint_put("k", "1", "2"), &target);
        assert_eq!(judgement.hazards, Hazards::default());
        let Predicted::Merge { rebased, delta } = judgement.predicted else {
            panic!("a merge");
        };
        assert!(rebased.is_empty());
        assert_eq!(delta.upserts, [("k".to_string(), Val::text("2"))].into());
    }

    #[test]
    fn an_overwritten_key_that_changed_is_a_lost_update_and_a_conflict() {
        let judgement = judge(&footprint_put("k", "1", "2"), &model(&[("k", "9")]));
        assert_eq!(judgement.hazards.names(), vec!["lost_update"]);
        assert_eq!(judgement.predicted, Predicted::Conflict(set(&["k"])));
        // Removed at the target: the key is gone, which is a change too.
        let judgement = judge(&footprint_put("k", "1", "2"), &Model::default());
        assert_eq!(judgement.hazards.names(), vec!["lost_update"]);
    }

    #[test]
    fn a_read_only_key_that_changed_is_a_stale_read() {
        let mut footprint = footprint_put("k", "1", "2");
        footprint
            .reads
            .insert("other".to_string(), Some(Val::text("5")));
        let judgement = judge(&footprint, &model(&[("k", "1"), ("other", "6")]));
        assert_eq!(judgement.hazards.names(), vec!["stale_read"]);
        assert_eq!(judgement.predicted, Predicted::Conflict(set(&["other"])));
    }

    #[test]
    fn a_scanned_prefix_that_gained_a_key_is_a_phantom_and_one_that_changed_a_value_is_stale() {
        let scanned = |entries: &[(&str, &str)]| Footprint {
            scans: [(
                "p:".to_string(),
                entries
                    .iter()
                    .map(|(key, value)| ((*key).to_string(), Val::text(*value)))
                    .collect(),
            )]
            .into(),
            ..Footprint::default()
        };
        let footprint = scanned(&[("p:a", "1")]);
        let phantom = judge(&footprint, &model(&[("p:a", "1"), ("p:b", "2")]));
        assert_eq!(phantom.hazards.names(), vec!["phantom"]);
        assert_eq!(phantom.predicted, Predicted::Conflict(set(&["p:*"])));
        let stale = judge(&footprint, &model(&[("p:a", "9")]));
        assert_eq!(stale.hazards.names(), vec!["stale_scan"]);
        assert_eq!(stale.predicted, Predicted::Conflict(set(&["p:*"])));
        let removed = judge(&footprint, &Model::default());
        assert_eq!(removed.hazards.names(), vec!["phantom"]);
        let unchanged = judge(&footprint, &model(&[("p:a", "1"), ("q:z", "3")]));
        assert_eq!(unchanged.hazards, Hazards::default());
    }

    #[test]
    fn a_touched_key_with_another_input_set_is_a_stale_input_even_when_no_value_changed() {
        let mut target = model(&[("a", "1"), ("b", "1"), ("t", "2")]);
        target
            .apply(&Delta {
                dependencies: [("t".to_string(), set(&["a", "b"]))].into(),
                upserts: [("t".to_string(), Val::text("2"))].into(),
                ..Delta::default()
            })
            .expect("t derives from a and b");
        let footprint = Footprint {
            reads: [
                ("t".to_string(), Some(Val::text("2"))),
                ("a".to_string(), Some(Val::text("1"))),
                ("b".to_string(), Some(Val::text("1"))),
            ]
            .into(),
            written: set(&["t"]),
            touched: [(
                "t".to_string(),
                Touched {
                    base: Some(Val::text("2")),
                    inputs: set(&["a", "b"]),
                },
            )]
            .into(),
            ops: vec![Op::Put {
                key: "t".to_string(),
                value: Val::text("3"),
            }],
            ..Footprint::default()
        };
        assert_eq!(judge(&footprint, &target).hazards, Hazards::default());
        // The same values, rewired to the inputs {a} alone.
        target
            .apply(&Delta {
                dependencies: [("t".to_string(), set(&["a"]))].into(),
                upserts: [("t".to_string(), Val::text("2"))].into(),
                ..Delta::default()
            })
            .expect("rewired");
        let judgement = judge(&footprint, &target);
        assert_eq!(judgement.hazards.names(), vec!["stale_input"]);
        assert_eq!(judgement.predicted, Predicted::Conflict(set(&["t"])));
    }

    #[test]
    fn an_input_the_branch_did_not_read_is_a_conflict_although_the_input_set_is_unchanged() {
        let mut target = model(&[("a", "1"), ("t", "1")]);
        target
            .apply(&Delta {
                dependencies: [("t".to_string(), set(&["a"]))].into(),
                upserts: [("t".to_string(), Val::text("1"))].into(),
                ..Delta::default()
            })
            .expect("derived");
        let footprint = Footprint {
            reads: [("t".to_string(), Some(Val::text("1")))].into(),
            written: set(&["t"]),
            touched: [(
                "t".to_string(),
                Touched {
                    base: Some(Val::text("1")),
                    inputs: set(&["a"]),
                },
            )]
            .into(),
            ops: vec![Op::Put {
                key: "t".to_string(),
                value: Val::text("2"),
            }],
            ..Footprint::default()
        };
        let judgement = judge(&footprint, &target);
        assert_eq!(judgement.hazards, Hazards::default());
        assert_eq!(judgement.predicted, Predicted::Conflict(set(&["a"])));
    }

    #[test]
    fn a_relied_generation_that_is_not_live_refuses_before_anything_else() {
        let mut target = model(&[("k", "9")]);
        target.lifecycle.set_live("policy-0", 1);
        let mut footprint = footprint_put("k", "1", "2");
        footprint.relied.insert("policy-0".to_string(), 1);
        assert_eq!(
            judge(&footprint, &target).predicted,
            Predicted::Conflict(set(&["k"]))
        );
        target.lifecycle.revoke("policy-0", 1);
        let judgement = judge(&footprint, &target);
        assert_eq!(
            judgement.predicted,
            Predicted::LifecycleChanged(set(&["policy-0"]))
        );
        assert!(judgement.hazards.stale_reliance);
        target.lifecycle.set_live("policy-0", 2);
        assert!(
            judge(&footprint, &target).hazards.stale_reliance,
            "superseded is not live either"
        );
        let unknown = Footprint {
            relied: [("policy-1".to_string(), 1)].into(),
            ..Footprint::default()
        };
        assert!(
            judge(&unknown, &Model::default()).hazards.stale_reliance,
            "an unknown target is not live"
        );
    }

    fn counter_footprint(base: i64, amount: i64) -> Footprint {
        Footprint {
            commutative: set(&["ctr:0"]),
            touched: [(
                "ctr:0".to_string(),
                Touched {
                    base: Some(Val::counter(base, "g")),
                    inputs: BTreeSet::new(),
                },
            )]
            .into(),
            ops: vec![Op::Add {
                key: "ctr:0".to_string(),
                amount,
            }],
            ..Footprint::default()
        }
    }

    fn with_counter(value: i64) -> Model {
        let mut model = Model::default();
        model
            .apply(&Delta {
                upserts: [("ctr:0".to_string(), Val::counter(value, "g"))].into(),
                ..Delta::default()
            })
            .expect("counter");
        model
    }

    #[test]
    fn a_commutative_key_that_changed_is_rebased_and_merged_over_the_current_value() {
        let judgement = judge(&counter_footprint(50, -7), &with_counter(60));
        assert_eq!(judgement.hazards, Hazards::default());
        let Predicted::Merge { rebased, delta } = judgement.predicted else {
            panic!("a merge");
        };
        assert_eq!(rebased, set(&["ctr:0"]));
        assert_eq!(delta.upserts["ctr:0"], Val::counter(53, "ptr-branch/op"));
        // Unchanged: clean, not rebased.
        let Predicted::Merge { rebased, .. } =
            judge(&counter_footprint(50, -7), &with_counter(50)).predicted
        else {
            panic!("a merge");
        };
        assert!(rebased.is_empty());
    }

    #[test]
    fn a_rebase_that_goes_negative_is_held_by_verification_and_an_empty_change_is_no_change() {
        let judgement = judge(&counter_footprint(50, -10), &with_counter(4));
        assert_eq!(
            judgement.predicted,
            Predicted::VerificationRejected {
                rebased: set(&["ctr:0"])
            }
        );
        // Adding nothing to a counter another operation wrote leaves it as it is;
        // one from another source becomes the operation's, which is a change.
        let mut from_operation = Model::default();
        from_operation
            .apply(&Delta {
                upserts: [("ctr:0".to_string(), Val::counter(50, "ptr-branch/op"))].into(),
                ..Delta::default()
            })
            .expect("counter");
        let judgement = judge(&counter_footprint(50, 0), &from_operation);
        assert_eq!(
            judgement.predicted,
            Predicted::NoChange {
                // Another source is another value than the base's, so it is rebased.
                rebased: set(&["ctr:0"])
            }
        );
        assert!(matches!(
            judge(&counter_footprint(50, 0), &with_counter(50)).predicted,
            Predicted::Merge { .. }
        ));
        assert!(negative_counter(&with_counter(-1)));
        assert!(!negative_counter(&with_counter(0)));
    }

    #[test]
    fn a_set_member_left_as_the_base_had_it_conflicts_when_the_target_moved_it() {
        let member = |after_insert: bool, in_base: bool| Footprint {
            commutative: set(&["set:0"]),
            touched: [(
                "set:0".to_string(),
                Touched {
                    base: None,
                    inputs: BTreeSet::new(),
                },
            )]
            .into(),
            ops: vec![if after_insert {
                Op::SetInsert {
                    key: "set:0".into(),
                    member: "m".into(),
                    in_base,
                }
            } else {
                Op::SetRemove {
                    key: "set:0".into(),
                    member: "m".into(),
                    in_base,
                }
            }],
            ..Footprint::default()
        };
        let with_set = |members: &[&str]| {
            let mut model = Model::default();
            model
                .apply(&Delta {
                    upserts: [("set:0".to_string(), Val::set(&set(members), "g"))].into(),
                    ..Delta::default()
                })
                .expect("set");
            model
        };
        // Insert of a member the base already held, and a target that removed it: undo.
        let judgement = judge(&member(true, true), &with_set(&[]));
        assert!(judgement.set_member_undo);
        assert_eq!(judgement.predicted, Predicted::Conflict(set(&["set:0"])));
        // Insert of a member the base lacked, and a target that inserted it too: kept, not duplicated.
        let judgement = judge(&member(true, false), &with_set(&["m"]));
        assert!(!judgement.set_member_undo);
        assert!(matches!(
            judgement.predicted,
            Predicted::Merge { .. } | Predicted::NoChange { .. }
        ));
        // Remove of a member the base lacked, and a target that inserted it: undo.
        assert!(judge(&member(false, false), &with_set(&["m"])).set_member_undo);
    }

    #[test]
    fn a_merge_leaves_out_an_unchanged_put_unless_a_key_it_is_derived_from_changes() {
        let mut target = model(&[("a", "1"), ("t", "1")]);
        target
            .apply(&Delta {
                dependencies: [("t".to_string(), set(&["a"]))].into(),
                upserts: [("t".to_string(), Val::text("1"))].into(),
                ..Delta::default()
            })
            .expect("derived");
        let put = |key: &str, value: &str| Op::Put {
            key: key.to_string(),
            value: Val::text(value),
        };
        let unchanged = merge_delta(&[put("t", "1")], &target).expect("delta");
        assert!(unchanged.upserts.is_empty(), "the put restates the value");
        let both = merge_delta(&[put("a", "2"), put("t", "1")], &target).expect("delta");
        assert_eq!(
            both.upserts.len(),
            2,
            "the input changes, so the derived put is kept"
        );
        let removed = merge_delta(
            &[Op::Remove {
                key: "a".to_string(),
            }],
            &target,
        )
        .expect("delta");
        assert_eq!(removed.removals, set(&["a"]));
        let nothing = merge_delta(
            &[Op::Remove {
                key: "absent".to_string(),
            }],
            &target,
        )
        .expect("delta");
        assert_eq!(nothing, Delta::default());
    }

    #[test]
    fn increments_are_lost_when_the_state_after_is_not_the_fold_over_the_state_before() {
        let footprint = counter_footprint(50, -7);
        let before = with_counter(60);
        let mut after = before.clone();
        after
            .apply(&Delta {
                upserts: [("ctr:0".to_string(), Val::counter(53, "ptr-branch/op"))].into(),
                ..Delta::default()
            })
            .expect("applies");
        assert!(!lost_increments(&footprint, &before, &after));
        // The branch's own value written over the current one: 43, the update of 60 lost.
        let mut lost = before.clone();
        lost.apply(&Delta {
            upserts: [("ctr:0".to_string(), Val::counter(43, "ptr-branch/op"))].into(),
            ..Delta::default()
        })
        .expect("applies");
        assert!(lost_increments(&footprint, &before, &lost));
    }
}
