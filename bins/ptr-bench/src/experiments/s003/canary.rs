//! Canaries: nine fixed footprints and histories, each with the verdict the
//! oracle must give, judged in every case. They make the oracle itself
//! falsifiable: an oracle that has stopped noticing one hazard misses the
//! canary built to show it, and the run counts a `canary_miss`.
//!
//! The footprints are written by hand, not by the programs, so a defect in a
//! program or in a view cannot hide a defect in the oracle.

use std::collections::BTreeSet;

use super::model::{Delta, Model, Op, Val};
use super::oracle::{self, Hazards, Predicted};
use super::program::{Footprint, Touched};

/// One canary: its name and whether the oracle gives the verdict expected.
pub struct Canary {
    pub name: &'static str,
    pub check: fn() -> bool,
}

/// The canaries, in a fixed order.
pub fn canaries() -> Vec<Canary> {
    vec![
        Canary {
            name: "clean",
            check: clean,
        },
        Canary {
            name: "lost-update",
            check: lost_update,
        },
        Canary {
            name: "write-skew",
            check: write_skew,
        },
        Canary {
            name: "phantom-insert",
            check: phantom_insert,
        },
        Canary {
            name: "stale-scan-value",
            check: stale_scan_value,
        },
        Canary {
            name: "stale-input-swap",
            check: stale_input_swap,
        },
        Canary {
            name: "stale-reliance",
            check: stale_reliance,
        },
        Canary {
            name: "lost-increment",
            check: lost_increment,
        },
        Canary {
            name: "set-member-undo",
            check: set_member_undo,
        },
    ]
}

/// Run `canaries`: how many ran and the names of the ones that missed.
pub fn run(canaries: &[Canary]) -> (u64, Vec<&'static str>) {
    let missed = canaries
        .iter()
        .filter(|canary| !(canary.check)())
        .map(|canary| canary.name)
        .collect();
    (canaries.len() as u64, missed)
}

fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|item| (*item).to_string()).collect()
}

fn text(value: &str) -> Val {
    Val::text(value)
}

fn model(values: &[(&str, Val)]) -> Model {
    let mut delta = Delta::default();
    for (key, value) in values {
        delta.upserts.insert((*key).to_string(), value.clone());
    }
    let mut model = Model::default();
    model.apply(&delta).expect("a canary state applies");
    model
}

fn put(key: &str, value: &str) -> Op {
    Op::Put {
        key: key.to_string(),
        value: text(value),
    }
}

fn touched(base: Option<Val>, inputs: &[&str]) -> Touched {
    Touched {
        base,
        inputs: set(inputs),
    }
}

/// A branch that read `k` as "1" and overwrote it.
fn overwrite_k() -> Footprint {
    Footprint {
        reads: [("k".to_string(), Some(text("1")))].into(),
        written: set(&["k"]),
        touched: [("k".to_string(), touched(Some(text("1")), &[]))].into(),
        ops: vec![put("k", "2")],
        ..Footprint::default()
    }
}

fn verdict(footprint: &Footprint, target: &Model) -> (Vec<&'static str>, Predicted) {
    let judgement = oracle::judge(footprint, target);
    (judgement.hazards.names(), judgement.predicted)
}

fn clean() -> bool {
    let (hazards, predicted) = verdict(&overwrite_k(), &model(&[("k", text("1"))]));
    hazards.is_empty()
        && matches!(predicted, Predicted::Merge { ref rebased, .. } if rebased.is_empty())
}

fn lost_update() -> bool {
    let (hazards, predicted) = verdict(&overwrite_k(), &model(&[("k", text("9"))]));
    hazards == ["lost_update"] && predicted == Predicted::Conflict(set(&["k"]))
}

/// The branch read `a` and `b`, wrote `a` from both, and `b` moved.
fn write_skew() -> bool {
    let footprint = Footprint {
        reads: [
            ("a".to_string(), Some(text("50"))),
            ("b".to_string(), Some(text("50"))),
        ]
        .into(),
        written: set(&["a"]),
        touched: [("a".to_string(), touched(Some(text("50")), &[]))].into(),
        ops: vec![put("a", "49")],
        ..Footprint::default()
    };
    let (hazards, predicted) = verdict(&footprint, &model(&[("a", text("50")), ("b", text("0"))]));
    hazards == ["stale_read"] && predicted == Predicted::Conflict(set(&["b"]))
}

fn scanned() -> Footprint {
    Footprint {
        scans: [("p:".to_string(), vec![("p:a".to_string(), text("1"))])].into(),
        ..Footprint::default()
    }
}

fn phantom_insert() -> bool {
    let (hazards, predicted) = verdict(
        &scanned(),
        &model(&[("p:a", text("1")), ("p:b", text("2"))]),
    );
    hazards == ["phantom"] && predicted == Predicted::Conflict(set(&["p:*"]))
}

fn stale_scan_value() -> bool {
    let (hazards, predicted) = verdict(&scanned(), &model(&[("p:a", text("7"))]));
    hazards == ["stale_scan"] && predicted == Predicted::Conflict(set(&["p:*"]))
}

/// The derived key `t` was computed from {a, b}; the target rewired it to
/// {a} with the same value, and no value the branch read changed.
fn stale_input_swap() -> bool {
    let mut target = model(&[("a", text("1")), ("b", text("1")), ("t", text("2"))]);
    let rewire = Delta {
        upserts: [("t".to_string(), text("2"))].into(),
        dependencies: [("t".to_string(), set(&["a"]))].into(),
        ..Delta::default()
    };
    target.apply(&rewire).expect("t is rewired to a");
    let footprint = Footprint {
        reads: [
            ("t".to_string(), Some(text("2"))),
            ("a".to_string(), Some(text("1"))),
            ("b".to_string(), Some(text("1"))),
        ]
        .into(),
        written: set(&["t"]),
        touched: [("t".to_string(), touched(Some(text("2")), &["a", "b"]))].into(),
        ops: vec![put("t", "3")],
        ..Footprint::default()
    };
    let (hazards, predicted) = verdict(&footprint, &target);
    hazards == ["stale_input"] && predicted == Predicted::Conflict(set(&["t"]))
}

fn stale_reliance() -> bool {
    let mut target = model(&[]);
    target.lifecycle.set_live("policy-0", 1);
    target.lifecycle.revoke("policy-0", 1);
    let footprint = Footprint {
        relied: [("policy-0".to_string(), 1)].into(),
        ..Footprint::default()
    };
    let (hazards, predicted) = verdict(&footprint, &target);
    hazards == ["stale_reliance"] && predicted == Predicted::LifecycleChanged(set(&["policy-0"]))
}

/// The branch added -7 to a counter that was 50 at its base; it is 60 now.
/// Merged as an increment the counter reads 53; a merge that wrote the
/// branch's own 43 over it lost the other writer's ten.
fn lost_increment() -> bool {
    let counter = |value: i64| model(&[("ctr:0", Val::counter(value, "g"))]);
    let footprint = Footprint {
        commutative: set(&["ctr:0"]),
        touched: [(
            "ctr:0".to_string(),
            touched(Some(Val::counter(50, "g")), &[]),
        )]
        .into(),
        ops: vec![Op::Add {
            key: "ctr:0".to_string(),
            amount: -7,
        }],
        ..Footprint::default()
    };
    let before = counter(60);
    let mut kept = before.clone();
    kept.apply(&Delta {
        upserts: [("ctr:0".to_string(), Val::counter(53, "ptr-branch/op"))].into(),
        ..Delta::default()
    })
    .expect("applies");
    let mut lost = before.clone();
    lost.apply(&Delta {
        upserts: [("ctr:0".to_string(), Val::counter(43, "ptr-branch/op"))].into(),
        ..Delta::default()
    })
    .expect("applies");
    !oracle::lost_increments(&footprint, &before, &kept)
        && oracle::lost_increments(&footprint, &before, &lost)
}

/// The branch inserts a member the base already held, so its view changes
/// nothing; the target removed it meanwhile, and merging would undo that.
fn set_member_undo() -> bool {
    let footprint = Footprint {
        commutative: set(&["set:0"]),
        touched: [(
            "set:0".to_string(),
            touched(Some(Val::set(&set(&["m"]), "g")), &[]),
        )]
        .into(),
        ops: vec![Op::SetInsert {
            key: "set:0".to_string(),
            member: "m".to_string(),
            in_base: true,
        }],
        ..Footprint::default()
    };
    let target = model(&[("set:0", Val::set(&BTreeSet::new(), "g"))]);
    let judgement = oracle::judge(&footprint, &target);
    judgement.hazards == Hazards::default()
        && judgement.set_member_undo
        && judgement.predicted == Predicted::Conflict(set(&["set:0"]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_canary_gets_the_verdict_it_was_built_for() {
        let (ran, missed) = run(&canaries());
        assert_eq!(ran, 9);
        assert_eq!(missed, Vec::<&str>::new());
    }

    #[test]
    fn a_canary_the_oracle_gets_wrong_is_a_miss() {
        let canaries = vec![
            Canary {
                name: "right",
                check: || true,
            },
            Canary {
                name: "wrong",
                check: || false,
            },
        ];
        assert_eq!(run(&canaries), (2, vec!["wrong"]));
    }

    #[test]
    fn the_canaries_are_named_once_each() {
        let names: BTreeSet<&str> = canaries().iter().map(|canary| canary.name).collect();
        assert_eq!(names.len(), canaries().len());
    }
}
