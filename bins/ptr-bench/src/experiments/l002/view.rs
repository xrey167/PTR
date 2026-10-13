//! What a recovered state means in the harness's terms: the logical history a snapshot and the
//! events above it stand for, the payload a member's snapshot carries, and the authority a
//! runtime restored from a compacted snapshot gives, compared with the oracle's.
//!
//! The oracle (`l001::model`) is shared with L001-v2 and shares no code with the product. The
//! judgement of a logical history reuses `l001::check::judge_history` (the exact prefix, the
//! indices 1 to n, the acknowledged revocations, and the authority a replay of the whole
//! history gives); the restored-runtime comparison is this file's own.

use super::super::l001::check::{add, bump, judge_history, Expectation};
use super::super::l001::model::{Ev, Model, Standing, Workload};
use crate::experiments::layered_kv::emit::Tally;
use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_ledger::CommittedEvent;
use ptr_runtime::compacted::CompactedAnchor;
use ptr_runtime::PtrRuntime;
use ptr_types::{CapabilityId, CommitIndex, Effect, Generation, TypeId, Validity};
use std::collections::BTreeSet;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// The history `prefix` stands for, committed at indices 1 to its length, followed by `events`
/// as the product holds them (their own indices).
pub fn logical(prefix: &[Ev], events: &[CommittedEvent]) -> Vec<CommittedEvent> {
    let mut view: Vec<CommittedEvent> = prefix
        .iter()
        .enumerate()
        .map(|(i, event)| CommittedEvent {
            index: CommitIndex(i as u64 + 1),
            event: event.to_ledger(),
        })
        .collect();
    view.extend(events.iter().cloned());
    view
}

/// Judge a logical history against the workload: `acked` events were acknowledged, `attempted`
/// had begun.
pub fn judge_logical(
    tally: &mut Tally,
    arm: &str,
    workload: &Workload,
    acked: usize,
    attempted: usize,
    view: &[CommittedEvent],
) {
    let expect = Expectation {
        acked,
        attempted,
        must_refuse: false,
        damage: None,
        class: None,
        witness: 0,
    };
    judge_history(tally, arm, workload, &expect, view);
}

/// The text of a snapshot payload: the events a member's state is made of, one per line.
pub fn encode_payload(events: &[Ev]) -> Vec<u8> {
    let mut text = String::new();
    for event in events {
        match event {
            Ev::Commit {
                subject,
                generation,
            } => text.push_str(&format!("C {subject} {generation}\n")),
            Ev::Supersede { subject, old, new } => {
                text.push_str(&format!("S {subject} {old} {new}\n"));
            }
            Ev::Revoke {
                subject,
                generation,
            } => text.push_str(&format!("R {subject} {generation}\n")),
        }
    }
    text.into_bytes()
}

/// The events a payload holds, or `None` when it is not one the harness wrote.
pub fn decode_payload(bytes: &[u8]) -> Option<Vec<Ev>> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut events = Vec::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.split(' ').collect();
        events.push(match parts.as_slice() {
            ["C", subject, generation] => Ev::Commit {
                subject: (*subject).to_string(),
                generation: generation.parse().ok()?,
            },
            ["S", subject, old, new] => Ev::Supersede {
                subject: (*subject).to_string(),
                old: old.parse().ok()?,
                new: new.parse().ok()?,
            },
            ["R", subject, generation] => Ev::Revoke {
                subject: (*subject).to_string(),
                generation: generation.parse().ok()?,
            },
            _ => return None,
        });
    }
    Some(events)
}

/// The runtime's answer for one pair: may an action proceed, and what is the generation's
/// standing.
fn decide(runtime: &PtrRuntime, subject: &str, generation: u64) -> (bool, Option<Standing>) {
    let action = ActionIr {
        operation: "update".into(),
        target: subject.to_owned(),
        capability: CapabilityId::from("memory.write"),
        effect: Effect::Mutation,
        input_type: TypeId::from("SemanticCapsule"),
        generation: Generation(generation),
        revision: runtime.revision(),
        payload: vec![],
    };
    let authorized = runtime.authorize_action(&action).is_ok();
    let standing = runtime
        .generation_validity(subject, Generation(generation))
        .map(|validity| match validity {
            Validity::Live => Standing::Live,
            Validity::Superseded => Standing::Superseded,
            Validity::Revoked => Standing::Revoked,
            Validity::Disputed => Standing::Disputed,
        });
    (authorized, standing)
}

/// Compare a runtime with the oracle for every pair of the universe, over the first `count`
/// events of the workload. The same counters as the replay comparison of `l001::check`.
pub fn compare_runtime(
    tally: &mut Tally,
    arm: &str,
    workload: &Workload,
    runtime: &mut PtrRuntime,
    count: usize,
    acked: usize,
) {
    let model = Model::of(workload.prefix(count.min(workload.events.len())));
    let acked_revoked: BTreeSet<(String, u64)> = workload.events
        [..acked.min(workload.events.len())]
        .iter()
        .filter_map(|event| match event {
            Ev::Revoke {
                subject,
                generation,
            } => Some((subject.clone(), *generation)),
            _ => None,
        })
        .collect();
    runtime
        .permissions_mut()
        .capabilities
        .insert(CapabilityId::from("memory.write"));
    runtime.permissions_mut().allow_mutation = true;
    for (subject, generation) in &workload.universe {
        let decided = catch_unwind(AssertUnwindSafe(|| decide(runtime, subject, *generation)));
        let (authorized, standing) = match decided {
            Ok(answer) => answer,
            Err(payload) => {
                super::cutover::explain(
                    "the runtime panicked on a decision",
                    &super::cutover::panic_text(payload.as_ref()),
                );
                bump(tally, arm, "product_panics");
                continue;
            }
        };
        let wanted = model.authorizable(subject, *generation);
        let revoked_by_ack = acked_revoked.contains(&(subject.clone(), *generation));
        if !authorized
            && revoked_by_ack
            && model.standing(subject, *generation) == Some(Standing::Revoked)
        {
            bump(tally, arm, "revoked_pairs_denied");
        }
        if authorized && (!wanted || revoked_by_ack) {
            bump(tally, arm, "stale_accepts");
        } else if authorized != wanted || standing != model.standing(subject, *generation) {
            bump(tally, arm, "authority_mismatches");
        }
    }
}

/// What a restore from a compacted snapshot gave.
pub enum Restored {
    Runtime(Box<PtrRuntime>),
    Refused,
    Panicked,
}

/// Restore the runtime from the snapshot the host retained and the records above the
/// snapshot's floor. The snapshot's floor can lie above the log's floor: only the records past
/// the snapshot are replayed.
pub fn restore(
    snapshot_bytes: &[u8],
    anchor: CompactedAnchor,
    retained: &[CommittedEvent],
) -> Restored {
    let above: Vec<CommittedEvent> = retained
        .iter()
        .filter(|event| event.index.0 > anchor.floor.index.0)
        .cloned()
        .collect();
    match catch_unwind(AssertUnwindSafe(|| {
        PtrRuntime::restore_compacted(PtrConfig::default(), snapshot_bytes, anchor, &above)
    })) {
        Err(_) => Restored::Panicked,
        Ok(Err(_)) => Restored::Refused,
        Ok(Ok(runtime)) => Restored::Runtime(Box::new(runtime)),
    }
}

/// Count a restore: a refusal of a state the table calls recoverable is an unjustified
/// refusal, a panic is the product's, and a runtime is compared with the oracle.
#[allow(clippy::too_many_arguments)]
pub fn judge_restore(
    tally: &mut Tally,
    arm: &str,
    workload: &Workload,
    snapshot_bytes: &[u8],
    anchor: CompactedAnchor,
    retained: &[CommittedEvent],
    held: usize,
    acked: usize,
) {
    match restore(snapshot_bytes, anchor, retained) {
        Restored::Panicked => {
            super::cutover::explain("the restore from a compacted snapshot panicked", "");
            bump(tally, arm, "product_panics");
        }
        Restored::Refused => bump(tally, arm, "unjustified_refusals"),
        Restored::Runtime(mut runtime) => {
            add(tally, arm, "runtimes_restored", 1);
            compare_runtime(tally, arm, workload, &mut runtime, held, acked);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::rng::Rng;

    #[test]
    fn a_payload_round_trips_and_a_foreign_one_is_refused() {
        let workload = super::super::super::l001::model::generate(&mut Rng::new(9), 10, 16);
        let bytes = encode_payload(&workload.events);
        assert_eq!(decode_payload(&bytes), Some(workload.events.clone()));
        assert_eq!(decode_payload(&[]), Some(Vec::new()));
        assert_eq!(decode_payload(b"X a 1\n"), None);
        assert_eq!(decode_payload(&[0xff, 0xfe]), None);
        assert_eq!(decode_payload(b"C a x\n"), None);
    }

    #[test]
    fn the_logical_history_numbers_the_prefix_from_one_and_keeps_the_products_indices() {
        let workload = super::super::super::l001::model::generate(&mut Rng::new(3), 10, 16);
        let tail: Vec<CommittedEvent> = workload.events[4..6]
            .iter()
            .enumerate()
            .map(|(i, event)| CommittedEvent {
                index: CommitIndex(5 + i as u64),
                event: event.to_ledger(),
            })
            .collect();
        let view = logical(&workload.events[..4], &tail);
        let indices: Vec<u64> = view.iter().map(|event| event.index.0).collect();
        assert_eq!(indices, vec![1, 2, 3, 4, 5, 6]);
        let mut tally = Tally::new(super::super::metrics::LIST);
        judge_logical(&mut tally, "t", &workload, 6, 6, &view);
        assert_eq!(super::super::metrics::hard_sum(&tally), 0);
    }
}
