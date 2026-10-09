//! The harness's own model of what the runtime must do: an independent revision mirror and an
//! independent scan of the committed ledger.
//!
//! Nothing here calls `PtrRuntime::revision`, `SemanticHost::is_stale` or the runtime's
//! generation tables to decide anything; the mirror counts the writes the harness itself makes
//! and the scan replays the committed records from their own contents. The runtime's numbers
//! are only ever compared with these, never read into them (a mutation of the harness that
//! feeds the mirror the runtime's number would make every comparison hold, which is why the
//! planted harness defects of `tests/mutations.toml` break the mirror the other way).

use ptr_ledger::{CommittedEvent, LedgerEvent};
use ptr_types::ActionIr;
use std::collections::{BTreeMap, BTreeSet};

/// Why an action is stale in the mirror's view.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Staleness {
    Fresh,
    /// The action's revision is not the semantic revision.
    Revision,
    /// The action's generation is not the live one, or was revoked.
    Generation,
}

/// The mirror: the semantic revision and the target's generations as the harness has made
/// them happen, plus the executor's log.
#[derive(Debug)]
pub struct Mirror {
    revision: u64,
    foreign: BTreeMap<String, String>,
    outputs: BTreeMap<String, Vec<u8>>,
    live_generation: u64,
    revoked: BTreeSet<u64>,
    /// The revision the active run is bound to: the mirror's revision after the run's own latest
    /// write, never after anybody else's.
    run_bound: u64,
    /// The executor's log: the mirror's verdict on each action at the moment the executor ran.
    pub calls: Vec<Staleness>,
}

impl Mirror {
    /// A fresh runtime: revision 0, generation 1 of the target live.
    pub fn new() -> Self {
        Self {
            revision: 0,
            foreign: BTreeMap::new(),
            outputs: BTreeMap::new(),
            live_generation: 1,
            revoked: BTreeSet::new(),
            run_bound: 0,
            calls: Vec::new(),
        }
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn generation_is_dead(&self) -> bool {
        self.live_generation != 1 || !self.revoked.is_empty()
    }

    pub fn run_bound(&self) -> u64 {
        self.run_bound
    }

    /// A run ingests its request text under a key of its own: a new key, so the revision moves.
    pub fn run_ingests(&mut self) {
        self.revision += 1;
        self.run_bound = self.revision;
    }

    /// A run promotes a Pod output under the key of (its request, the Pod): the revision moves
    /// iff the bytes differ from what that key holds.
    pub fn run_promotes(&mut self, request: &str, bytes: &[u8]) {
        if self
            .outputs
            .insert(request.to_string(), bytes.to_vec())
            .as_deref()
            != Some(bytes)
        {
            self.revision += 1;
        }
        self.run_bound = self.revision;
    }

    /// A write by somebody other than the run. `true` iff it moved the revision (a key that
    /// holds another value, or none).
    pub fn foreign_write(&mut self, key: &str, value: &str) -> bool {
        let moved = self
            .foreign
            .insert(key.to_string(), value.to_string())
            .as_deref()
            != Some(value);
        if moved {
            self.revision += 1;
        }
        moved
    }

    pub fn supersede(&mut self, new_generation: u64) {
        self.live_generation = new_generation;
    }

    pub fn revoke(&mut self, generation: u64) {
        self.revoked.insert(generation);
    }

    /// Whether the active run's bound is behind the revision: somebody else wrote since the
    /// run's own latest write.
    pub fn run_is_stale(&self) -> bool {
        self.run_bound != self.revision
    }

    pub fn staleness(&self, revision: u64, generation: u64) -> Staleness {
        if revision != self.revision {
            Staleness::Revision
        } else if generation != self.live_generation || self.revoked.contains(&generation) {
            Staleness::Generation
        } else {
            Staleness::Fresh
        }
    }

    /// The executor's log entry for `action`, judged against the mirror now.
    pub fn record_call(&mut self, action: &ActionIr) -> Staleness {
        let staleness = self.staleness(action.revision.0, action.generation.0);
        self.calls.push(staleness);
        staleness
    }

    pub fn stale_calls(&self) -> i64 {
        self.calls
            .iter()
            .filter(|staleness| **staleness != Staleness::Fresh)
            .count() as i64
    }
}

/// The semantic revision and the target generations as the committed ledger itself records
/// them, replayed in order from the records alone (the `revision` of each committed delta; the
/// capsule records and revocations). It is the harness' second, mirror-free opinion of "the
/// current state": a write the harness did not make, or a step it mis-modelled, still shows
/// here, because the runtime's own records carry it.
#[derive(Clone, Debug, Default)]
pub struct LedgerState {
    revision: u64,
    live: BTreeMap<String, u64>,
    revoked: BTreeSet<(String, u64)>,
}

impl LedgerState {
    pub fn replay(events: &[CommittedEvent]) -> Self {
        let mut state = Self::default();
        for committed in events {
            state.apply(&committed.event);
        }
        state
    }

    pub fn apply(&mut self, event: &LedgerEvent) {
        match event {
            LedgerEvent::SemanticDeltaCommitted { revision, .. } => self.revision = revision.0,
            LedgerEvent::CapsuleCommitted {
                capsule,
                generation,
                ..
            } => {
                self.live.insert(capsule.to_string(), generation.0);
            }
            LedgerEvent::CapsuleSuperseded { capsule, new, .. } => {
                self.live.insert(capsule.to_string(), new.0);
            }
            LedgerEvent::Revoked {
                subject,
                generation,
            } => {
                self.revoked.insert((subject.clone(), generation.0));
            }
            _ => {}
        }
    }

    /// Revision first, then generation, as the mirror judges.
    pub fn staleness(&self, target: &str, revision: u64, generation: u64) -> Staleness {
        if revision != self.revision {
            Staleness::Revision
        } else if self.live.get(target) != Some(&generation)
            || self.revoked.contains(&(target.to_string(), generation))
        {
            Staleness::Generation
        } else {
            Staleness::Fresh
        }
    }
}

/// What the scan of a committed ledger found.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Scan {
    pub attempts: i64,
    pub stale_attempts: i64,
}

/// Replays the committed records in order and counts the `EffectAttempted` records whose
/// revision or generation was not the live one at the moment they were committed.
pub fn scan(events: &[CommittedEvent]) -> Scan {
    let mut state = LedgerState::default();
    let mut found = Scan::default();
    for committed in events {
        if let LedgerEvent::EffectAttempted {
            target,
            generation,
            revision,
            ..
        } = &committed.event
        {
            found.attempts += 1;
            found.stale_attempts +=
                i64::from(state.staleness(target, revision.0, generation.0) != Staleness::Fresh);
        }
        state.apply(&committed.event);
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_types::{
        CapabilityId, CapsuleId, CommitIndex, Effect, Generation, ProjectId, Revision, TypeId,
    };

    fn action(revision: u64, generation: u64) -> ActionIr {
        ActionIr {
            operation: "write".into(),
            target: "t".into(),
            capability: CapabilityId::from("c"),
            effect: Effect::Mutation,
            input_type: TypeId::from("Bytes"),
            generation: Generation(generation),
            revision: Revision(revision),
            payload: vec![],
        }
    }

    #[test]
    fn the_mirror_counts_writes_and_tells_the_runs_from_everybody_elses() {
        let mut mirror = Mirror::new();
        assert!(!mirror.run_is_stale());
        mirror.run_ingests();
        assert_eq!((mirror.revision(), mirror.run_bound()), (1, 1));
        mirror.run_promotes("r", b"a");
        assert_eq!((mirror.revision(), mirror.run_bound()), (2, 2));
        assert!(!mirror.run_is_stale());
        assert!(mirror.foreign_write("k", "v"));
        assert_eq!((mirror.revision(), mirror.run_bound()), (3, 2));
        assert!(mirror.run_is_stale());
        // The same value again moves nothing; another value does.
        assert!(!mirror.foreign_write("k", "v"));
        assert_eq!(mirror.revision(), 3);
        assert!(mirror.foreign_write("k", "w"));
        // The run's next write rebases its bound onto the revision it produced.
        mirror.run_promotes("r", b"b");
        assert!(!mirror.run_is_stale());
        // The same bytes under the same request key move nothing.
        let before = mirror.revision();
        mirror.run_promotes("r", b"b");
        assert_eq!(mirror.revision(), before);
    }

    #[test]
    fn staleness_is_revision_first_then_generation() {
        let mut mirror = Mirror::new();
        mirror.run_ingests();
        assert_eq!(mirror.staleness(1, 1), Staleness::Fresh);
        assert_eq!(mirror.staleness(0, 1), Staleness::Revision);
        assert_eq!(mirror.staleness(2, 1), Staleness::Revision);
        mirror.supersede(2);
        assert_eq!(mirror.staleness(1, 1), Staleness::Generation);
        assert_eq!(mirror.staleness(1, 2), Staleness::Fresh);
        let mut revoked = Mirror::new();
        revoked.revoke(1);
        assert_eq!(revoked.staleness(0, 1), Staleness::Generation);
        assert!(revoked.generation_is_dead());
        assert!(!Mirror::new().generation_is_dead());
    }

    #[test]
    fn the_executor_log_flags_a_stale_call() {
        let mut mirror = Mirror::new();
        mirror.run_ingests();
        assert_eq!(mirror.record_call(&action(1, 1)), Staleness::Fresh);
        assert_eq!(mirror.record_call(&action(0, 1)), Staleness::Revision);
        assert_eq!(mirror.record_call(&action(1, 7)), Staleness::Generation);
        assert_eq!(mirror.stale_calls(), 2);
    }

    fn attempt(target: &str, generation: u64, revision: u64) -> LedgerEvent {
        LedgerEvent::EffectAttempted {
            key: None,
            project: ProjectId::from("p"),
            principal: "x".into(),
            target: target.into(),
            operation: "write".into(),
            capability: CapabilityId::from("c"),
            effect: Effect::Mutation,
            generation: Generation(generation),
            revision: Revision(revision),
            verification: ptr_types::VerificationLevel::Deterministic,
            action_digest: [0; 32],
        }
    }

    fn delta(revision: u64) -> LedgerEvent {
        LedgerEvent::SemanticDeltaCommitted {
            base_revision: Revision(revision - 1),
            revision: Revision(revision),
            encoded_delta: vec![],
            origin: ptr_ledger::SemanticOrigin::Legacy,
        }
    }

    fn at(events: Vec<LedgerEvent>) -> Vec<CommittedEvent> {
        events
            .into_iter()
            .enumerate()
            .map(|(index, event)| CommittedEvent {
                index: CommitIndex(index as u64 + 1),
                event,
            })
            .collect()
    }

    #[test]
    fn the_ledger_state_judges_an_action_from_the_records_alone_whatever_the_mirror_believes() {
        let committed = LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from("t"),
            generation: Generation(1),
        };
        let state = LedgerState::replay(&at(vec![committed, delta(1), delta(2)]));
        assert_eq!(state.staleness("t", 2, 1), Staleness::Fresh);
        // A write the harness never made is still a write: the action one behind is stale.
        assert_eq!(state.staleness("t", 1, 1), Staleness::Revision);
        assert_eq!(state.staleness("t", 2, 2), Staleness::Generation);
        assert_eq!(state.staleness("u", 2, 1), Staleness::Generation);
    }

    #[test]
    fn the_scan_replays_revision_and_generation_from_the_records_alone() {
        let committed = |generation: u64| LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from("t"),
            generation: Generation(generation),
        };
        let fresh = scan(&at(vec![
            committed(1),
            delta(1),
            delta(2),
            attempt("t", 1, 2),
        ]));
        assert_eq!(
            fresh,
            Scan {
                attempts: 1,
                stale_attempts: 0
            }
        );

        // The revision moved after the action was made: a stale attempt.
        let late = scan(&at(vec![
            committed(1),
            delta(1),
            delta(2),
            attempt("t", 1, 1),
        ]));
        assert_eq!(late.stale_attempts, 1);

        // The revision at the moment of the attempt is the one that counts, not the final one.
        let before_the_write = scan(&at(vec![
            committed(1),
            delta(1),
            attempt("t", 1, 1),
            delta(2),
        ]));
        assert_eq!(before_the_write.stale_attempts, 0);

        // A superseded and a revoked generation.
        let superseded = scan(&at(vec![
            committed(1),
            delta(1),
            LedgerEvent::CapsuleSuperseded {
                capsule: CapsuleId::from("t"),
                old: Generation(1),
                new: Generation(2),
            },
            attempt("t", 1, 1),
            attempt("t", 2, 1),
        ]));
        assert_eq!(
            superseded,
            Scan {
                attempts: 2,
                stale_attempts: 1
            }
        );
        let revoked = scan(&at(vec![
            committed(1),
            delta(1),
            LedgerEvent::Revoked {
                subject: "t".into(),
                generation: Generation(1),
            },
            attempt("t", 1, 1),
        ]));
        assert_eq!(revoked.stale_attempts, 1);
        // A target nothing committed is not live.
        assert_eq!(
            scan(&at(vec![delta(1), attempt("u", 1, 1)])).stale_attempts,
            1
        );
    }
}
