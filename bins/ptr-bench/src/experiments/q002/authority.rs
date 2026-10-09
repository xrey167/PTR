//! The lifecycle authority the mechanism asks, and the source it resolves against.
//!
//! The authority is a real [`PtrRuntime`] (owner decision of 2026-10-09): the world's
//! lifecycle is committed to its ledger as the events a runtime commits (`CapsuleCommitted`,
//! `CapsuleSuperseded`, `Revoked`), and every validity question is
//! [`PtrRuntime::generation_validity`]. The runtime has no notion of a disputed generation
//! (its answer is never `Validity::Disputed`), so P8 is the one class whose lifecycle comes
//! from the harness: [`Authority::validity`] answers `Disputed` for the generations the
//! generator marked disputed and the runtime's answer for every other. The runtime does
//! exercise P6 (superseded) and P7 (revoked although still the live generation).

use super::world::{Status, World};
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_memory::{KnowledgeStore, RawEvent};
use ptr_runtime::PtrRuntime;
use ptr_search::RawEventSource;
use ptr_types::{CapsuleId, Generation, ProjectId, RawEventId, Validity};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};

pub struct Authority {
    runtime: PtrRuntime,
    disputed: BTreeSet<(String, u64)>,
    queries: Cell<i64>,
}

impl Authority {
    /// Commit the world's lifecycle to a fresh runtime.
    pub fn build(world: &World) -> Result<Authority, String> {
        let mut runtime = PtrRuntime::new(PtrConfig::default())
            .map_err(|error| format!("the runtime does not open: {error:?}"))?;
        let mut disputed = BTreeSet::new();
        let project = ProjectId::from("q002");
        for capsule in &world.capsules {
            let name = CapsuleId::from(capsule.id.as_str());
            let commit = |runtime: &mut PtrRuntime, event: LedgerEvent| {
                runtime
                    .commit(event)
                    .map_err(|error| format!("the runtime refuses {}: {error:?}", capsule.id))
            };
            commit(
                &mut runtime,
                LedgerEvent::CapsuleCommitted {
                    project: project.clone(),
                    capsule: name.clone(),
                    generation: Generation(1),
                },
            )?;
            for generation in 2..=capsule.live_generation() {
                commit(
                    &mut runtime,
                    LedgerEvent::CapsuleSuperseded {
                        capsule: name.clone(),
                        old: Generation(generation - 1),
                        new: Generation(generation),
                    },
                )?;
            }
            match capsule.status {
                Status::Healthy => {}
                Status::Revoked => {
                    commit(
                        &mut runtime,
                        LedgerEvent::Revoked {
                            subject: capsule.id.clone(),
                            generation: Generation(capsule.live_generation()),
                        },
                    )?;
                }
                Status::Disputed => {
                    disputed.insert((capsule.id.clone(), capsule.live_generation()));
                }
            }
        }
        Ok(Authority {
            runtime,
            disputed,
            queries: Cell::new(0),
        })
    }

    /// Whether `capsule` at `generation` may be used now: the runtime's answer, except that a
    /// generation the generator marked disputed is `Disputed`.
    pub fn validity(&self, capsule: &CapsuleId, generation: Generation) -> Option<Validity> {
        self.queries.set(self.queries.get() + 1);
        if self.disputed.contains(&(capsule.0.clone(), generation.0)) {
            return Some(Validity::Disputed);
        }
        self.runtime.generation_validity(&capsule.0, generation)
    }

    /// How many times the authority was asked.
    pub fn queries(&self) -> i64 {
        self.queries.get()
    }

    /// The runtime's own answer, without the disputed overlay (for the tests).
    #[cfg(test)]
    pub fn runtime_validity(&self, capsule: &str, generation: u64) -> Option<Validity> {
        self.runtime
            .generation_validity(capsule, Generation(generation))
    }
}

/// The raw events the resolver reads: the store of record, and beside it the corrupt
/// replicas. A `KnowledgeStore` cannot hold an event whose bytes disagree with its digest
/// (it checks when it appends), so the P3 hits cite replicas that do, as a damaged copy of
/// the store would serve them; the replicas have ids of their own, so no honest hit reads one.
pub struct Source<'store> {
    store: &'store KnowledgeStore,
    replicas: BTreeMap<RawEventId, RawEvent>,
}

impl<'store> Source<'store> {
    /// A store holding the world's raw events, to be borrowed by a [`Source`].
    pub fn store_of(world: &World) -> Result<KnowledgeStore, String> {
        let mut store = KnowledgeStore::default();
        for (id, text) in world.events() {
            store
                .append_raw(World::raw_event(id, text))
                .map_err(|error| format!("the store refuses {id}: {error:?}"))?;
        }
        Ok(store)
    }

    pub fn new(store: &'store KnowledgeStore, world: &World) -> Source<'store> {
        Source {
            store,
            replicas: world
                .replicas
                .iter()
                .map(|replica| (replica.id.clone(), replica.clone()))
                .collect(),
        }
    }
}

impl RawEventSource for Source<'_> {
    fn raw_event(&self, id: &RawEventId) -> Option<&RawEvent> {
        self.replicas.get(id).or_else(|| self.store.raw(id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::q002::params::Shape;

    fn world() -> World {
        World::generate(17, Shape::PINNED).expect("a world")
    }

    #[test]
    fn the_runtime_answers_live_superseded_and_revoked_as_the_generator_says() {
        let world = world();
        let authority = Authority::build(&world).expect("an authority");
        for capsule in &world.capsules {
            let live = capsule.live_generation();
            let expected_live = match capsule.status {
                Status::Healthy | Status::Disputed => Validity::Live,
                Status::Revoked => Validity::Revoked,
            };
            assert_eq!(
                authority.runtime_validity(&capsule.id, live),
                Some(expected_live),
                "{}",
                capsule.id
            );
            for older in 1..live {
                assert_eq!(
                    authority.runtime_validity(&capsule.id, older),
                    Some(Validity::Superseded)
                );
            }
            assert_eq!(authority.runtime_validity(&capsule.id, live + 1), None);
        }
        assert_eq!(authority.runtime_validity("cap-nope", 1), None);
    }

    #[test]
    fn the_harness_overlay_makes_only_the_disputed_live_generations_disputed() {
        let world = world();
        let authority = Authority::build(&world).expect("an authority");
        for capsule in &world.capsules {
            let id = CapsuleId::from(capsule.id.as_str());
            let answer = authority.validity(&id, Generation(capsule.live_generation()));
            match capsule.status {
                Status::Disputed => assert_eq!(answer, Some(Validity::Disputed)),
                Status::Revoked => assert_eq!(answer, Some(Validity::Revoked)),
                Status::Healthy => assert_eq!(answer, Some(Validity::Live)),
            }
        }
        assert!(authority.queries() > 0);
    }

    #[test]
    fn the_source_serves_replicas_that_the_store_could_not_hold_and_real_events_otherwise() {
        let world = world();
        let mut store = Source::store_of(&world).expect("a store");
        let source = Source::new(&store, &world);
        let replica = world.replicas.first().expect("a replica");
        assert!(replica.validate_digest().is_err());
        assert!(store.raw(&replica.id).is_none());
        assert!(source.raw_event(&replica.id).is_some());
        let (id, text) = world.events().next().expect("an event");
        let served = source
            .raw_event(&RawEventId::from(id))
            .expect("a real event");
        assert_eq!(served.content, text.as_bytes());
        assert!(served.validate_digest().is_ok());
        assert!(source
            .raw_event(&RawEventId::from("raw-fab-0000"))
            .is_none());
        assert!(store.append_raw(replica.clone()).is_err());
    }
}
