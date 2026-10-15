//! The control baselines of Part A, each of which must produce the failure it exists to show
//! (`experiment.toml`: "a downstream cache keyed by frame digest that trusts the Revoke message
//! (stale serves must occur), a cache without lineage or raw ancestry in its key (reuse after a
//! removed Raw segment must occur), a cache with one namespace for both production modes (an
//! Approximate frame must be served to an Exact lookup)").
//!
//! A control that does not trip proves nothing about its probe, so the run stops with a
//! harness fault when one stays at zero.
//!
//! The controls are **external baselines**, small caches written for the harness that do what
//! `experiment.toml` names (a cache that trusts the Revoke message, a key without lineage or
//! raw ancestry, one namespace for both modes). They are not the product with one mechanism
//! removed: they share no code with `SplitStageCacheV2`, so a control that trips shows that
//! the probe can see the failure, not that the product's own mechanism is the one that
//! prevents it. That is the job of the matrix cells and the mutation plan.

use super::matrix::{cells, Path};
use super::world::{FrameSpec, World};
use crate::experiments::layered_kv::emit::Tally;
use crate::experiments::layered_kv::lab::Res;
use ptr_ledger::integrity::sha256;
use ptr_protocol::activation::{chain_bytes, lineage_bytes, ActivationFrameV2, Invalidation};
use ptr_runtime::StageLiveness;
use ptr_types::Generation;
use std::collections::BTreeMap;

/// The three controls, in the order they print.
pub const CONTROLS: [&str; 3] = [
    "naive_revoke_trusting_cache_serves_stale",
    "keyless_lineage_cache_reuses_after_raw_removal",
    "single_namespace_cache_serves_approximate_to_exact",
];

/// A downstream cache keyed by the frame digest that trusts the `Revoke` message: it drops
/// entries when told and revalidates nothing at use.
/// What the trusting cache remembers about an entry: the dependencies it must drop it for.
struct Remembered {
    dependencies: Vec<(String, u64)>,
    value: Vec<f32>,
}

#[derive(Default)]
struct RevokeTrusting {
    entries: BTreeMap<[u8; 32], Remembered>,
}

impl RevokeTrusting {
    fn store(&mut self, frame: &ActivationFrameV2, value: Vec<f32>) -> [u8; 32] {
        let key = frame.digest();
        let dependencies = frame
            .elements()
            .iter()
            .map(|element| (element.identity.to_string(), element.generation))
            .collect();
        self.entries.insert(
            key,
            Remembered {
                dependencies,
                value,
            },
        );
        key
    }

    fn revoke_message(&mut self, identity: &str, generation: u64) {
        self.entries.retain(|_, remembered| {
            !remembered
                .dependencies
                .iter()
                .any(|(name, number)| name == identity && *number == generation)
        });
    }

    /// The cached KV, with no check of anything it was derived from.
    fn lookup(&self, key: &[u8; 32]) -> Option<&[f32]> {
        self.entries
            .get(key)
            .map(|remembered| remembered.value.as_slice())
    }
}

/// Control 1: the cache that trusts the `Revoke` message serves a stale entry whenever the
/// message is lost, for every kind of the matrix and every invalidation type. With the
/// message delivered the same cache behaves (the control is wrong only because it trusts the
/// message), which keeps the probe honest.
pub fn revoke_trusting(tally: &mut Tally, seed: u64) -> Res<()> {
    for (number, cell) in cells()
        .into_iter()
        .filter(|cell| cell.path == Path::Lag)
        .enumerate()
    {
        for delivered in [false, true] {
            let mut world = World::new()?;
            let frame = world.frame(&FrameSpec::exact(
                vec![[1; 32], [2; 32]],
                seed ^ number as u64,
            ));
            let mut naive = RevokeTrusting::default();
            let key = naive.store(&frame, vec![1.0]);
            let target = super::world::targets(cell.kind)[0].clone();
            let target = target.as_str();
            let generation = world.generation(target);
            match cell.invalidation {
                Invalidation::Revocation => world.revoke(target)?,
                Invalidation::Supersession => world.supersede(target)?,
                Invalidation::Fence => world.fence(target)?,
            }
            if delivered {
                naive.revoke_message(target, generation);
            }
            let served = naive.lookup(&key).is_some();
            match (delivered, served) {
                (false, true) => {
                    tally.add("control_trips.naive_revoke_trusting_cache_serves_stale", 1)
                }
                (true, true) => tally.add("control_inconsistencies", 1),
                (_, false) => {}
            }
            // The mechanism, shown the same lost message, refuses the entry: the ledger says
            // the dependency is gone.
            if cell.invalidation != Invalidation::Fence {
                let live = world.live();
                let validity = StageLiveness::validity(&live, target, Generation(generation));
                if validity == Some(ptr_types::Validity::Live) {
                    tally.add("control_inconsistencies", 1);
                }
            }
        }
    }
    Ok(())
}

/// The key of a cache that has no lineage and no raw ancestry in it: the upstream chain, the
/// positions and the downstream stage's own chain.
pub fn keyless_key(
    frame: &ActivationFrameV2,
    own: &ptr_protocol::activation::UpstreamChain,
) -> [u8; 32] {
    let mut bytes = chain_bytes(&frame.upstream_chain);
    for position in &frame.positions {
        bytes.extend_from_slice(&position.to_le_bytes());
    }
    bytes.extend_from_slice(&chain_bytes(own));
    sha256(&bytes)
}

/// The key of a cache with one namespace for both production modes: everything the mechanism
/// keys on (chain, lineage including the raw ancestry, positions, activations, own chain) but
/// the mode and the taint.
pub fn single_namespace_key(
    frame: &ActivationFrameV2,
    own: &ptr_protocol::activation::UpstreamChain,
) -> [u8; 32] {
    let mut bytes = chain_bytes(&frame.upstream_chain);
    bytes.extend_from_slice(&lineage_bytes(&frame.input_lineage));
    for position in &frame.positions {
        bytes.extend_from_slice(&position.to_le_bytes());
    }
    bytes.extend_from_slice(&frame.tensor);
    bytes.extend_from_slice(&chain_bytes(own));
    sha256(&bytes)
}
