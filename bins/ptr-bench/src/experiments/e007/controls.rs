//! The controls: naive configurations of each protection, which must fail the same probes.
//!
//! Two of them run the real product code with the protection taken away (the cache with a
//! liveness frozen at storing, the assembler without its tombstones); one is a harness-only
//! reference model driven by the same real chunks (the assembler keyed by the chain alone),
//! and one pairs a harness-only deny-list with the real allowlist on the real mislabelled
//! page. The 16 MiB control is the real PodWire codec. None of them is the evidence that a
//! counter can fire: that evidence is the mutation set (47+ planted defects in the product
//! code, each of which must be killed).
//!
//! A probe that the naive version also survives proves nothing, so each control must trip
//! (counted in `control_trips`); one that does not is a failure of the harness, not of the
//! product, and ends the run with status 2.

use super::bitwise::{self, Transport};
use super::link::Link;
use super::params;
use super::rig::Rig;
use crate::experiments::layered_kv::emit::Tally;
use crate::experiments::layered_kv::lab::Res;
use ptr_pods::KvTensorSnapshot;
use ptr_podwire::Offered;
use ptr_protocol::activation::{
    admit_typed, chunk_payload, ActivationChunk, ActivationFrame, BoundaryDtype, SchemaError,
};
use ptr_runtime::{SplitStageCache, StageLiveness};
use ptr_types::{Generation, TypeId, Validity};
use std::collections::BTreeMap;

/// The liveness a cache that trusts the Revoke message holds: what was true when the entry
/// was stored (every generation live, the digests and the placement as committed then). It is
/// never asked again, so a revocation nobody announced changes nothing for it.
struct FrozenLiveness {
    manifests: BTreeMap<(String, u64), [u8; 32]>,
    epoch: u64,
}

impl StageLiveness for FrozenLiveness {
    fn validity(&self, _target: &str, _generation: Generation) -> Option<Validity> {
        Some(Validity::Live)
    }

    fn manifest_digest(&self, target: &str, generation: Generation) -> Option<[u8; 32]> {
        self.manifests
            .get(&(target.to_string(), generation.0))
            .copied()
    }

    fn placement_epoch(&self, _stage: &str) -> Option<u64> {
        Some(self.epoch)
    }

    fn fence_epoch(&self, _expert: &str) -> Option<u64> {
        None
    }
}

/// The real [`SplitStageCache`] with its use-time revalidation fed the liveness of the time
/// of storing, as a cache that relies on the Revoke message alone would be: after the ledger
/// revokes the stage and the message is dropped it still serves. The same probe against the
/// real liveness must refuse, or the probe distinguishes nothing.
pub fn naive_cache(tally: &mut Tally, seed: u64) -> Res<()> {
    let mut rig = Rig::new()?;
    let mut link = Link::new(&rig, seed, 910)?;
    let run = bitwise::split(&rig, Transport::Wire(&mut link), &bitwise::prompt(2), 2)?;
    let frame = run.prefill_frame.ok_or("no frame")?;
    let snapshot = run.prefill_snapshot.ok_or("no snapshot")?;
    let frozen = FrozenLiveness {
        manifests: rig.manifests.clone(),
        epoch: rig.epoch(),
    };
    let mut cache: SplitStageCache<KvTensorSnapshot> = SplitStageCache::new(2);
    let key = cache
        .store(&[&frame], &rig.own_chain(), snapshot, &rig.live())
        .map_err(|error| format!("the control's store: {error:?}"))?;
    // The ledger revokes; the Revoke message is dropped.
    rig.revoke_stage()?;
    let served_by_the_naive = cache.lookup(&key, &frozen).is_ok();
    let refused_by_the_real = cache.lookup(&key, &rig.live()).is_err();
    if served_by_the_naive && refused_by_the_real {
        tally.add("control_trips.naive_downstream_cache_serves_stale", 1);
    }
    Ok(())
}

/// A reassembler keyed by the upstream chain alone, as if one chain meant one frame.
pub fn chain_keyed_assembler(
    tally: &mut Tally,
    a: &(ActivationFrame, Vec<ActivationChunk>),
    b: &(ActivationFrame, Vec<ActivationChunk>),
) {
    let mut held: BTreeMap<u32, Vec<u8>> = BTreeMap::new();
    let total = a.1[0].header.chunk_total as usize;
    let mut out: Option<Vec<u8>> = None;
    for index in 0..a.1.len().max(b.1.len()) {
        for chunks in [&a.1, &b.1] {
            if let Some(chunk) = chunks.get(index) {
                held.insert(chunk.header.chunk_index, chunk.bytes.clone());
                if out.is_none() && held.len() == total {
                    out = Some(held.values().flatten().copied().collect());
                }
            }
        }
    }
    if let Some(bytes) = out {
        if bytes != a.0.to_bytes() && bytes != b.0.to_bytes() {
            tally.add("control_trips.chain_keyed_assembler_mixes_frames", 1);
        }
    }
}

/// Admission by a deny-list of KV type ids (a harness-only reference model: the product has
/// no such mode): every type id off the list is taken as it is.
fn denylist_admits(type_id: &str) -> bool {
    !["ptr.kv.page.v1", "ptr.kv.snapshot.v1", "ptr.kv.handle.v1"].contains(&type_id)
}

/// A real KV page under another name: the deny-list takes it, the real allowlist refuses it.
pub fn denylist(tally: &mut Tally, page: &[u8]) {
    let type_id = TypeId::from("ptr.tensor.blob.v1");
    let naive = denylist_admits(&type_id.0);
    let real = admit_typed(&type_id, page);
    if naive && !page.is_empty() && matches!(real, Err(SchemaError::NotAllowlisted(_))) {
        tally.add("control_trips.denylist_admits_mislabelled_payload", 1);
    }
}

/// The real `FrameAssembler` that forgets a frame the moment it completes (the assembler
/// is rebuilt, which drops the tombstones, as a process that lost its state would): the
/// same real chunks, replayed, are delivered a second time. The unmodified assembler must
/// deliver once.
pub fn tombstone_free(tally: &mut Tally, rig: &Rig, chunks: &[ActivationChunk]) -> Res<()> {
    let deliveries = |forgetful: bool| -> Res<usize> {
        let mut assembler = super::link::assembler(rig)?;
        let mut delivered = 0;
        for _ in 0..2 {
            for chunk in chunks {
                let payload = chunk_payload(chunk);
                if let Ok(Offered::Delivered(_)) =
                    assembler.offer(1_000, params::UPSTREAM_PRINCIPAL, 1, &payload)
                {
                    delivered += 1;
                    if forgetful {
                        assembler = super::link::assembler(rig)?;
                    }
                }
            }
        }
        Ok(delivered)
    };
    if deliveries(true)? > 1 && deliveries(false)? == 1 {
        tally.add("control_trips.tombstone_free_assembler_admits_replay", 1);
    }
    Ok(())
}

/// A small frame for the assembler controls.
pub fn frame(rig: &Rig, fill: u8) -> ActivationFrame {
    ActivationFrame {
        tokens: params::PROBE_FRAME_ROWS as u32,
        width: params::HIDDEN_SIZE as u32,
        dtype: BoundaryDtype::F32,
        positions: (0..params::PROBE_FRAME_ROWS as u32).collect(),
        upstream_chain: rig.chain(),
        placement_epoch: rig.epoch(),
        tensor: (0..params::PROBE_FRAME_ROWS as usize * params::HIDDEN_SIZE as usize * 4)
            .map(|byte| (byte as u8).wrapping_mul(17).wrapping_add(fill))
            .collect(),
    }
}
