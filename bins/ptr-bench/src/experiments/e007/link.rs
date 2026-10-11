//! The link between the two stages: frames are cut into signed chunks, sent over the PodWire
//! frame codec under a virtual clock and reassembled by the receiving assembler.

use super::params;
use super::rig::Rig;
use super::wire::{Role, Sent, Wire};
use crate::experiments::layered_kv::lab::Res;
use crate::experiments::rng::Rng;
use ptr_podwire::{
    chunk_frame, sign_chunk, AssemblyPolicy, DeliveredFrame, FrameAssembler, FrameMeta, Offered,
};
use ptr_protocol::activation::{
    frame_digest, ActivationChunk, ActivationFrame, ChunkHeader, UpstreamChain,
};

/// The receiver's limits as preregistered.
pub fn policy() -> AssemblyPolicy {
    AssemblyPolicy {
        reorder_window: params::REORDER_WINDOW as u32,
        max_incomplete_frames_per_peer: params::MAX_INCOMPLETE_FRAMES_PER_PEER as usize,
        max_chunks_in_flight: params::MAX_CHUNKS_IN_FLIGHT as usize,
        max_frame_tombstones_per_principal: params::MAX_FRAME_TOMBSTONES_PER_PRINCIPAL as usize,
        max_frame_deadline_lifetime_ms: params::MAX_FRAME_DEADLINE_LIFETIME_MS,
        max_clock_skew_ms: params::MAX_CLOCK_SKEW_MS,
        replay_window_ms: params::REPLAY_WINDOW_MS,
        max_frame_bytes: params::MAX_ASSEMBLED_FRAME_BYTES,
        max_total_buffered_bytes: params::MAX_TOTAL_BUFFERED_BYTES,
    }
}

/// A receiver with the upstream runtime's key registered.
pub fn assembler(rig: &Rig) -> Res<FrameAssembler> {
    let mut assembler = FrameAssembler::new(policy())
        .map_err(|violation| format!("the assembly policy: {violation:?}"))?;
    assembler.register_principal(
        params::UPSTREAM_PRINCIPAL,
        rig.signer.verifying_key(),
        params::STAGE_IDENTITY,
    );
    Ok(assembler)
}

/// What shipping one frame left behind, for the counters.
pub struct Shipped {
    pub delivered: DeliveredFrame,
    pub frame_id: u64,
    /// Whether every chunk carried one frame digest and it is the digest of the bytes sent.
    pub consistent_digest: bool,
}

pub struct Link {
    pub wire: Wire,
    pub assembler: FrameAssembler,
    pub now_ms: u64,
    next_frame: u64,
    pub operation: u64,
    rng: Rng,
}

impl Link {
    pub fn new(rig: &Rig, seed: u64, operation: u64) -> Res<Self> {
        Ok(Self {
            wire: Wire::default(),
            assembler: assembler(rig)?,
            now_ms: 1_000,
            next_frame: 1,
            operation,
            rng: Rng::new(seed ^ operation.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        })
    }

    pub fn next_frame_id(&mut self) -> u64 {
        let id = self.next_frame;
        self.next_frame += 1;
        id
    }

    /// Cut `frame` into signed chunks of `chunk_bytes`, deadline `budget_ms` ahead.
    pub fn chunks(
        &mut self,
        rig: &Rig,
        frame: &ActivationFrame,
        chunk_bytes: usize,
        budget_ms: u64,
    ) -> Res<(u64, Vec<ActivationChunk>)> {
        let frame_id = self.next_frame_id();
        let chunks = chunk_frame(
            &rig.signer,
            frame,
            FrameMeta {
                frame_id,
                parent_operation_id: self.operation,
                signed_absolute_deadline_ms: self.now_ms + budget_ms,
            },
            chunk_bytes,
        )
        .map_err(|error| format!("chunking a frame: {error:?}"))?;
        Ok((frame_id, chunks))
    }

    /// Signed chunks of arbitrary `bytes` taken as frame bytes (a consistent frame digest of
    /// those bytes, whatever they are): how a sender with the key ships a frame that is no
    /// valid frame.
    pub fn raw_chunks(
        &mut self,
        rig: &Rig,
        bytes: &[u8],
        chain: &UpstreamChain,
        chunk_bytes: usize,
        budget_ms: u64,
    ) -> (u64, Vec<ActivationChunk>) {
        let frame_id = self.next_frame_id();
        let digest = frame_digest(bytes);
        let total = bytes.len().div_ceil(chunk_bytes).max(1) as u32;
        let chunks = bytes
            .chunks(chunk_bytes)
            .enumerate()
            .map(|(index, part)| {
                sign_chunk(
                    &rig.signer,
                    ChunkHeader {
                        frame_id,
                        parent_operation_id: self.operation,
                        chunk_index: index as u32,
                        chunk_total: total,
                        signed_absolute_deadline_ms: self.now_ms + budget_ms,
                        frame_digest: digest,
                        chunk_digest: [0; 32],
                        upstream_chain: chain.clone(),
                    },
                    part.to_vec(),
                )
            })
            .collect();
        (frame_id, chunks)
    }

    /// Send every chunk in order under `role`; the answer to the last one.
    pub fn send_all(&mut self, chunks: &[ActivationChunk], role: Role) -> Option<Sent> {
        let mut last = None;
        for chunk in chunks {
            last = Some(
                self.wire
                    .send_chunk(&mut self.assembler, self.now_ms, 1, role, chunk),
            );
        }
        last
    }

    /// Send `frame` in order (adjacent chunks swapped at random, inside the reorder window),
    /// expect the last chunk to deliver it exactly once, and advance the clock one frame
    /// period.
    pub fn ship(&mut self, rig: &Rig, frame: &ActivationFrame, role: Role) -> Res<Shipped> {
        let (frame_id, chunks) = self.chunks(
            rig,
            frame,
            params::SMALL_FRAME_CHUNK_BYTES as usize,
            params::FRAME_BUDGET_MS,
        )?;
        let mut order: Vec<usize> = (0..chunks.len()).collect();
        // Disjoint adjacent pairs, so no chunk moves more than one place.
        for at in (0..order.len().saturating_sub(1)).step_by(2) {
            if self.rng.chance(0.5) {
                order.swap(at, at + 1);
            }
        }
        let mut delivered = None;
        for index in order {
            match self
                .wire
                .send_chunk(&mut self.assembler, self.now_ms, 1, role, &chunks[index])
            {
                Sent::Answered(Ok(Offered::Delivered(frame))) => {
                    if delivered.replace(frame).is_some() {
                        return Err("a frame was delivered twice".to_string());
                    }
                }
                Sent::Answered(Ok(_)) => {}
                other => return Err(format!("a valid chunk was not taken: {other:?}")),
            }
        }
        let delivered = *delivered.ok_or("the last chunk did not deliver the frame")?;
        let sent = frame.to_bytes();
        let digest = super::large::independent_digest(&sent);
        let consistent_digest = chunks
            .iter()
            .all(|chunk| chunk.header.frame_digest == digest)
            && delivered.frame_digest == digest;
        self.now_ms += params::FRAME_SPACING_MS;
        Ok(Shipped {
            delivered,
            frame_id,
            consistent_digest,
        })
    }
}
