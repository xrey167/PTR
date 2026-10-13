//! C2: a frame larger than one PodWire body, transport only.
//!
//! 2048 tokens x 2048 width in F32 is 16 MiB, four times `MAX_BODY_BYTES`. The control sends
//! it as one payload and PodWire's own codec must refuse it; the case sends it as five signed
//! chunks of at most 3.5 MiB, out of order inside the reorder window, and compares the
//! reassembled bytes with the bytes sent and the frame digest with an independent hash.

use super::link::Link;
use super::params;
use super::rig::Rig;
use super::wire::{Role, Sent};
use crate::experiments::layered_kv::emit::Tally;
use crate::experiments::layered_kv::lab::Res;
use crate::experiments::rng::Rng;
use ptr_podwire::{chunk_frame, FrameMeta, Offered};
use ptr_protocol::activation::{ActivationFrame, BoundaryDtype, ACTIVATION_CHUNK_TYPE};
use ptr_protocol::TypedPayload;
use ptr_types::TypeId;

/// The digest of a frame, composed here from SHA-256 and the frame's documented domain, so
/// that a defect in the product's digest function is a difference and not a shared error.
pub fn independent_digest(bytes: &[u8]) -> [u8; 32] {
    let mut material = b"ptr.split.frame.v1\0".to_vec();
    material.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    material.extend_from_slice(bytes);
    ptr_ledger::integrity::sha256(&material)
}

pub fn run(
    tally: &mut Tally,
    rig: &Rig,
    link: &mut Link,
    seed: u64,
    ids: &mut super::bitwise::Frames,
) -> Res<Vec<u8>> {
    let mut rng = Rng::new(seed ^ 0x1A46_E5CA_1E00);
    let mut tensor = Vec::with_capacity(params::LARGE_FRAME_BYTES as usize);
    while tensor.len() < params::LARGE_FRAME_BYTES as usize {
        tensor.extend_from_slice(&rng.next_u64().to_le_bytes());
    }
    let frame = ActivationFrame {
        tokens: params::LARGE_FRAME_TOKENS as u32,
        width: params::LARGE_FRAME_WIDTH as u32,
        dtype: BoundaryDtype::F32,
        positions: (0..params::LARGE_FRAME_TOKENS as u32).collect(),
        upstream_chain: rig.chain(),
        placement_epoch: rig.epoch(),
        tensor,
    };
    let sent = frame.to_bytes();
    // The control: the same bytes as one payload. PodWire's own bound must refuse it.
    let whole = TypedPayload {
        type_id: TypeId::from(ACTIVATION_CHUNK_TYPE),
        bytes: sent.clone(),
    };
    let unchunked = link.wire.send(
        &mut link.assembler,
        link.now_ms,
        1,
        Role::Probe("unchunked_large_frame"),
        &whole,
    );
    if matches!(unchunked, Sent::Codec(_)) {
        tally.add("control_trips.unchunked_large_frame_refused_by_podwire", 1);
    }
    link.operation = 9_000;
    let frame_id = link.next_frame_id();
    let chunks = chunk_frame(
        &rig.signer,
        &frame,
        FrameMeta {
            frame_id,
            parent_operation_id: link.operation,
            signed_absolute_deadline_ms: link.now_ms + params::FRAME_DEADLINE_MS,
        },
        params::MAX_CHUNK_BYTES as usize,
    )
    .map_err(|error| format!("chunking the large frame: {error:?}"))?;
    if chunks.len() as u64 != params::LARGE_FRAME_CHUNKS {
        return Err(format!(
            "the large frame is {} chunks, the preregistration says {}",
            chunks.len(),
            params::LARGE_FRAME_CHUNKS
        ));
    }
    let mut delivered = None;
    for index in params::LARGE_FRAME_SEND_ORDER {
        match link.wire.send_chunk(
            &mut link.assembler,
            link.now_ms,
            1,
            Role::Valid,
            &chunks[index as usize],
        ) {
            Sent::Answered(Ok(Offered::Delivered(done))) => delivered = Some(done),
            Sent::Answered(Ok(_)) => {}
            // A valid chunk the receiver refuses is a product defect: the frame does not arrive.
            _ => {}
        }
    }
    tally.add("large_frame_cases", 1);
    let name = format!("{}:{frame_id}", link.operation);
    ids.ids.push(name.clone());
    tally.add("valid_frames_sent", 1);
    match delivered {
        None => {
            tally.add("large_frame_reassembled_bytes_mismatch_count", 1);
            tally.add("large_frame_digest_mismatches", 1);
        }
        Some(done) => {
            if done.bytes != sent {
                tally.add("large_frame_reassembled_bytes_mismatch_count", 1);
                tally.add("valid_frame_bitwise_mismatches", 1);
            }
            let independent = independent_digest(&sent);
            if done.frame_digest != independent || independent_digest(&done.bytes) != independent {
                tally.add("large_frame_digest_mismatches", 1);
            }
            let consistent = chunks
                .iter()
                .all(|chunk| chunk.header.frame_digest == independent)
                && done.frame_digest == independent;
            if consistent {
                tally.add("frames_with_consistent_whole_frame_digest", 1);
                ids.consistent_ids.push(name);
            }
        }
    }
    link.now_ms += params::FRAME_SPACING_MS;
    Ok(sent)
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_independent_digest_is_the_products() {
        let bytes = vec![7u8; 1000];
        assert_eq!(
            super::independent_digest(&bytes),
            ptr_protocol::activation::frame_digest(&bytes)
        );
    }
}
