//! C4 and C5: the chunk protocol under attack and under load.
//!
//! Every adversarial attempt is a payload an authenticated sender, or one that holds a valid
//! chunk of its own, could really send. Each ends in a typed `Rejection` and the variant is
//! compared with the one the hand-written table below pins (`verdict_mismatches`). The valid
//! cases (interleaved frames, concurrent sessions, a frame after an abandonment) must complete
//! bit-identically and exactly once.

use super::bitwise::Frames;
use super::link::Link;
use super::params;
use super::rig::Rig;
use super::wire::{Role, Sent};
use crate::experiments::layered_kv::emit::{Json, Tally};
use crate::experiments::layered_kv::lab::{id_set_digest, Res};
use crate::experiments::rng::Rng;
use ed25519_dalek::SigningKey;
use ptr_podwire::{
    chunk_frame, resign_envelope, sign_chunk, sign_policy, AssemblyPolicy, FrameAssembler,
    FrameMeta, Offered,
};
use ptr_protocol::activation::{
    chunk_payload, encode_policy, frame_digest, ActivationChunk, ActivationFrame, BoundaryDtype,
    ChunkHeader, SplitPolicy, SPLIT_POLICY_TYPE,
};
use ptr_protocol::TypedPayload;
use ptr_types::TypeId;

fn probe_frame(rig: &Rig, rows: usize, fill: u8) -> ActivationFrame {
    ActivationFrame {
        tokens: rows as u32,
        width: params::HIDDEN_SIZE as u32,
        dtype: BoundaryDtype::F32,
        positions: (0..rows as u32).collect(),
        upstream_chain: rig.chain(),
        placement_epoch: rig.epoch(),
        tensor: (0..rows * params::HIDDEN_SIZE as usize * 4)
            .map(|byte| (byte as u8).wrapping_mul(31).wrapping_add(fill))
            .collect(),
    }
}

fn link(rig: &Rig, seed: u64, operation: u64) -> Res<Link> {
    Link::new(rig, seed, operation)
}

fn chunks_of(
    link: &mut Link,
    rig: &Rig,
    rows: usize,
    fill: u8,
    budget_ms: u64,
) -> Res<(u64, ActivationFrame, Vec<ActivationChunk>)> {
    let frame = probe_frame(rig, rows, fill);
    let (id, chunks) = link.chunks(rig, &frame, params::PROBE_CHUNK_BYTES as usize, budget_ms)?;
    Ok((id, frame, chunks))
}

/// Judge one attempt: it must end in a rejection of the pinned variant.
fn attempt(tally: &mut Tally, sent: &Sent, expected: &str) {
    match sent.rejection() {
        Some(rejection) => {
            if rejection.kind() != expected {
                tally.add("verdict_mismatches", 1);
            }
        }
        None => tally.add("attempts_without_typed_rejection", 1),
    }
}

fn send(link: &mut Link, kind: &'static str, session: u64, chunk: &ActivationChunk) -> Sent {
    link.wire.send_chunk(
        &mut link.assembler,
        link.now_ms,
        session,
        Role::Probe(kind),
        chunk,
    )
}

fn deliver(link: &mut Link, session: u64, chunk: &ActivationChunk) -> Sent {
    link.wire.send_chunk(
        &mut link.assembler,
        link.now_ms,
        session,
        Role::Valid,
        chunk,
    )
}

/// Every attack on the chunks of one frame, each on a fresh receiver.
pub fn attacks(tally: &mut Tally, rig: &Rig, seed: u64) -> Res<()> {
    let rows = params::PROBE_FRAME_ROWS as usize;
    let budget = params::FRAME_BUDGET_MS;

    // A different key signs: an unsigned chunk as far as the receiver is concerned.
    {
        let mut l = link(rig, seed, 100)?;
        let (_, _, chunks) = chunks_of(&mut l, rig, rows, 1, budget)?;
        let other = SigningKey::from_bytes(&[9; 32]);
        let forged = sign_chunk(&other, chunks[0].header.clone(), chunks[0].bytes.clone());
        let sent = send(&mut l, "unsigned_chunk", 1, &forged);
        tally.add("unsigned_chunk_attempts", 1);
        tally.add("unsigned_chunks_admitted", i64::from(sent.admitted()));
        attempt(tally, &sent, "bad_signature");
        let mut flipped = chunks[0].clone();
        flipped.signature[3] ^= 0x40;
        let sent = send(&mut l, "unsigned_chunk", 1, &flipped);
        tally.add("unsigned_chunk_attempts", 1);
        tally.add("unsigned_chunks_admitted", i64::from(sent.admitted()));
        attempt(tally, &sent, "bad_signature");
    }

    // chain_mismatch: a later chunk names another upstream chain.
    {
        let mut l = link(rig, seed, 101)?;
        let (_, _, chunks) = chunks_of(&mut l, rig, rows, 2, budget)?;
        deliver(&mut l, 1, &chunks[0]);
        let mut header = chunks[1].header.clone();
        header.upstream_chain.stage.generation += 1;
        let swapped = sign_chunk(&rig.signer, header, chunks[1].bytes.clone());
        let sent = send(&mut l, "chain_mismatch", 1, &swapped);
        tally.add("chain_mismatch_attempts", 1);
        attempt(tally, &sent, "chain_mismatch");
        if l.assembler.incomplete_frames(params::UPSTREAM_PRINCIPAL) != 0 {
            tally.add("attempts_without_typed_rejection", 1);
        }
    }

    // duplicate_chunk.
    {
        let mut l = link(rig, seed, 102)?;
        let (_, _, chunks) = chunks_of(&mut l, rig, rows, 3, budget)?;
        deliver(&mut l, 1, &chunks[0]);
        let sent = send(&mut l, "duplicate_chunk", 1, &chunks[0]);
        tally.add("duplicate_chunk_attempts", 1);
        attempt(tally, &sent, "duplicate_chunk");
    }

    // out_of_window: with chunk 0 held the window is 1..5, so chunk 5 is outside it.
    {
        let mut l = link(rig, seed, 103)?;
        let (_, _, chunks) = chunks_of(&mut l, rig, rows, 4, budget)?;
        if chunks.len() < 7 {
            return Err(format!(
                "the probe frame has {} chunks, fewer than 7",
                chunks.len()
            ));
        }
        deliver(&mut l, 1, &chunks[0]);
        let sent = send(&mut l, "out_of_window", 1, &chunks[5]);
        tally.add("out_of_window_attempts", 1);
        attempt(tally, &sent, "out_of_window");
    }

    // bad_chunk_digest: corrupted bytes under an intact envelope; and an envelope re-signed
    // over a digest that does not match the bytes.
    {
        let mut l = link(rig, seed, 104)?;
        let (_, _, chunks) = chunks_of(&mut l, rig, rows, 5, budget)?;
        let mut corrupted = chunks[0].clone();
        corrupted.bytes[7] ^= 0x01;
        let sent = send(&mut l, "bad_chunk_digest", 1, &corrupted);
        tally.add("bad_chunk_digest_attempts", 1);
        attempt(tally, &sent, "bad_chunk_digest");
        let mut resigned = chunks[1].clone();
        resigned.header.chunk_digest[0] ^= 0x01;
        resign_envelope(&rig.signer, &mut resigned);
        let sent = send(&mut l, "bad_chunk_digest", 1, &resigned);
        tally.add("bad_chunk_digest_attempts", 1);
        attempt(tally, &sent, "bad_chunk_digest");
    }

    // mixed_version_frame: the same frame announced with another chunk count.
    {
        let mut l = link(rig, seed, 105)?;
        let (_, _, chunks) = chunks_of(&mut l, rig, rows, 6, budget)?;
        deliver(&mut l, 1, &chunks[0]);
        let mut header = chunks[1].header.clone();
        header.chunk_total += 1;
        let skewed = sign_chunk(&rig.signer, header, chunks[1].bytes.clone());
        let sent = send(&mut l, "mixed_version_frame", 1, &skewed);
        tally.add("mixed_version_frame_attempts", 1);
        tally.add("digest_inconsistency_attempts", 1);
        attempt(tally, &sent, "frame_mismatch");
    }

    // splice: valid chunks of two different frames under one frame id.
    {
        let mut l = link(rig, seed, 106)?;
        let (_, _, first) = chunks_of(&mut l, rig, rows, 7, budget)?;
        let other = probe_frame(rig, rows, 77);
        let spliced = ptr_podwire::chunk_frame(
            &rig.signer,
            &other,
            ptr_podwire::FrameMeta {
                frame_id: first[0].header.frame_id,
                parent_operation_id: first[0].header.parent_operation_id,
                signed_absolute_deadline_ms: first[0].header.signed_absolute_deadline_ms,
            },
            params::PROBE_CHUNK_BYTES as usize,
        )
        .map_err(|error| format!("chunking the splice: {error:?}"))?;
        deliver(&mut l, 1, &first[0]);
        let sent = send(&mut l, "splice", 1, &spliced[1]);
        tally.add("splice_attempts", 1);
        tally.add("digest_inconsistency_attempts", 1);
        attempt(tally, &sent, "frame_mismatch");
    }

    // cross_operation_splice: a chunk relabelled into another parent operation. Without the
    // key it fails the signature; with the key it is a frame of its own and must not join.
    {
        let mut l = link(rig, seed, 107)?;
        let (_, frame, chunks) = chunks_of(&mut l, rig, rows, 8, budget)?;
        deliver(&mut l, 1, &chunks[0]);
        let mut relabelled = chunks[1].clone();
        relabelled.header.parent_operation_id += 1;
        let sent = send(&mut l, "cross_operation_splice", 1, &relabelled);
        tally.add("cross_operation_splice_attempts", 1);
        tally.add("digest_inconsistency_attempts", 1);
        attempt(tally, &sent, "bad_signature");
        let colluding = sign_chunk(
            &rig.signer,
            relabelled.header.clone(),
            relabelled.bytes.clone(),
        );
        let _ = send(&mut l, "cross_operation_splice", 1, &colluding);
        let mut delivered = None;
        for chunk in &chunks[1..] {
            if let Sent::Answered(Ok(Offered::Delivered(done))) = deliver(&mut l, 1, chunk) {
                delivered = Some(done);
            }
        }
        match delivered {
            Some(done) if done.bytes == frame.to_bytes() && done.parent_operation_id == 107 => {}
            _ => tally.add("cross_operation_splices_admitted", 1),
        }
    }

    // mislabelled_frame_digest: every chunk carries one consistent but wrong digest.
    {
        let mut l = link(rig, seed, 108)?;
        let (_, _, chunks) = chunks_of(&mut l, rig, rows, 9, budget)?;
        let mut last = None;
        for chunk in &chunks {
            let mut header = chunk.header.clone();
            header.frame_digest = [0xEE; 32];
            let relabelled = sign_chunk(&rig.signer, header, chunk.bytes.clone());
            last = Some(deliver(&mut l, 1, &relabelled));
        }
        let last = last.ok_or("no chunks")?;
        tally.add("mislabelled_frame_digest_attempts", 1);
        if last.admitted() {
            tally.add("mislabelled_frame_digests_admitted", 1);
        }
        attempt(tally, &last, "frame_digest_mismatch");
    }

    // expired deadline: the signed deadline has passed (minus skew) before the chunk arrives.
    {
        let mut l = link(rig, seed, 109)?;
        l.now_ms = 20_000;
        let frame = probe_frame(rig, rows, 10);
        let (_, chunks) = {
            let id = l.next_frame_id();
            let chunks = ptr_podwire::chunk_frame(
                &rig.signer,
                &frame,
                ptr_podwire::FrameMeta {
                    frame_id: id,
                    parent_operation_id: 109,
                    signed_absolute_deadline_ms: 20_000 - params::MAX_CLOCK_SKEW_MS - 1,
                },
                params::PROBE_CHUNK_BYTES as usize,
            )
            .map_err(|error| format!("chunking: {error:?}"))?;
            (id, chunks)
        };
        let sent = send(&mut l, "expired_deadline", 1, &chunks[0]);
        tally.add("expired_deadline_frame_attempts", 1);
        if sent.admitted() {
            tally.add("expired_frames_admitted", 1);
        }
        attempt(tally, &sent, "expired");
    }

    // relabelled envelopes of a completed frame: every field of the envelope is signed.
    {
        let mut l = link(rig, seed, 110)?;
        let (_, _, chunks) = chunks_of(&mut l, rig, rows, 11, budget)?;
        for chunk in &chunks {
            deliver(&mut l, 1, chunk);
        }
        let relabelings: [fn(&mut ActivationChunk); 5] = [
            |c| c.header.frame_id += 1,
            |c| c.header.parent_operation_id += 1,
            |c| c.header.chunk_index = (c.header.chunk_index + 1) % c.header.chunk_total,
            |c| c.header.chunk_total += 1,
            |c| c.header.signed_absolute_deadline_ms += 1,
        ];
        for relabel in relabelings {
            let mut chunk = chunks[0].clone();
            relabel(&mut chunk);
            let sent = send(&mut l, "relabeled_envelope", 1, &chunk);
            tally.add("relabeled_envelope_replay_cases", 1);
            tally.add("relabeled_chunks_admitted", i64::from(sent.admitted()));
            attempt(tally, &sent, "bad_signature");
        }
    }
    hardening(tally, rig, seed)
}

/// The receiver's checks on who sent a payload and on what the payload says about itself:
/// the replay boundary tick, a forged policy, a principal speaking for a stage that is not
/// its own, a signed chain that is not the chain inside the frame, and the global byte cap.
fn hardening(tally: &mut Tally, rig: &Rig, seed: u64) -> Res<()> {
    let rows = params::PROBE_FRAME_ROWS as usize;
    let budget = params::FRAME_BUDGET_MS;

    // The boundary tick: a completed frame replayed at exactly deadline + skew. The tombstone
    // is gone at that tick, so only the deadline can still refuse it.
    {
        let mut l = link(rig, seed, 120)?;
        let (_, _, chunks) = chunks_of(&mut l, rig, rows, 12, budget)?;
        for chunk in &chunks {
            deliver(&mut l, 1, chunk);
        }
        l.now_ms = chunks[0].header.signed_absolute_deadline_ms + params::MAX_CLOCK_SKEW_MS;
        let sent = send(&mut l, "boundary_tick_replay", 1, &chunks[0]);
        tally.add("boundary_tick_replay_attempts", 1);
        tally.add("boundary_tick_replays_admitted", i64::from(sent.admitted()));
        attempt(tally, &sent, "expired");
    }

    // A policy nobody signed, and one signed by another key: never delivered.
    {
        let mut l = link(rig, seed, 121)?;
        let policy = SplitPolicy {
            operation_id: 121,
            split_layer: params::SPLIT_LAYER as u32,
            width: params::HIDDEN_SIZE as u32,
            dtype: BoundaryDtype::F32,
            signed_absolute_deadline_ms: l.now_ms + budget,
            signature: [0; 64],
        };
        let other = SigningKey::from_bytes(&[9; 32]);
        for forged in [policy, sign_policy(&other, policy)] {
            let payload = TypedPayload {
                type_id: TypeId::from(SPLIT_POLICY_TYPE),
                bytes: encode_policy(&forged),
            };
            let sent = l.wire.send(
                &mut l.assembler,
                l.now_ms,
                1,
                Role::Probe("forged_policy"),
                &payload,
            );
            tally.add("unsigned_chunk_attempts", 1);
            tally.add("unsigned_chunks_admitted", i64::from(sent.admitted()));
            attempt(tally, &sent, "bad_signature");
        }
    }

    // A valid signature of the registered principal over a chain that names another stage.
    {
        let mut l = link(rig, seed, 122)?;
        let mut frame = probe_frame(rig, rows, 13);
        frame.upstream_chain.stage.identity = params::DOWNSTREAM_STAGE_IDENTITY.to_string();
        let (_, chunks) = l.chunks(rig, &frame, params::PROBE_CHUNK_BYTES as usize, budget)?;
        let sent = send(&mut l, "unbound_stage", 1, &chunks[0]);
        tally.add("unbound_stage_attempts", 1);
        tally.add("unbound_stage_frames_admitted", i64::from(sent.admitted()));
        attempt(tally, &sent, "stage_not_bound");
    }

    // The signed chain says X, the frame bytes say Y: refused whole, nothing delivered.
    {
        let mut l = link(rig, seed, 123)?;
        let mut inner = probe_frame(rig, rows, 14);
        inner.upstream_chain.base_model.generation += 1;
        let bytes = inner.to_bytes();
        let digest = frame_digest(&bytes);
        let size = params::PROBE_CHUNK_BYTES as usize;
        let total = bytes.chunks(size).count() as u32;
        let id = l.next_frame_id();
        tally.add("chain_frame_mismatch_attempts", 1);
        let mut last = None;
        for (index, part) in bytes.chunks(size).enumerate() {
            let header = ChunkHeader {
                frame_id: id,
                parent_operation_id: l.operation,
                chunk_index: index as u32,
                chunk_total: total,
                signed_absolute_deadline_ms: l.now_ms + budget,
                frame_digest: digest,
                chunk_digest: [0; 32],
                upstream_chain: rig.chain(),
            };
            let chunk = sign_chunk(&rig.signer, header, part.to_vec());
            last = Some(send(&mut l, "chain_frame_mismatch", 1, &chunk));
        }
        let last = last.ok_or("no chunks")?;
        tally.add(
            "chain_frame_mismatches_delivered",
            i64::from(matches!(last, Sent::Answered(Ok(Offered::Delivered(_))))),
        );
        attempt(tally, &last, "chain_frame_mismatch");
    }

    // A signed policy for the operation says another width: the frame is refused whole.
    {
        let mut l = link(rig, seed, 125)?;
        let policy = sign_policy(
            &rig.signer,
            SplitPolicy {
                operation_id: l.operation,
                split_layer: params::SPLIT_LAYER as u32,
                width: params::HIDDEN_SIZE as u32 * 2,
                dtype: BoundaryDtype::F32,
                signed_absolute_deadline_ms: l.now_ms + budget,
                signature: [0; 64],
            },
        );
        let payload = TypedPayload {
            type_id: TypeId::from(SPLIT_POLICY_TYPE),
            bytes: encode_policy(&policy),
        };
        let told = l
            .wire
            .send(&mut l.assembler, l.now_ms, 1, Role::Valid, &payload);
        if !told.admitted() {
            tally.add("attempts_without_typed_rejection", 1);
        }
        let (_, _, chunks) = chunks_of(&mut l, rig, rows, 17, budget)?;
        tally.add("chain_frame_mismatch_attempts", 1);
        let mut last = None;
        for chunk in &chunks {
            last = Some(send(&mut l, "policy_mismatch", 1, chunk));
        }
        let last = last.ok_or("no chunks")?;
        tally.add(
            "chain_frame_mismatches_delivered",
            i64::from(matches!(last, Sent::Answered(Ok(Offered::Delivered(_))))),
        );
        attempt(tally, &last, "policy_mismatch");
    }

    // The global cap: two principals, a cap of two chunks' worth; the third chunk is refused
    // whoever sends it, and the buffered bytes never pass the cap.
    {
        let size = params::PROBE_CHUNK_BYTES;
        let cap = 2 * size;
        // A frame bound of the same size: the policy needs a total of at least one frame.
        let limits = AssemblyPolicy {
            max_total_buffered_bytes: cap,
            max_frame_bytes: cap,
            ..super::link::policy()
        };
        let mut assembler =
            FrameAssembler::new(limits).map_err(|violation| format!("{violation:?}"))?;
        assembler.register_principal(
            params::UPSTREAM_PRINCIPAL,
            rig.signer.verifying_key(),
            params::STAGE_IDENTITY,
        );
        let other_key = SigningKey::from_bytes(&[11; 32]);
        assembler.register_principal(
            "other-runtime",
            other_key.verifying_key(),
            params::DOWNSTREAM_STAGE_IDENTITY,
        );
        let now = 1_000;
        let meta = |frame_id: u64| FrameMeta {
            frame_id,
            parent_operation_id: 124,
            signed_absolute_deadline_ms: now + budget,
        };
        let chunk_size = size as usize;
        let mine = chunk_frame(
            &rig.signer,
            &probe_frame(rig, rows, 15),
            meta(1),
            chunk_size,
        )
        .map_err(|error| format!("{error:?}"))?;
        let mut theirs_frame = probe_frame(rig, rows, 16);
        theirs_frame.upstream_chain.stage.identity = params::DOWNSTREAM_STAGE_IDENTITY.to_string();
        let theirs = chunk_frame(&other_key, &theirs_frame, meta(1), chunk_size)
            .map_err(|error| format!("{error:?}"))?;
        let mut offer = |principal: &str, chunk: &ActivationChunk| {
            assembler.offer(now, principal, 1, &chunk_payload(chunk))
        };
        offer(params::UPSTREAM_PRINCIPAL, &mine[0]).map_err(|e| format!("{e:?}"))?;
        offer("other-runtime", &theirs[0]).map_err(|e| format!("{e:?}"))?;
        for (principal, chunk) in [
            (params::UPSTREAM_PRINCIPAL, &mine[1]),
            ("other-runtime", &theirs[1]),
        ] {
            tally.add("global_cap_attempts", 1);
            match offer(principal, chunk) {
                Err(rejection) if rejection.kind() == "global_buffer_full" => {}
                Err(_) => tally.add("verdict_mismatches", 1),
                Ok(_) => tally.add("attempts_without_typed_rejection", 1),
            }
        }
        if assembler.buffered_bytes() as u64 > cap {
            tally.add("global_cap_breaches", 1);
        }
    }
    Ok(())
}

/// The bounds of the receiver: peers, buffered chunks.
pub fn bounds(tally: &mut Tally, rig: &Rig, seed: u64) -> Res<()> {
    let budget = params::FRAME_BUDGET_MS;
    {
        let mut l = link(rig, seed, 200)?;
        for _ in 0..params::MAX_INCOMPLETE_FRAMES_PER_PEER {
            let (_, _, chunks) =
                chunks_of(&mut l, rig, params::PROBE_FRAME_ROWS as usize, 1, budget)?;
            deliver(&mut l, 1, &chunks[0]);
        }
        let (_, _, chunks) = chunks_of(&mut l, rig, params::PROBE_FRAME_ROWS as usize, 1, budget)?;
        let sent = send(&mut l, "incomplete_frames_over_peer_limit", 1, &chunks[0]);
        tally.add("incomplete_frames_over_peer_limit", 1);
        attempt(tally, &sent, "too_many_incomplete_frames");
    }
    {
        // Eight frames of many chunks each, the first four chunks of each buffered.
        let mut l = link(rig, seed, 201)?;
        let mut held = Vec::new();
        for _ in 0..params::MAX_INCOMPLETE_FRAMES_PER_PEER {
            let (_, _, chunks) = chunks_of(&mut l, rig, 64, 1, budget)?;
            for chunk in &chunks[..4] {
                deliver(&mut l, 1, chunk);
            }
            held.push(chunks);
        }
        if l.assembler.buffered_chunks(params::UPSTREAM_PRINCIPAL) as u64
            != params::MAX_CHUNKS_IN_FLIGHT
        {
            return Err("the in-flight bound was not reached".to_string());
        }
        let sent = send(&mut l, "queue_overflow", 1, &held[0][4]);
        tally.add("queue_overflow_attempts", 1);
        attempt(tally, &sent, "queue_overflow");
    }
    Ok(())
}

/// Replays of completed frames: same session, another session, concurrent assembly.
pub fn replays(tally: &mut Tally, rig: &Rig, seed: u64, frames: &mut Frames) -> Res<()> {
    let rows = params::PROBE_FRAME_ROWS as usize;
    let budget = params::FRAME_BUDGET_MS;
    let mut l = link(rig, seed, 300)?;
    let (id, frame, chunks) = chunks_of(&mut l, rig, rows, 21, budget)?;
    let mut deliveries = 0;
    for chunk in &chunks {
        if let Sent::Answered(Ok(Offered::Delivered(done))) = deliver(&mut l, 1, chunk) {
            deliveries += 1;
            if done.bytes != frame.to_bytes() {
                tally.add("valid_frame_bitwise_mismatches", 1);
            }
        }
    }
    note_valid(tally, frames, 300, id, &chunks, &frame);
    if deliveries != 1 {
        tally.add("duplicate_activation_deliveries", 1);
    }
    // Full-frame replay, same session, then after a reconnect (another session), each chunk
    // under a fresh PodWire request id.
    for session in [1, 2] {
        for (index, chunk) in chunks.iter().enumerate() {
            let sent = send(&mut l, "full_frame_replay", session, chunk);
            if index == 0 {
                if session == 1 {
                    tally.add("full_frame_replay_attempts", 1);
                } else {
                    tally.add("cross_session_replay_attempts", 1);
                }
            }
            if sent.admitted() {
                tally.add(
                    if session == 1 {
                        "replayed_frames_admitted"
                    } else {
                        "cross_session_replays_admitted"
                    },
                    1,
                );
            }
            attempt(tally, &sent, "frame_replayed");
        }
    }

    // Concurrent assembly of one tuple from two sessions: exactly one delivery.
    let mut l = link(rig, seed, 301)?;
    let (id, frame, chunks) = chunks_of(&mut l, rig, rows, 22, budget)?;
    let mut deliveries = 0;
    let mut sessions = 0;
    for (index, chunk) in chunks.iter().enumerate() {
        let session = 1 + (index as u64 % 2);
        if let Sent::Answered(Ok(Offered::Delivered(done))) = deliver(&mut l, session, chunk) {
            deliveries += 1;
            sessions = done.sessions;
            if done.bytes != frame.to_bytes() {
                tally.add("valid_frame_bitwise_mismatches", 1);
            }
        }
    }
    note_valid(tally, frames, 301, id, &chunks, &frame);
    // The other session sends the whole frame again, as a reconnect that raced the first.
    for chunk in &chunks {
        if let Sent::Answered(Ok(Offered::Delivered(_))) = deliver(&mut l, 2, chunk) {
            deliveries += 1;
        }
    }
    tally.add("concurrent_cross_session_assembly_cases", 1);
    if sessions != 2 {
        tally.add("verdict_mismatches", 1);
    }
    if deliveries > 1 {
        tally.add("duplicate_activation_deliveries", deliveries - 1);
    }
    if deliveries == 0 {
        tally.add("duplicate_activation_deliveries", 1);
    }
    Ok(())
}

/// Count a valid frame the cases sent: its id, and its digest consistency.
fn note_valid(
    tally: &mut Tally,
    frames: &mut Frames,
    operation: u64,
    id: u64,
    chunks: &[ActivationChunk],
    frame: &ActivationFrame,
) {
    let name = format!("{operation}:{id}");
    tally.add("valid_frames_sent", 1);
    frames.ids.push(name.clone());
    let digest = ptr_protocol::activation::frame_digest(&frame.to_bytes());
    if chunks
        .iter()
        .all(|chunk| chunk.header.frame_digest == digest)
    {
        tally.add("frames_with_consistent_whole_frame_digest", 1);
        frames.consistent_ids.push(name);
    }
}

/// Abandonment: explicit, by expiry, replays inside the window, and a frame after it.
pub fn abandonment(
    tally: &mut Tally,
    rig: &Rig,
    seed: u64,
    frames: &mut Frames,
    delays: &mut Vec<i64>,
) -> Res<()> {
    let rows = params::PROBE_FRAME_ROWS as usize;
    let principal = params::UPSTREAM_PRINCIPAL;
    let mut sent_ids = Vec::new();
    let mut released_ids = Vec::new();
    let name = |id: u64, op: u64, index: u32| format!("{principal}|{op}|{id}|{index}");
    let mut rng = Rng::new(seed ^ 0xAB4D_0123);

    // Explicit abandonment with chunks buffered.
    let mut l = link(rig, seed, 400)?;
    if l.assembler.buffered_bytes() as u64 != params::BUFFERED_BYTES_BASELINE {
        return Err("the receiver starts with buffered bytes".to_string());
    }
    let (id, _, chunks) = chunks_of(&mut l, rig, rows, 31, 4_000)?;
    for chunk in &chunks[..3] {
        deliver(&mut l, 1, chunk);
        sent_ids.push(name(id, 400, chunk.header.chunk_index));
    }
    tally.add("abandoned_chunks_sent", 3);
    tally.add(
        "buffered_bytes_before_abandon",
        l.assembler.buffered_bytes() as i64,
    );
    let released = l
        .assembler
        .abandon(l.now_ms, principal, 400, id)
        .map_err(|error| format!("abandoning: {error:?}"))?;
    for chunk in &released.chunks {
        released_ids.push(name(
            chunk.frame_id,
            chunk.parent_operation_id,
            chunk.chunk_index,
        ));
    }
    tally.add("abandoned_chunks_released", released.chunks.len() as i64);
    tally.add(
        "buffered_bytes_after_release",
        l.assembler.buffered_bytes() as i64,
    );
    tally.add("abandoned_frame_cases", 1);
    // The rest of the frame arrives after the abandonment.
    for chunk in &chunks[3..] {
        let sent = send(&mut l, "late_chunk_after_abandon", 1, chunk);
        tally.add("late_chunks_after_abandon_attempts", 1);
        tally.add("reassembly_attempts_after_abandon", 1);
        if sent.admitted() {
            tally.add("late_chunks_after_abandon_admitted", 1);
        }
        attempt(tally, &sent, "frame_abandoned");
    }
    // An immediate replay of the first chunk.
    let sent = send(&mut l, "post_abandonment_replay", 2, &chunks[0]);
    tally.add("immediate_replay_cases", 1);
    tally.add("post_abandonment_replay_cases", 1);
    if sent.admitted() {
        tally.add("abandoned_frame_replays_admitted", 1);
    }
    attempt(tally, &sent, "frame_abandoned");
    // A replay near the end of the window.
    let delay = params::LATE_WINDOW_REPLAY_DELAY_RANGE_MS[0]
        + rng.below(
            params::LATE_WINDOW_REPLAY_DELAY_RANGE_MS[1]
                - params::LATE_WINDOW_REPLAY_DELAY_RANGE_MS[0],
        );
    l.now_ms += delay;
    let sent = send(&mut l, "post_abandonment_replay", 2, &chunks[1]);
    tally.add("late_window_replay_cases", 1);
    tally.add("post_abandonment_replay_cases", 1);
    delays.push(delay as i64);
    if sent.admitted() {
        tally.add("abandoned_frame_replays_admitted", 1);
    }
    attempt(tally, &sent, "frame_abandoned");
    // The frame that follows reassembles: the abandoned one released its chunks first.
    let (id, frame, chunks) = chunks_of(&mut l, rig, rows, 32, params::FRAME_BUDGET_MS)?;
    let mut delivered = 0;
    for chunk in &chunks {
        if let Sent::Answered(Ok(Offered::Delivered(done))) = deliver(&mut l, 1, chunk) {
            delivered += 1;
            if done.bytes != frame.to_bytes() {
                tally.add("valid_frame_bitwise_mismatches", 1);
            }
        }
    }
    if delivered != 1 {
        tally.add("verdict_mismatches", 1);
    }
    note_valid(tally, frames, 400, id, &chunks, &frame);

    // Abandonment by expiry: the deadline passes with chunks buffered.
    let mut l = link(rig, seed, 401)?;
    let (id, _, chunks) = chunks_of(&mut l, rig, rows, 33, params::FRAME_BUDGET_MS)?;
    for chunk in &chunks[..2] {
        deliver(&mut l, 1, chunk);
        sent_ids.push(name(id, 401, chunk.header.chunk_index));
    }
    tally.add("abandoned_chunks_sent", 2);
    tally.add(
        "buffered_bytes_before_abandon",
        l.assembler.buffered_bytes() as i64,
    );
    l.now_ms += params::FRAME_BUDGET_MS + params::MAX_CLOCK_SKEW_MS + 1;
    let released = l.assembler.expire(l.now_ms);
    for chunk in &released {
        released_ids.push(name(
            chunk.frame_id,
            chunk.parent_operation_id,
            chunk.chunk_index,
        ));
    }
    tally.add("abandoned_chunks_released", released.len() as i64);
    tally.add(
        "buffered_bytes_after_release",
        l.assembler.buffered_bytes() as i64,
    );
    tally.add("abandoned_frame_cases", 1);
    let sent = send(&mut l, "late_chunk_after_abandon", 1, &chunks[2]);
    tally.add("late_chunks_after_abandon_attempts", 1);
    tally.add("reassembly_attempts_after_abandon", 1);
    if sent.admitted() {
        tally.add("late_chunks_after_abandon_admitted", 1);
    }
    attempt(tally, &sent, "frame_abandoned");

    tally.set_json(
        "abandoned_chunk_id_set_digest",
        Json::str(id_set_digest(sent_ids.iter().map(String::as_str))),
    );
    tally.set_json(
        "released_chunk_id_set_digest",
        Json::str(id_set_digest(released_ids.iter().map(String::as_str))),
    );

    // Several admitted frames abandoned together, at the peer limit.
    let mut l = link(rig, seed, 402)?;
    let mut ids = Vec::new();
    for _ in 0..params::MAX_INCOMPLETE_FRAMES_PER_PEER {
        let (id, _, chunks) = chunks_of(&mut l, rig, rows, 34, 4_000)?;
        deliver(&mut l, 1, &chunks[0]);
        ids.push(id);
    }
    for id in ids {
        l.assembler
            .abandon(l.now_ms, principal, 402, id)
            .map_err(|error| format!("abandoning: {error:?}"))?;
        if l.assembler.reserved_slots(principal) as u64 > params::MAX_FRAME_TOMBSTONES_PER_PRINCIPAL
        {
            tally.add("tombstone_bound_violations", 1);
        }
    }
    tally.add("concurrent_abandonment_cases", 1);
    Ok(())
}

/// The table of tombstones fills, refuses, and frees a slot as the oldest expires.
pub fn saturation(tally: &mut Tally, rig: &Rig, seed: u64) -> Res<()> {
    let rows = params::PROBE_FRAME_ROWS as usize;
    let principal = params::UPSTREAM_PRINCIPAL;
    let mut l = link(rig, seed, 500)?;
    let start = l.now_ms;
    let mut abandoned = Vec::new();
    for index in 0..params::TOMBSTONE_OVERCAPACITY_FRAMES {
        l.now_ms = start + index * params::TOMBSTONE_STAGGER_MS;
        let (id, _, chunks) = chunks_of(&mut l, rig, rows, 41, 4_000)?;
        let sent = deliver(&mut l, 1, &chunks[0]);
        if index < params::MAX_FRAME_TOMBSTONES_PER_PRINCIPAL {
            if !sent.admitted() {
                return Err(format!(
                    "frame {index} was refused before the table was full"
                ));
            }
            l.assembler
                .abandon(l.now_ms, principal, 500, id)
                .map_err(|error| format!("abandoning: {error:?}"))?;
            abandoned.push((id, chunks[0].clone(), l.now_ms));
            tally.add("abandoned_frames_in_saturation_case", 1);
        } else {
            tally.add("overcapacity_frame_attempts", 1);
            if sent.rejection().map(|r| r.kind()) == Some("frame_table_full") {
                tally.add("new_frames_refused_typed", 1);
            } else {
                tally.add("attempts_without_typed_rejection", 1);
            }
        }
        if l.assembler.reserved_slots(principal) as u64 > params::MAX_FRAME_TOMBSTONES_PER_PRINCIPAL
        {
            tally.add("tombstone_bound_violations", 1);
        }
    }
    tally.add("tombstone_saturation_cases", 1);
    tally.set(
        "tombstones_retained_at_saturation",
        l.assembler.retained_tombstones(principal) as i64,
    );
    // Nothing is evicted early: every abandoned frame still refuses a replay, and the table
    // keeps its tombstones, until the first one's retention ends.
    let first_expiry = abandoned[0].2 + params::REPLAY_WINDOW_MS;
    l.now_ms = first_expiry - 1;
    for (_, chunk, _) in &abandoned {
        let sent = send(&mut l, "late_chunk_after_abandon", 2, chunk);
        if sent.rejection().map(|r| r.kind()) != Some("frame_abandoned") {
            tally.add("tombstone_evictions_before_retention", 1);
        }
    }
    if l.assembler.retained_tombstones(principal) as u64
        != params::MAX_FRAME_TOMBSTONES_PER_PRINCIPAL
    {
        tally.add("tombstone_evictions_before_retention", 1);
    }
    // One tick later the first tombstone is gone: exactly one new frame fits.
    l.now_ms = first_expiry;
    let (_, _, fresh) = chunks_of(&mut l, rig, rows, 42, 4_000)?;
    let sent = deliver(&mut l, 1, &fresh[0]);
    if sent.admitted() {
        tally.add("frames_accepted_after_tombstone_expiry", 1);
    }
    let (_, _, again) = chunks_of(&mut l, rig, rows, 43, 4_000)?;
    let sent = send(&mut l, "overcapacity_frame", 1, &again[0]);
    tally.add("overcapacity_frame_attempts", 1);
    attempt(tally, &sent, "frame_table_full");
    if sent.rejection().map(|r| r.kind()) == Some("frame_table_full") {
        tally.add("new_frames_refused_typed", 1);
    }
    if l.assembler.reserved_slots(principal) as u64 > params::MAX_FRAME_TOMBSTONES_PER_PRINCIPAL {
        tally.add("tombstone_bound_violations", 1);
    }
    Ok(())
}

/// The slots of the table are shared by incomplete frames and tombstones: with 60 tombstones
/// held, only four frames may be incomplete at once, and the fifth is refused.
pub fn shared_slots(tally: &mut Tally, rig: &Rig, seed: u64) -> Res<()> {
    let rows = params::PROBE_FRAME_ROWS as usize;
    let principal = params::UPSTREAM_PRINCIPAL;
    let max = params::MAX_FRAME_TOMBSTONES_PER_PRINCIPAL;
    let held = max - 4;
    let mut l = link(rig, seed, 510)?;
    for _ in 0..held {
        let (id, _, chunks) = chunks_of(&mut l, rig, rows, 44, 4_000)?;
        deliver(&mut l, 1, &chunks[0]);
        l.assembler
            .abandon(l.now_ms, principal, 510, id)
            .map_err(|error| format!("abandoning: {error:?}"))?;
    }
    for _ in 0..4 {
        let (_, _, chunks) = chunks_of(&mut l, rig, rows, 45, 4_000)?;
        let sent = deliver(&mut l, 1, &chunks[0]);
        if !sent.admitted() {
            return Err("a frame inside the bound was refused".to_string());
        }
    }
    let (_, _, chunks) = chunks_of(&mut l, rig, rows, 46, 4_000)?;
    let sent = send(&mut l, "overcapacity_frame", 1, &chunks[0]);
    tally.add("overcapacity_frame_attempts", 1);
    attempt(tally, &sent, "frame_table_full");
    if sent.rejection().map(|r| r.kind()) == Some("frame_table_full") {
        tally.add("new_frames_refused_typed", 1);
    }
    if l.assembler.reserved_slots(principal) as u64 > max {
        tally.add("tombstone_bound_violations", 1);
    }
    Ok(())
}

/// Two frames with one upstream chain reassembled at once, chunks interleaved.
pub fn interleaving(tally: &mut Tally, rig: &Rig, seed: u64, frames: &mut Frames) -> Res<()> {
    let rows = params::PROBE_FRAME_ROWS as usize;
    let mut l = link(rig, seed, 600)?;
    let (id_a, frame_a, chunks_a) = chunks_of(&mut l, rig, rows, 51, params::FRAME_BUDGET_MS)?;
    let (id_b, frame_b, chunks_b) = chunks_of(&mut l, rig, rows, 52, params::FRAME_BUDGET_MS)?;
    if chunks_a[0].header.upstream_chain != chunks_b[0].header.upstream_chain {
        return Err("the interleaved frames do not share a chain".to_string());
    }
    let mut order: Vec<(bool, usize)> = Vec::new();
    for index in 0..chunks_a.len().max(chunks_b.len()) {
        if index < chunks_a.len() {
            order.push((true, index));
        }
        if index < chunks_b.len() {
            order.push((false, index));
        }
    }
    let mut delivered: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut previous = None;
    for (is_a, index) in order {
        if previous.is_some_and(|last| last != is_a) {
            tally.add("interleaved_chunk_switches", 1);
        }
        previous = Some(is_a);
        let chunk = if is_a {
            &chunks_a[index]
        } else {
            &chunks_b[index]
        };
        if let Sent::Answered(Ok(Offered::Delivered(done))) = deliver(&mut l, 1, chunk) {
            delivered.push((done.frame_id, done.bytes));
        }
    }
    tally.add("interleaved_frame_pairs", 1);
    let expected = [(id_a, frame_a.to_bytes()), (id_b, frame_b.to_bytes())];
    if delivered.len() != 2 {
        tally.add("cross_frame_mixing_count", 1);
    }
    for (id, bytes) in &delivered {
        let sent = expected.iter().find(|(wanted, _)| wanted == id);
        if sent.is_none_or(|(_, wanted)| wanted != bytes) {
            tally.add("cross_frame_mixing_count", 1);
        }
    }
    note_valid(tally, frames, 600, id_a, &chunks_a, &frame_a);
    note_valid(tally, frames, 600, id_b, &chunks_b, &frame_b);
    Ok(())
}
