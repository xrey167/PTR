//! The cases of Part B. Each sends payloads through [`Net`] (a real `PodHost` and real
//! `PodClient` endpoints over loopback iroh, one chunk per request) and judges what the receiving
//! assembler made of them. The frames, the adversarial chunks and the typed rejections they must
//! end in are those of E007 (`e007::chunks`, `e007::smuggle`, `e007::large`), replayed over the
//! network instead of through the in-process codec.
//!
//! v1 frame types throughout (`ActivationFrame`, `chunk_frame`, `ACTIVATION_CHUNK_TYPE`). Where
//! Part B would move to v2 is marked `V2:` below.

use super::net::{other_key, Net, Peer, Sent};
use super::{Counters, Res};
use crate::experiments::e007::bitwise::{prompt, whole, Output};
use crate::experiments::e007::large::independent_digest;
use crate::experiments::e007::params;
use crate::experiments::e007::rig::{bits, Rig, Stage};
use crate::experiments::layered_kv::lab::fail;
use crate::experiments::rng::Rng;
use ed25519_dalek::{SigningKey, VerifyingKey};
use ptr_podwire::{
    chunk_frame, resign_envelope, sign_chunk, sign_policy, AssemblyPolicy, DeliveredFrame,
    FrameMeta, Offered, RefusalCode, Rejection,
};
use ptr_protocol::activation::{
    chunk_payload, encode_chunk, encode_policy, frame_digest, ActivationChunk, ActivationFrame,
    BoundaryDtype, ChunkHeader, SchemaError, SplitPolicy, UpstreamChain, ACTIVATION_CHUNK_TYPE,
    SPLIT_POLICY_TYPE,
};
use ptr_protocol::TypedPayload;
use ptr_runtime::admit_frame;
use ptr_types::TypeId;

/// The receiver's limits as E007 preregistered them.
pub fn limits() -> AssemblyPolicy {
    crate::experiments::e007::link::policy()
}

/// The peers of the host: the upstream runtime's endpoint and the ones that must be told apart
/// from it.
pub struct Peers {
    /// The upstream runtime: admitted by the host, registered in the assembler.
    pub a: Peer,
    /// Admitted with a misconfigured (wide) scope, registered in the assembler with its own key.
    pub lax: Peer,
    /// A registered runtime of another stage that holds copies of A's chunks.
    pub captor: Peer,
    /// Admitted by the host, never registered in the assembler.
    pub plain: Peer,
    /// Never admitted by the host.
    pub stranger: Peer,
    /// A new endpoint identity for the runtime of A (a restarted runtime).
    pub reborn: Peer,
}

/// One run's working state.
pub struct H<'a> {
    pub net: &'a mut Net,
    pub p: &'a Peers,
    pub rig: &'a Rig,
    pub c: &'a mut Counters,
    pub seed: u64,
    frame: u64,
    operation: u64,
    rng: Rng,
}

impl<'a> H<'a> {
    pub fn new(
        net: &'a mut Net,
        p: &'a Peers,
        rig: &'a Rig,
        c: &'a mut Counters,
        seed: u64,
    ) -> Self {
        Self {
            net,
            p,
            rig,
            c,
            seed,
            frame: 1,
            operation: 0,
            rng: Rng::new(seed ^ 0xB0B0_0008),
        }
    }

    fn registrations(&self) -> Vec<(&'a str, VerifyingKey, &'static str)> {
        let p = self.p;
        vec![
            (p.a.id.as_str(), p.a.key.verifying_key(), p.a.stage),
            (p.lax.id.as_str(), p.lax.key.verifying_key(), p.lax.stage),
            (
                p.captor.id.as_str(),
                p.captor.key.verifying_key(),
                p.captor.stage,
            ),
        ]
    }

    /// A fresh receiver under the preregistered limits for `operation`.
    pub fn fresh(&mut self, operation: u64) -> Res<()> {
        self.fresh_with(limits(), operation)
    }

    pub fn fresh_with(&mut self, limits: AssemblyPolicy, operation: u64) -> Res<()> {
        let registrations = self.registrations();
        self.net.reset(limits, &registrations)?;
        self.frame = 1;
        self.operation = operation;
        Ok(())
    }

    fn next_frame(&mut self) -> u64 {
        let id = self.frame;
        self.frame += 1;
        id
    }

    /// Cut `frame` into chunks signed by the upstream runtime, deadline `budget_ms` ahead.
    fn cut(
        &mut self,
        frame: &ActivationFrame,
        size: usize,
        budget_ms: u64,
        min_chunks: usize,
    ) -> Res<(u64, Vec<ActivationChunk>)> {
        let frame_id = self.next_frame();
        let chunks = chunk_frame(
            &self.rig.signer,
            frame,
            FrameMeta {
                frame_id,
                parent_operation_id: self.operation,
                signed_absolute_deadline_ms: self.net.now_ms + budget_ms,
            },
            size,
        )
        .map_err(|error| format!("chunking a frame: {error:?}"))?;
        need(&chunks, min_chunks)?;
        Ok((frame_id, chunks))
    }

    /// Signed chunks of arbitrary `bytes` taken as frame bytes (a consistent frame digest of
    /// those bytes, whatever they are).
    fn raw_chunks(
        &mut self,
        bytes: &[u8],
        chain: &UpstreamChain,
        size: usize,
    ) -> Vec<ActivationChunk> {
        let frame_id = self.next_frame();
        let digest = frame_digest(bytes);
        let total = bytes.len().div_ceil(size).max(1) as u32;
        let deadline = self.net.now_ms + params::FRAME_BUDGET_MS;
        bytes
            .chunks(size)
            .enumerate()
            .map(|(index, part)| {
                sign_chunk(
                    &self.rig.signer,
                    ChunkHeader {
                        frame_id,
                        parent_operation_id: self.operation,
                        chunk_index: index as u32,
                        chunk_total: total,
                        signed_absolute_deadline_ms: deadline,
                        frame_digest: digest,
                        chunk_digest: [0; 32],
                        upstream_chain: chain.clone(),
                    },
                    part.to_vec(),
                )
            })
            .collect()
    }

    /// Record what the assembler said, by rejection kind.
    fn observe(&mut self, sent: &Sent) {
        if let Some(rejection) = sent.rejection() {
            self.c
                .add(&format!("rejection_kinds.{}", rejection.kind()), 1);
        }
        if let Some(code) = sent.host_refusal() {
            self.c.add(&format!("host_refusals.{code:?}"), 1);
        }
    }

    /// Send a payload from `from`.
    pub fn send_payload(&mut self, from: &Peer, payload: &TypedPayload) -> Res<Sent> {
        let sent = self.net.send(self.c, from, payload)?;
        self.observe(&sent);
        Ok(sent)
    }

    /// Send one chunk from the upstream runtime's endpoint.
    pub fn chunk(&mut self, chunk: &ActivationChunk) -> Res<Sent> {
        self.c.add("chunks_sent", 1);
        let p = self.p;
        self.send_payload(&p.a, &chunk_payload(chunk))
    }

    /// Send one chunk from `from`.
    fn chunk_from(&mut self, from: &Peer, chunk: &ActivationChunk) -> Res<Sent> {
        self.c.add("chunks_sent", 1);
        self.send_payload(from, &chunk_payload(chunk))
    }

    /// Judge one attempt: it must end in a rejection of the pinned variant.
    pub fn attempt(&mut self, sent: &Sent, expected: &str) {
        self.c.add("rejection_cases", 1);
        match sent.rejection() {
            Some(rejection) => {
                if rejection.kind() != expected {
                    self.c.add("verdict_mismatches", 1);
                }
            }
            None => self.c.add("attempts_without_typed_rejection", 1),
        }
    }

    /// A valid frame the cases sent and the receiver delivered: compare bit for bit.
    fn check_delivery(
        &mut self,
        frame: &ActivationFrame,
        chunks: &[ActivationChunk],
        delivered: &DeliveredFrame,
    ) {
        let sent = frame.to_bytes();
        let independent = independent_digest(&sent);
        let identical = delivered.bytes == sent
            && delivered.frame == *frame
            && delivered.frame_digest == independent
            && independent_digest(&delivered.bytes) == independent
            && chunks
                .iter()
                .all(|chunk| chunk.header.frame_digest == independent);
        self.c.add("frames_delivered", 1);
        if !identical {
            self.c.add("bitwise_mismatch_count", 1);
        }
    }

    /// Ship `frame` as valid traffic: chunks of the small-frame size, adjacent pairs swapped at
    /// random inside the reorder window, one chunk per request. `None` when the receiver did not
    /// deliver it exactly once (a product defect, counted).
    pub fn ship(&mut self, frame: &ActivationFrame) -> Res<Option<DeliveredFrame>> {
        let (_, chunks) = self.cut(
            frame,
            params::SMALL_FRAME_CHUNK_BYTES as usize,
            params::FRAME_BUDGET_MS,
            1,
        )?;
        let mut order: Vec<usize> = (0..chunks.len()).collect();
        for at in (0..order.len().saturating_sub(1)).step_by(2) {
            if self.rng.chance(0.5) {
                order.swap(at, at + 1);
            }
        }
        self.c.add("valid_frames_sent", 1);
        let mut delivered = None;
        let mut deliveries = 0;
        let mut refused = false;
        for index in order {
            let sent = self.chunk(&chunks[index])?;
            match sent {
                Sent::Exchange(exchange) => match exchange.verdict {
                    Some(Ok(Offered::Delivered(done))) => {
                        deliveries += 1;
                        delivered = Some(*done);
                    }
                    Some(Ok(_)) => {}
                    _ => refused = true,
                },
                Sent::Codec => refused = true,
            }
        }
        self.net.now_ms += params::FRAME_SPACING_MS;
        match delivered {
            Some(done) if deliveries == 1 && !refused => {
                self.check_delivery(frame, &chunks, &done);
                Ok(Some(done))
            }
            _ => {
                self.c.add("valid_frames_not_delivered", 1);
                self.c.add("bitwise_mismatch_count", 1);
                Ok(None)
            }
        }
    }
}

/// The case indexes into `chunks`: a frame that cut into fewer is a harness failure (exit 2).
fn need(chunks: &[ActivationChunk], count: usize) -> Res<()> {
    if chunks.len() < count {
        return Err(format!(
            "a frame cut into {} chunks, the case needs {count}",
            chunks.len()
        ));
    }
    Ok(())
}

fn probe_frame(rig: &Rig, rows: usize, fill: u8) -> ActivationFrame {
    // V2: the probe frames are built here; Part B would build the v2 frame instead.
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

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (index, value) in row.iter().enumerate() {
        if *value > row[best] {
            best = index;
        }
    }
    best as u32
}

/// C1: the split model of E007 with every activation frame crossing the loopback wire as signed
/// chunks, against the single-pod run: the frames arrive bit-identical and stage B, fed the
/// delivered frames, produces the single-pod output bit for bit.
pub fn corpus(h: &mut H<'_>, sequences: u64) -> Res<()> {
    let steps = params::DECODE_STEPS as usize;
    h.fresh(1_000)?;
    for index in 0..sequences {
        let tokens = prompt(index);
        let reference = whole(h.rig, &tokens, steps)?;
        h.operation = 1_000 + index;
        let observed = split_over_wire(h, &tokens, steps)?;
        h.c.add("corpus_sequences", 1);
        if observed != reference {
            h.c.add("split_output_mismatch_count", 1);
        }
    }
    Ok(())
}

/// The two stages of E007, A holding `0..split` and B `split..L`, with the residual rows carried
/// over the loopback wire.
fn split_over_wire(h: &mut H<'_>, tokens: &[u32], steps: usize) -> Res<Output> {
    let rig = h.rig;
    let stack = &rig.stack;
    let (split_at, layers) = (
        params::SPLIT_LAYER as usize,
        params::TEST_MODEL_LAYERS as usize,
    );
    let mut stage_a = Stage::new(stack, 0..split_at, "stage-a")?;
    let mut stage_b = Stage::new(stack, split_at..layers, "stage-b")?;
    let regime = stack
        .admit_rope_regime(tokens.len(), steps)
        .map_err(fail("the regime"))?;
    let mut logits = Vec::new();
    let mut generated = Vec::new();
    let embedded = stack.embed_tokens(tokens).map_err(fail("the embedding"))?;
    let mid = stage_a.run(stack, &embedded, tokens.len(), false, &regime)?;
    let arrived = carry(h, &mid.residual, mid.rows, mid.first_position)?;
    let end = stage_b.run(stack, &arrived, mid.rows, false, &regime)?;
    let rows = stack
        .logits_from_residual(&end.residual)
        .map_err(fail("the logits"))?;
    logits.extend(rows.iter().map(|row| bits(row)));
    let mut next = argmax(rows.last().ok_or("no logits")?);
    generated.push(next);
    for _ in 1..steps {
        let embedded = stack.embed_tokens(&[next]).map_err(fail("the embedding"))?;
        let mid = stage_a.run(stack, &embedded, 1, true, &regime)?;
        let arrived = carry(h, &mid.residual, 1, mid.first_position)?;
        let end = stage_b.run(stack, &arrived, 1, true, &regime)?;
        let row = stack
            .logits_from_residual(&end.residual)
            .map_err(fail("the logits"))?;
        logits.push(bits(&row[0]));
        next = argmax(&row[0]);
        generated.push(next);
    }
    let mut kv = stage_a.kv_bytes()?;
    kv.extend(stage_b.kv_bytes()?);
    Ok(Output {
        logits,
        generated,
        kv,
    })
}

/// Stage A's residual rows to stage B, as one frame over the wire. A frame that does not arrive
/// is counted and stage B continues from the sender's rows, so the output comparison still runs.
fn carry(h: &mut H<'_>, residual: &[f32], rows: usize, first_position: usize) -> Res<Vec<f32>> {
    // V2: `Rig::frame` builds the v1 frame; Part B would build the v2 frame here.
    let frame = h.rig.frame(residual, rows, first_position);
    match h.ship(&frame)? {
        None => Ok(residual.to_vec()),
        Some(done) => {
            let admitted = admit_frame(
                &done.frame,
                ptr_model_api::KvReuseMode::Exact,
                &h.rig.live(),
            )
            .map_err(|fault| format!("a live frame was not admitted: {fault:?}"))?;
            Ok(admitted.activations().to_vec())
        }
    }
}

/// C2: the 16 MiB frame as five signed chunks of at most 3.5 MiB, one per request, out of order
/// inside the reorder window; and the control that sends the same bytes as one payload.
pub fn large(h: &mut H<'_>) -> Res<()> {
    h.fresh(9_000)?;
    let mut rng = Rng::new(h.seed ^ 0x1A46_E5CA_1E00);
    let mut tensor = Vec::with_capacity(params::LARGE_FRAME_BYTES as usize);
    while tensor.len() < params::LARGE_FRAME_BYTES as usize {
        tensor.extend_from_slice(&rng.next_u64().to_le_bytes());
    }
    let frame = ActivationFrame {
        tokens: params::LARGE_FRAME_TOKENS as u32,
        width: params::LARGE_FRAME_WIDTH as u32,
        dtype: BoundaryDtype::F32,
        positions: (0..params::LARGE_FRAME_TOKENS as u32).collect(),
        upstream_chain: h.rig.chain(),
        placement_epoch: h.rig.epoch(),
        tensor,
    };
    let sent = frame.to_bytes();
    // The control: the same bytes as one payload. PodWire's own bound must refuse it before
    // anything leaves.
    let whole = TypedPayload {
        type_id: TypeId::from(ACTIVATION_CHUNK_TYPE),
        bytes: sent.clone(),
    };
    let p = h.p;
    h.c.add("unchunked_large_frame_attempts", 1);
    if matches!(h.send_payload(&p.a, &whole)?, Sent::Codec) {
        h.c.add("control_trips.unchunked_large_frame_refused_by_podwire", 1);
    }
    let (_, chunks) = h.cut(
        &frame,
        params::MAX_CHUNK_BYTES as usize,
        params::FRAME_DEADLINE_MS,
        5,
    )?;
    if chunks.len() as u64 != params::LARGE_FRAME_CHUNKS {
        return Err(format!(
            "the large frame is {} chunks, the preregistration says {}",
            chunks.len(),
            params::LARGE_FRAME_CHUNKS
        ));
    }
    h.c.add("valid_frames_sent", 1);
    h.c.add("large_frame_cases", 1);
    let mut delivered = None;
    for index in params::LARGE_FRAME_SEND_ORDER {
        h.c.add("large_frame_chunks_sent", 1);
        if let Sent::Exchange(exchange) = h.chunk(&chunks[index as usize])? {
            if let Some(Ok(Offered::Delivered(done))) = exchange.verdict {
                delivered = Some(*done);
            }
        }
    }
    match delivered {
        Some(done) => h.check_delivery(&frame, &chunks, &done),
        None => {
            h.c.add("valid_frames_not_delivered", 1);
            h.c.add("bitwise_mismatch_count", 1);
        }
    }
    Ok(())
}

/// Every attack on the chunks of one frame, each on a fresh receiver.
pub fn attacks(h: &mut H<'_>) -> Res<()> {
    let rows = params::PROBE_FRAME_ROWS as usize;
    let budget = params::FRAME_BUDGET_MS;
    let size = params::PROBE_CHUNK_BYTES as usize;
    let rig = h.rig;
    let frame = |fill: u8| probe_frame(rig, rows, fill);

    // A different key signs: an unsigned chunk as far as the receiver is concerned.
    {
        h.fresh(100)?;
        let (_, chunks) = h.cut(&frame(1), size, budget, 7)?;
        let other = SigningKey::from_bytes(&[9; 32]);
        let forged = sign_chunk(&other, chunks[0].header.clone(), chunks[0].bytes.clone());
        let sent = h.chunk(&forged)?;
        h.c.add("unsigned_chunks_admitted", i64::from(sent.admitted()));
        h.attempt(&sent, "bad_signature");
        let mut flipped = chunks[0].clone();
        flipped.signature[3] ^= 0x40;
        let sent = h.chunk(&flipped)?;
        h.c.add("unsigned_chunks_admitted", i64::from(sent.admitted()));
        h.attempt(&sent, "bad_signature");
    }

    // chain_mismatch: a later chunk names another upstream chain.
    {
        h.fresh(101)?;
        let (_, chunks) = h.cut(&frame(2), size, budget, 7)?;
        h.chunk(&chunks[0])?;
        let mut header = chunks[1].header.clone();
        header.upstream_chain.stage.generation += 1;
        let swapped = sign_chunk(&rig.signer, header, chunks[1].bytes.clone());
        let sent = h.chunk(&swapped)?;
        h.attempt(&sent, "chain_mismatch");
        if h.net.assembler.incomplete_frames(&h.p.a.id) != 0 {
            h.c.add("attempts_without_typed_rejection", 1);
        }
    }

    // duplicate_chunk.
    {
        h.fresh(102)?;
        let (_, chunks) = h.cut(&frame(3), size, budget, 7)?;
        h.chunk(&chunks[0])?;
        let sent = h.chunk(&chunks[0])?;
        h.attempt(&sent, "duplicate_chunk");
    }

    // out_of_window: with chunk 0 held the window is 1..5, so chunk 5 is outside it.
    {
        h.fresh(103)?;
        let (_, chunks) = h.cut(&frame(4), size, budget, 7)?;
        if chunks.len() < 7 {
            return Err(format!(
                "the probe frame has {} chunks, fewer than 7",
                chunks.len()
            ));
        }
        h.chunk(&chunks[0])?;
        let sent = h.chunk(&chunks[5])?;
        h.attempt(&sent, "out_of_window");
    }

    // bad_chunk_digest: corrupted bytes under an intact envelope; and an envelope re-signed over
    // a digest that does not match the bytes.
    {
        h.fresh(104)?;
        let (_, chunks) = h.cut(&frame(5), size, budget, 7)?;
        let mut corrupted = chunks[0].clone();
        corrupted.bytes[7] ^= 0x01;
        let sent = h.chunk(&corrupted)?;
        h.attempt(&sent, "bad_chunk_digest");
        let mut resigned = chunks[1].clone();
        resigned.header.chunk_digest[0] ^= 0x01;
        resign_envelope(&rig.signer, &mut resigned);
        let sent = h.chunk(&resigned)?;
        h.attempt(&sent, "bad_chunk_digest");
    }

    // mixed_version_frame: the same frame announced with another chunk count.
    {
        h.fresh(105)?;
        let (_, chunks) = h.cut(&frame(6), size, budget, 7)?;
        h.chunk(&chunks[0])?;
        let mut header = chunks[1].header.clone();
        header.chunk_total += 1;
        let skewed = sign_chunk(&rig.signer, header, chunks[1].bytes.clone());
        let sent = h.chunk(&skewed)?;
        h.attempt(&sent, "frame_mismatch");
    }

    // splice: valid chunks of two different frames under one frame id.
    {
        h.fresh(106)?;
        let (_, first) = h.cut(&frame(7), size, budget, 7)?;
        let spliced = chunk_frame(
            &rig.signer,
            &frame(77),
            FrameMeta {
                frame_id: first[0].header.frame_id,
                parent_operation_id: first[0].header.parent_operation_id,
                signed_absolute_deadline_ms: first[0].header.signed_absolute_deadline_ms,
            },
            size,
        )
        .map_err(|error| format!("chunking the splice: {error:?}"))?;
        need(&spliced, 2)?;
        h.chunk(&first[0])?;
        let sent = h.chunk(&spliced[1])?;
        h.attempt(&sent, "frame_mismatch");
    }

    // cross_operation_splice: a chunk relabelled into another parent operation. Without the key
    // it fails the signature; with the key it is a frame of its own and must not join.
    {
        h.fresh(107)?;
        let f = frame(8);
        let (_, chunks) = h.cut(&f, size, budget, 7)?;
        h.chunk(&chunks[0])?;
        let mut relabelled = chunks[1].clone();
        relabelled.header.parent_operation_id += 1;
        let sent = h.chunk(&relabelled)?;
        h.attempt(&sent, "bad_signature");
        let colluding = sign_chunk(
            &rig.signer,
            relabelled.header.clone(),
            relabelled.bytes.clone(),
        );
        h.chunk(&colluding)?;
        let mut delivered = None;
        for chunk in &chunks[1..] {
            if let Sent::Exchange(exchange) = h.chunk(chunk)? {
                if let Some(Ok(Offered::Delivered(done))) = exchange.verdict {
                    delivered = Some(done);
                }
            }
        }
        match delivered {
            Some(done) if done.bytes == f.to_bytes() && done.parent_operation_id == 107 => {}
            _ => h.c.add("cross_operation_splices_admitted", 1),
        }
    }

    // mislabelled_frame_digest: every chunk carries one consistent but wrong digest.
    {
        h.fresh(108)?;
        let (_, chunks) = h.cut(&frame(9), size, budget, 7)?;
        let mut last = None;
        for chunk in &chunks {
            let mut header = chunk.header.clone();
            header.frame_digest = [0xEE; 32];
            let relabelled = sign_chunk(&rig.signer, header, chunk.bytes.clone());
            last = Some(h.chunk(&relabelled)?);
        }
        let last = last.ok_or("no chunks")?;
        if last.admitted() {
            h.c.add("mislabelled_frame_digests_admitted", 1);
        }
        h.attempt(&last, "frame_digest_mismatch");
    }

    // expired deadline: the signed deadline has passed (minus skew) before the chunk arrives.
    {
        h.fresh(109)?;
        h.net.now_ms = 20_000;
        let id = h.next_frame();
        let chunks = chunk_frame(
            &rig.signer,
            &frame(10),
            FrameMeta {
                frame_id: id,
                parent_operation_id: 109,
                signed_absolute_deadline_ms: 20_000 - params::MAX_CLOCK_SKEW_MS - 1,
            },
            size,
        )
        .map_err(|error| format!("chunking: {error:?}"))?;
        need(&chunks, 1)?;
        h.c.add("expired_deadline_frame_attempts", 1);
        let sent = h.chunk(&chunks[0])?;
        if sent.admitted() {
            h.c.add("expired_frames_admitted", 1);
        }
        h.attempt(&sent, "expired");
    }

    // relabelled envelopes of a completed frame: every field of the envelope is signed.
    {
        h.fresh(110)?;
        let (_, chunks) = h.cut(&frame(11), size, budget, 7)?;
        for chunk in &chunks {
            h.chunk(chunk)?;
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
            h.c.add("relabeled_envelope_replay_cases", 1);
            let sent = h.chunk(&chunk)?;
            if sent.admitted() {
                h.c.add("relabeled_chunks_admitted", 1);
            }
            h.attempt(&sent, "bad_signature");
        }
    }
    hardening(h)
}

/// The receiver's checks on what a payload says about itself: the replay boundary tick, a forged
/// policy, a principal speaking for a stage that is not its own, a signed chain that is not the
/// chain inside the frame, a signed policy of another width, and the global byte cap.
fn hardening(h: &mut H<'_>) -> Res<()> {
    let rows = params::PROBE_FRAME_ROWS as usize;
    let budget = params::FRAME_BUDGET_MS;
    let size = params::PROBE_CHUNK_BYTES as usize;
    let rig = h.rig;
    let p = h.p;

    // The boundary tick: a completed frame replayed at exactly deadline + skew.
    {
        h.fresh(120)?;
        let (_, chunks) = h.cut(&probe_frame(rig, rows, 12), size, budget, 7)?;
        for chunk in &chunks {
            h.chunk(chunk)?;
        }
        h.net.now_ms = chunks[0].header.signed_absolute_deadline_ms + params::MAX_CLOCK_SKEW_MS;
        h.c.add("boundary_tick_replay_attempts", 1);
        let sent = h.chunk(&chunks[0])?;
        if sent.admitted() {
            h.c.add("boundary_tick_replays_admitted", 1);
        }
        h.attempt(&sent, "expired");
    }

    // A policy nobody signed, and one signed by another key: never delivered.
    {
        h.fresh(121)?;
        let policy = SplitPolicy {
            operation_id: 121,
            split_layer: params::SPLIT_LAYER as u32,
            width: params::HIDDEN_SIZE as u32,
            dtype: BoundaryDtype::F32,
            signed_absolute_deadline_ms: h.net.now_ms + budget,
            signature: [0; 64],
        };
        let other = SigningKey::from_bytes(&[9; 32]);
        for forged in [policy, sign_policy(&other, policy)] {
            let payload = TypedPayload {
                type_id: TypeId::from(SPLIT_POLICY_TYPE),
                bytes: encode_policy(&forged),
            };
            let sent = h.send_payload(&p.a, &payload)?;
            h.c.add("unsigned_chunks_admitted", i64::from(sent.admitted()));
            h.attempt(&sent, "bad_signature");
        }
    }

    // A valid signature of the registered principal over a chain that names another stage.
    {
        h.fresh(122)?;
        let mut frame = probe_frame(rig, rows, 13);
        frame.upstream_chain.stage.identity = params::DOWNSTREAM_STAGE_IDENTITY.to_string();
        let (_, chunks) = h.cut(&frame, size, budget, 7)?;
        let sent = h.chunk(&chunks[0])?;
        if sent.admitted() {
            h.c.add("unbound_stage_frames_admitted", 1);
        }
        h.attempt(&sent, "stage_not_bound");
    }

    // The signed chain says X, the frame bytes say Y: refused whole, nothing delivered.
    {
        h.fresh(123)?;
        let mut inner = probe_frame(rig, rows, 14);
        inner.upstream_chain.base_model.generation += 1;
        let bytes = inner.to_bytes();
        let chunks = h.raw_chunks(&bytes, &rig.chain(), size);
        let mut last = None;
        for chunk in &chunks {
            last = Some(h.chunk(chunk)?);
        }
        let last = last.ok_or("no chunks")?;
        if matches!(
            &last,
            Sent::Exchange(exchange) if matches!(exchange.verdict, Some(Ok(Offered::Delivered(_))))
        ) {
            h.c.add("chain_frame_mismatches_delivered", 1);
        }
        h.attempt(&last, "chain_frame_mismatch");
    }

    // A signed policy for the operation says another width: the frame is refused whole.
    {
        h.fresh(125)?;
        let policy = sign_policy(
            &rig.signer,
            SplitPolicy {
                operation_id: 125,
                split_layer: params::SPLIT_LAYER as u32,
                width: params::HIDDEN_SIZE as u32 * 2,
                dtype: BoundaryDtype::F32,
                signed_absolute_deadline_ms: h.net.now_ms + budget,
                signature: [0; 64],
            },
        );
        let payload = TypedPayload {
            type_id: TypeId::from(SPLIT_POLICY_TYPE),
            bytes: encode_policy(&policy),
        };
        let told = h.send_payload(&p.a, &payload)?;
        if !told.admitted() {
            h.c.add("attempts_without_typed_rejection", 1);
        }
        let (_, chunks) = h.cut(&probe_frame(rig, rows, 17), size, budget, 7)?;
        let mut last = None;
        for chunk in &chunks {
            last = Some(h.chunk(chunk)?);
        }
        let last = last.ok_or("no chunks")?;
        if matches!(
            &last,
            Sent::Exchange(exchange) if matches!(exchange.verdict, Some(Ok(Offered::Delivered(_))))
        ) {
            h.c.add("chain_frame_mismatches_delivered", 1);
        }
        h.attempt(&last, "policy_mismatch");
    }

    // The global cap: two principals, a cap of two chunks' worth; the third chunk is refused
    // whoever sends it, and the buffered bytes never pass the cap.
    {
        let cap = 2 * params::PROBE_CHUNK_BYTES;
        let capped = AssemblyPolicy {
            max_total_buffered_bytes: cap,
            max_frame_bytes: cap,
            ..limits()
        };
        h.fresh_with(capped, 124)?;
        let meta = |frame_id: u64, now: u64| FrameMeta {
            frame_id,
            parent_operation_id: 124,
            signed_absolute_deadline_ms: now + budget,
        };
        let now = h.net.now_ms;
        let mine = chunk_frame(&rig.signer, &probe_frame(rig, rows, 15), meta(1, now), size)
            .map_err(|e| format!("{e:?}"))?;
        need(&mine, 2)?;
        let mut theirs_frame = probe_frame(rig, rows, 16);
        theirs_frame.upstream_chain.stage.identity = params::DOWNSTREAM_STAGE_IDENTITY.to_string();
        let theirs = chunk_frame(&p.captor.key, &theirs_frame, meta(1, now), size)
            .map_err(|e| format!("{e:?}"))?;
        need(&theirs, 2)?;
        let first = h.chunk_from(&p.a, &mine[0])?;
        let second = h.chunk_from(&p.captor, &theirs[0])?;
        if !first.admitted() || !second.admitted() {
            return Err("the first chunks under the cap were refused".to_string());
        }
        for (peer, chunk) in [(&p.a, &mine[1]), (&p.captor, &theirs[1])] {
            let sent = h.chunk_from(peer, chunk)?;
            h.attempt(&sent, "global_buffer_full");
        }
        if h.net.assembler.buffered_bytes() as u64 > cap {
            h.c.add("global_cap_breaches", 1);
        }
    }
    Ok(())
}

/// The bounds of the receiver: peers and buffered chunks.
pub fn bounds(h: &mut H<'_>) -> Res<()> {
    let size = params::PROBE_CHUNK_BYTES as usize;
    let budget = params::FRAME_BUDGET_MS;
    let rig = h.rig;
    {
        h.fresh(200)?;
        for _ in 0..params::MAX_INCOMPLETE_FRAMES_PER_PEER {
            let (_, chunks) = h.cut(
                &probe_frame(rig, params::PROBE_FRAME_ROWS as usize, 1),
                size,
                budget,
                7,
            )?;
            h.chunk(&chunks[0])?;
        }
        let (_, chunks) = h.cut(
            &probe_frame(rig, params::PROBE_FRAME_ROWS as usize, 1),
            size,
            budget,
            7,
        )?;
        let sent = h.chunk(&chunks[0])?;
        h.attempt(&sent, "too_many_incomplete_frames");
    }
    {
        // Eight frames of many chunks each, the first four chunks of each buffered.
        h.fresh(201)?;
        let mut held = Vec::new();
        for _ in 0..params::MAX_INCOMPLETE_FRAMES_PER_PEER {
            let (_, chunks) = h.cut(&probe_frame(rig, 64, 1), size, budget, 7)?;
            for chunk in &chunks[..4] {
                h.chunk(chunk)?;
            }
            held.push(chunks);
        }
        if h.net.assembler.buffered_chunks(&h.p.a.id) as u64 != params::MAX_CHUNKS_IN_FLIGHT {
            return Err("the in-flight bound was not reached".to_string());
        }
        let first_held = held.first().ok_or("no frame is held")?;
        need(first_held, 5)?;
        let sent = h.chunk(&first_held[4])?;
        h.attempt(&sent, "queue_overflow");
    }
    Ok(())
}

/// Replays of completed frames: on fresh connections (every request is one), after the
/// deployment registers the same principal again (a re-registration, not a network event), under
/// a new principal id with the same key and stage (a restarted runtime), against an assembler
/// without tombstones (the control), and one frame assembled from chunks that arrived on many
/// connections.
pub fn replays(h: &mut H<'_>) -> Res<()> {
    let rows = params::PROBE_FRAME_ROWS as usize;
    let budget = params::FRAME_BUDGET_MS;
    let size = params::PROBE_CHUNK_BYTES as usize;
    let rig = h.rig;
    let p = h.p;

    h.fresh(300)?;
    let frame = probe_frame(rig, rows, 21);
    let (_, chunks) = h.cut(&frame, size, budget, 7)?;
    h.c.add("valid_frames_sent", 1);
    let mut deliveries = 0;
    let mut delivered = None;
    for chunk in &chunks {
        if let Sent::Exchange(exchange) = h.chunk(chunk)? {
            if let Some(Ok(Offered::Delivered(done))) = exchange.verdict {
                deliveries += 1;
                delivered = Some(*done);
            }
        }
    }
    match delivered {
        Some(done) if deliveries == 1 => h.check_delivery(&frame, &chunks, &done),
        _ => {
            h.c.add("duplicate_activation_deliveries", 1);
            h.c.add("bitwise_mismatch_count", 1);
        }
    }
    // The whole frame again, each chunk on a fresh connection (the first connections are long
    // closed: PodClient::request opens and closes one per request).
    h.c.add("replay_cases", 1);
    for chunk in &chunks {
        let sent = h.chunk(chunk)?;
        h.c.add("replay_attempts", 1);
        if sent.admitted() {
            h.c.add("replays_admitted", 1);
        }
        h.attempt(&sent, "frame_replayed");
    }
    // A re-registration of the same principal: the deployment registers the peer again with the
    // same key and stage. Tombstones survive it, so the same chunks are still replays. This is
    // the assembler's side only; the network side of a reconnect is the fresh connection every
    // request opens.
    h.net
        .assembler
        .register_principal(&p.a.id, p.a.key.verifying_key(), p.a.stage);
    h.c.add("reregistration_replay_cases", 1);
    for chunk in &chunks {
        let sent = h.chunk(chunk)?;
        h.c.add("replay_attempts", 1);
        if sent.admitted() {
            h.c.add("replays_admitted", 1);
        }
        h.attempt(&sent, "frame_replayed");
    }

    // The runtime of A behind a new endpoint identity (a process restart draws a new iroh key):
    // a new principal id with the same key and stage inherits the tombstones, so the same
    // chunks are still replays. Gated: an admission counts toward the replay metric.
    {
        let reborn = &p.reborn;
        h.net
            .assembler
            .register_principal(&reborn.id, p.a.key.verifying_key(), p.a.stage);
        h.c.add("new_identity_replay_cases", 1);
        for chunk in &chunks {
            let sent = h.chunk_from(reborn, chunk)?;
            h.c.add("new_identity_replay_attempts", 1);
            if sent.admitted() {
                h.c.add("replays_admitted", 1);
            }
            h.attempt(&sent, "frame_replayed");
        }
    }

    // The negative control of the replay mechanism, on the same path (host, intake Pod,
    // assembler): the assembler has no switch to ignore its tombstones, so each replayed chunk
    // is offered to a fresh assembler, which holds none. Every one of them is admitted.
    for chunk in &chunks {
        h.fresh(302)?;
        let sent = h.chunk(chunk)?;
        h.c.add("replay_control_attempts", 1);
        if sent.admitted() {
            h.c.add("control_trips.replay_admitted_without_tombstones", 1);
        }
    }

    // One tuple assembled from chunks that arrived on many connections: exactly one delivery,
    // however the chunks and a second full copy of them were spread.
    h.fresh(301)?;
    let frame = probe_frame(rig, rows, 22);
    let (_, chunks) = h.cut(&frame, size, budget, 7)?;
    h.c.add("valid_frames_sent", 1);
    let mut deliveries = 0;
    let mut sessions = 0;
    let mut delivered = None;
    for chunk in &chunks {
        if let Sent::Exchange(exchange) = h.chunk(chunk)? {
            if let Some(Ok(Offered::Delivered(done))) = exchange.verdict {
                deliveries += 1;
                sessions = done.sessions;
                delivered = Some(*done);
            }
        }
    }
    // The second copy, as a second connection that raced the first.
    for chunk in &chunks {
        if let Sent::Exchange(exchange) = h.chunk(chunk)? {
            if let Some(Ok(Offered::Delivered(_))) = exchange.verdict {
                deliveries += 1;
            }
        }
    }
    h.c.add("cross_connection_assembly_cases", 1);
    match delivered {
        Some(done) => h.check_delivery(&frame, &chunks, &done),
        None => {
            h.c.add("bitwise_mismatch_count", 1);
        }
    }
    if sessions != chunks.len() {
        h.c.add("verdict_mismatches", 1);
    }
    if deliveries != 1 {
        h.c.add("duplicate_activation_deliveries", 1);
    }
    Ok(())
}

/// Abandonment: explicit, by expiry, replays inside the window, and a frame after it.
pub fn abandonment(h: &mut H<'_>) -> Res<()> {
    let rows = params::PROBE_FRAME_ROWS as usize;
    let size = params::PROBE_CHUNK_BYTES as usize;
    let rig = h.rig;
    let principal = h.p.a.id.clone();
    let mut sent_ids: Vec<(u64, u64, u32)> = Vec::new();
    let mut released_ids: Vec<(u64, u64, u32)> = Vec::new();

    // Explicit abandonment with chunks buffered.
    h.fresh(400)?;
    if h.net.assembler.buffered_bytes() as u64 != params::BUFFERED_BYTES_BASELINE {
        return Err("the receiver starts with buffered bytes".to_string());
    }
    let (id, chunks) = h.cut(&probe_frame(rig, rows, 31), size, 4_000, 7)?;
    for chunk in &chunks[..3] {
        h.chunk(chunk)?;
        sent_ids.push((400, id, chunk.header.chunk_index));
    }
    let now = h.net.now_ms;
    let released = h
        .net
        .assembler
        .abandon(now, &principal, 400, id)
        .map_err(|error| format!("abandoning: {error:?}"))?;
    released_ids.extend(
        released
            .chunks
            .iter()
            .map(|chunk| (chunk.parent_operation_id, chunk.frame_id, chunk.chunk_index)),
    );
    h.c.add("abandoned_frame_cases", 1);
    if h.net.assembler.buffered_bytes() != 0 {
        h.c.add("buffered_bytes_after_release", 1);
    }
    for chunk in &chunks[3..] {
        let sent = h.chunk(chunk)?;
        h.c.add("late_chunk_after_abandon_attempts", 1);
        if sent.admitted() {
            h.c.add("late_chunks_after_abandon_admitted", 1);
        }
        h.attempt(&sent, "frame_abandoned");
    }
    let sent = h.chunk(&chunks[0])?;
    h.c.add("post_abandonment_replay_cases", 1);
    if sent.admitted() {
        h.c.add("abandoned_frame_replays_admitted", 1);
    }
    h.attempt(&sent, "frame_abandoned");
    let delay = params::LATE_WINDOW_REPLAY_DELAY_RANGE_MS[0]
        + h.rng.below(
            params::LATE_WINDOW_REPLAY_DELAY_RANGE_MS[1]
                - params::LATE_WINDOW_REPLAY_DELAY_RANGE_MS[0],
        );
    h.net.now_ms += delay;
    let sent = h.chunk(&chunks[1])?;
    h.c.add("post_abandonment_replay_cases", 1);
    if sent.admitted() {
        h.c.add("abandoned_frame_replays_admitted", 1);
    }
    h.attempt(&sent, "frame_abandoned");
    // The frame that follows reassembles: the abandoned one released its chunks first.
    let frame = probe_frame(rig, rows, 32);
    let (_, chunks) = h.cut(&frame, size, params::FRAME_BUDGET_MS, 7)?;
    h.c.add("valid_frames_sent", 1);
    let mut delivered = None;
    let mut deliveries = 0;
    for chunk in &chunks {
        if let Sent::Exchange(exchange) = h.chunk(chunk)? {
            if let Some(Ok(Offered::Delivered(done))) = exchange.verdict {
                deliveries += 1;
                delivered = Some(*done);
            }
        }
    }
    match delivered {
        Some(done) if deliveries == 1 => h.check_delivery(&frame, &chunks, &done),
        _ => {
            h.c.add("verdict_mismatches", 1);
            h.c.add("bitwise_mismatch_count", 1);
        }
    }

    // Abandonment by expiry: the deadline passes with chunks buffered.
    h.fresh(401)?;
    let (id, chunks) = h.cut(
        &probe_frame(rig, rows, 33),
        size,
        params::FRAME_BUDGET_MS,
        7,
    )?;
    for chunk in &chunks[..2] {
        h.chunk(chunk)?;
        sent_ids.push((401, id, chunk.header.chunk_index));
    }
    h.net.now_ms += params::FRAME_BUDGET_MS + params::MAX_CLOCK_SKEW_MS + 1;
    let now = h.net.now_ms;
    let released = h.net.assembler.expire(now);
    released_ids.extend(
        released
            .iter()
            .map(|chunk| (chunk.parent_operation_id, chunk.frame_id, chunk.chunk_index)),
    );
    h.c.add("abandoned_frame_cases", 1);
    if h.net.assembler.buffered_bytes() != 0 {
        h.c.add("buffered_bytes_after_release", 1);
    }
    let sent = h.chunk(&chunks[2])?;
    h.c.add("late_chunk_after_abandon_attempts", 1);
    if sent.admitted() {
        h.c.add("late_chunks_after_abandon_admitted", 1);
    }
    h.attempt(&sent, "frame_abandoned");
    sent_ids.sort_unstable();
    released_ids.sort_unstable();
    h.c.add("abandoned_chunks_sent", sent_ids.len() as i64);
    h.c.add("abandoned_chunks_released", released_ids.len() as i64);
    if sent_ids != released_ids {
        h.c.add("abandon_release_mismatches", 1);
    }
    Ok(())
}

/// The table of tombstones fills, refuses, and frees a slot as the oldest expires.
pub fn saturation(h: &mut H<'_>) -> Res<()> {
    let rows = params::PROBE_FRAME_ROWS as usize;
    let size = params::PROBE_CHUNK_BYTES as usize;
    let rig = h.rig;
    let principal = h.p.a.id.clone();
    let max = params::MAX_FRAME_TOMBSTONES_PER_PRINCIPAL;
    h.fresh(500)?;
    let start = h.net.now_ms;
    let mut abandoned = Vec::new();
    for index in 0..params::TOMBSTONE_OVERCAPACITY_FRAMES {
        h.net.now_ms = start + index * params::TOMBSTONE_STAGGER_MS;
        let (id, chunks) = h.cut(&probe_frame(rig, rows, 41), size, 4_000, 7)?;
        let sent = h.chunk(&chunks[0])?;
        if index < max {
            if !sent.admitted() {
                return Err(format!(
                    "frame {index} was refused before the table was full"
                ));
            }
            let now = h.net.now_ms;
            h.net
                .assembler
                .abandon(now, &principal, 500, id)
                .map_err(|error| format!("abandoning: {error:?}"))?;
            abandoned.push((chunks[0].clone(), now));
        } else {
            h.c.add("overcapacity_frame_attempts", 1);
            h.attempt(&sent, "frame_table_full");
        }
        if h.net.assembler.reserved_slots(&principal) as u64 > max {
            h.c.add("tombstone_bound_violations", 1);
        }
    }
    h.c.add("tombstone_saturation_cases", 1);
    // Nothing is evicted early: every abandoned frame still refuses a replay until the first
    // one's retention ends.
    let first_expiry = abandoned
        .first()
        .ok_or("no frame was abandoned in the saturation case")?
        .1
        + params::REPLAY_WINDOW_MS;
    h.net.now_ms = first_expiry - 1;
    for (chunk, _) in &abandoned {
        let sent = h.chunk(chunk)?;
        if sent.rejection().map(Rejection::kind) != Some("frame_abandoned") {
            h.c.add("tombstone_evictions_before_retention", 1);
        }
    }
    if h.net.assembler.retained_tombstones(&principal) as u64 != max {
        h.c.add("tombstone_evictions_before_retention", 1);
    }
    // One tick later the first tombstone is gone: exactly one new frame fits.
    h.net.now_ms = first_expiry;
    let (_, fresh) = h.cut(&probe_frame(rig, rows, 42), size, 4_000, 7)?;
    let sent = h.chunk(&fresh[0])?;
    if sent.admitted() {
        h.c.add("frames_accepted_after_tombstone_expiry", 1);
    }
    let (_, again) = h.cut(&probe_frame(rig, rows, 43), size, 4_000, 7)?;
    let sent = h.chunk(&again[0])?;
    h.c.add("overcapacity_frame_attempts", 1);
    h.attempt(&sent, "frame_table_full");
    if h.net.assembler.reserved_slots(&principal) as u64 > max {
        h.c.add("tombstone_bound_violations", 1);
    }

    // The slots are shared by incomplete frames and tombstones: with 60 tombstones held only
    // four frames may be incomplete at once, and the fifth is refused.
    h.fresh(510)?;
    for _ in 0..(max - 4) {
        let (id, chunks) = h.cut(&probe_frame(rig, rows, 44), size, 4_000, 7)?;
        h.chunk(&chunks[0])?;
        let now = h.net.now_ms;
        h.net
            .assembler
            .abandon(now, &principal, 510, id)
            .map_err(|error| format!("abandoning: {error:?}"))?;
    }
    for _ in 0..4 {
        let (_, chunks) = h.cut(&probe_frame(rig, rows, 45), size, 4_000, 7)?;
        if !h.chunk(&chunks[0])?.admitted() {
            return Err("a frame inside the bound was refused".to_string());
        }
    }
    let (_, chunks) = h.cut(&probe_frame(rig, rows, 46), size, 4_000, 7)?;
    let sent = h.chunk(&chunks[0])?;
    h.c.add("overcapacity_frame_attempts", 1);
    h.attempt(&sent, "frame_table_full");
    if h.net.assembler.reserved_slots(&principal) as u64 > max {
        h.c.add("tombstone_bound_violations", 1);
    }
    Ok(())
}

/// Two frames with one upstream chain reassembled at once, chunks interleaved.
pub fn interleaving(h: &mut H<'_>) -> Res<()> {
    let rows = params::PROBE_FRAME_ROWS as usize;
    let size = params::PROBE_CHUNK_BYTES as usize;
    let rig = h.rig;
    h.fresh(600)?;
    let frame_a = probe_frame(rig, rows, 51);
    let frame_b = probe_frame(rig, rows, 52);
    let (id_a, chunks_a) = h.cut(&frame_a, size, params::FRAME_BUDGET_MS, 7)?;
    let (id_b, chunks_b) = h.cut(&frame_b, size, params::FRAME_BUDGET_MS, 7)?;
    h.c.add("valid_frames_sent", 2);
    let mut order: Vec<(bool, usize)> = Vec::new();
    for index in 0..chunks_a.len().max(chunks_b.len()) {
        if index < chunks_a.len() {
            order.push((true, index));
        }
        if index < chunks_b.len() {
            order.push((false, index));
        }
    }
    let mut delivered: Vec<(u64, DeliveredFrame)> = Vec::new();
    let mut previous = None;
    for (is_a, index) in order {
        if previous.is_some_and(|last| last != is_a) {
            h.c.add("interleaved_chunk_switches", 1);
        }
        previous = Some(is_a);
        let chunk = if is_a {
            &chunks_a[index]
        } else {
            &chunks_b[index]
        };
        if let Sent::Exchange(exchange) = h.chunk(chunk)? {
            if let Some(Ok(Offered::Delivered(done))) = exchange.verdict {
                delivered.push((done.frame_id, *done));
            }
        }
    }
    h.c.add("interleaved_frame_pairs", 1);
    if delivered.len() != 2 {
        h.c.add("cross_frame_mixing_count", 1);
    }
    for (id, done) in &delivered {
        let (frame, chunks) = if *id == id_a {
            (&frame_a, &chunks_a)
        } else if *id == id_b {
            (&frame_b, &chunks_b)
        } else {
            h.c.add("cross_frame_mixing_count", 1);
            continue;
        };
        h.check_delivery(frame, chunks, done);
    }
    Ok(())
}

/// The allowlist on the real endpoint, and the payloads that try to move a KV page across it.
pub fn allowlist(h: &mut H<'_>) -> Res<()> {
    let rig = h.rig;
    let p = h.p;
    let pods = crate::experiments::e007::smuggle::pods(rig)?;
    let page = pods
        .inventory
        .pages()
        .first()
        .cloned()
        .ok_or("the pods hold no KV page")?;
    let other_page = pods
        .inventory
        .pages()
        .get(5)
        .cloned()
        .ok_or("the pods hold too few KV pages")?;
    h.fresh(7_000)?;

    // A valid, signed chunk of a valid frame: its bytes would be taken as a chunk by an
    // admission that looked at the bytes and not at the type.
    let tiny = ActivationFrame {
        tokens: 1,
        width: params::HIDDEN_SIZE as u32,
        dtype: BoundaryDtype::F32,
        positions: vec![0],
        upstream_chain: rig.chain(),
        placement_epoch: rig.epoch(),
        tensor: vec![0; params::HIDDEN_SIZE as usize * 4],
    };
    let (_, tiny_chunks) = h.cut(&tiny, 4096, params::FRAME_BUDGET_MS, 1)?;
    let signed_chunk_bytes = encode_chunk(&tiny_chunks[0]);

    // 1. A payload under a type id that is not on the allowlist, from a peer whose scope is the
    // allowlist: the host's access decision refuses it before the Pod runs. The same payloads
    // from a peer with a deliberately wide scope reach the Pod (the control that shows the scope
    // is what held them) and then meet the assembler's own allowlist.
    let probes: [(&str, &[u8]); 4] = [
        ("ptr.kv.page.v1", other_page.as_slice()),
        ("ptr.kv.snapshot.v1", other_page.as_slice()),
        ("ptr.split.activation-chunk.v2", other_page.as_slice()),
        ("ptr.tensor.blob.v1", signed_chunk_bytes.as_slice()),
    ];
    for (type_id, bytes) in probes {
        let payload = TypedPayload {
            type_id: TypeId::from(type_id),
            bytes: bytes.to_vec(),
        };
        // The strict peer.
        let sent = h.send_payload(&p.a, &payload)?;
        h.c.add("non_allowlisted_host_attempts", 1);
        if sent.reached_pod() || sent.admitted() {
            h.c.add("non_allowlisted_admitted", 1);
        }
        if sent.host_refusal() != Some(RefusalCode::Unavailable) {
            h.c.add("verdict_mismatches", 1);
        }
        // The wide-scope peer.
        let sent = h.send_payload(&p.lax, &payload)?;
        h.c.add("non_allowlisted_assembler_attempts", 1);
        if sent.reached_pod() {
            h.c.add("control_trips.wide_scope_reaches_pod", 1);
        }
        if sent.admitted() {
            h.c.add("non_allowlisted_admitted", 1);
        }
        h.attempt(&sent, "not_allowlisted");
    }

    // 2. A KV page mislabelled as an allowlisted type.
    for type_id in [ACTIVATION_CHUNK_TYPE, SPLIT_POLICY_TYPE] {
        let payload = TypedPayload {
            type_id: TypeId::from(type_id),
            bytes: page.clone(),
        };
        let sent = h.send_payload(&p.a, &payload)?;
        h.c.add("mislabelled_kv_payload_attempts", 1);
        if sent.admitted() {
            h.c.add("non_allowlisted_admitted", 1);
        }
        if !matches!(sent.rejection(), Some(Rejection::Malformed(_))) {
            h.c.add("verdict_mismatches", 1);
        }
        h.c.add("rejection_cases", 1);
    }

    // 3. KV-sized payloads inside a correctly typed, signed, digest-consistent frame.
    let chain = rig.chain();
    let valid = ActivationFrame {
        tokens: 4,
        width: params::HIDDEN_SIZE as u32,
        dtype: BoundaryDtype::F32,
        positions: vec![0, 1, 2, 3],
        upstream_chain: chain.clone(),
        placement_epoch: rig.epoch(),
        tensor: vec![0; 4 * params::HIDDEN_SIZE as usize * 4],
    };
    let mut trailing = valid.to_bytes();
    trailing.extend_from_slice(&page);
    let mut shaped = valid.clone();
    shaped.tensor = page.clone();
    let mut positions = valid.clone();
    let rows_bytes = 4 * params::HIDDEN_SIZE as usize * 4;
    if page.len() < rows_bytes {
        return Err(format!(
            "a KV page of {} bytes is shorter than {rows_bytes}",
            page.len()
        ));
    }
    positions.tensor = page[..rows_bytes].to_vec();
    positions.positions.pop();
    let mut positions_long = positions.clone();
    positions_long.positions = vec![0, 1, 2, 3, 4];
    let cases: [(Vec<u8>, &str); 4] = [
        (trailing, "trailing"),
        (shaped.to_bytes(), "shape"),
        (positions.to_bytes(), "positions"),
        (positions_long.to_bytes(), "positions"),
    ];
    for (bytes, expected) in cases {
        let chunks = h.raw_chunks(&bytes, &chain, 1024);
        let mut last = None;
        for chunk in &chunks {
            last = Some(h.chunk(chunk)?);
        }
        let last = last.ok_or("no chunks")?;
        h.c.add("kv_sized_allowlisted_attempts", 1);
        if last.admitted() {
            h.c.add("non_allowlisted_admitted", 1);
        }
        let ok = matches!(
            (expected, last.rejection()),
            (
                "trailing",
                Some(Rejection::InvalidFrame(SchemaError::TrailingBytes))
            ) | (
                "shape",
                Some(Rejection::InvalidFrame(SchemaError::ShapeMismatch { .. }))
            ) | (
                "positions",
                Some(Rejection::InvalidFrame(SchemaError::PositionsLength { .. }))
            )
        );
        if !ok {
            h.c.add("verdict_mismatches", 1);
        }
        h.c.add("rejection_cases", 1);
        h.net.now_ms += params::FRAME_SPACING_MS;
    }
    Ok(())
}

/// The chunk signature is bound to the principal the connection authenticated.
///
/// The assembler takes every payload under the peer id the host's `serve_once` reports for the
/// connection. A chunk that is valid under the upstream runtime's key is therefore admitted from
/// that runtime's endpoint and from no other: not from an endpoint the host never admitted, not
/// from one it admitted but the assembler does not know, not from a registered runtime of another
/// key or another stage. The control offers the same captured chunk under a principal taken from
/// anywhere but the connection and it is admitted, which is what the probes would show if the
/// binding were the harness's.
pub fn principals(h: &mut H<'_>) -> Res<()> {
    let rig = h.rig;
    let p = h.p;
    let rows = params::PROBE_FRAME_ROWS as usize;
    let size = params::PROBE_CHUNK_BYTES as usize;
    h.fresh(8_000)?;
    let (_, chunks) = h.cut(
        &probe_frame(rig, rows, 61),
        size,
        params::FRAME_BUDGET_MS,
        7,
    )?;

    // 1. A peer the host never admitted sends a chunk the upstream runtime signed: refused by
    // the host's access decision, the Pod never runs.
    let sent = h.chunk_from(&p.stranger, &chunks[0])?;
    h.c.add("principal_probe_attempts", 1);
    h.c.add("principal_attempts.unadmitted_peer", 1);
    if sent.reached_pod() || sent.admitted() {
        h.c.add("unauthenticated_principal_admitted", 1);
    }
    if sent.host_refusal() != Some(RefusalCode::NotAdmitted) {
        h.c.add("verdict_mismatches", 1);
    }
    h.c.add("rejection_cases", 1);

    // 2. A peer the host admitted but the assembler has no key for.
    let sent = h.chunk_from(&p.plain, &chunks[0])?;
    h.c.add("principal_probe_attempts", 1);
    h.c.add("principal_attempts.unregistered_principal", 1);
    if sent.admitted() {
        h.c.add("unauthenticated_principal_admitted", 1);
    }
    h.attempt(&sent, "unknown_principal");

    // 3. A registered runtime of another key sends A's captured chunk over its own connection.
    let sent = h.chunk_from(&p.captor, &chunks[0])?;
    h.c.add("principal_probe_attempts", 1);
    h.c.add("principal_attempts.captured_chunk_on_other_connection", 1);
    if sent.admitted() {
        h.c.add("unauthenticated_principal_admitted", 1);
    }
    h.attempt(&sent, "bad_signature");

    // 4. A's own endpoint carries a chunk signed by a key that is not A's.
    let forged = sign_chunk(
        &other_key(9),
        chunks[0].header.clone(),
        chunks[0].bytes.clone(),
    );
    let sent = h.chunk_from(&p.a, &forged)?;
    h.c.add("principal_probe_attempts", 1);
    h.c.add("principal_attempts.foreign_key_on_bound_connection", 1);
    if sent.admitted() {
        h.c.add("unauthenticated_principal_admitted", 1);
    }
    h.attempt(&sent, "bad_signature");

    // 5. A registered runtime signs with its own key a chunk whose chain speaks for A's stage.
    let own = sign_chunk(
        &p.captor.key,
        chunks[0].header.clone(),
        chunks[0].bytes.clone(),
    );
    let sent = h.chunk_from(&p.captor, &own)?;
    h.c.add("principal_probe_attempts", 1);
    h.c.add("principal_attempts.stage_of_another_principal", 1);
    if sent.admitted() {
        h.c.add("unauthenticated_principal_admitted", 1);
    }
    h.attempt(&sent, "stage_not_bound");

    // The positive control: the same chunk from A's endpoint is taken.
    let sent = h.chunk(&chunks[0])?;
    if sent.admitted() {
        h.c.add("principal_positive_admitted", 1);
    }

    // The negative control: the captured chunk on the captor's connection, offered under a
    // principal that did not come from the connection. It is admitted.
    let p_captor = &p.captor;
    h.c.add("chunks_sent", 1);
    let payload = chunk_payload(&chunks[1]);
    let claimed = p.a.id.clone();
    let sent = h.net.send_as(h.c, p_captor, Some(&claimed), &payload)?;
    if sent.admitted() {
        h.c.add("control_trips.claimed_principal_admits_captured_chunk", 1);
    }
    Ok(())
}
