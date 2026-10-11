//! C1: the split model against the single-pod run, bit for bit, and C1b: the downstream
//! cache's reuse against a recomputation.
//!
//! Output means the final logits bytes (every prompt row and every decode step) plus the
//! per-layer KV bytes on each pod. Three runs per sequence: the whole stack on one pod (the
//! baseline), the same two stages in process with the boundary cast (a no-op in F32), and the
//! two stages with every activation frame crossing the PodWire codec as signed chunks.

use super::link::Link;
use super::params;
use super::rig::{bits, bytes_f32, f32_bytes, Rig, Stage};
use super::wire::Role;
use crate::experiments::layered_kv::corpus;
use crate::experiments::layered_kv::emit::Tally;
use crate::experiments::layered_kv::lab::{fail, id_set_digest, Res};
use ptr_pods::KvTensorSnapshot;
use ptr_runtime::{admit_frame, SplitStageCache};

/// What a run produced.
#[derive(Debug, PartialEq)]
pub struct Output {
    pub logits: Vec<Vec<u32>>,
    pub generated: Vec<u32>,
    /// Per layer of the whole stack, keys then values.
    pub kv: Vec<Vec<u8>>,
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

/// The prompt of sequence `index`: a pure function of the corpus seed.
pub fn prompt(index: u64) -> Vec<u32> {
    let span = params::PROMPT_TOKENS_MAX - params::PROMPT_TOKENS_MIN + 1;
    let len = params::PROMPT_TOKENS_MIN + (index * 7) % span;
    corpus::sequence(super::super::layered_kv::SEQUENCE_SEED, index, len as usize)
}

/// The whole stack on one pod.
pub fn whole(rig: &Rig, tokens: &[u32], steps: usize) -> Res<Output> {
    let stack = &rig.stack;
    let layers = params::TEST_MODEL_LAYERS as usize;
    let mut pod = Stage::new(stack, 0..layers, "single-pod")?;
    let regime = stack
        .admit_rope_regime(tokens.len(), steps)
        .map_err(fail("the regime"))?;
    let out = stack
        .prefill_in_regime(
            &pod.backend,
            &mut pod.cache,
            ptr_pods::Qwen2StackInput {
                tokens: tokens.to_vec(),
            },
            &regime,
        )
        .map_err(fail("the single-pod prefill"))?;
    let mut logits: Vec<Vec<u32>> = out.logits.iter().map(|row| bits(row)).collect();
    let mut next = argmax(out.logits.last().ok_or("no logits")?);
    let mut generated = vec![next];
    for _ in 1..steps {
        let out = stack
            .decode_in_regime(
                &pod.backend,
                &mut pod.cache,
                ptr_pods::Qwen2StackInput { tokens: vec![next] },
                &regime,
            )
            .map_err(fail("a single-pod decode"))?;
        logits.push(bits(&out.logits[0]));
        next = argmax(&out.logits[0]);
        generated.push(next);
    }
    Ok(Output {
        logits,
        generated,
        kv: pod.kv_bytes()?,
    })
}

/// How the residual rows get from stage A to stage B.
pub enum Transport<'a> {
    /// In process, through the boundary cast (a no-op in F32).
    Cast,
    /// As signed chunks over the PodWire codec, admitted against the live chain.
    Wire(&'a mut Link),
}

/// What the split run leaves for the counters and for the reuse case.
pub struct SplitRun {
    pub output: Output,
    pub frames: usize,
    pub frame_ids: Vec<u64>,
    pub consistent: usize,
    pub consistent_ids: Vec<u64>,
    pub bytes_mismatches: usize,
    /// Stage B's snapshot right after the prompt, the frames of the prompt, and the stage A
    /// residual rows of each decode step with the expected logits of that step.
    pub prefill_snapshot: Option<KvTensorSnapshot>,
    pub prefill_frame: Option<ptr_runtime::AdmittedFrame>,
    pub decode_inputs: Vec<(Vec<f32>, Vec<u32>)>,
    /// The identifiers of the two caches, for the scan of the wire for KV handles.
    pub handles: Vec<Vec<u8>>,
    /// The bytes of each valid frame as the receiver reassembled it: a page that straddles two
    /// chunks of 2048 B is whole only here, so the scan reads these as well as the chunks.
    pub reassembled: Vec<Vec<u8>>,
}

fn carry(
    rig: &Rig,
    transport: &mut Transport<'_>,
    residual: &[f32],
    rows: usize,
    first_position: usize,
    run: &mut SplitRun,
) -> Res<(Vec<f32>, Option<ptr_runtime::AdmittedFrame>)> {
    match transport {
        Transport::Cast => Ok((bytes_f32(&f32_bytes(residual)), None)),
        Transport::Wire(link) => {
            let frame = rig.frame(residual, rows, first_position);
            // A valid frame that does not arrive is a product defect, counted and not fatal.
            let shipped = match link.ship(rig, &frame, Role::Valid) {
                Ok(shipped) => shipped,
                Err(_) => {
                    run.frames += 1;
                    run.bytes_mismatches += 1;
                    return Ok((residual.to_vec(), None));
                }
            };
            run.frames += 1;
            run.reassembled.push(shipped.delivered.bytes.clone());
            run.frame_ids.push(shipped.frame_id);
            run.consistent += usize::from(shipped.consistent_digest);
            if shipped.consistent_digest {
                run.consistent_ids.push(shipped.frame_id);
            }
            if shipped.delivered.bytes != frame.to_bytes() || shipped.delivered.frame != frame {
                run.bytes_mismatches += 1;
            }
            let admitted = admit_frame(
                &shipped.delivered.frame,
                ptr_model_api::KvReuseMode::Exact,
                &rig.live(),
            )
            .map_err(|fault| format!("a live frame was not admitted: {fault:?}"))?;
            Ok((admitted.activations().to_vec(), Some(admitted)))
        }
    }
}

/// The two stages, A holding `0..split` and B `split..L`.
pub fn split(
    rig: &Rig,
    mut transport: Transport<'_>,
    tokens: &[u32],
    steps: usize,
) -> Res<SplitRun> {
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
    let mut run = SplitRun {
        output: Output {
            logits: Vec::new(),
            generated: Vec::new(),
            kv: Vec::new(),
        },
        frames: 0,
        frame_ids: Vec::new(),
        consistent: 0,
        consistent_ids: Vec::new(),
        bytes_mismatches: 0,
        prefill_snapshot: None,
        prefill_frame: None,
        decode_inputs: Vec::new(),
        handles: Vec::new(),
        reassembled: Vec::new(),
    };
    let embedded = stack.embed_tokens(tokens).map_err(fail("the embedding"))?;
    let mid = stage_a.run(stack, &embedded, tokens.len(), false, &regime)?;
    let (arrived, admitted) = carry(
        rig,
        &mut transport,
        &mid.residual,
        mid.rows,
        mid.first_position,
        &mut run,
    )?;
    let end = stage_b.run(stack, &arrived, mid.rows, false, &regime)?;
    note_handles(&mut run, &stage_a, &stage_b)?;
    run.prefill_snapshot = Some(stage_b.snapshot()?);
    run.prefill_frame = admitted;
    let rows = stack
        .logits_from_residual(&end.residual)
        .map_err(fail("the logits"))?;
    run.output.logits = rows.iter().map(|row| bits(row)).collect();
    let mut next = argmax(rows.last().ok_or("no logits")?);
    run.output.generated.push(next);
    for _ in 1..steps {
        let embedded = stack.embed_tokens(&[next]).map_err(fail("the embedding"))?;
        let mid = stage_a.run(stack, &embedded, 1, true, &regime)?;
        let (arrived, _) = carry(
            rig,
            &mut transport,
            &mid.residual,
            1,
            mid.first_position,
            &mut run,
        )?;
        let end = stage_b.run(stack, &arrived, 1, true, &regime)?;
        note_handles(&mut run, &stage_a, &stage_b)?;
        let row = stack
            .logits_from_residual(&end.residual)
            .map_err(fail("the logits"))?;
        run.output.logits.push(bits(&row[0]));
        run.decode_inputs.push((arrived, bits(&row[0])));
        next = argmax(&row[0]);
        run.output.generated.push(next);
    }
    let mut kv = stage_a.kv_bytes()?;
    kv.extend(stage_b.kv_bytes()?);
    run.output.kv = kv;
    run.handles.push(b"kv-state-stage-a".to_vec());
    run.handles.push(b"kv-state-stage-b".to_vec());
    Ok(run)
}

/// Record the identifiers of both caches as they stand after a step: a stage that shipped a
/// cache handle in a frame would ship the one it holds at that moment, not only the last.
fn note_handles(run: &mut SplitRun, stage_a: &Stage, stage_b: &Stage) -> Res<()> {
    run.handles.push(stage_a.snapshot()?.digest.to_vec());
    run.handles.push(stage_b.snapshot()?.digest.to_vec());
    Ok(())
}

/// The sets and counts of the valid frames of the bit-identity cases.
#[derive(Default)]
pub struct Frames {
    pub ids: Vec<String>,
    pub consistent_ids: Vec<String>,
}

/// Run every sequence of the corpus three ways and count.
pub fn run(
    tally: &mut Tally,
    rig: &Rig,
    link: &mut Link,
    sequences: u64,
    frames: &mut Frames,
) -> Res<()> {
    let steps = params::DECODE_STEPS as usize;
    for index in 0..sequences {
        let tokens = prompt(index);
        let reference = whole(rig, &tokens, steps)?;
        let cast = split(rig, Transport::Cast, &tokens, steps)?;
        // A fresh operation per sequence: frame ids and the parent operation are part of the
        // reassembly identity.
        link.operation = 1_000 + index;
        let start = link.wire.records.len();
        let wired = split(rig, Transport::Wire(link), &tokens, steps)?;
        let split_at = params::SPLIT_LAYER as usize;
        let row_bytes = (params::NUM_KV_HEADS * params::HEAD_DIM * 4) as usize;
        let inventory = super::wire::KvInventory::new(
            &[
                wired.output.kv[..split_at].to_vec(),
                wired.output.kv[split_at..].to_vec(),
            ],
            row_bytes,
            wired.handles.clone(),
        );
        let (mut page_bytes, mut handle_count) =
            super::wire::scan(&link.wire.records[start..], &inventory);
        let (frame_bytes, frame_handles) = super::wire::scan_frames(&wired.reassembled, &inventory);
        page_bytes += frame_bytes;
        handle_count += frame_handles;
        tally.add("wire_kv_page_bytes", page_bytes);
        tally.add("wire_kv_handle_count", handle_count);
        for observed in [&cast.output, &wired.output] {
            if *observed != reference {
                tally.add("bitwise_mismatch_count", 1);
            }
        }
        tally.add("activation_frames_crossed", wired.frames as i64);
        tally.add("valid_frames_sent", wired.frames as i64);
        tally.add(
            "frames_with_consistent_whole_frame_digest",
            wired.consistent as i64,
        );
        tally.add(
            "valid_frame_bitwise_mismatches",
            wired.bytes_mismatches as i64,
        );
        for id in &wired.frame_ids {
            frames.ids.push(format!("{}:{id}", link.operation));
        }
        for id in &wired.consistent_ids {
            frames
                .consistent_ids
                .push(format!("{}:{id}", link.operation));
        }
        let _ = (
            &wired.prefill_snapshot,
            &wired.prefill_frame,
            &wired.decode_inputs,
        );
    }
    Ok(())
}

/// C1b: the downstream entry derived from the prompt frame is reused to continue the decode,
/// and the continuation equals the recomputation bit for bit.
pub fn reuse(tally: &mut Tally, rig: &Rig, link: &mut Link, index: u64) -> Res<()> {
    let steps = params::DECODE_STEPS as usize;
    let tokens = prompt(index);
    link.operation = 5_000 + index;
    let original = split(rig, Transport::Wire(link), &tokens, steps)?;
    let snapshot = original
        .prefill_snapshot
        .clone()
        .ok_or("no prefill snapshot")?;
    let frame = original.prefill_frame.clone().ok_or("no prefill frame")?;
    let mut cache: SplitStageCache<KvTensorSnapshot> = SplitStageCache::new(4);
    let key = cache
        .store(&[&frame], &rig.own_chain(), snapshot, &rig.live())
        .map_err(|e| format!("store: {e:?}"))?;
    let live = rig.live();
    let ticket = cache
        .begin_use(&key, &live)
        .map_err(|e| format!("begin_use: {e:?}"))?;
    let hit = cache
        .lookup(&key, &live)
        .map_err(|e| format!("lookup: {e:?}"))?
        .clone();
    let layers = params::TEST_MODEL_LAYERS as usize;
    let mut stage_b = Stage::restored(&rig.stack, params::SPLIT_LAYER as usize..layers, hit)?;
    let regime = rig
        .stack
        .admit_rope_regime(tokens.len(), steps)
        .map_err(fail("the regime"))?;
    for (arrived, expected) in &original.decode_inputs {
        let end = stage_b.run(&rig.stack, arrived, 1, true, &regime)?;
        let row = rig
            .stack
            .logits_from_residual(&end.residual)
            .map_err(fail("the logits"))?;
        if bits(&row[0]) != *expected {
            tally.add("bitwise_mismatch_count", 1);
        }
    }
    cache
        .finish_use(ticket, &live)
        .map_err(|e| format!("finish_use: {e:?}"))?;
    tally.add("downstream_reuse_cases", 1);
    Ok(())
}

/// The identity-set digest of a list of names.
pub fn digest(names: &[String]) -> String {
    id_set_digest(names.iter().map(String::as_str))
}
