//! C3: the allowlist, the smuggling probes and the activation control.
//!
//! Two pods hold real KV (prompt prefilled through the split). The probes try to move a KV
//! page across PodWire three ways: mislabelled as an allowlisted type, under a type id that is
//! not allowlisted, and inside an allowlisted frame (KV-sized, with a wrong shape, trailing
//! bytes or a wrong positions length). The control is an activation-shaped KV payload that is
//! valid, is admitted and is consumed as activation only. Whether a page crossed is measured
//! by the independent scan of the capture (`wire::scan`), and whether the control installed
//! one by counting the smuggled rows in the receiving cache before and after.

use super::link::Link;
use super::params;
use super::rig::{bytes_f32, f32_bytes, Rig, Stage};
use super::wire::{KvInventory, Role, Sent};
use crate::experiments::layered_kv::emit::Tally;
use crate::experiments::layered_kv::lab::{fail, Res};
use ptr_podwire::{Offered, Rejection};
use ptr_protocol::activation::{
    ActivationFrame, BoundaryDtype, SchemaError, ACTIVATION_CHUNK_TYPE, SPLIT_POLICY_TYPE,
};
use ptr_protocol::TypedPayload;
use ptr_runtime::admit_frame;
use ptr_types::TypeId;

/// Count how often `needle` occurs in `haystack`.
fn occurrences(haystack: &[u8], needle: &[u8]) -> usize {
    if needle.is_empty() || haystack.len() < needle.len() {
        return 0;
    }
    haystack
        .windows(needle.len())
        .filter(|window| *window == needle)
        .count()
}

/// How a probe must end, for the verdict table.
fn expect(tally: &mut Tally, kind: &str, sent: &Sent, check: impl Fn(&Rejection) -> bool) {
    match sent.rejection() {
        Some(rejection) => {
            if !check(rejection) {
                tally.add("verdict_mismatches", 1);
            }
        }
        None => tally.add("attempts_without_typed_rejection", 1),
    }
    if sent.admitted() {
        tally.add("smuggling_probes_admitted", 1);
        tally.add("non_allowlisted_payloads_admitted", 1);
    }
    let _ = kind;
}

/// The pods after a prompt, and the inventory of what they hold.
pub struct Pods {
    pub stage_a: Stage,
    pub stage_b: Stage,
    pub inventory: KvInventory,
}

pub fn pods(rig: &Rig) -> Res<Pods> {
    let tokens = crate::experiments::layered_kv::corpus::sequence(
        crate::experiments::layered_kv::SEQUENCE_SEED,
        9_000,
        48,
    );
    let layers = params::TEST_MODEL_LAYERS as usize;
    let split = params::SPLIT_LAYER as usize;
    let mut stage_a = Stage::new(&rig.stack, 0..split, "probe-stage-a")?;
    let mut stage_b = Stage::new(&rig.stack, split..layers, "probe-stage-b")?;
    let regime = rig
        .stack
        .admit_rope_regime(tokens.len() + 16, 0)
        .map_err(fail("the regime"))?;
    let embedded = rig.stack.embed_tokens(&tokens).map_err(fail("embed"))?;
    let mid = stage_a.run(&rig.stack, &embedded, tokens.len(), false, &regime)?;
    stage_b.run(&rig.stack, &mid.residual, mid.rows, false, &regime)?;
    let inventory = inventory_of(&stage_a, &stage_b)?;
    Ok(Pods {
        stage_a,
        stage_b,
        inventory,
    })
}

/// The inventory of two pods' KV: their pages and the handles of their caches.
pub fn inventory_of(a: &Stage, b: &Stage) -> Res<KvInventory> {
    let row_bytes = (params::NUM_KV_HEADS * params::HEAD_DIM * 4) as usize;
    let handles = vec![
        a.snapshot()?.digest.to_vec(),
        b.snapshot()?.digest.to_vec(),
        b"kv-state-stage-a".to_vec(),
        b"kv-state-stage-b".to_vec(),
    ];
    Ok(KvInventory::new(
        &[a.kv_bytes()?, b.kv_bytes()?],
        row_bytes,
        handles,
    ))
}

pub fn run(tally: &mut Tally, rig: &Rig, link: &mut Link, pods: &mut Pods) -> Res<()> {
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
    link.operation = 7_000;

    // 1. A KV page mislabelled as an allowlisted type.
    for type_id in [ACTIVATION_CHUNK_TYPE, SPLIT_POLICY_TYPE] {
        let payload = TypedPayload {
            type_id: TypeId::from(type_id),
            bytes: page.clone(),
        };
        let sent = link.wire.send(
            &mut link.assembler,
            link.now_ms,
            1,
            Role::Probe("mislabelled_kv_payload"),
            &payload,
        );
        tally.add("mislabelled_kv_payload_attempts", 1);
        tally.add("smuggling_probe_attempts.mislabelled_kv_payload", 1);
        expect(tally, "mislabelled_kv_payload", &sent, |rejection| {
            matches!(rejection, Rejection::Malformed(_))
        });
    }

    // 2. A KV page under a type id that is not on the allowlist.
    for type_id in [
        "ptr.kv.page.v1",
        "ptr.kv.snapshot.v1",
        "ptr.split.activation-chunk.v2",
    ] {
        let payload = TypedPayload {
            type_id: TypeId::from(type_id),
            bytes: other_page.clone(),
        };
        let sent = link.wire.send(
            &mut link.assembler,
            link.now_ms,
            1,
            Role::Probe("non_allowlisted_type"),
            &payload,
        );
        tally.add("non_allowlisted_type_attempts", 1);
        tally.add("smuggling_probe_attempts.non_allowlisted_type", 1);
        expect(tally, "non_allowlisted_type", &sent, |rejection| {
            matches!(rejection, Rejection::NotAllowlisted(_))
        });
    }

    // A valid, signed chunk of a valid frame under a type id that is not on the allowlist: the
    // bytes would be taken as a chunk by an admission that looked at the bytes and not the type.
    {
        let frame = ActivationFrame {
            tokens: 1,
            width: params::HIDDEN_SIZE as u32,
            dtype: BoundaryDtype::F32,
            positions: vec![0],
            upstream_chain: rig.chain(),
            placement_epoch: rig.epoch(),
            tensor: vec![0; params::HIDDEN_SIZE as usize * 4],
        };
        let (_, chunks) = link.chunks(rig, &frame, 4096, params::FRAME_BUDGET_MS)?;
        let payload = TypedPayload {
            type_id: TypeId::from("ptr.tensor.blob.v1"),
            bytes: ptr_protocol::activation::encode_chunk(&chunks[0]),
        };
        let sent = link.wire.send(
            &mut link.assembler,
            link.now_ms,
            1,
            Role::Probe("non_allowlisted_type"),
            &payload,
        );
        tally.add("non_allowlisted_type_attempts", 1);
        tally.add("smuggling_probe_attempts.non_allowlisted_type", 1);
        expect(tally, "non_allowlisted_type", &sent, |rejection| {
            matches!(rejection, Rejection::NotAllowlisted(_))
        });
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
    // (a) a valid frame with a KV page appended after it.
    let mut trailing = valid.to_bytes();
    trailing.extend_from_slice(&page);
    // (b) a KV page as the tensor, with a header that declares another shape.
    let mut shaped = valid.clone();
    shaped.tensor = page.clone();
    // (c) a tensor of the right shape for the header whose positions are one short.
    let mut positions = valid.clone();
    positions.tensor = page[..4 * params::HIDDEN_SIZE as usize * 4].to_vec();
    positions.positions.pop();
    // (d) the same with one position too many.
    let mut positions_long = positions.clone();
    positions_long.positions = vec![0, 1, 2, 3, 4];
    let cases: [(&str, Vec<u8>, &str); 4] = [
        ("kv_sized_allowlisted_smuggling", trailing, "trailing"),
        ("wrong_shape", shaped.to_bytes(), "shape"),
        ("positions_length", positions.to_bytes(), "positions"),
        ("positions_length", positions_long.to_bytes(), "positions"),
    ];
    for (kind, bytes, expected) in cases {
        let (_, chunks) = link.raw_chunks(rig, &bytes, &chain, 1024, params::FRAME_BUDGET_MS);
        let sent = link
            .send_all(&chunks, Role::Probe("kv_sized_allowlisted_smuggling"))
            .ok_or("no chunks")?;
        let ok = |rejection: &Rejection| {
            matches!(
                (expected, rejection),
                (
                    "trailing",
                    Rejection::InvalidFrame(SchemaError::TrailingBytes)
                ) | (
                    "shape",
                    Rejection::InvalidFrame(SchemaError::ShapeMismatch { .. })
                ) | (
                    "positions",
                    Rejection::InvalidFrame(SchemaError::PositionsLength { .. })
                )
            )
        };
        match kind {
            "kv_sized_allowlisted_smuggling" => {
                tally.add("kv_sized_allowlisted_smuggling_attempts", 1);
                tally.add("smuggling_probe_attempts.kv_sized_allowlisted_smuggling", 1);
            }
            "wrong_shape" => {
                tally.add("kv_sized_allowlisted_smuggling_attempts", 1);
                tally.add("smuggling_probe_attempts.kv_sized_allowlisted_smuggling", 1);
                tally.add("shape_validation_rejections", 1);
            }
            _ => tally.add("positions_length_rejection_attempts", 1),
        }
        let admitted = sent.admitted();
        expect(tally, kind, &sent, ok);
        if kind == "positions_length" && admitted {
            tally.add("positions_length_mismatches_admitted", 1);
        }
        link.now_ms += params::FRAME_SPACING_MS;
    }

    // 4. The control: an activation-shaped KV payload, valid, consumed only as activation.
    let rows = params::PAGE_TOKENS as usize;
    let snapshot = pods.stage_b.snapshot()?;
    let layer = snapshot.layers.first().ok_or("no layer")?;
    let kv_width = (params::NUM_KV_HEADS * params::HEAD_DIM) as usize;
    let mut activations = Vec::with_capacity(rows * params::HIDDEN_SIZE as usize);
    for row in 0..rows {
        activations.extend_from_slice(&layer.keys.values[row * kv_width..(row + 1) * kv_width]);
        activations.extend_from_slice(&layer.values.values[row * kv_width..(row + 1) * kv_width]);
    }
    let smuggled_rows: Vec<Vec<u8>> = activations.chunks(kv_width).map(f32_bytes).collect();
    let cache_bytes = |stage: &Stage| -> Res<Vec<u8>> { Ok(stage.kv_bytes()?.concat()) };
    let count_installed = |stage: &Stage| -> Res<usize> {
        let bytes = cache_bytes(stage)?;
        Ok(smuggled_rows
            .iter()
            .map(|row| occurrences(&bytes, row))
            .sum())
    };
    let before = count_installed(&pods.stage_b)?;
    let first_position = snapshot.sequence_length;
    let control = rig.frame(&activations, rows, first_position);
    let shipped = link.ship(rig, &control, Role::Control)?;
    tally.add("activation_control_cases", 1);
    if shipped.delivered.bytes != control.to_bytes() {
        tally.add("valid_frame_bitwise_mismatches", 1);
    }
    let admitted = admit_frame(
        &shipped.delivered.frame,
        ptr_model_api::KvReuseMode::Exact,
        &rig.live(),
    )
    .map_err(|fault| format!("the control was not admitted: {fault:?}"))?;
    let regime = rig
        .stack
        .admit_rope_regime(first_position + rows + 8, 0)
        .map_err(fail("the regime"))?;
    // Consumed as activation only: it is fed to compute, nothing else.
    pods.stage_b.run(
        &rig.stack,
        &bytes_f32(&f32_bytes(admitted.activations())),
        rows,
        false,
        &regime,
    )?;
    let after = count_installed(&pods.stage_b)?;
    let installed = after.saturating_sub(before) as i64;
    tally.add("activation_control_kv_installs", installed);
    tally.add("kv_pages_installed_from_wire", installed);
    let grown = pods.stage_b.snapshot()?.sequence_length - first_position;
    if grown != rows {
        return Err(format!(
            "the control grew the cache by {grown} rows, not {rows}"
        ));
    }
    let _ = Offered::Policy;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::occurrences;

    #[test]
    fn occurrences_counts_overlapping_windows_exactly() {
        assert_eq!(occurrences(b"abcabcab", b"abc"), 2);
        assert_eq!(occurrences(b"aaaa", b"aa"), 3);
        assert_eq!(occurrences(b"ab", b"abc"), 0);
        assert_eq!(occurrences(b"ab", b""), 0);
    }
}
