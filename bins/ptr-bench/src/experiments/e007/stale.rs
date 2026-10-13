//! C6: a downstream pod never serves KV derived from an invalidated upstream chain.
//!
//! The downstream cache holds an entry derived from a prompt frame. The upstream stage's
//! generation is then revoked or superseded in the real ledger, and the entry is used. Per
//! cell (invalidation type x path): `normal`, the `Revoke` message reaches the cache and an
//! ordinary lookup follows; `lag`, the message is dropped (never delivered) or delayed
//! (delivered after the lookup); `inflight`, a use begun before the invalidation finishes
//! after it, with the message dropped or delivered. A stale hit, or an output released after
//! the invalidation, trips the cell. A zombie upstream sending a frame of the old chain, and a
//! stage re-placed without a generation change, are the same defence from other sides.

use super::bitwise::{self, Transport};
use super::link::Link;
use super::params;
use super::rig::{bits, Rig, Stage};
use super::wire::Role;
use crate::experiments::layered_kv::emit::Tally;
use crate::experiments::layered_kv::lab::{fail, Res};
use ptr_pods::KvTensorSnapshot;
use ptr_runtime::{admit_frame, ChainFault, SplitStageCache, UseFault};
use ptr_types::Generation;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Message {
    /// The Revoke message reaches the cache before the use.
    Delivered,
    /// It is never delivered.
    Dropped,
    /// It is delivered after the use.
    Delayed,
}

fn invalidate(rig: &mut Rig, ty: &str) -> Res<()> {
    match ty {
        "revocation" => rig.revoke_stage(),
        _ => rig.supersede_stage_in_place(),
    }
}

/// A downstream entry derived from a prompt frame, and what continues it.
struct Entry {
    cache: SplitStageCache<KvTensorSnapshot>,
    key: [u8; 32],
    decode_input: (Vec<f32>, Vec<u32>),
    generation: u64,
    frame: ptr_runtime::AdmittedFrame,
}

fn entry(rig: &Rig, link: &mut Link, index: u64) -> Res<Entry> {
    let run = bitwise::split(rig, Transport::Wire(link), &bitwise::prompt(index), 2)?;
    let frame = run.prefill_frame.clone().ok_or("no prefill frame")?;
    let snapshot = run.prefill_snapshot.clone().ok_or("no snapshot")?;
    let mut cache = SplitStageCache::new(4);
    let key = cache
        .store(&[&frame], &rig.own_chain(), snapshot, &rig.live())
        .map_err(|error| format!("storing the entry: {error:?}"))?;
    Ok(Entry {
        cache,
        key,
        decode_input: run
            .decode_inputs
            .first()
            .cloned()
            .ok_or("no decode input")?,
        generation: rig.stage_generation(),
        frame,
    })
}

fn continue_with(
    rig: &Rig,
    snapshot: KvTensorSnapshot,
    entry: &Entry,
    steps_hint: usize,
) -> Res<Vec<u32>> {
    let layers = params::TEST_MODEL_LAYERS as usize;
    let mut stage_b = Stage::restored(&rig.stack, params::SPLIT_LAYER as usize..layers, snapshot)?;
    let regime = rig
        .stack
        .admit_rope_regime(entry.frame.rows() as usize + steps_hint, 0)
        .map_err(fail("the regime"))?;
    let end = stage_b.run(&rig.stack, &entry.decode_input.0, 1, true, &regime)?;
    let row = rig
        .stack
        .logits_from_residual(&end.residual)
        .map_err(fail("the logits"))?;
    Ok(bits(&row[0]))
}

pub fn run(tally: &mut Tally, seed: u64) -> Res<()> {
    let mut operation = 700;
    for ty in params::INVALIDATION_TYPES {
        for path in params::INVALIDATION_PATHS {
            let cell = format!("upstream_stage_generation_{path}_{ty}");
            let variants: &[Message] = match path {
                "normal" => &[Message::Delivered],
                "lag" => &[Message::Dropped, Message::Delayed],
                _ => &[Message::Dropped, Message::Delivered],
            };
            for variant in variants {
                operation += 1;
                let mut rig = Rig::new()?;
                let mut link = Link::new(&rig, seed, operation)?;
                let mut e = entry(&rig, &mut link, 1 + operation % 5)?;
                tally.add(
                    "stale_upstream_entries_before_revocation",
                    e.cache.len() as i64,
                );
                tally.add(&format!("stale_upstream_attempts.{ty}"), 1);
                tally.add(&format!("invalidation_matrix_cases.{cell}"), 1);
                // Before anything is invalidated the entry serves, and the continuation from
                // it is the recomputation's.
                {
                    let live = rig.live();
                    let snapshot = e
                        .cache
                        .lookup(&e.key, &live)
                        .map_err(|error| format!("a live entry did not serve: {error:?}"))?
                        .clone();
                    let served = continue_with(&rig, snapshot, &e, 2)?;
                    if served != e.decode_input.1 {
                        tally.add("bitwise_mismatch_count", 1);
                    }
                }
                let mut released = false;
                let mut served_stale = false;
                match path {
                    "inflight" => {
                        inflight_race(tally, &mut rig, &mut e, ty, *variant, &mut released)?;
                        tally.add(&format!("inflight_split_request_race_cases.{ty}"), 1);
                    }
                    _ => {
                        invalidate(&mut rig, ty)?;
                        if *variant == Message::Delivered {
                            e.cache
                                .invalidate_stage(params::STAGE_IDENTITY, Generation(e.generation));
                        }
                        if matches!(variant, Message::Dropped | Message::Delayed) {
                            tally.add("revoke_messages_dropped_or_delayed", 1);
                        }
                        let live = rig.live();
                        let ticket = e.cache.begin_use(&e.key, &live);
                        let lookup = e.cache.lookup(&e.key, &live).map(|_| ());
                        served_stale = ticket.is_ok() || lookup.is_ok();
                        if let Ok(ticket) = ticket {
                            let _ = e.cache.finish_use(ticket, &live);
                        }
                        if *variant == Message::Delayed {
                            e.cache
                                .invalidate_stage(params::STAGE_IDENTITY, Generation(e.generation));
                        }
                    }
                }
                if served_stale {
                    tally.add("stale_upstream_kv_served", 1);
                    tally.add(&format!("invalidation_matrix_stale_hits.{cell}"), 1);
                }
                if released {
                    tally.add("outputs_after_upstream_invalidation", 1);
                    tally.add(&format!("invalidation_matrix_stale_hits.{cell}"), 1);
                }
                // A zombie of the old chain sends a frame that was produced before.
                let zombie =
                    admit_frame(&reframe(&e), ptr_model_api::KvReuseMode::Exact, &rig.live());
                match zombie {
                    Err(ChainFault::NotLive { .. }) => {}
                    Err(_) => tally.add("verdict_mismatches", 1),
                    Ok(_) => {
                        tally.add("attempts_without_typed_rejection", 1);
                        tally.add("stale_upstream_kv_served", 1);
                        tally.add(&format!("invalidation_matrix_stale_hits.{cell}"), 1);
                    }
                }
                if ty == "supersession" {
                    // The new generation works: its frames are admitted and its entry serves.
                    rig.place_successor()?;
                    let mut next = entry(&rig, &mut link, 9)?;
                    let live = rig.live();
                    next.cache
                        .lookup(&next.key, &live)
                        .map_err(|error| format!("the new generation does not serve: {error:?}"))?;
                }
            }
        }
    }
    stale_worker(tally, seed)?;
    cache_hardening(tally, seed)
}

/// What the downstream stage refuses before a frame is ever stored: an upstream that ran
/// under an Approximate mode (version 1 is Exact-only), activations that are not finite; and
/// what the cache does when a Revoke never arrives and it is full of dead entries.
fn cache_hardening(tally: &mut Tally, seed: u64) -> Res<()> {
    let mut rig = Rig::new()?;
    let mut link = Link::new(&rig, seed, 900)?;
    let run = bitwise::split(&rig, Transport::Wire(&mut link), &bitwise::prompt(2), 2)?;
    let good = run.prefill_frame.clone().ok_or("no prefill frame")?;
    let wire_frame = ptr_protocol::activation::ActivationFrame {
        tokens: good.rows(),
        width: good.width(),
        dtype: ptr_protocol::activation::BoundaryDtype::F32,
        positions: good.positions().to_vec(),
        upstream_chain: good.chain().clone(),
        placement_epoch: good.placement_epoch(),
        tensor: super::rig::f32_bytes(good.activations()),
    };
    tally.add("non_exact_upstream_attempts", 1);
    match admit_frame(
        &wire_frame,
        ptr_model_api::KvReuseMode::Approximate,
        &rig.live(),
    ) {
        Err(ChainFault::NotExact { .. }) => {}
        Err(_) => tally.add("verdict_mismatches", 1),
        Ok(_) => tally.add("non_exact_frames_admitted", 1),
    }
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut activations = good.activations().to_vec();
        activations[0] = poison;
        let poisoned = ptr_protocol::activation::ActivationFrame {
            tensor: super::rig::f32_bytes(&activations),
            ..wire_frame.clone()
        };
        tally.add("non_finite_activation_attempts", 1);
        match admit_frame(&poisoned, ptr_model_api::KvReuseMode::Exact, &rig.live()) {
            Err(ChainFault::NonFinite { index: 0 }) => {}
            Err(_) => tally.add("verdict_mismatches", 1),
            Ok(_) => tally.add("non_finite_frames_admitted", 1),
        }
    }

    // A full cache whose entries the authority has since superseded, with no Revoke message
    // ever delivered: a fresh entry still fits.
    let mut cache: SplitStageCache<KvTensorSnapshot> = SplitStageCache::new(2);
    for index in [3, 4] {
        let first = bitwise::split(&rig, Transport::Wire(&mut link), &bitwise::prompt(index), 2)?;
        let frame = first.prefill_frame.clone().ok_or("no prefill frame")?;
        let snapshot = first.prefill_snapshot.clone().ok_or("no snapshot")?;
        cache
            .store(&[&frame], &rig.own_chain(), snapshot, &rig.live())
            .map_err(|error| format!("filling the cache: {error:?}"))?;
    }
    // The downstream stage's own generation is revoked: its entry stops serving, whatever the
    // upstream chain says. And an entry the check at use refused is evicted on the spot.
    {
        let mut own: SplitStageCache<KvTensorSnapshot> = SplitStageCache::new(2);
        let probe = bitwise::split(&rig, Transport::Wire(&mut link), &bitwise::prompt(6), 2)?;
        let frame = probe.prefill_frame.clone().ok_or("no prefill frame")?;
        let snapshot = probe.prefill_snapshot.clone().ok_or("no snapshot")?;
        let key = own
            .store(&[&frame], &rig.own_chain(), snapshot, &rig.live())
            .map_err(|error| format!("the downstream entry: {error:?}"))?;
        crate::experiments::layered_kv::lab::commit_revocation(
            &mut rig.authority.runtime,
            params::DOWNSTREAM_STAGE_IDENTITY,
            1,
        )?;
        if own.lookup(&key, &rig.live()).is_ok() {
            tally.add("stale_upstream_kv_served", 1);
        }
        if !own.is_empty() {
            tally.add("cache_slots_held_by_dead_entries", own.len() as i64);
        }
    }
    tally.add("lost_revoke_full_cache_cases", 1);
    rig.supersede_stage_in_place()?;
    rig.place_successor()?;
    let next = bitwise::split(&rig, Transport::Wire(&mut link), &bitwise::prompt(5), 2)?;
    let frame = next.prefill_frame.clone().ok_or("no prefill frame")?;
    let snapshot = next.prefill_snapshot.clone().ok_or("no snapshot")?;
    if cache
        .store(&[&frame], &rig.own_chain(), snapshot, &rig.live())
        .is_err()
    {
        tally.add("cache_slots_held_by_dead_entries", cache.len() as i64);
    }
    Ok(())
}

/// The frame of the entry as a wire frame again (what a zombie upstream would resend).
fn reframe(entry: &Entry) -> ptr_protocol::activation::ActivationFrame {
    use ptr_protocol::activation::{ActivationFrame, BoundaryDtype};
    ActivationFrame {
        tokens: entry.frame.rows(),
        width: entry.frame.width(),
        dtype: BoundaryDtype::F32,
        positions: entry.frame.positions().to_vec(),
        upstream_chain: entry.frame.chain().clone(),
        placement_epoch: entry.frame.placement_epoch(),
        tensor: super::rig::f32_bytes(entry.frame.activations()),
    }
}

fn inflight_race(
    tally: &mut Tally,
    rig: &mut Rig,
    entry: &mut Entry,
    ty: &str,
    message: Message,
    released: &mut bool,
) -> Res<()> {
    let snapshot;
    let ticket;
    {
        let live = rig.live();
        ticket = entry
            .cache
            .begin_use(&entry.key, &live)
            .map_err(|error| format!("a live entry did not start a use: {error:?}"))?;
        snapshot = entry
            .cache
            .lookup(&entry.key, &live)
            .map_err(|error| format!("lookup: {error:?}"))?
            .clone();
    }
    // The use is running; the generation is invalidated under it.
    invalidate(rig, ty)?;
    if message == Message::Delivered {
        entry
            .cache
            .invalidate_stage(params::STAGE_IDENTITY, Generation(entry.generation));
    } else {
        tally.add("revoke_messages_dropped_or_delayed", 1);
    }
    let output = continue_with(rig, snapshot, entry, 2)?;
    let _ = output;
    let live = rig.live();
    match entry.cache.finish_use(ticket, &live) {
        Ok(()) => *released = true,
        Err(UseFault::Stale(_)) | Err(UseFault::Cancelled) => {}
        Err(_) => tally.add("verdict_mismatches", 1),
    }
    Ok(())
}

/// A stage re-placed without a generation change: the old worker's frame is stale.
fn stale_worker(tally: &mut Tally, seed: u64) -> Res<()> {
    let mut rig = Rig::new()?;
    let mut link = Link::new(&rig, seed, 800)?;
    let mut e = entry(&rig, &mut link, 3)?;
    let old_epoch = rig.epoch();
    // A frame the old worker produced, still in flight.
    let regime = rig
        .stack
        .admit_rope_regime(8, 0)
        .map_err(fail("the regime"))?;
    let mut stage_a = Stage::new(&rig.stack, 0..params::SPLIT_LAYER as usize, "zombie-a")?;
    let embedded = rig
        .stack
        .embed_tokens(&[1, 2, 3, 4])
        .map_err(fail("embed"))?;
    let mid = stage_a.run(&rig.stack, &embedded, 4, false, &regime)?;
    let old_frame = rig.frame(&mid.residual, 4, 0);
    rig.replace_stage()?;
    if rig.epoch() == old_epoch {
        return Err("re-placing the stage did not move its epoch".to_string());
    }
    tally.add("stale_worker_frame_race_cases", 1);
    let shipped = link.ship(&rig, &old_frame, Role::Probe("stale_upstream_request"))?;
    match admit_frame(
        &shipped.delivered.frame,
        ptr_model_api::KvReuseMode::Exact,
        &rig.live(),
    ) {
        Err(ChainFault::StalePlacement { .. }) => {}
        Err(_) => tally.add("verdict_mismatches", 1),
        Ok(_) => tally.add("stale_placement_frames_admitted", 1),
    }
    // The entry derived under the old placement does not serve either.
    let live = rig.live();
    if e.cache.lookup(&e.key, &live).is_ok() {
        tally.add("stale_placement_frames_admitted", 1);
    }
    // A frame that names the stage's current generation with a digest it never committed.
    let mut forged = rig.frame(&mid.residual, 4, 0);
    forged.upstream_chain.stage.digest = [0xAB; 32];
    let shipped = link.ship(&rig, &forged, Role::Probe("stale_upstream_request"))?;
    match admit_frame(
        &shipped.delivered.frame,
        ptr_model_api::KvReuseMode::Exact,
        &rig.live(),
    ) {
        Err(ChainFault::DigestMismatch { .. }) => {}
        Err(_) => tally.add("verdict_mismatches", 1),
        Ok(_) => tally.add("stale_upstream_kv_served", 1),
    }
    // A frame the new worker produces is admitted.
    let fresh = rig.frame(&mid.residual, 4, 0);
    let shipped = link.ship(&rig, &fresh, Role::Probe("stale_upstream_request"))?;
    admit_frame(
        &shipped.delivered.frame,
        ptr_model_api::KvReuseMode::Exact,
        &rig.live(),
    )
    .map_err(|fault| format!("the new worker's frame was refused: {fault:?}"))?;
    Ok(())
}
