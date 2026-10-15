//! A1, A3 and A4: the seeded case families that are not cells of the matrix.
//!
//! * A1 (`raw_ancestry_cases`): an earlier Raw segment removed, changed, reordered or dropped
//!   from the front, with the typed inputs, the chain, the positions and even the activations
//!   equal, is never a hit of the downstream cache (`reuse_after_raw_ancestor_removed`). The
//!   control, a cache keyed without lineage or raw ancestry, reuses in every one of them.
//! * A3 (`mode_cases`): an Approximate frame never populates or hits the Exact namespace, the
//!   output of a hit on it carries the taint, and that taint reaches the runtime's effect
//!   boundary, which refuses the action. The control, one namespace for both modes, serves the
//!   Approximate entry to an Exact lookup.
//! * A4 (`persistence_cases`): a snapshot, a tier write, a persist and a restore of a
//!   downstream entry are each refused with a typed fault, a restart (a new cache, or a clear)
//!   leaves no entry and no open use, and nothing restores one.

use super::controls::{keyless_key, single_namespace_key};
use super::params;
use super::world::{raw_ancestry, FrameSpec, World};
use crate::experiments::layered_kv::emit::Tally;
use crate::experiments::layered_kv::lab::{fail, Res};
use crate::experiments::rng::Rng;
use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_ledger::LedgerEvent;
use ptr_model_api::{KvReuseMode, KvReusePolicy, KvReuseTaint};
use ptr_runtime::{
    admit_frame_v2, prefix_key_v2, AdmittedFrameV2, DownstreamKey, FrameFault, PersistFault,
    PersistOp, PtrRuntime, RuntimeError, SplitStageCacheV2, UseFaultV2,
};
use ptr_types::{CapabilityId, CapsuleId, Effect, Generation, ProjectId, RequestId, TypeId};
use std::collections::BTreeSet;

fn derive(frame: &AdmittedFrameV2) -> Vec<f32> {
    frame
        .activations()
        .iter()
        .map(|value| value * 2.0 + 1.0)
        .collect()
}

/// A1.
pub fn raw_ancestry_cases(tally: &mut Tally, seed: u64, cases: u64) -> Res<()> {
    for index in 0..cases {
        let mut rng = Rng::new(seed).fork(1_000 + index);
        let world = World::new()?;
        let length = 3 + rng.index(4);
        let raw = raw_ancestry(&mut rng, length);
        let tensor_seed = rng.next_u64();
        let mut changed = raw.clone();
        match index % 4 {
            // An earlier segment removed (never the last one).
            0 => {
                changed.remove(rng.index(length - 1));
            }
            // A segment changed in place.
            1 => {
                let at = rng.index(length);
                changed[at] = rng.digest();
            }
            // The first two segments in the other order.
            2 => changed.swap(0, 1),
            // The first segment dropped from the front.
            _ => {
                changed.remove(0);
            }
        }
        let full = world.frame(&FrameSpec::exact(raw.clone(), tensor_seed));
        let reduced = world.frame(&FrameSpec::exact(changed, tensor_seed));
        // Everything but the Raw ancestry is equal.
        let only_raw_differs = full.input_lineage.inputs == reduced.input_lineage.inputs
            && full.upstream_chain == reduced.upstream_chain
            && full.positions == reduced.positions
            && full.tensor == reduced.tensor
            && full.input_lineage.raw_ancestry != reduced.input_lineage.raw_ancestry;
        if !only_raw_differs {
            tally.add("verdict_mismatches", 1);
        }
        tally.add("raw_ancestor_omission_cases", 1);

        let live = world.live();
        let own = world.own_chain();
        let mut cache: SplitStageCacheV2<Vec<f32>> = SplitStageCacheV2::new(params::CACHE_CAPACITY);
        let admitted = admit_frame_v2(&full, KvReusePolicy::ExactOnly, &live, &live)
            .map_err(fail("admitting the full frame"))?;
        let reduced_admitted = admit_frame_v2(&reduced, KvReusePolicy::ExactOnly, &live, &live)
            .map_err(fail("admitting the reduced frame"))?;
        let key = cache
            .store(&[&admitted], &own, derive(&admitted), &live, &live)
            .map_err(fail("storing the entry"))?;
        if cache.lookup(&key, &live, &live).is_err() {
            tally.add("verdict_mismatches", 1);
        }
        let probe = prefix_key_v2(&[&reduced_admitted], &own);
        if cache.lookup(&probe, &live, &live).is_ok() {
            tally.add("reuse_after_raw_ancestor_removed", 1);
        }
        if cache.begin_use(&probe, &live, &live).is_ok() {
            tally.add("reuse_after_raw_ancestor_removed", 1);
        }

        // The control: a cache whose key has no lineage and no raw ancestry.
        let mut keyless: BTreeSet<[u8; 32]> = BTreeSet::new();
        keyless.insert(keyless_key(&full, &own));
        if keyless.contains(&keyless_key(&reduced, &own)) {
            tally.add(
                "control_trips.keyless_lineage_cache_reuses_after_raw_removal",
                1,
            );
        }
    }
    Ok(())
}

/// An effect runtime and an action reachable from the output of a hit.
fn effect_runtime() -> Res<(PtrRuntime, ActionIr)> {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).map_err(fail("the effect runtime"))?;
    let revision = runtime
        .ingest_text(RequestId::from("e008-request"), "hello")
        .map_err(fail("ingesting the request"))?;
    let action = ActionIr {
        operation: "write".into(),
        target: "artifact:a".into(),
        capability: CapabilityId::from("file.write"),
        effect: Effect::Mutation,
        input_type: TypeId::from("Bytes"),
        generation: Generation(1),
        revision,
        payload: b"derived from a split frame".to_vec(),
    };
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from(action.target.as_str()),
            generation: action.generation,
        })
        .map_err(fail("committing the target"))?;
    runtime
        .permissions_mut()
        .capabilities
        .insert(action.capability.clone());
    runtime.permissions_mut().allow_mutation = true;
    Ok((runtime, action))
}

/// What the runtime's effect boundary does with an action carrying `taint`.
#[derive(Debug, Eq, PartialEq)]
enum Boundary {
    Allowed,
    RefusedAsApproximate,
    RefusedOtherwise,
}

fn at_boundary(runtime: &PtrRuntime, action: &ActionIr, taint: KvReuseTaint) -> Boundary {
    match runtime.authorize_action_with_taint(action, taint) {
        Ok(()) => Boundary::Allowed,
        Err(RuntimeError::ApproximateStateAtEffectBoundary { .. }) => {
            Boundary::RefusedAsApproximate
        }
        Err(_) => Boundary::RefusedOtherwise,
    }
}

/// A3.
pub fn mode_cases(tally: &mut Tally, seed: u64, cases: u64) -> Res<()> {
    let (runtime, action) = effect_runtime()?;
    for index in 0..cases {
        let mut rng = Rng::new(seed).fork(2_000 + index);
        let world = World::new()?;
        let count = 2 + rng.index(4);
        let raw = raw_ancestry(&mut rng, count);
        let tensor_seed = rng.next_u64();
        // The Approximate frame and its Exact twin: the same chain, lineage, positions and
        // activations, differing in the mode and taint alone.
        let approximate = world.frame(&FrameSpec::approximate(raw.clone(), tensor_seed));
        let exact_twin = world.frame(&FrameSpec::exact(raw, tensor_seed));
        let live = world.live();
        let own = world.own_chain();
        tally.add("approximate_frame_downstream_exact_lookup_cases", 1);

        // An Approximate frame is admitted only when the policy allows it.
        match admit_frame_v2(&approximate, KvReusePolicy::ExactOnly, &live, &live) {
            Err(FrameFault::ApproximateNotAllowed) => {}
            _ => tally.add("approximate_frames_served_to_exact", 1),
        }
        let approx = admit_frame_v2(&approximate, KvReusePolicy::AllowApproximate, &live, &live)
            .map_err(fail("admitting the Approximate frame"))?;
        let exact = admit_frame_v2(&exact_twin, KvReusePolicy::ExactOnly, &live, &live)
            .map_err(fail("admitting the Exact twin"))?;

        let mut cache: SplitStageCacheV2<Vec<f32>> = SplitStageCacheV2::new(params::CACHE_CAPACITY);
        let approx_key = cache
            .store(&[&approx], &own, derive(&approx), &live, &live)
            .map_err(fail("storing the Approximate entry"))?;
        // It populated the Approximate namespace and nothing else.
        if approx_key.namespace != KvReuseMode::Approximate || cache.len_in(KvReuseMode::Exact) != 0
        {
            tally.add("approximate_frames_served_to_exact", 1);
        }
        // Exact lookups: by the twin's own key, and by the Approximate digest asked in the
        // Exact namespace.
        let twin_key = prefix_key_v2(&[&exact], &own);
        let forged = DownstreamKey {
            namespace: KvReuseMode::Exact,
            digest: approx_key.digest,
        };
        for probe in [&twin_key, &forged] {
            if cache.lookup(probe, &live, &live).is_ok()
                || cache.begin_use(probe, &live, &live).is_ok()
            {
                tally.add("approximate_frames_served_to_exact", 1);
            }
        }
        // The Exact twin then populates its own namespace and serves its own value.
        let twin_stored = cache
            .store(&[&exact], &own, derive(&exact), &live, &live)
            .map_err(fail("storing the Exact twin"))?;
        if twin_stored.namespace != KvReuseMode::Exact
            || cache.len_in(KvReuseMode::Exact) != 1
            || cache.len_in(KvReuseMode::Approximate) != 1
        {
            tally.add("verdict_mismatches", 1);
        }
        match cache.lookup(&twin_stored, &live, &live) {
            Ok(served) if served.taint == KvReuseTaint::Exact => {}
            _ => tally.add("verdict_mismatches", 1),
        }
        if at_boundary(&runtime, &action, KvReuseTaint::Exact) != Boundary::Allowed {
            tally.add("verdict_mismatches", 1);
        }

        // The output of an Approximate hit carries the taint, and the boundary refuses it.
        let ticket = cache
            .begin_use(&approx_key, &live, &live)
            .map_err(fail("beginning the Approximate use"))?;
        let released = cache.finish_use(ticket, &live, &live);
        match released {
            Ok(taint) if taint.is_approximate() && taint == approx.taint() => {
                if at_boundary(&runtime, &action, taint) != Boundary::RefusedAsApproximate {
                    tally.add("untainted_outputs_from_approximate_frames", 1);
                }
            }
            Ok(_) => tally.add("untainted_outputs_from_approximate_frames", 1),
            Err(_) => tally.add("verdict_mismatches", 1),
        }
        match cache.lookup(&approx_key, &live, &live) {
            Ok(served) if served.taint == approx.taint() => {}
            Ok(_) => tally.add("untainted_outputs_from_approximate_frames", 1),
            Err(_) => tally.add("verdict_mismatches", 1),
        }

        // The control: one namespace for both modes.
        let mut single: std::collections::BTreeMap<[u8; 32], bool> =
            std::collections::BTreeMap::new();
        single.insert(single_namespace_key(&approximate, &own), true);
        if single.contains_key(&single_namespace_key(&exact_twin, &own)) {
            tally.add(
                "control_trips.single_namespace_cache_serves_approximate_to_exact",
                1,
            );
        }
    }
    Ok(())
}

/// A4.
pub fn persistence_cases(tally: &mut Tally, seed: u64, cases: u64) -> Res<()> {
    for index in 0..cases {
        let mut rng = Rng::new(seed).fork(3_000 + index);
        let world = World::new()?;
        let count = 2 + rng.index(3);
        let raw = raw_ancestry(&mut rng, count);
        let tensor_seed = rng.next_u64();
        let live = world.live();
        let own = world.own_chain();
        let exact = admit_frame_v2(
            &world.frame(&FrameSpec::exact(raw.clone(), tensor_seed)),
            KvReusePolicy::ExactOnly,
            &live,
            &live,
        )
        .map_err(fail("admitting the Exact frame"))?;
        let approximate = admit_frame_v2(
            &world.frame(&FrameSpec::approximate(raw, tensor_seed)),
            KvReusePolicy::AllowApproximate,
            &live,
            &live,
        )
        .map_err(fail("admitting the Approximate frame"))?;
        let mut cache: SplitStageCacheV2<Vec<f32>> = SplitStageCacheV2::new(params::CACHE_CAPACITY);
        let keys = [
            cache
                .store(&[&exact], &own, derive(&exact), &live, &live)
                .map_err(fail("storing the Exact entry"))?,
            cache
                .store(&[&approximate], &own, derive(&approximate), &live, &live)
                .map_err(fail("storing the Approximate entry"))?,
        ];
        for key in &keys {
            for op in [
                PersistOp::Snapshot,
                PersistOp::TierWrite,
                PersistOp::Persist,
                PersistOp::Restore,
            ] {
                tally.add("downstream_persist_rejection_cases", 1);
                match cache.persist(op, key) {
                    Err(PersistFault { op: refused }) if refused == op => {}
                    Err(_) => tally.add("verdict_mismatches", 1),
                    // The success type is uninhabited: this arm cannot be reached.
                    Ok(never) => match never {},
                }
            }
        }
        // A use is open across the restart.
        let open = cache
            .begin_use(&keys[0], &live, &live)
            .map_err(fail("beginning a use"))?;
        // Restart one: the process dies and a new cache starts. Nothing brings an entry back.
        let mut restarted: SplitStageCacheV2<Vec<f32>> =
            SplitStageCacheV2::new(params::CACHE_CAPACITY);
        tally.add("downstream_entries_after_restart", restarted.len() as i64);
        for key in &keys {
            if restarted.lookup(key, &live, &live).is_ok() {
                tally.add("downstream_restore_attempts_admitted", 1);
            }
        }
        if restarted.finish_use(open, &live, &live).is_ok() {
            tally.add("downstream_restore_attempts_admitted", 1);
        }
        // Restart two: the same cache cleared.
        let open = cache
            .begin_use(&keys[1], &live, &live)
            .map_err(fail("beginning a use"))?;
        cache.clear();
        tally.add(
            "downstream_entries_after_restart",
            (cache.len() + cache.open_uses()) as i64,
        );
        for key in &keys {
            if cache.lookup(key, &live, &live).is_ok() {
                tally.add("downstream_restore_attempts_admitted", 1);
            }
        }
        match cache.finish_use(open, &live, &live) {
            Err(UseFaultV2::UnknownTicket) => {}
            Ok(_) => tally.add("downstream_restore_attempts_admitted", 1),
            Err(_) => tally.add("verdict_mismatches", 1),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::e008::part_a::metrics;

    fn tally() -> Tally {
        Tally::new(metrics::LIST)
    }

    #[test]
    fn a_removed_or_changed_raw_segment_is_never_reused_and_the_keyless_control_reuses() {
        let mut tally = tally();
        raw_ancestry_cases(&mut tally, 17, 8).unwrap();
        assert_eq!(tally.get("raw_ancestor_omission_cases"), 8);
        assert_eq!(tally.get("reuse_after_raw_ancestor_removed"), 0);
        assert_eq!(tally.get("verdict_mismatches"), 0);
        assert!(
            tally.get("control_trips.keyless_lineage_cache_reuses_after_raw_removal") >= 1,
            "the control must show the failure it exists to show"
        );
    }

    #[test]
    fn an_approximate_frame_never_reaches_exact_and_its_taint_reaches_the_boundary() {
        let mut tally = tally();
        mode_cases(&mut tally, 17, 4).unwrap();
        assert_eq!(
            tally.get("approximate_frame_downstream_exact_lookup_cases"),
            4
        );
        assert_eq!(tally.get("approximate_frames_served_to_exact"), 0);
        assert_eq!(tally.get("untainted_outputs_from_approximate_frames"), 0);
        assert_eq!(tally.get("verdict_mismatches"), 0);
        assert_eq!(
            tally.get("control_trips.single_namespace_cache_serves_approximate_to_exact"),
            4
        );
    }

    #[test]
    fn persistence_is_refused_and_a_restart_leaves_nothing() {
        let mut tally = tally();
        persistence_cases(&mut tally, 17, 3).unwrap();
        assert_eq!(tally.get("downstream_persist_rejection_cases"), 3 * 2 * 4);
        assert_eq!(tally.get("downstream_restore_attempts_admitted"), 0);
        assert_eq!(tally.get("downstream_entries_after_restart"), 0);
        assert_eq!(tally.get("verdict_mismatches"), 0);
    }

    #[test]
    fn the_effect_boundary_refuses_an_approximate_taint_and_lets_an_exact_one_through() {
        let (runtime, action) = effect_runtime().unwrap();
        assert_eq!(
            at_boundary(&runtime, &action, KvReuseTaint::Exact),
            Boundary::Allowed
        );
        assert_eq!(
            at_boundary(
                &runtime,
                &action,
                KvReuseTaint::Approximate {
                    report_digest: [1; 32]
                }
            ),
            Boundary::RefusedAsApproximate
        );
        assert_ne!(
            at_boundary(&runtime, &action, KvReuseTaint::Exact),
            Boundary::RefusedOtherwise
        );
    }
}
