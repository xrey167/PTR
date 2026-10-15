//! A5: the invalidation coverage matrix, every cell run on its own.
//!
//! **The 45 cells, derived from `docs/architecture/36-kv-reuse.md` sections D and E.** The
//! dependency kinds are a closed list of seven (`DependencyKind`, in bijection with the frame
//! slots by `bijection_violations`): the upstream stage generation, the semantic, memory and
//! derived inputs of the frame's lineage, and the expert, adapter and base model of its
//! chain. The invalidation types are a revocation and a supersession for every kind and, "for
//! the expert kind, fence" (doc 36, "A fence is an invalidating event for KV too"): 7 x 2 + 1
//! = 15 kind x type pairs. The paths are normal lookup, lagging invalidation (the `Revoke`
//! message dropped or delayed) and in-flight race; the doc's fourth path, restore, "does not
//! apply" to E007 and E008 because no KV is restored on a Pod. 15 x 3 = 45.
//!
//! The doc splits the 45 between two rows: the E007 row owns the 6 stage cells, and the E008
//! row lists the other 39 (base model 6, experts 9, adapters 6, and the semantic, memory and
//! derived inputs of the lineage 18). The manifest of E008 counts 45 and names the stage among
//! its seven kinds, so this harness runs the stage's six again with the rest: nothing in the doc
//! gives another count.
//!
//! Per cell, one or two cases run, each on a fresh world (a real ledger, a real placement
//! controller): the entry is stored and shown to serve, the dependency is invalidated, and
//! the cell passes only if nothing derived from it is served or released afterwards, whether
//! or not the `Revoke` message reached the cache:
//!
//! * `normal`: the message is delivered, then an ordinary use follows;
//! * `lag`: the message is dropped (never delivered) or delayed (delivered after the use);
//! * `inflight`: a use begun before the invalidation finishes after it, the message dropped
//!   or delivered; the frame buffered while it ran is revalidated too.
//!
//! A stale hit (the entry served, a use finished, the buffered frame still live) or a frame of
//! the old generation admitted again is a stale hit of its cell.

use super::params;
use super::world::{self, raw_ancestry, FrameSpec, World};
use crate::experiments::layered_kv::emit::Tally;
use crate::experiments::layered_kv::lab::{fail, Res};
use crate::experiments::rng::Rng;
use ptr_model_api::KvReusePolicy;
use ptr_protocol::activation::{ActivationFrameV2, DependencyKind, Invalidation};
use ptr_runtime::{
    admit_frame_v2, AdmittedFrameV2, ChainFault, FrameFault, SplitStageCacheV2, UseFaultV2,
};
use ptr_types::Generation;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Path {
    Normal,
    Lag,
    InFlight,
}

impl Path {
    pub const ALL: [Self; 3] = [Self::Normal, Self::Lag, Self::InFlight];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Lag => "lag",
            Self::InFlight => "inflight",
        }
    }
}

/// One cell of the matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cell {
    pub kind: DependencyKind,
    pub invalidation: Invalidation,
    pub path: Path,
}

impl Cell {
    /// The cell's name, the member of the grouped metrics: `<kind>_<path>_<type>`.
    pub fn name(self) -> String {
        format!(
            "{}_{}_{}",
            self.kind.name(),
            self.path.name(),
            self.invalidation.name()
        )
    }
}

/// Every cell of the matrix, in a fixed order: kind, then invalidation type, then path.
pub fn cells() -> Vec<Cell> {
    let mut out = Vec::new();
    for kind in DependencyKind::ALL {
        for invalidation in kind.invalidations() {
            for path in Path::ALL {
                out.push(Cell {
                    kind,
                    invalidation: *invalidation,
                    path,
                });
            }
        }
    }
    out
}

pub fn cell_names() -> Vec<String> {
    cells().into_iter().map(Cell::name).collect()
}

/// What happens to the `Revoke` message.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Message {
    /// It reaches the cache before the use.
    Delivered,
    /// It is never delivered.
    Dropped,
    /// It is delivered after the use.
    Delayed,
}

fn variants(path: Path) -> &'static [Message] {
    match path {
        Path::Normal => &[Message::Delivered],
        Path::Lag => &[Message::Dropped, Message::Delayed],
        Path::InFlight => &[Message::Dropped, Message::Delivered],
    }
}

/// The KV an entry holds: derived from the activations of its frame.
fn derive(frame: &AdmittedFrameV2) -> Vec<f32> {
    frame
        .activations()
        .iter()
        .map(|value| value * 2.0 + 1.0)
        .collect()
}

fn invalidate(world: &mut World, cell: Cell, target: &str) -> Res<()> {
    match cell.invalidation {
        Invalidation::Revocation => world.revoke(target),
        Invalidation::Supersession => world.supersede(target),
        Invalidation::Fence => world.fence(target),
    }
}

/// Whether `fault` is the typed refusal the invalidation of `cell` must produce when the old
/// frame is admitted again.
fn expected_refusal(cell: Cell, fault: &FrameFault) -> bool {
    match (cell.kind, cell.invalidation, fault) {
        (
            DependencyKind::SemanticInput
            | DependencyKind::MemoryInput
            | DependencyKind::DerivedInput,
            _,
            FrameFault::InputNotLive { .. },
        ) => true,
        (DependencyKind::UpstreamExpert, Invalidation::Fence, FrameFault::Chain(chain)) => {
            matches!(chain, ChainFault::StaleFence { .. })
        }
        (_, _, FrameFault::Chain(chain)) => matches!(chain, ChainFault::NotLive { .. }),
        _ => false,
    }
}

/// Run every cell for `seed`. `frames` collects the frame of every case, for the bijection
/// check.
pub fn run(tally: &mut Tally, seed: u64, frames: &mut Vec<ActivationFrameV2>) -> Res<()> {
    let mut case_number = 0u64;
    for cell in cells() {
        for message in variants(cell.path) {
            case_number += 1;
            run_case(tally, seed, case_number, cell, *message, frames)?;
        }
    }
    Ok(())
}

fn run_case(
    tally: &mut Tally,
    seed: u64,
    case_number: u64,
    cell: Cell,
    message: Message,
    frames: &mut Vec<ActivationFrameV2>,
) -> Res<()> {
    let name = cell.name();
    let mut rng = Rng::new(seed).fork(case_number);
    let mut world = World::new()?;
    let count = 2 + rng.index(4);
    let raw = raw_ancestry(&mut rng, count);
    let frame = world.frame(&FrameSpec::exact(raw, rng.next_u64()));
    let choices = world::targets(cell.kind);
    let target = choices[rng.index(choices.len())].as_str();
    let target_generation = Generation(world.generation(target));
    // The target is an element of the frame, in the slot of the kind under test.
    let in_slot = frame
        .elements()
        .iter()
        .any(|element| element.identity == target && element.kind() == cell.kind);
    if !in_slot {
        return Err(format!(
            "{name}: {target} is not a {} of the frame",
            cell.kind.name()
        ));
    }
    tally.add(&format!("invalidation_matrix_cases.{name}"), 1);
    count_case(tally, cell);

    let mut cache: SplitStageCacheV2<Vec<f32>> = SplitStageCacheV2::new(params::CACHE_CAPACITY);
    // The downstream stage's own chain the buffered frame is computed with.
    let own = world.own_chain();
    let (buffered, key) = {
        let live = world.live();
        let admitted = admit_frame_v2(&frame, KvReusePolicy::ExactOnly, &live, &live)
            .map_err(fail("admitting the live frame"))?;
        let key = cache
            .store(
                &[&admitted],
                &world.own_chain(),
                derive(&admitted),
                &live,
                &live,
            )
            .map_err(fail("storing the entry"))?;
        (admitted, key)
    };
    // Before anything is invalidated the entry serves what the frame derives.
    {
        let live = world.live();
        match cache.lookup(&key, &live, &live) {
            Ok(served) if *served.value == derive(&buffered) => {}
            _ => tally.add("verdict_mismatches", 1),
        }
    }

    let mut stale_served = false;
    let mut released = false;
    match cell.path {
        Path::Normal | Path::Lag => {
            invalidate(&mut world, cell, target)?;
            if message == Message::Delivered {
                cache.invalidate_dependency(target, target_generation);
            } else {
                tally.add("revoke_messages_dropped_or_delayed", 1);
            }
            let live = world.live();
            let ticket = cache.begin_use(&key, &live, &live);
            let lookup_ok = cache.lookup(&key, &live, &live).is_ok();
            stale_served = ticket.is_ok() || lookup_ok;
            if let Ok(ticket) = ticket {
                released |= cache.finish_use(ticket, &live, &live).is_ok();
            }
            if message == Message::Delayed {
                cache.invalidate_dependency(target, target_generation);
            }
        }
        Path::InFlight => {
            let (ticket, snapshot) = {
                let live = world.live();
                let ticket = cache
                    .begin_use(&key, &live, &live)
                    .map_err(fail("beginning the use"))?;
                let snapshot = cache
                    .lookup(&key, &live, &live)
                    .map_err(fail("the lookup of the use"))?
                    .value
                    .clone();
                (ticket, snapshot)
            };
            // The use is running; the dependency is invalidated under it.
            invalidate(&mut world, cell, target)?;
            if message == Message::Delivered {
                cache.invalidate_dependency(target, target_generation);
            } else {
                tally.add("revoke_messages_dropped_or_delayed", 1);
            }
            let _output: f32 = snapshot.iter().sum();
            let live = world.live();
            match cache.finish_use(ticket, &live, &live) {
                Ok(_) => released = true,
                Err(UseFaultV2::Stale(_)) | Err(UseFaultV2::Cancelled) => {}
                Err(_) => tally.add("verdict_mismatches", 1),
            }
        }
    }
    // The frame that was buffered while all this happened: computing from it and releasing
    // the output would be an output after the invalidation.
    {
        let live = world.live();
        if buffered.revalidate(&own, &live, &live).is_ok() {
            released = true;
        }
    }
    // A zombie of the old generation sends the frame again.
    {
        let live = world.live();
        match admit_frame_v2(&frame, KvReusePolicy::ExactOnly, &live, &live) {
            Err(fault) if expected_refusal(cell, &fault) => {}
            Err(_) => tally.add("verdict_mismatches", 1),
            Ok(_) => {
                tally.add("attempts_without_typed_rejection", 1);
                stale_served = true;
            }
        }
    }
    if stale_served || released {
        tally.add(&format!("invalidation_matrix_stale_hits.{name}"), 1);
        count_stale(tally, cell);
    }
    if released {
        tally.add("outputs_after_upstream_invalidation", 1);
    }
    // The successor works: a revoked generation has none, but a superseded one and a fenced
    // expert's next epoch are admitted, stored and served again.
    if matches!(
        cell.invalidation,
        Invalidation::Supersession | Invalidation::Fence
    ) {
        if cell.invalidation == Invalidation::Fence {
            world.readmit(target)?;
        }
        let next = world.frame(&FrameSpec::exact(vec![[7; 32]], rng.next_u64()));
        let live = world.live();
        let served = admit_frame_v2(&next, KvReusePolicy::ExactOnly, &live, &live)
            .ok()
            .and_then(|admitted| {
                let key = cache
                    .store(
                        &[&admitted],
                        &world.own_chain(),
                        derive(&admitted),
                        &live,
                        &live,
                    )
                    .ok()?;
                let ok = cache.lookup(&key, &live, &live).is_ok();
                ok.then_some(())
            });
        if served.is_none() {
            tally.add("verdict_mismatches", 1);
        }
    }
    frames.push(frame);
    Ok(())
}

/// The case counters named in the manifest, by kind.
fn count_case(tally: &mut Tally, cell: Cell) {
    match cell.kind {
        DependencyKind::SemanticInput
        | DependencyKind::MemoryInput
        | DependencyKind::DerivedInput => tally.add("input_lineage_revalidation_cases", 1),
        DependencyKind::UpstreamExpert if cell.invalidation == Invalidation::Fence => {
            tally.add("upstream_expert_fence_cases", 1)
        }
        DependencyKind::UpstreamAdapter if cell.path == Path::InFlight => {
            tally.add("upstream_adapter_invalidation_race_cases", 1)
        }
        DependencyKind::UpstreamBaseModel => tally.add("upstream_base_model_invalidation_cases", 1),
        DependencyKind::UpstreamStageGeneration
        | DependencyKind::UpstreamExpert
        | DependencyKind::UpstreamAdapter => {}
    }
}

/// The stale counters named in the manifest, by kind.
fn count_stale(tally: &mut Tally, cell: Cell) {
    match cell.kind {
        DependencyKind::SemanticInput
        | DependencyKind::MemoryInput
        | DependencyKind::DerivedInput => tally.add("stale_input_lineage_kv_served", 1),
        DependencyKind::UpstreamExpert if cell.invalidation == Invalidation::Fence => {
            tally.add("downstream_kv_reuse_after_upstream_fence", 1)
        }
        DependencyKind::UpstreamAdapter => {
            tally.add("downstream_kv_reuse_after_upstream_adapter_invalidation", 1)
        }
        DependencyKind::UpstreamBaseModel => tally.add(
            "downstream_kv_reuse_after_upstream_base_model_invalidation",
            1,
        ),
        DependencyKind::UpstreamStageGeneration | DependencyKind::UpstreamExpert => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::e008::part_a::metrics;

    #[test]
    fn the_matrix_has_forty_five_distinct_cells_of_seven_kinds() {
        let cells = cells();
        assert_eq!(cells.len(), params::MATRIX_CELLS);
        let names = cell_names();
        let distinct: std::collections::BTreeSet<_> = names.iter().collect();
        assert_eq!(distinct.len(), 45);
        for kind in DependencyKind::ALL {
            let of_kind = cells.iter().filter(|cell| cell.kind == kind).count();
            let expected = if kind == DependencyKind::UpstreamExpert {
                9
            } else {
                6
            };
            assert_eq!(of_kind, expected, "{}", kind.name());
        }
        let fences = cells
            .iter()
            .filter(|cell| cell.invalidation == Invalidation::Fence)
            .count();
        assert_eq!(fences, 3);
        assert!(cells
            .iter()
            .filter(|cell| cell.invalidation == Invalidation::Fence)
            .all(|cell| cell.kind == DependencyKind::UpstreamExpert));
        assert_eq!(
            names.first().map(String::as_str),
            Some("upstream_stage_generation_normal_revocation")
        );
    }

    #[test]
    fn every_cell_runs_at_least_once_and_none_serves_stale_state() {
        let mut tally = Tally::new(metrics::LIST);
        let mut frames = Vec::new();
        run(&mut tally, 17, &mut frames).unwrap();
        let cases = tally.group("invalidation_matrix_cases");
        assert_eq!(cases.len(), 45);
        assert!(cases.iter().all(|(_, count)| *count >= 1));
        assert_eq!(tally.get("verdict_mismatches"), 0);
        assert_eq!(tally.get("attempts_without_typed_rejection"), 0);
        assert!(tally.group("invalidation_matrix_stale_hits").is_empty());
        assert_eq!(tally.get("outputs_after_upstream_invalidation"), 0);
        assert!(tally.get("revoke_messages_dropped_or_delayed") >= 1);
        assert_eq!(
            frames.len(),
            cases.iter().map(|(_, c)| *c as usize).sum::<usize>()
        );
    }
}
