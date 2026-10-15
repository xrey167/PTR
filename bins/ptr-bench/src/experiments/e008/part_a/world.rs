//! What the cases share: a real `PtrRuntime` ledger as the lifecycle authority of every
//! generation-bearing dependency, a real `PodPlacementController` for the upstream stage, two
//! experts born and fenced through the runtime's real authority path (their fence epoch is the
//! one `RuntimeStageLiveness` answers from the expert registry), and the builder of version 2
//! activation frames over the live generations.

use super::params;
use crate::experiments::layered_kv::experts;
use crate::experiments::layered_kv::lab::{self, fail, Res};
use crate::experiments::rng::Rng;
use ptr_config::PtrConfig;
use ptr_ledger::integrity::sha256;
use ptr_ledger::ExpertFenceCause;
use ptr_protocol::activation::{
    ActivationFrameV2, BoundaryDtype, ChainElement, DependencyKind, FrameMode, FrameTaint,
    InputLineage, LineageInput, LineageKind, UpstreamChain,
};
use ptr_runtime::{
    input_target, DeviceHealth, DeviceRecord, InputLiveness, NodeHealth, NodeRecord,
    PodPlacementController, PtrRuntime, RuntimeInputLiveness, RuntimeStageLiveness, StageLiveness,
};
use ptr_types::{ArtifactId, DeviceId, Generation, NodeId, PodId, Timestamp, Validity};
use std::collections::BTreeMap;

pub const STAGE: &str = "stage-a";
pub const DOWNSTREAM: &str = "stage-b";
pub const BASE: &str = "base-model";
/// The downstream pod's own slice of the base model: a lifecycle target of its own, so that
/// invalidating the *upstream* base model (a matrix cell) is told apart from the downstream
/// stage's own dependencies, which the cache revalidates as well.
pub const DOWNSTREAM_BASE: &str = "base-model-b";
pub const ADAPTERS: [&str; 2] = ["adapter-1", "adapter-2"];
/// The layer the two experts live at; their identities are `expert:{project}/{pod}`.
pub const EXPERT_LAYER: usize = 3;
pub const EXPERT_COUNT: usize = 2;
pub const SEMANTIC: [&str; 2] = ["sem-1", "sem-2"];
pub const MEMORY: [&str; 2] = ["mem-1", "mem-2"];
pub const DERIVED: [&str; 2] = ["drv-1", "drv-2"];

/// The identities that hold elements of `kind` in a frame the world builds: the dependencies
/// a case of that kind can invalidate (it picks one by the seed).
pub fn targets(kind: DependencyKind) -> Vec<String> {
    let names = |list: &[&str]| list.iter().map(|name| (*name).to_string()).collect();
    match kind {
        DependencyKind::UpstreamStageGeneration => names(&[STAGE]),
        DependencyKind::SemanticInput => names(&SEMANTIC),
        DependencyKind::MemoryInput => names(&MEMORY),
        DependencyKind::DerivedInput => names(&DERIVED),
        DependencyKind::UpstreamExpert => expert_ids(),
        DependencyKind::UpstreamAdapter => names(&ADAPTERS),
        DependencyKind::UpstreamBaseModel => names(&[BASE]),
    }
}

/// The identities of the two experts: `expert:{project}/{pod}`.
pub fn expert_ids() -> Vec<String> {
    (0..EXPERT_COUNT)
        .map(|index| experts::identity_of(EXPERT_LAYER, index))
        .collect()
}

/// The kind of a typed input identity, if it is one.
fn input_kind(identity: &str) -> Option<LineageKind> {
    [
        (LineageKind::Semantic, &SEMANTIC),
        (LineageKind::Memory, &MEMORY),
        (LineageKind::Derived, &DERIVED),
    ]
    .into_iter()
    .find(|(_, list)| list.contains(&identity))
    .map(|(kind, _)| kind)
}

/// The lifecycle target the ledger tracks `identity` under: an input is a target of its kind.
fn ledger_target(identity: &str) -> String {
    input_kind(identity).map_or_else(|| identity.to_string(), |kind| input_target(kind, identity))
}

/// The digest committed for `(identity, generation)`.
pub fn digest_of(identity: &str, generation: u64) -> [u8; 32] {
    sha256(format!("e008.manifest|{identity}|{generation}").as_bytes())
}

/// The stage liveness of the runtime and its placement controller (expert fence epochs
/// included: the runtime answers them from its registry) and the input liveness of the runtime.
pub struct Live<'a> {
    stage: RuntimeStageLiveness<'a>,
    inputs: RuntimeInputLiveness<'a>,
}

impl StageLiveness for Live<'_> {
    fn validity(&self, target: &str, generation: Generation) -> Option<Validity> {
        self.stage.validity(target, generation)
    }

    fn manifest_digest(&self, target: &str, generation: Generation) -> Option<[u8; 32]> {
        self.stage.manifest_digest(target, generation)
    }

    fn placement_epoch(&self, stage: &str) -> Option<u64> {
        self.stage.placement_epoch(stage)
    }

    fn fence_epoch(&self, expert: &str) -> Option<u64> {
        self.stage.fence_epoch(expert)
    }
}

impl InputLiveness for Live<'_> {
    fn validity(
        &self,
        kind: LineageKind,
        identity: &str,
        generation: Generation,
    ) -> Option<Validity> {
        self.inputs.validity(kind, identity, generation)
    }

    fn content_digest(
        &self,
        kind: LineageKind,
        identity: &str,
        generation: Generation,
    ) -> Option<[u8; 32]> {
        self.inputs.content_digest(kind, identity, generation)
    }
}

/// What a frame built by the world varies in.
#[derive(Clone, Debug)]
pub struct FrameSpec {
    pub mode: FrameMode,
    /// The ordered Raw ancestry.
    pub raw: Vec<[u8; 32]>,
    /// Seeds the activations.
    pub tensor_seed: u64,
}

impl FrameSpec {
    pub fn exact(raw: Vec<[u8; 32]>, tensor_seed: u64) -> Self {
        Self {
            mode: FrameMode::Exact,
            raw,
            tensor_seed,
        }
    }

    pub fn approximate(raw: Vec<[u8; 32]>, tensor_seed: u64) -> Self {
        Self {
            mode: FrameMode::Approximate,
            raw,
            tensor_seed,
        }
    }
}

pub struct World {
    pub runtime: PtrRuntime,
    placement: PodPlacementController,
    manifests: BTreeMap<(String, u64), [u8; 32]>,
    input_digests: BTreeMap<(LineageKind, String, u64), [u8; 32]>,
    generations: BTreeMap<String, u64>,
}

impl World {
    /// A world in which every dependency is live at generation 1, both experts serve under
    /// their first fence epoch (born through the real admission path) and the upstream stage
    /// is placed.
    pub fn new() -> Res<Self> {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).map_err(fail("the runtime"))?;
        let identities: Vec<String> = [STAGE, DOWNSTREAM, BASE, DOWNSTREAM_BASE]
            .into_iter()
            .chain(ADAPTERS)
            .chain(SEMANTIC)
            .chain(MEMORY)
            .chain(DERIVED)
            .map(str::to_string)
            .collect();
        // Experts are registry objects only (no capsule): their generation, state, manifest
        // digest and fence epoch all come from the one registry entry the runtime answers from.
        let mut generations = BTreeMap::new();
        let mut manifests = BTreeMap::new();
        let mut input_digests = BTreeMap::new();
        for identity in identities {
            lab::commit_capsule(&mut runtime, &ledger_target(&identity), 1)?;
            generations.insert(identity.clone(), 1);
            match input_kind(&identity) {
                Some(kind) => {
                    input_digests.insert((kind, identity.clone(), 1), digest_of(&identity, 1));
                }
                None => {
                    manifests.insert((identity.clone(), 1), digest_of(&identity, 1));
                }
            }
        }
        for (index, identity) in expert_ids().into_iter().enumerate() {
            experts::birth(&mut runtime, EXPERT_LAYER, index)?;
            generations.insert(identity, 1);
        }
        let mut placement = PodPlacementController::default();
        placement
            .register_node(NodeRecord {
                node_id: NodeId::from("node-1"),
                zone: "z".into(),
                health: NodeHealth::Healthy,
                last_heartbeat: Timestamp(1),
                devices: vec![DeviceRecord {
                    device_id: DeviceId::from("cpu-0"),
                    vram_bytes: 1 << 20,
                    used_vram_bytes: 0,
                    health: DeviceHealth::Healthy,
                }],
            })
            .map_err(fail("a node"))?;
        placement
            .assign(
                PodId::from(STAGE),
                ArtifactId::from("stage-a-artifact"),
                Generation(1),
                NodeId::from("node-1"),
                DeviceId::from("cpu-0"),
                1,
            )
            .map_err(fail("placing the stage"))?;
        Ok(Self {
            runtime,
            placement,
            manifests,
            input_digests,
            generations,
        })
    }

    pub fn live(&self) -> Live<'_> {
        Live {
            stage: RuntimeStageLiveness {
                runtime: &self.runtime,
                placement: &self.placement,
                manifests: &self.manifests,
            },
            inputs: RuntimeInputLiveness {
                runtime: &self.runtime,
                digests: &self.input_digests,
            },
        }
    }

    /// The generation `identity` is at now (a revoked one stays at its revoked generation).
    pub fn generation(&self, identity: &str) -> u64 {
        self.generations.get(identity).copied().unwrap_or(0)
    }

    /// The expert's committed fence epoch: its state version in the runtime's registry (zero
    /// for anything that is not an expert).
    pub fn fence_epoch(&self, identity: &str) -> u64 {
        expert_index(identity)
            .and_then(|index| experts::state_of(&self.runtime, EXPERT_LAYER, index))
            .map_or(0, |(_, _, version)| version)
    }

    fn element(&self, identity: &str) -> ChainElement {
        let generation = self.generation(identity);
        ChainElement {
            identity: identity.to_string(),
            generation,
            digest: self.manifest_of(identity, generation),
            fence_epoch: self.fence_epoch(identity),
        }
    }

    /// The manifest digest committed for `(identity, generation)`: an expert's is its registry
    /// entry's, everything else's the one this world admitted.
    fn manifest_of(&self, identity: &str, generation: u64) -> [u8; 32] {
        expert_index(identity)
            .and_then(|_| {
                self.live()
                    .stage
                    .manifest_digest(identity, Generation(generation))
            })
            .unwrap_or_else(|| digest_of(identity, generation))
    }

    fn input(&self, kind: LineageKind, identity: &str) -> LineageInput {
        let generation = self.generation(identity);
        LineageInput {
            kind,
            identity: identity.to_string(),
            generation,
            digest: digest_of(identity, generation),
        }
    }

    /// The downstream stage's own chain.
    pub fn own_chain(&self) -> UpstreamChain {
        UpstreamChain {
            stage: self.element(DOWNSTREAM),
            base_model: self.element(DOWNSTREAM_BASE),
            adapters: Vec::new(),
            experts: Vec::new(),
        }
    }

    /// The placement epoch the upstream stage holds now.
    pub fn placement_epoch(&self) -> u64 {
        self.placement
            .placement(&PodId::from(STAGE))
            .map(|placement| placement.epoch.0)
            .unwrap_or(0)
    }

    /// A frame the upstream stage would produce now: the live generation of every dependency
    /// (two adapters, two experts, two inputs of each lineage kind) and the Raw ancestry of
    /// `spec`.
    pub fn frame(&self, spec: &FrameSpec) -> ActivationFrameV2 {
        let mut rng = Rng::new(spec.tensor_seed);
        let tokens = params::FRAME_ROWS;
        let width = params::FRAME_WIDTH;
        let tensor = (0..tokens * width)
            .flat_map(|_| rng.f32_between(-1.0, 1.0).to_le_bytes())
            .collect();
        let mut inputs = Vec::new();
        for (kind, list) in [
            (LineageKind::Semantic, SEMANTIC),
            (LineageKind::Memory, MEMORY),
            (LineageKind::Derived, DERIVED),
        ] {
            inputs.extend(list.iter().map(|identity| self.input(kind, identity)));
        }
        let taint = match spec.mode {
            FrameMode::Exact => FrameTaint::Exact,
            FrameMode::Approximate => FrameTaint::Approximate {
                report_digest: sha256(&spec.tensor_seed.to_le_bytes()),
            },
        };
        ActivationFrameV2 {
            tokens,
            width,
            dtype: BoundaryDtype::F32,
            positions: (0..tokens).collect(),
            upstream_chain: UpstreamChain {
                stage: self.element(STAGE),
                base_model: self.element(BASE),
                adapters: ADAPTERS.iter().map(|id| self.element(id)).collect(),
                experts: expert_ids().iter().map(|id| self.element(id)).collect(),
            },
            input_lineage: InputLineage::canonical(inputs, spec.raw.clone()),
            mode: spec.mode,
            taint,
            placement_epoch: self.placement_epoch(),
            tensor,
        }
    }

    /// Commit the revocation of the current generation of `identity`.
    pub fn revoke(&mut self, identity: &str) -> Res<()> {
        // An expert is revoked through the registry: a committed `Retire`.
        if let Some(index) = expert_index(identity) {
            return experts::retire(&mut self.runtime, EXPERT_LAYER, index);
        }
        let generation = self.generation(identity);
        lab::commit_revocation(&mut self.runtime, &ledger_target(identity), generation)
    }

    /// Commit the supersession of the current generation of `identity` by the next; frames
    /// built afterwards carry the new generation.
    pub fn supersede(&mut self, identity: &str) -> Res<()> {
        // An expert is superseded through the registry: a committed next generation of the pod.
        if let Some(index) = expert_index(identity) {
            let next = experts::supersede(&mut self.runtime, EXPERT_LAYER, index)?;
            self.generations.insert(identity.to_string(), next);
            return Ok(());
        }
        let old = self.generation(identity);
        lab::commit_supersession(&mut self.runtime, &ledger_target(identity), old, old + 1)?;
        self.generations.insert(identity.to_string(), old + 1);
        let digest = digest_of(identity, old + 1);
        match input_kind(identity) {
            Some(kind) => {
                self.input_digests
                    .insert((kind, identity.to_string(), old + 1), digest);
            }
            None => {
                self.manifests
                    .insert((identity.to_string(), old + 1), digest);
            }
        }
        Ok(())
    }

    /// Fence the expert `identity` through the real authority path (`fence_expert` under the
    /// revocation authority's grant): it goes to `Fenced` at an unchanged generation and the
    /// next fence epoch, and serves nowhere until it is readmitted.
    pub fn fence(&mut self, identity: &str) -> Res<()> {
        let index = expert_index(identity).ok_or_else(|| format!("{identity} is no expert"))?;
        experts::fence(
            &mut self.runtime,
            EXPERT_LAYER,
            index,
            ExpertFenceCause::AuthorityQuarantine,
            None,
        )
    }

    /// Readmit the fenced expert `identity` through the gate authority path: it serves again
    /// at the next epoch, which no frame computed before the fence carries.
    pub fn readmit(&mut self, identity: &str) -> Res<()> {
        let index = expert_index(identity).ok_or_else(|| format!("{identity} is no expert"))?;
        experts::readmit(
            &mut self.runtime,
            EXPERT_LAYER,
            index,
            ExpertFenceCause::AuthorityQuarantine,
        )
    }

    /// Whether a kind/identity names a dependency this world holds.
    #[cfg(test)]
    pub fn holds(&self, identity: &str) -> bool {
        self.generations.contains_key(identity)
    }
}

fn expert_index(identity: &str) -> Option<usize> {
    expert_ids().iter().position(|id| id == identity)
}

/// The ordered Raw ancestry the seed gives a case: `count` distinct segment digests.
pub fn raw_ancestry(rng: &mut Rng, count: usize) -> Vec<[u8; 32]> {
    (0..count).map(|_| rng.digest()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_protocol::activation::bijection_violations;

    #[test]
    fn the_world_commits_every_dependency_and_places_the_stage() {
        let world = World::new().unwrap();
        let live = world.live();
        for kind in DependencyKind::ALL {
            for identity in targets(kind) {
                assert!(world.holds(&identity));
                let validity = match input_kind(&identity) {
                    Some(kind) => InputLiveness::validity(&live, kind, &identity, Generation(1)),
                    None => StageLiveness::validity(&live, &identity, Generation(1)),
                };
                assert_eq!(validity, Some(Validity::Live), "{identity}");
            }
        }
        assert!(world.placement_epoch() > 0);
        let expert = &expert_ids()[0];
        assert_eq!(world.fence_epoch(expert), 1);
        assert_eq!(live.fence_epoch(expert), Some(1));
    }

    #[test]
    fn a_frame_holds_an_element_of_every_kind_and_the_world_moves_what_it_says() {
        let mut world = World::new().unwrap();
        let frame = world.frame(&FrameSpec::exact(vec![[1; 32]], 5));
        assert_eq!(bijection_violations(&[&frame]), 0);
        assert_eq!(frame.elements().len(), 2 + 2 + 2 + 6);
        assert_eq!(frame, world.frame(&FrameSpec::exact(vec![[1; 32]], 5)));
        world.supersede("sem-1").unwrap();
        let next = world.frame(&FrameSpec::exact(vec![[1; 32]], 5));
        assert_ne!(next.digest(), frame.digest());
        assert_eq!(
            InputLiveness::validity(&world.live(), LineageKind::Semantic, "sem-1", Generation(1)),
            Some(Validity::Superseded)
        );
        let expert = expert_ids()[1].clone();
        world.fence(&expert).unwrap();
        assert_eq!(world.fence_epoch(&expert), 2);
        assert_eq!(
            world.live().fence_epoch(&expert),
            None,
            "fenced serves nowhere"
        );
        world.readmit(&expert).unwrap();
        assert_eq!(world.fence_epoch(&expert), 3);
        assert_eq!(world.live().fence_epoch(&expert), Some(3));
        world.revoke("adapter-1").unwrap();
        assert_eq!(
            StageLiveness::validity(&world.live(), "adapter-1", Generation(1)),
            Some(Validity::Revoked)
        );
    }
}
