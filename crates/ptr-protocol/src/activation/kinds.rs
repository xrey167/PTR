//! The closed list of dependency kinds of the invalidation coverage matrix (doc 36, section E)
//! and the checker that they are in bijection with the element slots of a version 2 frame.
//!
//! The matrix covers every kind a cached split-pod object can depend on: the upstream stage
//! generation, the semantic, memory and derived inputs of the frame's `input_lineage`, and the
//! base model, adapters and experts of its `upstream_chain`. Each element of a frame sits in
//! exactly one [`FrameSlot`], each slot has exactly one [`DependencyKind`] and each kind exactly
//! one slot, so a frame element with no matrix row, or a matrix row no frame element can
//! reach, is a violation the checker counts. Adding a frame slot or a kind without the other
//! side is a compile error (every `match` here is exhaustive) and, in a mapping that compiles,
//! a counted violation.

use super::lineage::{ActivationFrameV2, LineageKind};
use super::{ChainElement, UpstreamChain};

/// The kinds of generation-bearing dependency of a split-pod cache entry: the closed list.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DependencyKind {
    UpstreamStageGeneration,
    SemanticInput,
    MemoryInput,
    DerivedInput,
    UpstreamExpert,
    UpstreamAdapter,
    UpstreamBaseModel,
}

/// How a dependency is invalidated.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Invalidation {
    /// An explicit revocation of the generation.
    Revocation,
    /// A committed supersession (a generation replacement or an edit).
    Supersession,
    /// A fence of an expert, for any cause: the generation does not change but the serving
    /// status does (doc 36, "A fence is an invalidating event for KV too").
    Fence,
}

impl Invalidation {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Revocation => "revocation",
            Self::Supersession => "supersession",
            Self::Fence => "fence",
        }
    }
}

const REVOCATION_AND_SUPERSESSION: [Invalidation; 2] =
    [Invalidation::Revocation, Invalidation::Supersession];
const EXPERT_INVALIDATIONS: [Invalidation; 3] = [
    Invalidation::Revocation,
    Invalidation::Supersession,
    Invalidation::Fence,
];

impl DependencyKind {
    /// Every kind, in a fixed order.
    pub const ALL: [Self; 7] = [
        Self::UpstreamStageGeneration,
        Self::SemanticInput,
        Self::MemoryInput,
        Self::DerivedInput,
        Self::UpstreamExpert,
        Self::UpstreamAdapter,
        Self::UpstreamBaseModel,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::UpstreamStageGeneration => "upstream_stage_generation",
            Self::SemanticInput => "semantic_input",
            Self::MemoryInput => "memory_input",
            Self::DerivedInput => "derived_input",
            Self::UpstreamExpert => "upstream_expert",
            Self::UpstreamAdapter => "upstream_adapter",
            Self::UpstreamBaseModel => "upstream_base_model",
        }
    }

    /// The invalidation types the matrix covers for this kind: revocation and supersession
    /// for every kind and, for an expert only, a fence.
    pub const fn invalidations(self) -> &'static [Invalidation] {
        match self {
            Self::UpstreamExpert => &EXPERT_INVALIDATIONS,
            Self::UpstreamStageGeneration
            | Self::SemanticInput
            | Self::MemoryInput
            | Self::DerivedInput
            | Self::UpstreamAdapter
            | Self::UpstreamBaseModel => &REVOCATION_AND_SUPERSESSION,
        }
    }

    /// The one slot of a frame that holds elements of this kind.
    pub const fn slot(self) -> FrameSlot {
        match self {
            Self::UpstreamStageGeneration => FrameSlot::ChainStage,
            Self::SemanticInput => FrameSlot::Lineage(LineageKind::Semantic),
            Self::MemoryInput => FrameSlot::Lineage(LineageKind::Memory),
            Self::DerivedInput => FrameSlot::Lineage(LineageKind::Derived),
            Self::UpstreamExpert => FrameSlot::ChainExpert,
            Self::UpstreamAdapter => FrameSlot::ChainAdapter,
            Self::UpstreamBaseModel => FrameSlot::ChainBaseModel,
        }
    }
}

/// Where in a version 2 frame a generation-bearing element sits. A Raw segment has no slot:
/// it carries no identity or generation and is a cache-key contribution, not a dependency.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum FrameSlot {
    ChainStage,
    ChainBaseModel,
    ChainAdapter,
    ChainExpert,
    Lineage(LineageKind),
}

impl FrameSlot {
    /// Every slot, in a fixed order.
    pub const ALL: [Self; 7] = [
        Self::ChainStage,
        Self::ChainBaseModel,
        Self::ChainAdapter,
        Self::ChainExpert,
        Self::Lineage(LineageKind::Semantic),
        Self::Lineage(LineageKind::Memory),
        Self::Lineage(LineageKind::Derived),
    ];

    /// The matrix kind of an element in this slot: the one function that maps a frame
    /// element to a kind.
    pub const fn kind(self) -> DependencyKind {
        match self {
            Self::ChainStage => DependencyKind::UpstreamStageGeneration,
            Self::ChainBaseModel => DependencyKind::UpstreamBaseModel,
            Self::ChainAdapter => DependencyKind::UpstreamAdapter,
            Self::ChainExpert => DependencyKind::UpstreamExpert,
            Self::Lineage(LineageKind::Semantic) => DependencyKind::SemanticInput,
            Self::Lineage(LineageKind::Memory) => DependencyKind::MemoryInput,
            Self::Lineage(LineageKind::Derived) => DependencyKind::DerivedInput,
        }
    }
}

/// One generation-bearing element of a frame, with the slot it sits in.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameElement<'a> {
    pub slot: FrameSlot,
    pub identity: &'a str,
    pub generation: u64,
    pub digest: &'a [u8; 32],
    /// The fence epoch an expert was computed under (zero for every other slot).
    pub fence_epoch: u64,
}

impl FrameElement<'_> {
    /// The matrix kind of this element.
    pub const fn kind(&self) -> DependencyKind {
        self.slot.kind()
    }
}

fn chain_element(slot: FrameSlot, element: &ChainElement) -> FrameElement<'_> {
    FrameElement {
        slot,
        identity: &element.identity,
        generation: element.generation,
        digest: &element.digest,
        fence_epoch: element.fence_epoch,
    }
}

/// Every generation-bearing element of an upstream chain, each in its slot.
pub fn chain_elements(chain: &UpstreamChain) -> Vec<FrameElement<'_>> {
    let mut out = Vec::with_capacity(2 + chain.adapters.len() + chain.experts.len());
    out.push(chain_element(FrameSlot::ChainStage, &chain.stage));
    out.push(chain_element(FrameSlot::ChainBaseModel, &chain.base_model));
    out.extend(
        chain
            .adapters
            .iter()
            .map(|element| chain_element(FrameSlot::ChainAdapter, element)),
    );
    out.extend(
        chain
            .experts
            .iter()
            .map(|element| chain_element(FrameSlot::ChainExpert, element)),
    );
    out
}

impl ActivationFrameV2 {
    /// Every generation-bearing element of the frame (the upstream chain and the typed
    /// inputs of the lineage), each in its slot. The Raw ancestry is not among them.
    pub fn elements(&self) -> Vec<FrameElement<'_>> {
        let mut out = chain_elements(&self.upstream_chain);
        out.extend(self.input_lineage.inputs.iter().map(|input| FrameElement {
            slot: FrameSlot::Lineage(input.kind),
            identity: &input.identity,
            generation: input.generation,
            digest: &input.digest,
            fence_epoch: 0,
        }));
        out
    }
}

/// The violations of a slot/kind mapping pair: for every slot the number of kinds that name
/// it as their slot must be exactly one and its own kind must name it back, and the same for
/// every kind. A mapping that sends two slots to one kind, leaves a kind with no slot or does
/// not invert counts one per defect.
pub fn mapping_violations(
    kind_of_slot: impl Fn(FrameSlot) -> DependencyKind,
    slot_of_kind: impl Fn(DependencyKind) -> FrameSlot,
) -> u64 {
    let mut violations = 0;
    for slot in FrameSlot::ALL {
        let owners = DependencyKind::ALL
            .iter()
            .filter(|kind| slot_of_kind(**kind) == slot)
            .count();
        if owners != 1 {
            violations += 1;
        }
        if slot_of_kind(kind_of_slot(slot)) != slot {
            violations += 1;
        }
    }
    for kind in DependencyKind::ALL {
        let holders = FrameSlot::ALL
            .iter()
            .filter(|slot| kind_of_slot(**slot) == kind)
            .count();
        if holders != 1 {
            violations += 1;
        }
        if kind_of_slot(slot_of_kind(kind)) != kind {
            violations += 1;
        }
    }
    violations
}

/// The bijection check of the invalidation coverage matrix over `frames`: the slot/kind
/// mapping ([`mapping_violations`]) plus, for the frames given, every element mapped to
/// exactly one kind (the elements of a frame are all accounted for) and every one of the
/// seven kinds reached by some element (a kind no frame element reaches has a matrix row with
/// nothing to exercise it). The count is zero for a sound mapping and frames that between them
/// hold an element of every kind.
pub fn bijection_violations(frames: &[&ActivationFrameV2]) -> u64 {
    let mut violations = mapping_violations(FrameSlot::kind, DependencyKind::slot);
    let mut reached = [false; 7];
    for frame in frames {
        let elements = frame.elements();
        let expected = 2
            + frame.upstream_chain.adapters.len()
            + frame.upstream_chain.experts.len()
            + frame.input_lineage.inputs.len();
        if elements.len() != expected {
            violations += 1;
        }
        for element in &elements {
            let kind = element.kind();
            match DependencyKind::ALL.iter().position(|known| *known == kind) {
                Some(index) => reached[index] = true,
                None => violations += 1,
            }
        }
    }
    violations + reached.iter().filter(|hit| !**hit).count() as u64
}

#[cfg(test)]
mod tests {
    use super::super::lineage::{FrameMode, FrameTaint, InputLineage, LineageInput};
    use super::super::BoundaryDtype;
    use super::*;

    fn element(name: &str) -> ChainElement {
        ChainElement {
            identity: name.to_owned(),
            generation: 1,
            digest: [1; 32],
            fence_epoch: 0,
        }
    }

    fn input(kind: LineageKind, name: &str) -> LineageInput {
        LineageInput {
            kind,
            identity: name.to_owned(),
            generation: 1,
            digest: [2; 32],
        }
    }

    fn full_frame() -> ActivationFrameV2 {
        ActivationFrameV2 {
            tokens: 1,
            width: 1,
            dtype: BoundaryDtype::F32,
            positions: vec![0],
            upstream_chain: UpstreamChain {
                stage: element("stage"),
                base_model: element("model"),
                adapters: vec![element("adapter-1"), element("adapter-2")],
                experts: vec![ChainElement {
                    fence_epoch: 1,
                    ..element("expert")
                }],
            },
            input_lineage: InputLineage::canonical(
                vec![
                    input(LineageKind::Semantic, "sem"),
                    input(LineageKind::Memory, "mem"),
                    input(LineageKind::Derived, "drv"),
                ],
                vec![[7; 32]],
            ),
            mode: FrameMode::Exact,
            taint: FrameTaint::Exact,
            placement_epoch: 1,
            tensor: vec![0; 4],
        }
    }

    #[test]
    fn the_slots_and_the_kinds_are_in_bijection() {
        assert_eq!(mapping_violations(FrameSlot::kind, DependencyKind::slot), 0);
        for kind in DependencyKind::ALL {
            assert_eq!(kind.slot().kind(), kind);
        }
        for slot in FrameSlot::ALL {
            assert_eq!(slot.kind().slot(), slot);
        }
        let names: std::collections::BTreeSet<_> =
            DependencyKind::ALL.iter().map(|kind| kind.name()).collect();
        assert_eq!(names.len(), DependencyKind::ALL.len());
    }

    #[test]
    fn the_matrix_has_forty_five_kind_type_path_cells() {
        // Revocation and supersession for the seven kinds, plus the expert's fence: fifteen
        // kind x type pairs, times the three paths (normal, lag, in flight).
        let pairs: usize = DependencyKind::ALL
            .iter()
            .map(|kind| kind.invalidations().len())
            .sum();
        assert_eq!(pairs, 15);
        assert_eq!(pairs * 3, 45);
        for kind in DependencyKind::ALL {
            let fence = kind.invalidations().contains(&Invalidation::Fence);
            assert_eq!(fence, kind == DependencyKind::UpstreamExpert);
        }
    }

    #[test]
    fn every_element_of_a_full_frame_has_exactly_one_kind_and_none_is_left_out() {
        let frame = full_frame();
        let elements = frame.elements();
        assert_eq!(elements.len(), 2 + 2 + 1 + 3);
        let mut seen = std::collections::BTreeMap::new();
        for element in &elements {
            *seen.entry(element.kind()).or_insert(0usize) += 1;
        }
        assert_eq!(seen.len(), 7);
        assert_eq!(seen[&DependencyKind::UpstreamAdapter], 2);
        assert_eq!(seen[&DependencyKind::UpstreamStageGeneration], 1);
        // The Raw ancestry is not a dependency element.
        assert!(elements.iter().all(|element| element.digest != &[7; 32]));
        assert_eq!(bijection_violations(&[&frame]), 0);
    }

    #[test]
    fn a_kind_no_frame_reaches_and_a_broken_mapping_are_counted() {
        let mut no_adapter = full_frame();
        no_adapter.upstream_chain.adapters.clear();
        assert_eq!(bijection_violations(&[&no_adapter]), 1);
        let mut no_inputs = full_frame();
        no_inputs.input_lineage.inputs.clear();
        assert_eq!(bijection_violations(&[&no_inputs]), 3);
        // Two frames that between them reach every kind pass.
        assert_eq!(bijection_violations(&[&no_adapter, &full_frame()]), 0);
        assert_eq!(bijection_violations(&[]), 7);
        // A mapping that sends two slots to one kind is caught by the table check.
        let broken = |slot: FrameSlot| match slot {
            FrameSlot::ChainAdapter => DependencyKind::UpstreamExpert,
            other => other.kind(),
        };
        assert!(mapping_violations(broken, DependencyKind::slot) > 0);
    }
}
