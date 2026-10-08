//! The expert dependency index: which experts rest on which input generation, which experts
//! descend from which, the content digest the first committed use fixed for each input
//! generation, and the invalidations (revocations and supersession floors) committed so far.
//!
//! It is a projection of committed history, rebuilt by applying version 3 structural records
//! and `InputInvalidated` events in order, live and on replay. It changes nothing by itself:
//! `ExpertRegistryProjection` consults it to admit a record, to answer a route probe and to derive
//! the fence set of an invalidation.
//!
//! # What it holds
//!
//! * `by_expert`: for every expert generation created by a version 3 record, the inputs it rests
//!   on: its training list, its evaluation list and its base model, each as `(identity,
//!   generation)` (the content digests live in `digests`), and the canonical digest of the
//!   record's [`ptr_pods::ExpertInputs`] block as the record committed it. The union of the three
//!   parts, de-duplicated and ascending, is the expert's dependency set. Experts created by
//!   version 1 and 2 records name no inputs and have no entry. A snapshot restore recomputes the
//!   block digest from the lists and the registered digests and refuses a disagreement.
//! * `by_input`: the same relation inverted, input generation to experts. Derived, never stored.
//! * `children`: source expert generation to the result generations of a `Merge`, `Split` or
//!   `Reactivate` that consumed it. An edge is recorded only when its source is tracked (it has
//!   a dependency block or is itself a recorded descendant), because only a tracked expert can be
//!   reached from an invalidated input; an operation over untracked experts therefore adds
//!   nothing and a history without version 3 records keeps an empty index. `parents` is the same
//!   relation inverted (derived, never stored): an expert's liveness is judged over its own
//!   inputs and those of every ancestor (see [`ExpertDependencyIndex::first_dead_input`]).
//! * `digests`: the content digest fixed for an `(identity, generation)`. The first committed use
//!   (a version 3 record naming it) fixes it; only a committed `Superseded` invalidation moves a
//!   generation's digest forward, by fixing the digest of its new generation.
//! * `revoked` and `floors`: the `(identity, generation)` pairs revoked and, per identity, the
//!   lowest generation not yet superseded, by committed `InputInvalidated` events.

use ptr_types::Generation;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// An expert generation: `(project, pod, generation)`.
pub(crate) type ExpertRef = (String, String, Generation);

/// An input generation: `(identity, generation)`.
pub(crate) type InputKey = (String, Generation);

/// One dependency a record contributes for the experts it creates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RecordedInput {
    pub(crate) identity: String,
    pub(crate) generation: Generation,
    pub(crate) digest: [u8; 32],
}

/// The dependency block of a version 3 record as the index takes it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RecordedBlock {
    pub(crate) training: Vec<RecordedInput>,
    pub(crate) evaluation: Vec<RecordedInput>,
    /// The base model, derived from the record's operation.
    pub(crate) base: RecordedInput,
    /// `ExpertInputs::canonical_digest` of the block the record carried.
    pub(crate) commitment: [u8; 32],
}

/// What applying one structural record changes in the index.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct DependencyUpdate {
    /// The expert generations the record creates.
    pub(crate) results: Vec<ExpertRef>,
    /// The expert generations the record consumes (only meaningful when it creates results).
    pub(crate) sources: Vec<ExpertRef>,
    /// The inputs of a version 3 record; `None` for version 1 and 2.
    pub(crate) inputs: Option<RecordedBlock>,
}

/// What an expert generation rests on, as its version 3 record named it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExpertInputSet {
    pub(crate) training: Vec<InputKey>,
    pub(crate) evaluation: Vec<InputKey>,
    pub(crate) base: InputKey,
    pub(crate) commitment: [u8; 32],
    /// The union of the three parts, ascending and de-duplicated. Derived.
    pairs: Vec<InputKey>,
}

impl ExpertInputSet {
    pub(crate) fn new(
        training: Vec<InputKey>,
        evaluation: Vec<InputKey>,
        base: InputKey,
        commitment: [u8; 32],
    ) -> Self {
        let mut pairs: Vec<InputKey> = training
            .iter()
            .chain(&evaluation)
            .chain(std::iter::once(&base))
            .cloned()
            .collect();
        pairs.sort();
        pairs.dedup();
        Self {
            training,
            evaluation,
            base,
            commitment,
            pairs,
        }
    }

    pub(crate) fn pairs(&self) -> &[InputKey] {
        &self.pairs
    }
}

/// Why an input of an expert or of one of its ancestors is not live: the input, why, and whether
/// the expert itself or an ancestor named it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DeadInput {
    pub(crate) identity: String,
    pub(crate) generation: Generation,
    pub(crate) status: InputStatus,
    pub(crate) inherited: bool,
}

/// Why an input generation is not live, as the index knows it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InputStatus {
    Revoked,
    /// Below the supersession floor of its identity.
    Superseded {
        floor: Generation,
    },
}

/// The dependency index. See the module documentation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExpertDependencyIndex {
    by_expert: BTreeMap<ExpertRef, ExpertInputSet>,
    by_input: BTreeMap<InputKey, BTreeSet<ExpertRef>>,
    children: BTreeMap<ExpertRef, BTreeSet<ExpertRef>>,
    parents: BTreeMap<ExpertRef, BTreeSet<ExpertRef>>,
    descendants: BTreeSet<ExpertRef>,
    digests: BTreeMap<InputKey, [u8; 32]>,
    revoked: BTreeSet<InputKey>,
    floors: BTreeMap<String, Generation>,
}

impl ExpertDependencyIndex {
    /// Whether the index holds nothing at all.
    pub fn is_empty(&self) -> bool {
        self.by_expert.is_empty()
            && self.children.is_empty()
            && self.digests.is_empty()
            && self.revoked.is_empty()
            && self.floors.is_empty()
    }

    /// The `(identity, generation)` pairs the expert `pod` of `project` at `generation` rests on,
    /// ascending, or `None` when its creating record named no inputs (a version 1 or 2 record).
    pub fn recorded_inputs(
        &self,
        project: &str,
        pod: &str,
        generation: Generation,
    ) -> Option<impl Iterator<Item = (&str, Generation)>> {
        self.by_expert
            .get(&(project.to_owned(), pod.to_owned(), generation))
            .map(|set| {
                set.pairs()
                    .iter()
                    .map(|(id, generation)| (id.as_str(), *generation))
            })
    }

    /// Every `(identity, generation)` the expert generation `node` or any ancestor (through the
    /// source edges of `Merge`, `Split` and `Reactivate`) rests on, ascending. Empty when neither
    /// it nor an ancestor is tracked.
    pub fn lineage_inputs(
        &self,
        project: &str,
        pod: &str,
        generation: Generation,
    ) -> BTreeSet<(String, Generation)> {
        let start = (project.to_owned(), pod.to_owned(), generation);
        let mut seen: BTreeSet<ExpertRef> = BTreeSet::new();
        let mut queue = VecDeque::from([start]);
        let mut inputs = BTreeSet::new();
        while let Some(node) = queue.pop_front() {
            if !seen.insert(node.clone()) {
                continue;
            }
            if let Some(set) = self.by_expert.get(&node) {
                inputs.extend(set.pairs().iter().cloned());
            }
            queue.extend(self.parents.get(&node).into_iter().flatten().cloned());
        }
        inputs
    }

    /// The experts whose own dependency block names `identity` at exactly `generation`, in
    /// `(project, pod, generation)` order. Descendants that do not name it themselves are not
    /// included; see `invalidate_input` for the closure.
    pub fn direct_dependents(
        &self,
        identity: &str,
        generation: Generation,
    ) -> impl Iterator<Item = (&str, &str, Generation)> {
        self.by_input
            .get(&(identity.to_owned(), generation))
            .into_iter()
            .flatten()
            .map(|(project, pod, generation)| (project.as_str(), pod.as_str(), *generation))
    }

    /// The content digest fixed for `identity` at `generation`, or `None` when no committed
    /// record named it and no supersession moved to it.
    pub fn registered_digest(&self, identity: &str, generation: Generation) -> Option<[u8; 32]> {
        self.digests
            .get(&(identity.to_owned(), generation))
            .copied()
    }

    /// Whether a committed `InputInvalidated` revoked `identity` at `generation`.
    pub fn is_revoked(&self, identity: &str, generation: Generation) -> bool {
        self.revoked.contains(&(identity.to_owned(), generation))
    }

    /// The lowest generation of `identity` not superseded by a committed `InputInvalidated`, or
    /// `None` when none superseded it.
    pub fn supersession_floor(&self, identity: &str) -> Option<Generation> {
        self.floors.get(identity).copied()
    }

    /// How many experts have a dependency block, and how many source edges are tracked.
    pub fn sizes(&self) -> (usize, usize) {
        (
            self.by_expert.len(),
            self.children.values().map(BTreeSet::len).sum(),
        )
    }

    /// Why `identity` at `generation` is not live as far as the index knows, or `None`. A
    /// revocation is reported before a supersession.
    pub(crate) fn status(&self, identity: &str, generation: Generation) -> Option<InputStatus> {
        if self.is_revoked(identity, generation) {
            return Some(InputStatus::Revoked);
        }
        match self.floors.get(identity) {
            Some(floor) if generation < *floor => Some(InputStatus::Superseded { floor: *floor }),
            _ => None,
        }
    }

    /// Record what a committed structural record contributes.
    pub(crate) fn register(&mut self, update: &DependencyUpdate) {
        if update.results.is_empty() {
            return;
        }
        for source in &update.sources {
            if self.by_expert.contains_key(source) || self.descendants.contains(source) {
                self.children
                    .entry(source.clone())
                    .or_default()
                    .extend(update.results.iter().cloned());
                for result in &update.results {
                    self.parents
                        .entry(result.clone())
                        .or_default()
                        .insert(source.clone());
                }
                self.descendants.extend(update.results.iter().cloned());
            }
        }
        let Some(block) = &update.inputs else {
            return;
        };
        let key = |input: &RecordedInput| (input.identity.clone(), input.generation);
        for input in block
            .training
            .iter()
            .chain(&block.evaluation)
            .chain(std::iter::once(&block.base))
        {
            self.digests.entry(key(input)).or_insert(input.digest);
        }
        let set = ExpertInputSet::new(
            block.training.iter().map(key).collect(),
            block.evaluation.iter().map(key).collect(),
            key(&block.base),
            block.commitment,
        );
        for result in &update.results {
            for pair in set.pairs() {
                self.by_input
                    .entry(pair.clone())
                    .or_default()
                    .insert(result.clone());
            }
            self.by_expert.insert(result.clone(), set.clone());
        }
    }

    /// Record a committed revocation of `identity` at `generation`.
    pub(crate) fn record_revocation(&mut self, identity: &str, generation: Generation) {
        self.revoked.insert((identity.to_owned(), generation));
    }

    /// Record a committed supersession: the floor of `identity` moves up to `new_generation` and
    /// that generation's digest is fixed to `new_digest`.
    pub(crate) fn record_supersession(
        &mut self,
        identity: &str,
        new_generation: Generation,
        new_digest: [u8; 32],
    ) {
        let floor = self
            .floors
            .entry(identity.to_owned())
            .or_insert(new_generation);
        *floor = (*floor).max(new_generation);
        self.digests
            .insert((identity.to_owned(), new_generation), new_digest);
    }

    /// Every expert generation the invalidation reaches: those whose block names the input
    /// (`Revoked`: exactly at `generation`; `Superseded`: at any generation below
    /// `superseded_by`), closed over the source edges, so a descendant of an affected expert is
    /// affected. For a supersession a descendant is not followed when its own block names the
    /// input at `superseded_by` or above: it was retrained on the replacement.
    pub(crate) fn affected(
        &self,
        identity: &str,
        generation: Generation,
        superseded_by: Option<Generation>,
    ) -> BTreeSet<ExpertRef> {
        let seeds: BTreeSet<ExpertRef> = match superseded_by {
            None => self
                .by_input
                .get(&(identity.to_owned(), generation))
                .cloned()
                .unwrap_or_default(),
            Some(new_generation) => self
                .by_input
                .range((identity.to_owned(), Generation(0))..(identity.to_owned(), new_generation))
                .flat_map(|(_, experts)| experts.iter().cloned())
                .collect(),
        };
        let retrained = |node: &ExpertRef| {
            superseded_by.is_some_and(|new_generation| {
                self.by_expert.get(node).is_some_and(|set| {
                    set.pairs()
                        .iter()
                        .any(|(id, g)| id == identity && *g >= new_generation)
                })
            })
        };
        let mut affected = seeds.clone();
        let mut queue: VecDeque<ExpertRef> = seeds.into_iter().collect();
        while let Some(node) = queue.pop_front() {
            for child in self.children.get(&node).into_iter().flatten() {
                if affected.contains(child) || retrained(child) {
                    continue;
                }
                affected.insert(child.clone());
                queue.push_back(child.clone());
            }
        }
        affected
    }

    /// The first input that is not live among those of an expert and of its ancestors.
    ///
    /// The expert is described by its own dependency `own` and its direct `sources` (the experts
    /// a `Merge`, `Split` or `Reactivate` consumed), so an expert that is about to be created can
    /// be judged the same way as one that exists. `status` says why an `(identity, generation)`
    /// is not live, or `None` when it is.
    ///
    /// A revoked input anywhere in the ancestry is dead, with no exception: revoked data is not
    /// laundered by a retraining. A superseded input is dead unless every path from the expert
    /// that named it down to the judged expert crosses an expert whose own dependency names the
    /// same identity at the supersession floor or above (it was retrained on the replacement),
    /// which is the exemption `affected` applies when it derives a fence set, so the probe and
    /// the fence set agree. The first dead input in breadth-first order (own inputs first, each
    /// node's inputs ascending) is returned.
    pub(crate) fn first_dead_input(
        &self,
        own: &[InputKey],
        sources: &[ExpertRef],
        status: &dyn Fn(&str, Generation) -> Option<InputStatus>,
    ) -> Option<DeadInput> {
        let mut order: Vec<&[InputKey]> = vec![own];
        let mut seen: BTreeSet<&ExpertRef> = BTreeSet::new();
        let mut queue: VecDeque<&ExpertRef> = sources.iter().collect();
        while let Some(node) = queue.pop_front() {
            if !seen.insert(node) {
                continue;
            }
            order.push(self.node_pairs(node));
            queue.extend(self.parents.get(node).into_iter().flatten());
        }
        for (position, inputs) in order.iter().enumerate() {
            for (identity, generation) in *inputs {
                let found = |status| DeadInput {
                    identity: identity.clone(),
                    generation: *generation,
                    status,
                    inherited: position > 0,
                };
                match status(identity, *generation) {
                    None => {}
                    Some(InputStatus::Revoked) => return Some(found(InputStatus::Revoked)),
                    Some(InputStatus::Superseded { floor })
                        if self.superseded_reaches(own, sources, identity, floor, status) =>
                    {
                        return Some(found(InputStatus::Superseded { floor }));
                    }
                    Some(InputStatus::Superseded { .. }) => {}
                }
            }
        }
        None
    }

    fn node_pairs(&self, node: &ExpertRef) -> &[InputKey] {
        self.by_expert.get(node).map_or(&[], |set| set.pairs())
    }

    /// Whether some expert in the ancestry (or the described expert itself) names `identity`
    /// below `floor` and a path from it down to the described expert crosses no expert that
    /// names `identity` at `floor` or above.
    fn superseded_reaches(
        &self,
        own: &[InputKey],
        sources: &[ExpertRef],
        identity: &str,
        floor: Generation,
        status: &dyn Fn(&str, Generation) -> Option<InputStatus>,
    ) -> bool {
        // Walk from the described expert towards its ancestors; a node that names the identity
        // at the floor or above is a retrained successor and blocks the path through it.
        let classify = |pairs: &[InputKey]| {
            let seed = pairs.iter().any(|(id, generation)| {
                id == identity
                    && matches!(
                        status(id, *generation),
                        Some(InputStatus::Superseded { .. })
                    )
            });
            let retrained = pairs
                .iter()
                .any(|(id, generation)| id == identity && *generation >= floor);
            (seed, retrained)
        };
        let (seed, retrained) = classify(own);
        if seed {
            return true;
        }
        if retrained {
            return false;
        }
        let mut seen: BTreeSet<&ExpertRef> = BTreeSet::new();
        let mut queue: VecDeque<&ExpertRef> = sources.iter().collect();
        while let Some(node) = queue.pop_front() {
            if !seen.insert(node) {
                continue;
            }
            let (seed, retrained) = classify(self.node_pairs(node));
            if seed {
                return true;
            }
            if !retrained {
                queue.extend(self.parents.get(node).into_iter().flatten());
            }
        }
        false
    }

    /// [`ExpertDependencyIndex::first_dead_input`] for the recorded expert generation `node`.
    pub(crate) fn first_dead_input_of(
        &self,
        node: &ExpertRef,
        status: &dyn Fn(&str, Generation) -> Option<InputStatus>,
    ) -> Option<DeadInput> {
        let sources: Vec<ExpertRef> = self
            .parents
            .get(node)
            .into_iter()
            .flatten()
            .cloned()
            .collect();
        self.first_dead_input(self.node_pairs(node), &sources, status)
    }

    // ---- snapshot support -------------------------------------------------------------------

    pub(crate) fn expert_entries(&self) -> impl Iterator<Item = (&ExpertRef, &ExpertInputSet)> {
        self.by_expert.iter()
    }

    pub(crate) fn edge_entries(&self) -> impl Iterator<Item = (&ExpertRef, &ExpertRef)> {
        self.children
            .iter()
            .flat_map(|(source, results)| results.iter().map(move |result| (source, result)))
    }

    pub(crate) fn digest_entries(&self) -> impl Iterator<Item = (&InputKey, &[u8; 32])> {
        self.digests.iter()
    }

    pub(crate) fn revoked_entries(&self) -> impl Iterator<Item = &InputKey> {
        self.revoked.iter()
    }

    pub(crate) fn floor_entries(&self) -> impl Iterator<Item = (&str, Generation)> {
        self.floors
            .iter()
            .map(|(identity, floor)| (identity.as_str(), *floor))
    }

    /// Rebuild an index from snapshot parts; the derived relations (`by_input`, `parents`, `descendants`)
    /// are recomputed, so a snapshot cannot carry them inconsistently.
    pub(crate) fn from_parts(
        by_expert: BTreeMap<ExpertRef, ExpertInputSet>,
        edges: Vec<(ExpertRef, ExpertRef)>,
        digests: BTreeMap<InputKey, [u8; 32]>,
        revoked: BTreeSet<InputKey>,
        floors: BTreeMap<String, Generation>,
    ) -> Self {
        let mut by_input: BTreeMap<InputKey, BTreeSet<ExpertRef>> = BTreeMap::new();
        for (expert, set) in &by_expert {
            for input in set.pairs() {
                by_input
                    .entry(input.clone())
                    .or_default()
                    .insert(expert.clone());
            }
        }
        let mut children: BTreeMap<ExpertRef, BTreeSet<ExpertRef>> = BTreeMap::new();
        let mut parents: BTreeMap<ExpertRef, BTreeSet<ExpertRef>> = BTreeMap::new();
        let mut descendants = BTreeSet::new();
        for (source, result) in edges {
            descendants.insert(result.clone());
            parents
                .entry(result.clone())
                .or_default()
                .insert(source.clone());
            children.entry(source).or_default().insert(result);
        }
        Self {
            by_expert,
            by_input,
            children,
            parents,
            descendants,
            digests,
            revoked,
            floors,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(pod: &str) -> ExpertRef {
        ("p".to_owned(), pod.to_owned(), Generation(1))
    }

    fn pair(identity: &str, generation: u64) -> InputKey {
        (identity.to_owned(), Generation(generation))
    }

    fn recorded(identity: &str, generation: u64) -> RecordedInput {
        RecordedInput {
            identity: identity.to_owned(),
            generation: Generation(generation),
            digest: [generation as u8 + 1; 32],
        }
    }

    /// Register `pod` as a version 3 record resting on `inputs`, consuming `sources`.
    fn register(
        index: &mut ExpertDependencyIndex,
        pod: &str,
        sources: &[&str],
        inputs: &[(&str, u64)],
    ) {
        index.register(&DependencyUpdate {
            results: vec![node(pod)],
            sources: sources.iter().map(|s| node(s)).collect(),
            inputs: Some(RecordedBlock {
                training: inputs.iter().map(|(i, g)| recorded(i, *g)).collect(),
                evaluation: vec![],
                base: recorded("base-model:b", 1),
                commitment: [1; 32],
            }),
        });
    }

    fn superseded_below(floor: u64) -> impl Fn(&str, Generation) -> Option<InputStatus> {
        move |identity, generation| {
            (identity == "data" && generation.0 < floor).then_some(InputStatus::Superseded {
                floor: Generation(floor),
            })
        }
    }

    #[test]
    fn a_revoked_input_anywhere_in_the_ancestry_is_dead_without_exemption() {
        let mut index = ExpertDependencyIndex::default();
        register(&mut index, "a", &[], &[("data", 1)]);
        register(&mut index, "b", &["a"], &[("data", 2)]);
        let revoked = |identity: &str, generation: Generation| {
            (identity == "data" && generation.0 == 1).then_some(InputStatus::Revoked)
        };
        // b names the newer generation itself, which does not launder the ancestor's revoked one.
        let dead = index.first_dead_input_of(&node("b"), &revoked).unwrap();
        assert_eq!(
            (dead.identity.as_str(), dead.generation, dead.inherited),
            ("data", Generation(1), true)
        );
        assert_eq!(dead.status, InputStatus::Revoked);
        // The ancestor itself judges its own input as not inherited.
        assert!(
            !index
                .first_dead_input_of(&node("a"), &revoked)
                .unwrap()
                .inherited
        );
    }

    #[test]
    fn a_retrained_successor_is_exempt_from_a_superseded_ancestor_input_only_on_its_own_path() {
        let mut index = ExpertDependencyIndex::default();
        register(&mut index, "a", &[], &[("data", 1)]);
        // b retrained on data@2, c did not.
        register(&mut index, "b", &["a"], &[("data", 2)]);
        register(&mut index, "c", &["a"], &[("other", 1)]);
        let status = superseded_below(2);
        assert_eq!(index.first_dead_input_of(&node("b"), &status), None);
        let dead = index.first_dead_input_of(&node("c"), &status).unwrap();
        assert_eq!((dead.identity.as_str(), dead.inherited), ("data", true));
        // d merges both: the path through c is not retrained, so d is dead.
        register(&mut index, "d", &["b", "c"], &[("third", 1)]);
        assert!(index.first_dead_input_of(&node("d"), &status).is_some());
        // e descends only from b: the retrained successor blocks the path to a.
        register(&mut index, "e", &["b"], &[("third", 1)]);
        assert_eq!(index.first_dead_input_of(&node("e"), &status), None);
        // The fence-set derivation agrees: superseding data@1 by data@2 reaches a, c and d, not
        // b or e.
        let affected = index.affected("data", Generation(1), Some(Generation(2)));
        assert_eq!(affected, BTreeSet::from([node("a"), node("c"), node("d")]));
    }

    #[test]
    fn an_expert_about_to_be_created_is_judged_like_one_that_exists() {
        let mut index = ExpertDependencyIndex::default();
        register(&mut index, "a", &[], &[("data", 1)]);
        let status = superseded_below(2);
        // Without its own block it inherits the dead input; naming the replacement exempts it.
        let own: Vec<InputKey> = vec![];
        assert!(index
            .first_dead_input(&own, &[node("a")], &status)
            .is_some());
        let retrained = vec![pair("data", 2)];
        assert_eq!(
            index.first_dead_input(&retrained, &[node("a")], &status),
            None
        );
        // An untracked source brings nothing.
        assert_eq!(
            index.first_dead_input(&own, &[node("ghost")], &status),
            None
        );
    }

    #[test]
    fn lineage_inputs_are_the_union_over_the_ancestry() {
        let mut index = ExpertDependencyIndex::default();
        register(&mut index, "a", &[], &[("data", 1)]);
        register(&mut index, "b", &["a"], &[("other", 1)]);
        let inputs = index.lineage_inputs("p", "b", Generation(1));
        assert!(inputs.contains(&pair("data", 1)));
        assert!(inputs.contains(&pair("other", 1)));
        assert!(inputs.contains(&pair("base-model:b", 1)));
        assert!(index.lineage_inputs("p", "ghost", Generation(1)).is_empty());
    }
}
