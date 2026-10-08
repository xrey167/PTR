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
//! * `by_expert`: for every expert generation created by a version 3 record, the
//!   `(identity, generation)` pairs it rests on (training inputs, evaluation inputs and the base
//!   model, de-duplicated and ascending). Experts created by version 1 and 2 records name no
//!   inputs and have no entry.
//! * `by_input`: the same relation inverted, input generation to experts. Derived, never stored.
//! * `children`: source expert generation to the result generations of a `Merge`, `Split` or
//!   `Reactivate` that consumed it. An edge is recorded only when its source is tracked (it has
//!   a dependency block or is itself a recorded descendant), because only a tracked expert can be
//!   reached from an invalidated input; an operation over untracked experts therefore adds
//!   nothing and a history without version 3 records keeps an empty index.
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

/// What applying one structural record changes in the index.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct DependencyUpdate {
    /// The expert generations the record creates.
    pub(crate) results: Vec<ExpertRef>,
    /// The expert generations the record consumes (only meaningful when it creates results).
    pub(crate) sources: Vec<ExpertRef>,
    /// The inputs of a version 3 record; `None` for version 1 and 2.
    pub(crate) inputs: Option<Vec<RecordedInput>>,
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
    by_expert: BTreeMap<ExpertRef, Vec<InputKey>>,
    by_input: BTreeMap<InputKey, BTreeSet<ExpertRef>>,
    children: BTreeMap<ExpertRef, BTreeSet<ExpertRef>>,
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
            .map(|inputs| {
                inputs
                    .iter()
                    .map(|(id, generation)| (id.as_str(), *generation))
            })
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
                self.descendants.extend(update.results.iter().cloned());
            }
        }
        let Some(inputs) = &update.inputs else {
            return;
        };
        let mut pairs: Vec<InputKey> = Vec::with_capacity(inputs.len());
        for input in inputs {
            self.digests
                .entry((input.identity.clone(), input.generation))
                .or_insert(input.digest);
            pairs.push((input.identity.clone(), input.generation));
        }
        pairs.sort();
        pairs.dedup();
        for result in &update.results {
            for pair in &pairs {
                self.by_input
                    .entry(pair.clone())
                    .or_default()
                    .insert(result.clone());
            }
            self.by_expert.insert(result.clone(), pairs.clone());
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
                self.by_expert.get(node).is_some_and(|inputs| {
                    inputs
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

    // ---- snapshot support -------------------------------------------------------------------

    pub(crate) fn expert_entries(&self) -> impl Iterator<Item = (&ExpertRef, &[InputKey])> {
        self.by_expert
            .iter()
            .map(|(key, inputs)| (key, inputs.as_slice()))
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

    /// Rebuild an index from snapshot parts; the derived relations (`by_input`, `descendants`)
    /// are recomputed, so a snapshot cannot carry them inconsistently.
    pub(crate) fn from_parts(
        by_expert: BTreeMap<ExpertRef, Vec<InputKey>>,
        edges: Vec<(ExpertRef, ExpertRef)>,
        digests: BTreeMap<InputKey, [u8; 32]>,
        revoked: BTreeSet<InputKey>,
        floors: BTreeMap<String, Generation>,
    ) -> Self {
        let mut by_input: BTreeMap<InputKey, BTreeSet<ExpertRef>> = BTreeMap::new();
        for (expert, inputs) in &by_expert {
            for input in inputs {
                by_input
                    .entry(input.clone())
                    .or_default()
                    .insert(expert.clone());
            }
        }
        let mut children: BTreeMap<ExpertRef, BTreeSet<ExpertRef>> = BTreeMap::new();
        let mut descendants = BTreeSet::new();
        for (source, result) in edges {
            descendants.insert(result.clone());
            children.entry(source).or_default().insert(result);
        }
        Self {
            by_expert,
            by_input,
            children,
            descendants,
            digests,
            revoked,
            floors,
        }
    }
}
