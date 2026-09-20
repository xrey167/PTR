//! Revisioned semantic values, dependency invalidation and immutable snapshots.

mod codec;

use ptr_types::{Revision, TypeId};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

pub use codec::{MAX_DELTA_BYTES, MAX_DELTA_ITEMS, MAX_KEY_BYTES};

/// Source identity and exact typed bytes; not a verifier proof or a capability.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticPayload {
    pub type_id: TypeId,
    pub source: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SemanticValue {
    Text(String),
    Payload(SemanticPayload),
}

impl From<String> for SemanticValue {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}
impl From<&str> for SemanticValue {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}
impl From<SemanticPayload> for SemanticValue {
    fn from(value: SemanticPayload) -> Self {
        Self::Payload(value)
    }
}

/// One atomic update. Dependency entries replace the complete input set for a
/// derived key; omitted entries preserve its previous dependency set.
/// Explicit upserts assert recomputation against the post-transaction inputs.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SemanticDelta {
    pub upserts: BTreeMap<String, SemanticValue>,
    pub removals: BTreeSet<String>,
    pub dependencies: BTreeMap<String, BTreeSet<String>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SemanticError {
    LimitExceeded,
    InvalidEncoding,
    InvalidKey,
    ConflictingOperation,
    CyclicDependency,
    MissingDependency {
        derived: String,
        input: String,
    },
    RevisionExhausted,
    StalePreparation,
    RevisionMismatch {
        expected: Revision,
        actual: Revision,
    },
    NoChangeRecord,
}
impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PTR_SEMANTIC_ERROR: {self:?}")
    }
}
impl std::error::Error for SemanticError {}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DependencyGraph {
    inputs: BTreeMap<String, BTreeSet<String>>,
}
impl DependencyGraph {
    pub fn depends_on(&mut self, derived: impl Into<String>, input: impl Into<String>) {
        self.inputs
            .entry(derived.into())
            .or_default()
            .insert(input.into());
    }
    pub fn affected_by<I, S>(&self, changed: I) -> BTreeSet<String>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut reverse: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (derived, inputs) in &self.inputs {
            for input in inputs {
                reverse.entry(input).or_default().push(derived);
            }
        }
        let mut out = BTreeSet::new();
        let mut stack: Vec<String> = changed.into_iter().map(Into::into).collect();
        while let Some(key) = stack.pop() {
            if out.insert(key.clone()) {
                if let Some(children) = reverse.get(key.as_str()) {
                    stack.extend(children.iter().map(|key| (*key).to_owned()));
                }
            }
        }
        out
    }
    fn acyclic(&self) -> bool {
        let mut degree: BTreeMap<&str, usize> = BTreeMap::new();
        let mut reverse: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (derived, inputs) in &self.inputs {
            degree.insert(derived, inputs.len());
            for input in inputs {
                reverse.entry(input).or_default().push(derived);
            }
        }
        for input in reverse.keys() {
            degree.entry(input).or_default();
        }
        let mut ready: Vec<&str> = degree
            .iter()
            .filter_map(|(key, n)| (*n == 0).then_some(*key))
            .collect();
        let mut visited = 0;
        while let Some(key) = ready.pop() {
            visited += 1;
            if let Some(children) = reverse.get(key) {
                for child in children {
                    let n = degree.get_mut(child).expect("graph node exists");
                    *n -= 1;
                    if *n == 0 {
                        ready.push(child);
                    }
                }
            }
        }
        visited == degree.len()
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct SemanticState {
    revision: Revision,
    ground: BTreeMap<String, SemanticValue>,
    dependencies: DependencyGraph,
}

/// Immutable historical view. Possessing it does not establish live admission.
#[derive(Clone, Debug)]
pub struct SemanticSnapshot {
    pub revision: Revision,
    state: Arc<SemanticState>,
    owner: Arc<()>,
}
impl SemanticSnapshot {
    pub fn get(&self, key: &str) -> Option<&str> {
        match self.value(key)? {
            SemanticValue::Text(text) => Some(text),
            _ => None,
        }
    }
    pub fn value(&self, key: &str) -> Option<&SemanticValue> {
        self.state.ground.get(key)
    }
    pub fn payload(&self, key: &str) -> Option<&SemanticPayload> {
        match self.value(key)? {
            SemanticValue::Payload(value) => Some(value),
            _ => None,
        }
    }
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.state.ground.keys().map(String::as_str)
    }
    pub fn inputs(&self, key: &str) -> impl Iterator<Item = &str> {
        self.state
            .dependencies
            .inputs
            .get(key)
            .into_iter()
            .flatten()
            .map(String::as_str)
    }
}

/// A validated but unpublished change. It is bound to one host and base revision.
#[derive(Debug)]
pub struct PreparedDelta {
    owner: Arc<()>,
    base: Revision,
    next: SemanticState,
    affected: BTreeSet<String>,
}
impl PreparedDelta {
    pub fn revision(&self) -> Revision {
        self.next.revision
    }
    pub fn affected(&self) -> &BTreeSet<String> {
        &self.affected
    }
}

#[derive(Debug, Default)]
pub struct SemanticHost {
    state: SemanticState,
    owner: Arc<()>,
}
impl SemanticHost {
    pub fn revision(&self) -> Revision {
        self.state.revision
    }

    /// Validate and stage without publishing. The reference implementation clones
    /// state; persistent data structures are an independent performance task.
    pub fn prepare_delta(&self, delta: SemanticDelta) -> Result<PreparedDelta, SemanticError> {
        // Encoding enforces the same bounds/conflict rules as durable replay.
        delta.encode()?;
        let mut next = self.state.clone();
        let mut changed = BTreeSet::new();
        for key in &delta.removals {
            if next.ground.remove(key).is_some() {
                changed.insert(key.clone());
            }
            if next.dependencies.inputs.remove(key).is_some() {
                changed.insert(key.clone());
            }
        }
        for (key, value) in &delta.upserts {
            if next.ground.get(key) != Some(value) {
                next.ground.insert(key.clone(), value.clone());
                changed.insert(key.clone());
            }
        }
        for (key, inputs) in &delta.dependencies {
            let old = next.dependencies.inputs.get(key);
            if old != Some(inputs) && !(old.is_none() && inputs.is_empty()) {
                if inputs.is_empty() {
                    next.dependencies.inputs.remove(key);
                } else {
                    next.dependencies.inputs.insert(key.clone(), inputs.clone());
                }
                changed.insert(key.clone());
            }
        }
        if !next.dependencies.acyclic() {
            return Err(SemanticError::CyclicDependency);
        }
        let mut affected = self.state.dependencies.affected_by(changed.iter().cloned());
        affected.extend(next.dependencies.affected_by(changed));
        // A cached derivation is not kept merely because its source was removed.
        // Keep dependency metadata so a later upsert still has to supply inputs.
        for key in &affected {
            if !delta.upserts.contains_key(key) {
                next.ground.remove(key);
            }
        }
        for key in delta.upserts.keys() {
            if let Some(inputs) = next.dependencies.inputs.get(key) {
                for input in inputs {
                    if !next.ground.contains_key(input) {
                        return Err(SemanticError::MissingDependency {
                            derived: key.clone(),
                            input: input.clone(),
                        });
                    }
                }
            }
        }
        if next != self.state {
            next.revision = Revision(
                self.revision()
                    .0
                    .checked_add(1)
                    .ok_or(SemanticError::RevisionExhausted)?,
            );
        }
        Ok(PreparedDelta {
            owner: self.owner.clone(),
            base: self.revision(),
            next,
            affected,
        })
    }

    pub fn apply_prepared(
        &mut self,
        prepared: PreparedDelta,
    ) -> Result<(Revision, BTreeSet<String>), SemanticError> {
        if !Arc::ptr_eq(&self.owner, &prepared.owner) || self.revision() != prepared.base {
            return Err(SemanticError::StalePreparation);
        }
        self.state = prepared.next;
        Ok((self.revision(), prepared.affected))
    }
    pub fn apply_delta(
        &mut self,
        delta: SemanticDelta,
    ) -> Result<(Revision, BTreeSet<String>), SemanticError> {
        let prepared = self.prepare_delta(delta)?;
        self.apply_prepared(prepared)
    }
    /// Export the complete published state as one canonical delta.
    ///
    /// With the revision this is everything [`Self::restore`] needs, expressed in
    /// the encoding the journal already uses. Reusing that encoding is the point:
    /// a snapshot cannot disagree with a replay about what a semantic value is.
    /// Dependency entries whose derived key is currently absent are preserved, so
    /// a later upsert still has to supply its inputs.
    pub fn export_state(&self) -> SemanticDelta {
        SemanticDelta {
            upserts: self.state.ground.clone(),
            removals: BTreeSet::new(),
            dependencies: self.state.dependencies.inputs.clone(),
        }
    }

    /// Rebuild a host at an exact revision from an exported state.
    ///
    /// `revision` is trusted input; the caller must have authenticated the state
    /// it arrived with, because nothing here can tell a genuine revision from a
    /// chosen one. Validation is the ordinary delta path, so a restored state can
    /// never be one [`Self::prepare_delta`] would have refused. An exported state
    /// has nothing to remove, so removals are rejected rather than ignored.
    pub fn restore(revision: Revision, state: SemanticDelta) -> Result<Self, SemanticError> {
        if !state.removals.is_empty() {
            return Err(SemanticError::ConflictingOperation);
        }
        let mut host = Self::default();
        host.apply_delta(state)?;
        host.state.revision = revision;
        Ok(host)
    }

    pub fn snapshot(&self) -> SemanticSnapshot {
        SemanticSnapshot {
            revision: self.revision(),
            state: Arc::new(self.state.clone()),
            owner: self.owner.clone(),
        }
    }
    pub fn is_stale(&self, snapshot: &SemanticSnapshot) -> bool {
        !Arc::ptr_eq(&self.owner, &snapshot.owner)
            || snapshot.revision != self.revision()
            || snapshot.state.revision != snapshot.revision
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProjectSkeleton {
    pub goals: BTreeSet<String>,
    pub entities: BTreeSet<String>,
    pub hard_constraints: BTreeSet<String>,
    pub capabilities: BTreeSet<String>,
    pub open_questions: BTreeSet<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Verifies that revision exhaustion leaves state unchanged.
    #[test]
    fn revision_exhaustion_leaves_state_unchanged() {
        let mut host = SemanticHost::default();
        host.state.revision = Revision(u64::MAX);
        let mut delta = SemanticDelta::default();
        delta.upserts.insert("source".into(), "new".into());
        assert!(matches!(
            host.apply_delta(delta),
            Err(SemanticError::RevisionExhausted)
        ));
        assert!(host.snapshot().keys().next().is_none());
        assert_eq!(host.revision(), Revision(u64::MAX));
    }
}
