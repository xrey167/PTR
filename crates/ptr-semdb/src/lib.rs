//! Revisioned, incremental semantic ground state and immutable snapshots.

use ptr_types::Revision;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SemanticDelta {
    pub upserts: BTreeMap<String, String>,
    pub removals: BTreeSet<String>,
}

#[derive(Clone, Debug, Default)]
pub struct DependencyGraph {
    forward: BTreeMap<String, BTreeSet<String>>,
}

impl DependencyGraph {
    pub fn depends_on(&mut self, derived: impl Into<String>, input: impl Into<String>) {
        self.forward
            .entry(input.into())
            .or_default()
            .insert(derived.into());
    }

    pub fn affected_by<I, S>(&self, changed: I) -> BTreeSet<String>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut out = BTreeSet::new();
        let mut stack: Vec<String> = changed.into_iter().map(Into::into).collect();
        while let Some(key) = stack.pop() {
            if !out.insert(key.clone()) {
                continue;
            }
            if let Some(children) = self.forward.get(&key) {
                stack.extend(children.iter().cloned());
            }
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct SemanticSnapshot {
    pub revision: Revision,
    ground: Arc<BTreeMap<String, String>>,
}

impl SemanticSnapshot {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.ground.get(key).map(String::as_str)
    }
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.ground.keys().map(String::as_str)
    }
}

#[derive(Clone, Debug, Default)]
pub struct SemanticHost {
    revision: Revision,
    ground: BTreeMap<String, String>,
    dependencies: DependencyGraph,
}

impl SemanticHost {
    pub fn revision(&self) -> Revision {
        self.revision
    }
    pub fn dependencies_mut(&mut self) -> &mut DependencyGraph {
        &mut self.dependencies
    }

    pub fn apply_delta(&mut self, delta: SemanticDelta) -> (Revision, BTreeSet<String>) {
        let mut changed = BTreeSet::new();
        for key in delta.removals {
            if self.ground.remove(&key).is_some() {
                changed.insert(key);
            }
        }
        for (key, value) in delta.upserts {
            if self.ground.get(&key) != Some(&value) {
                self.ground.insert(key.clone(), value);
                changed.insert(key);
            }
        }
        if !changed.is_empty() {
            self.revision = self.revision.next();
        }
        let affected = self.dependencies.affected_by(changed.iter().cloned());
        (self.revision, affected)
    }

    pub fn snapshot(&self) -> SemanticSnapshot {
        SemanticSnapshot {
            revision: self.revision,
            ground: Arc::new(self.ground.clone()),
        }
    }

    pub fn is_stale(&self, snapshot: &SemanticSnapshot) -> bool {
        snapshot.revision != self.revision
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

    #[test]
    fn local_change_invalidates_only_dependency_closure() {
        let mut host = SemanticHost::default();
        host.dependencies_mut().depends_on("claim:x", "input:b");
        host.dependencies_mut().depends_on("plan:p", "claim:x");
        host.dependencies_mut().depends_on("unrelated", "input:a");
        let mut d = SemanticDelta::default();
        d.upserts.insert("input:b".into(), "new".into());
        let (_, affected) = host.apply_delta(d);
        assert!(affected.contains("input:b"));
        assert!(affected.contains("claim:x"));
        assert!(affected.contains("plan:p"));
        assert!(!affected.contains("unrelated"));
    }
}
