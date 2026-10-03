use crate::retention::{KnowledgeKey, KnowledgeLifecycle, KnowledgeObject, RawEvent};
use ptr_types::{Generation, KnowledgeObjectId, RawEventId};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LifecycleRecord {
    pub key: KnowledgeKey,
    pub lifecycle: KnowledgeLifecycle,
    pub revision: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KnowledgeStoreError {
    InvalidRawDigest(RawEventId),
    InvalidToolPair(RawEventId),
    ConflictingRawEvent(RawEventId),
    UnknownObject(KnowledgeObjectId),
    InvalidLifecycleTransition {
        from: KnowledgeLifecycle,
        to: KnowledgeLifecycle,
    },
    ConflictingKnowledgeGeneration(KnowledgeKey),
    UnknownSource(RawEventId),
    InvalidDependency(KnowledgeObjectId),
    GenerationExhausted(KnowledgeObjectId),
}

#[derive(Default)]
pub struct KnowledgeStore {
    raw: BTreeMap<RawEventId, RawEvent>,
    objects: BTreeMap<KnowledgeKey, KnowledgeObject>,
    current: BTreeMap<KnowledgeObjectId, Generation>,
    lifecycle: BTreeMap<KnowledgeKey, LifecycleRecord>,
    revisions: BTreeMap<KnowledgeObjectId, u64>,
}

impl KnowledgeStore {
    /// Appends immutable history. Replaying the exact same event is idempotent;
    /// reusing an id for different content is rejected.
    pub fn append_raw(&mut self, event: RawEvent) -> Result<bool, KnowledgeStoreError> {
        event
            .validate_digest()
            .map_err(|_| KnowledgeStoreError::InvalidRawDigest(event.id.clone()))?;
        event
            .validate_tool_pair()
            .map_err(|_| KnowledgeStoreError::InvalidToolPair(event.id.clone()))?;
        match self.raw.get(&event.id) {
            Some(existing) if existing == &event => Ok(false),
            Some(_) => Err(KnowledgeStoreError::ConflictingRawEvent(event.id)),
            None => {
                self.raw.insert(event.id.clone(), event);
                Ok(true)
            }
        }
    }

    pub fn raw(&self, id: &RawEventId) -> Option<&RawEvent> {
        self.raw.get(id)
    }

    pub fn register_object(&mut self, object: KnowledgeObject) -> Result<(), KnowledgeStoreError> {
        let key = KnowledgeKey {
            logical_id: object.id.clone(),
            generation: object.generation,
        };
        if let Some(existing) = self.objects.get(&key) {
            if existing.digest() != object.digest() || existing != &object {
                return Err(KnowledgeStoreError::ConflictingKnowledgeGeneration(key));
            }
            return Ok(());
        }
        let logical_id = object.id.clone();
        let generation = object.generation;
        self.objects.insert(key, object);
        self.current.entry(logical_id).or_insert(generation);
        Ok(())
    }

    /// Admission path used by promotion: every raw source and dependency must
    /// already be present in this store.
    pub fn admit_object(&mut self, object: KnowledgeObject) -> Result<(), KnowledgeStoreError> {
        for source in &object.sources {
            if !self.raw.contains_key(source) {
                return Err(KnowledgeStoreError::UnknownSource(source.clone()));
            }
        }
        for dependency in &object.dependencies {
            let Some(dependency_object) = self.objects.get(dependency) else {
                return Err(KnowledgeStoreError::InvalidDependency(
                    dependency.logical_id.clone(),
                ));
            };
            if self.lifecycle_of(dependency) == KnowledgeLifecycle::Invalidated {
                return Err(KnowledgeStoreError::InvalidDependency(
                    dependency.logical_id.clone(),
                ));
            }
            if dependency_object.generation != dependency.generation {
                return Err(KnowledgeStoreError::InvalidDependency(
                    dependency.logical_id.clone(),
                ));
            }
        }
        if object.generation.0 > 1 {
            let superseded = object
                .supersedes
                .as_ref()
                .ok_or_else(|| KnowledgeStoreError::InvalidDependency(object.id.clone()))?;
            if superseded.logical_id != object.id
                || superseded.generation.0.checked_add(1) != Some(object.generation.0)
                || !self.objects.contains_key(superseded)
            {
                return Err(KnowledgeStoreError::InvalidDependency(object.id.clone()));
            }
        }
        self.register_object(object)
    }

    pub fn activate_generation(
        &mut self,
        id: &KnowledgeObjectId,
        generation: Generation,
    ) -> Result<(), KnowledgeStoreError> {
        let key = KnowledgeKey {
            logical_id: id.clone(),
            generation,
        };
        let object = self
            .objects
            .get(&key)
            .ok_or_else(|| KnowledgeStoreError::UnknownObject(id.clone()))?;
        let object_lifecycle = object.lifecycle;
        let supersedes = object.supersedes.clone();
        if generation.0 > 1 {
            let superseded = supersedes
                .as_ref()
                .ok_or_else(|| KnowledgeStoreError::InvalidDependency(id.clone()))?;
            if superseded.generation.0.checked_add(1) != Some(generation.0) {
                return Err(KnowledgeStoreError::InvalidDependency(id.clone()));
            }
        }
        if self.lifecycle_of(&key) == KnowledgeLifecycle::Invalidated {
            return Err(KnowledgeStoreError::InvalidLifecycleTransition {
                from: KnowledgeLifecycle::Invalidated,
                to: KnowledgeLifecycle::Hot,
            });
        }
        let previous = self
            .current
            .get(id)
            .copied()
            .filter(|previous| *previous != generation);
        if let Some(previous) = previous {
            // Decide before mutating anything: if the cascade from the
            // superseded generation would reach the generation being activated
            // (it depends on it, directly or transitively), activation must fail
            // instead of leaving a half-applied cascade behind.
            let cascade = self.invalidation_closure(id, previous)?;
            if cascade.contains(&(id.clone(), generation)) {
                return Err(KnowledgeStoreError::InvalidLifecycleTransition {
                    from: self.lifecycle_of(&key),
                    to: KnowledgeLifecycle::Hot,
                });
            }
            self.apply_invalidation(&cascade);
        }
        self.current.insert(id.clone(), generation);
        // Only a generation without any recorded lifecycle starts from the
        // object's own state. An existing record (for example a demotion) must
        // survive re-activation instead of being reset.
        if !self.lifecycle.contains_key(&key) {
            self.record_lifecycle(&key, object_lifecycle);
        }
        Ok(())
    }

    pub fn object(&self, id: &KnowledgeObjectId) -> Option<KnowledgeObject> {
        self.current.get(id).and_then(|generation| {
            let key = KnowledgeKey {
                logical_id: id.clone(),
                generation: *generation,
            };
            self.object_at(&key)
        })
    }

    pub fn object_at(&self, key: &KnowledgeKey) -> Option<KnowledgeObject> {
        let mut object = self.objects.get(key)?.clone();
        object.lifecycle = self.lifecycle_of(key);
        Some(object)
    }

    pub fn lifecycle(&self, id: &KnowledgeObjectId) -> Option<KnowledgeLifecycle> {
        self.object(id).map(|object| object.lifecycle)
    }

    pub fn transition(
        &mut self,
        id: &KnowledgeObjectId,
        to: KnowledgeLifecycle,
    ) -> Result<(), KnowledgeStoreError> {
        let generation = self
            .current
            .get(id)
            .copied()
            .ok_or_else(|| KnowledgeStoreError::UnknownObject(id.clone()))?;
        let key = KnowledgeKey {
            logical_id: id.clone(),
            generation,
        };
        let from = self.lifecycle_of(&key);
        if from != to && !lifecycle_allowed(from, to) {
            return Err(KnowledgeStoreError::InvalidLifecycleTransition { from, to });
        }
        if from != to {
            if to == KnowledgeLifecycle::Invalidated {
                // Invalidation must always cascade to dependents, exactly like
                // `invalidate`; recording it on one generation alone would leave
                // dependents trusting an invalidated object.
                return self.invalidate_generation(id, generation);
            }
            self.record_lifecycle(&key, to);
        }
        Ok(())
    }

    pub fn invalidate(&mut self, id: KnowledgeObjectId) -> Result<(), KnowledgeStoreError> {
        let generation = self
            .current
            .get(&id)
            .copied()
            .ok_or_else(|| KnowledgeStoreError::UnknownObject(id.clone()))?;
        self.invalidate_generation(&id, generation)
    }

    fn invalidate_generation(
        &mut self,
        id: &KnowledgeObjectId,
        generation: Generation,
    ) -> Result<(), KnowledgeStoreError> {
        let closure = self.invalidation_closure(id, generation)?;
        self.apply_invalidation(&closure);
        Ok(())
    }

    /// Every generation that becomes invalid when `(id, generation)` does: the
    /// generation itself plus all transitive dependents. Read-only, so callers
    /// can inspect the effect before committing to it.
    fn invalidation_closure(
        &self,
        id: &KnowledgeObjectId,
        generation: Generation,
    ) -> Result<BTreeSet<(KnowledgeObjectId, Generation)>, KnowledgeStoreError> {
        let mut pending = vec![(id.clone(), generation)];
        let mut seen = BTreeSet::new();
        while let Some((current, current_generation)) = pending.pop() {
            if !seen.insert((current.clone(), current_generation)) {
                continue;
            }
            let key = KnowledgeKey {
                logical_id: current.clone(),
                generation: current_generation,
            };
            if !self.objects.contains_key(&key) {
                return Err(KnowledgeStoreError::UnknownObject(current.clone()));
            }
            for (candidate_key, candidate) in &self.objects {
                if candidate.dependencies.iter().any(|dependency| {
                    dependency.logical_id == current && dependency.generation == current_generation
                }) {
                    pending.push((candidate_key.logical_id.clone(), candidate_key.generation));
                }
            }
        }
        Ok(seen)
    }

    fn apply_invalidation(&mut self, closure: &BTreeSet<(KnowledgeObjectId, Generation)>) {
        for (logical_id, generation) in closure {
            let key = KnowledgeKey {
                logical_id: logical_id.clone(),
                generation: *generation,
            };
            self.record_lifecycle(&key, KnowledgeLifecycle::Invalidated);
        }
    }

    pub fn is_invalidated(&self, id: &KnowledgeObjectId) -> bool {
        self.current
            .get(id)
            .map(|generation| {
                self.lifecycle_of(&KnowledgeKey {
                    logical_id: id.clone(),
                    generation: *generation,
                }) == KnowledgeLifecycle::Invalidated
            })
            .unwrap_or(false)
    }

    pub fn is_generation_invalidated(
        &self,
        id: &KnowledgeObjectId,
        generation: Generation,
    ) -> bool {
        self.lifecycle_of(&KnowledgeKey {
            logical_id: id.clone(),
            generation,
        }) == KnowledgeLifecycle::Invalidated
    }

    fn lifecycle_of(&self, key: &KnowledgeKey) -> KnowledgeLifecycle {
        self.lifecycle
            .get(key)
            .map(|record| record.lifecycle)
            .or_else(|| self.objects.get(key).map(|object| object.lifecycle))
            .unwrap_or(KnowledgeLifecycle::Invalidated)
    }

    fn record_lifecycle(&mut self, key: &KnowledgeKey, lifecycle: KnowledgeLifecycle) {
        let revision = self.revisions.entry(key.logical_id.clone()).or_insert(0);
        *revision = revision.saturating_add(1);
        self.lifecycle.insert(
            key.clone(),
            LifecycleRecord {
                key: key.clone(),
                lifecycle,
                revision: *revision,
            },
        );
    }
}

fn lifecycle_allowed(from: KnowledgeLifecycle, to: KnowledgeLifecycle) -> bool {
    matches!(
        (from, to),
        (KnowledgeLifecycle::Hot, KnowledgeLifecycle::Warm)
            | (KnowledgeLifecycle::Warm, KnowledgeLifecycle::Cold)
            | (KnowledgeLifecycle::Cold, KnowledgeLifecycle::Pod)
            | (KnowledgeLifecycle::Pod, KnowledgeLifecycle::Archived)
            | (_, KnowledgeLifecycle::Invalidated)
    )
}
