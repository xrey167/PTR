use ptr_memory::{KnowledgeKey, KnowledgeLifecycle, KnowledgeStore};
use ptr_pods::{ArtifactCatalog, ArtifactError, ExecutionManifest, LineageBinding};
use ptr_types::{Digest, Generation, IdentityContext, KnowledgeObjectId, PrincipalId, Revision};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidatedExecutionManifest {
    pub manifest: ExecutionManifest,
    pub knowledge: Vec<KnowledgeKey>,
    pub artifacts: Vec<LineageBinding>,
    pub runtime_revision: Revision,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeManifestError {
    InvalidManifest,
    UnknownManifestDigest(Digest),
    ConflictingManifest(Digest),
    MissingKnowledgeGeneration {
        key: KnowledgeKey,
    },
    RevokedKnowledgeGeneration {
        key: KnowledgeKey,
    },
    UnknownKnowledgeSource {
        key: KnowledgeKey,
    },
    InvalidKnowledgeDependency {
        key: KnowledgeKey,
    },
    InvalidKnowledgeSupersedes {
        key: KnowledgeKey,
    },
    KnowledgeDigestMismatch {
        key: KnowledgeKey,
    },
    MissingArtifactGeneration {
        key: String,
        generation: Generation,
    },
    RevokedArtifactGeneration {
        key: String,
        generation: Generation,
    },
    InvalidArtifactLineage {
        key: String,
        generation: Generation,
    },
    ArtifactDigestMismatch {
        key: String,
        generation: Generation,
    },
    MissingReaderGeneration {
        key: String,
        generation: Generation,
    },
    RevokedReaderGeneration {
        key: String,
        generation: Generation,
    },
    InvalidReaderLineage {
        key: String,
        generation: Generation,
    },
    ReaderDigestMismatch {
        key: String,
        generation: Generation,
    },
    MissingAdapterGeneration {
        key: String,
        generation: Generation,
    },
    RevokedAdapterGeneration {
        key: String,
        generation: Generation,
    },
    InvalidAdapterLineage {
        key: String,
        generation: Generation,
    },
    AdapterDigestMismatch {
        key: String,
        generation: Generation,
    },
    SnapshotRevisionMismatch {
        expected: Revision,
        actual: Revision,
    },
    PolicyRevisionMissing,
    PrincipalNotAdmitted(PrincipalId),
    PolicyRevisionMismatch {
        expected: Revision,
        actual: Revision,
    },
    UnknownSnapshotRevision(Revision),
    ProtectedSnapshotMissing(Revision),
    ProtectedSnapshotRevisionMismatch {
        revision: Revision,
        expected: Digest,
        actual: Digest,
    },
    SnapshotDigestMismatch {
        revision: Revision,
        expected: Digest,
        actual: Digest,
    },
}

/// Runtime-owned authority indexes that are needed in addition to immutable
/// Knowledge and Artifact lineage. The manifest remains transport-neutral; the
/// runtime supplies the current identity, policy and semantic snapshot facts.
pub trait ManifestBindingAuthority {
    fn principal_admitted(&self, principal: &PrincipalId) -> bool;
    fn current_policy_revision(&self) -> Revision;
    fn snapshot_digest(&self, revision: Revision) -> Option<Digest>;
}

/// Replayable projection of authority bindings carried by admitted manifests.
/// Identity and policy providers remain the sources of truth; this registry
/// only materializes bindings already committed to the PTR ledger.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ManifestAuthorityRegistry {
    principals: BTreeSet<PrincipalId>,
    policy_revisions: BTreeSet<Revision>,
    policy_digests: BTreeMap<Revision, Digest>,
    revoked_policy_revisions: BTreeSet<Revision>,
    snapshots: BTreeMap<Revision, Digest>,
    protected_snapshots: BTreeMap<Revision, Digest>,
}

impl ManifestAuthorityRegistry {
    pub fn admit_identity(
        &mut self,
        identity: &IdentityContext,
        now: ptr_types::Timestamp,
    ) -> Result<PrincipalId, RuntimeManifestError> {
        if !identity.is_valid_at(now) {
            return Err(RuntimeManifestError::PrincipalNotAdmitted(PrincipalId(
                identity.subject.clone(),
            )));
        }
        let principal = PrincipalId(format!("{}:{}", identity.issuer, identity.subject));
        self.admit_principal(principal.clone())?;
        Ok(principal)
    }

    pub fn admit_principal(&mut self, principal: PrincipalId) -> Result<(), RuntimeManifestError> {
        if principal.0.is_empty() {
            return Err(RuntimeManifestError::PrincipalNotAdmitted(principal));
        }
        self.principals.insert(principal);
        Ok(())
    }

    pub fn set_policy_revision(&mut self, revision: Revision) {
        self.policy_revisions.insert(revision);
    }

    pub fn activate_policy(
        &mut self,
        revision: Revision,
        digest: Digest,
    ) -> Result<(), RuntimeManifestError> {
        if revision.0 == 0 || digest == [0; 32] {
            return Err(RuntimeManifestError::PolicyRevisionMissing);
        }
        if self.revoked_policy_revisions.contains(&revision) {
            return Err(RuntimeManifestError::PolicyRevisionMismatch {
                expected: self.current_policy_revision(),
                actual: revision,
            });
        }
        if let Some(existing) = self.policy_digests.get(&revision) {
            if existing != &digest {
                return Err(RuntimeManifestError::PolicyRevisionMismatch {
                    expected: revision,
                    actual: revision,
                });
            }
            return Ok(());
        }
        if self
            .policy_digests
            .keys()
            .next_back()
            .is_some_and(|current| revision <= *current)
        {
            return Err(RuntimeManifestError::PolicyRevisionMismatch {
                expected: self.current_policy_revision().next(),
                actual: revision,
            });
        }
        self.policy_revisions.insert(revision);
        self.policy_digests.insert(revision, digest);
        Ok(())
    }

    pub fn revoke_policy(&mut self, revision: Revision) -> Result<(), RuntimeManifestError> {
        if !self.policy_revisions.contains(&revision) {
            return Err(RuntimeManifestError::PolicyRevisionMismatch {
                expected: self.current_policy_revision(),
                actual: revision,
            });
        }
        self.revoked_policy_revisions.insert(revision);
        Ok(())
    }

    pub fn policy_digest(&self, revision: Revision) -> Option<Digest> {
        self.policy_digests.get(&revision).copied()
    }

    pub fn is_policy_revoked(&self, revision: Revision) -> bool {
        self.revoked_policy_revisions.contains(&revision)
    }

    pub fn register_snapshot(
        &mut self,
        revision: Revision,
        digest: Digest,
    ) -> Result<(), RuntimeManifestError> {
        match self.snapshots.get(&revision) {
            Some(existing) if existing != &digest => {
                Err(RuntimeManifestError::SnapshotDigestMismatch {
                    revision,
                    expected: *existing,
                    actual: digest,
                })
            }
            _ => {
                if let Some(protected) = self.protected_snapshots.get(&revision) {
                    if protected != &digest {
                        return Err(RuntimeManifestError::ProtectedSnapshotRevisionMismatch {
                            revision,
                            expected: *protected,
                            actual: digest,
                        });
                    }
                }
                self.snapshots.insert(revision, digest);
                Ok(())
            }
        }
    }

    /// Bind a protected snapshot reference to the live SemDB projection. The
    /// protected record is not a second source of truth: it must agree with a
    /// live digest whenever both are available.
    pub fn register_protected_snapshot(
        &mut self,
        revision: Revision,
        digest: Digest,
    ) -> Result<(), RuntimeManifestError> {
        if digest == [0; 32] {
            return Err(RuntimeManifestError::ProtectedSnapshotRevisionMismatch {
                revision,
                expected: digest,
                actual: digest,
            });
        }
        if let Some(existing) = self.protected_snapshots.get(&revision) {
            if existing != &digest {
                return Err(RuntimeManifestError::ProtectedSnapshotRevisionMismatch {
                    revision,
                    expected: *existing,
                    actual: digest,
                });
            }
            return Ok(());
        }
        if let Some(live) = self.snapshots.get(&revision) {
            if live != &digest {
                return Err(RuntimeManifestError::ProtectedSnapshotRevisionMismatch {
                    revision,
                    expected: *live,
                    actual: digest,
                });
            }
        }
        self.protected_snapshots.insert(revision, digest);
        Ok(())
    }

    pub fn protected_snapshot_digest(&self, revision: Revision) -> Option<Digest> {
        self.protected_snapshots.get(&revision).copied()
    }

    pub fn validate_protected_snapshot(
        &self,
        revision: Revision,
        digest: Digest,
    ) -> Result<(), RuntimeManifestError> {
        let Some(protected) = self.protected_snapshot_digest(revision) else {
            return Err(RuntimeManifestError::ProtectedSnapshotMissing(revision));
        };
        if protected != digest {
            return Err(RuntimeManifestError::ProtectedSnapshotRevisionMismatch {
                revision,
                expected: protected,
                actual: digest,
            });
        }
        if self.snapshot_digest(revision) != Some(digest) {
            return Err(RuntimeManifestError::SnapshotDigestMismatch {
                revision,
                expected: self.snapshot_digest(revision).unwrap_or([0; 32]),
                actual: digest,
            });
        }
        Ok(())
    }

    pub fn observe_manifest(
        &mut self,
        manifest: &ExecutionManifest,
    ) -> Result<(), RuntimeManifestError> {
        self.admit_principal(manifest.principal.clone())?;
        self.set_policy_revision(manifest.policy_revision);
        self.register_snapshot(manifest.snapshot_revision, manifest.snapshot_digest)
    }
}

impl ManifestBindingAuthority for ManifestAuthorityRegistry {
    fn principal_admitted(&self, principal: &PrincipalId) -> bool {
        self.principals.contains(principal)
    }

    fn current_policy_revision(&self) -> Revision {
        self.policy_revisions
            .iter()
            .rev()
            .find(|revision| !self.revoked_policy_revisions.contains(revision))
            .copied()
            .unwrap_or(Revision(0))
    }

    fn snapshot_digest(&self, revision: Revision) -> Option<Digest> {
        self.snapshots.get(&revision).copied()
    }
}

pub fn validate_authority(
    manifest: &ExecutionManifest,
    authority: &dyn ManifestBindingAuthority,
) -> Result<(), RuntimeManifestError> {
    if !authority.principal_admitted(&manifest.principal) {
        return Err(RuntimeManifestError::PrincipalNotAdmitted(
            manifest.principal.clone(),
        ));
    }
    let current_policy_revision = authority.current_policy_revision();
    if manifest.policy_revision != current_policy_revision {
        return Err(RuntimeManifestError::PolicyRevisionMismatch {
            expected: current_policy_revision,
            actual: manifest.policy_revision,
        });
    }
    let Some(expected_digest) = authority.snapshot_digest(manifest.snapshot_revision) else {
        return Err(RuntimeManifestError::UnknownSnapshotRevision(
            manifest.snapshot_revision,
        ));
    };
    if expected_digest != manifest.snapshot_digest {
        return Err(RuntimeManifestError::SnapshotDigestMismatch {
            revision: manifest.snapshot_revision,
            expected: expected_digest,
            actual: manifest.snapshot_digest,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ManifestAuthorityRegistry, ManifestBindingAuthority, RuntimeManifestError};
    use ptr_types::{Digest, Revision};

    fn digest(byte: u8) -> Digest {
        [byte; 32]
    }

    #[test]
    fn protected_snapshot_must_match_live_projection() {
        let mut authority = ManifestAuthorityRegistry::default();
        authority.register_snapshot(Revision(4), digest(4)).unwrap();
        authority
            .register_protected_snapshot(Revision(4), digest(4))
            .unwrap();
        authority
            .validate_protected_snapshot(Revision(4), digest(4))
            .unwrap();
    }

    #[test]
    fn protected_snapshot_mismatch_is_rejected_without_mutation() {
        let mut authority = ManifestAuthorityRegistry::default();
        authority.register_snapshot(Revision(4), digest(4)).unwrap();
        assert!(matches!(
            authority.register_protected_snapshot(Revision(4), digest(9)),
            Err(RuntimeManifestError::ProtectedSnapshotRevisionMismatch { .. })
        ));
        assert_eq!(authority.protected_snapshot_digest(Revision(4)), None);
    }

    #[test]
    fn protected_snapshot_before_live_snapshot_is_checked_when_live_arrives() {
        let mut authority = ManifestAuthorityRegistry::default();
        authority
            .register_protected_snapshot(Revision(4), digest(4))
            .unwrap();
        assert!(matches!(
            authority.register_snapshot(Revision(4), digest(5)),
            Err(RuntimeManifestError::ProtectedSnapshotRevisionMismatch { .. })
        ));
        assert_eq!(authority.snapshot_digest(Revision(4)), None);
    }

    #[test]
    fn missing_protected_snapshot_is_fail_closed() {
        let authority = ManifestAuthorityRegistry::default();
        assert_eq!(
            authority.validate_protected_snapshot(Revision(4), digest(4)),
            Err(RuntimeManifestError::ProtectedSnapshotMissing(Revision(4)))
        );
    }
}

pub trait RuntimeExecutionManifestResolver: Send + Sync {
    fn resolve(&self, digest: &Digest) -> Result<ValidatedExecutionManifest, RuntimeManifestError>;
}

#[derive(Clone, Debug, Default)]
pub struct ExecutionManifestRegistry {
    admitted: BTreeMap<Digest, ValidatedExecutionManifest>,
    revoked: BTreeMap<Digest, ()>,
}

impl ExecutionManifestRegistry {
    pub fn admit(
        &mut self,
        candidate: ValidatedExecutionManifest,
    ) -> Result<(), RuntimeManifestError> {
        let digest = candidate.manifest.manifest_digest;
        if self.revoked.contains_key(&digest) {
            return Err(RuntimeManifestError::UnknownManifestDigest(digest));
        }
        match self.admitted.get(&digest) {
            Some(existing) if existing == &candidate => Ok(()),
            Some(_) => Err(RuntimeManifestError::ConflictingManifest(digest)),
            None => {
                self.admitted.insert(digest, candidate);
                Ok(())
            }
        }
    }

    pub fn revoke(&mut self, digest: Digest) -> Result<(), RuntimeManifestError> {
        if self.admitted.remove(&digest).is_none() && !self.revoked.contains_key(&digest) {
            return Err(RuntimeManifestError::UnknownManifestDigest(digest));
        }
        self.revoked.insert(digest, ());
        Ok(())
    }

    pub fn admit_replayed(
        &mut self,
        manifest: ExecutionManifest,
        runtime_revision: Revision,
    ) -> Result<(), RuntimeManifestError> {
        manifest
            .validate()
            .map_err(|_| RuntimeManifestError::InvalidManifest)?;
        self.admit(ValidatedExecutionManifest {
            knowledge: manifest
                .knowledge
                .iter()
                .map(|binding| KnowledgeKey {
                    logical_id: KnowledgeObjectId::from(binding.key.as_str()),
                    generation: binding.generation,
                })
                .collect(),
            artifacts: manifest.artifacts.clone(),
            manifest,
            runtime_revision,
        })
    }

    pub fn is_revoked(&self, digest: &Digest) -> bool {
        self.revoked.contains_key(digest)
    }
}

impl RuntimeExecutionManifestResolver for ExecutionManifestRegistry {
    fn resolve(&self, digest: &Digest) -> Result<ValidatedExecutionManifest, RuntimeManifestError> {
        if self.revoked.contains_key(digest) {
            return Err(RuntimeManifestError::UnknownManifestDigest(*digest));
        }
        self.admitted
            .get(digest)
            .cloned()
            .ok_or(RuntimeManifestError::UnknownManifestDigest(*digest))
    }
}

pub fn validate_and_build(
    manifest: ExecutionManifest,
    knowledge: &KnowledgeStore,
    artifacts: &dyn ArtifactCatalog,
    runtime_revision: Revision,
) -> Result<ValidatedExecutionManifest, RuntimeManifestError> {
    manifest
        .validate()
        .map_err(|_| RuntimeManifestError::InvalidManifest)?;
    if manifest.policy_revision.0 == 0 {
        return Err(RuntimeManifestError::PolicyRevisionMissing);
    }
    if manifest.snapshot_revision != runtime_revision {
        return Err(RuntimeManifestError::SnapshotRevisionMismatch {
            expected: runtime_revision,
            actual: manifest.snapshot_revision,
        });
    }

    let mut resolved_knowledge = Vec::with_capacity(manifest.knowledge.len());
    for binding in &manifest.knowledge {
        let key = KnowledgeKey {
            logical_id: KnowledgeObjectId::from(binding.key.as_str()),
            generation: binding.generation,
        };
        let Some(object) = knowledge.object_at(&key) else {
            return Err(RuntimeManifestError::MissingKnowledgeGeneration { key });
        };
        if object.lifecycle == KnowledgeLifecycle::Invalidated {
            return Err(RuntimeManifestError::RevokedKnowledgeGeneration { key });
        }
        if object
            .sources
            .iter()
            .any(|source| knowledge.raw(source).is_none())
        {
            return Err(RuntimeManifestError::UnknownKnowledgeSource { key });
        }
        if object.dependencies.iter().any(|dependency| {
            knowledge.object_at(dependency).is_none()
                || knowledge
                    .is_generation_invalidated(&dependency.logical_id, dependency.generation)
        }) {
            return Err(RuntimeManifestError::InvalidKnowledgeDependency { key });
        }
        if object.generation.0 > 1
            && object.supersedes.as_ref().is_none_or(|supersedes| {
                supersedes.logical_id != object.id
                    || supersedes.generation.0.checked_add(1) != Some(object.generation.0)
                    || knowledge.object_at(supersedes).is_none()
            })
        {
            return Err(RuntimeManifestError::InvalidKnowledgeSupersedes { key });
        }
        if object.content_digest() != binding.digest {
            return Err(RuntimeManifestError::KnowledgeDigestMismatch { key });
        }
        resolved_knowledge.push(key);
    }

    let mut resolved_artifacts = Vec::with_capacity(manifest.artifacts.len());
    for binding in &manifest.artifacts {
        let artifact_id = ptr_types::ArtifactId::from(binding.key.as_str());
        let descriptor = match artifacts.resolve(&artifact_id, binding.generation) {
            Ok(descriptor) => descriptor,
            Err(ArtifactError::Revoked) => {
                return Err(RuntimeManifestError::RevokedArtifactGeneration {
                    key: binding.key.clone(),
                    generation: binding.generation,
                })
            }
            Err(ArtifactError::NotFound(_, generation)) => {
                return Err(RuntimeManifestError::MissingArtifactGeneration {
                    key: binding.key.clone(),
                    generation,
                })
            }
            Err(_) => {
                return Err(RuntimeManifestError::InvalidArtifactLineage {
                    key: binding.key.clone(),
                    generation: binding.generation,
                })
            }
        };
        artifacts.verify_lineage(&descriptor).map_err(|_| {
            RuntimeManifestError::InvalidArtifactLineage {
                key: binding.key.clone(),
                generation: binding.generation,
            }
        })?;
        if descriptor.lineage_digest() != binding.digest {
            return Err(RuntimeManifestError::ArtifactDigestMismatch {
                key: binding.key.clone(),
                generation: binding.generation,
            });
        }
        resolved_artifacts.push(binding.clone());
    }

    if let Some(reader) = &manifest.reader {
        let artifact_id = ptr_types::ArtifactId::from(reader.key.as_str());
        let descriptor = match artifacts.resolve(&artifact_id, reader.generation) {
            Ok(descriptor) => descriptor,
            Err(ArtifactError::Revoked) => {
                return Err(RuntimeManifestError::RevokedReaderGeneration {
                    key: reader.key.clone(),
                    generation: reader.generation,
                })
            }
            Err(ArtifactError::NotFound(_, generation)) => {
                return Err(RuntimeManifestError::MissingReaderGeneration {
                    key: reader.key.clone(),
                    generation,
                })
            }
            Err(_) => {
                return Err(RuntimeManifestError::InvalidReaderLineage {
                    key: reader.key.clone(),
                    generation: reader.generation,
                })
            }
        };
        artifacts.verify_lineage(&descriptor).map_err(|_| {
            RuntimeManifestError::InvalidReaderLineage {
                key: reader.key.clone(),
                generation: reader.generation,
            }
        })?;
        if descriptor.lineage_digest() != reader.digest {
            return Err(RuntimeManifestError::ReaderDigestMismatch {
                key: reader.key.clone(),
                generation: reader.generation,
            });
        }
    }

    if let Some(adapter) = &manifest.adapter {
        let artifact_id = ptr_types::ArtifactId::from(adapter.key.as_str());
        let descriptor = match artifacts.resolve(&artifact_id, adapter.generation) {
            Ok(descriptor) => descriptor,
            Err(ArtifactError::Revoked) => {
                return Err(RuntimeManifestError::RevokedAdapterGeneration {
                    key: adapter.key.clone(),
                    generation: adapter.generation,
                })
            }
            Err(ArtifactError::NotFound(_, generation)) => {
                return Err(RuntimeManifestError::MissingAdapterGeneration {
                    key: adapter.key.clone(),
                    generation,
                })
            }
            Err(_) => {
                return Err(RuntimeManifestError::InvalidAdapterLineage {
                    key: adapter.key.clone(),
                    generation: adapter.generation,
                })
            }
        };
        artifacts.verify_lineage(&descriptor).map_err(|_| {
            RuntimeManifestError::InvalidAdapterLineage {
                key: adapter.key.clone(),
                generation: adapter.generation,
            }
        })?;
        if descriptor.lineage_digest() != adapter.digest {
            return Err(RuntimeManifestError::AdapterDigestMismatch {
                key: adapter.key.clone(),
                generation: adapter.generation,
            });
        }
    }

    Ok(ValidatedExecutionManifest {
        manifest,
        knowledge: resolved_knowledge,
        artifacts: resolved_artifacts,
        runtime_revision,
    })
}
