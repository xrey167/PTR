//! Head of a stable pod identity.
//!
//! A [`PodIdentity`] names a pod across artifacts and generations; two projects with the same pod
//! semantics derive the same identity, so a head belongs to a `(ProjectId, PodIdentity)` pair and
//! never crosses a project. Its *head* is the
//! [`PodRevisionAddress`] that currently answers for it. The head moves only through a committed
//! [`ptr_ledger::LedgerEvent::PodHeadAdvanced`] record, which is a compare-and-swap: the writer
//! names the head it observed (`expected`) and the record is refused unless that is still the
//! head and the new generation is strictly above it.
//!
//! The heads are a projection of the `pod-head:` entries of the materialized state, so they
//! survive replay and a compacted snapshot without a second copy in the snapshot format. A head
//! whose generation carries a revocation tombstone is never returned.

use ptr_state::POD_HEAD_PREFIX;
use ptr_types::{Generation, PodIdentity, PodRevisionAddress, ProjectId};
use std::collections::BTreeMap;

use crate::{PtrRuntime, RuntimeError};
use ptr_ledger::LedgerEvent;
use ptr_types::CommitIndex;

/// A request to move the head of `identity` in `project` to `new`, provided it is still
/// `expected`. Every address of the request names `project`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdvancePodHead {
    pub project: ProjectId,
    pub identity: PodIdentity,
    /// The head the caller observed; `None` when the caller expects no head yet.
    pub expected: Option<PodRevisionAddress>,
    pub new: PodRevisionAddress,
}

/// Why a head move or a stored head was refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PodHeadError {
    /// The committed head is not the one the writer observed.
    Conflict {
        project: ProjectId,
        identity: PodIdentity,
        expected: Option<PodRevisionAddress>,
        actual: Option<PodRevisionAddress>,
    },
    /// The new generation is below the generation the lifecycle authority holds live for the pod.
    BelowLiveGeneration {
        project: ProjectId,
        identity: PodIdentity,
        live: Generation,
        requested: Generation,
    },
    /// The new generation is not strictly above the current one.
    NotAdvancing {
        project: ProjectId,
        identity: PodIdentity,
        current: Generation,
        requested: Generation,
    },
    /// The new generation carries a revocation tombstone.
    RevokedGeneration {
        project: ProjectId,
        identity: PodIdentity,
        target: String,
        generation: Generation,
    },
    /// The address of a request is malformed.
    InvalidAddress {
        project: ProjectId,
        identity: PodIdentity,
    },
    /// An address of the request belongs to another project than the head being moved.
    ProjectMismatch {
        project: ProjectId,
        identity: PodIdentity,
        address_project: ProjectId,
    },
    /// A materialized `pod-head:` entry does not parse; the state is refused rather than
    /// restored without a head.
    MalformedEntry { key: String },
}

/// The head of an identity as the lifecycle authority sees it now.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PodHeadStatus<'a> {
    /// The head's generation is not revoked.
    Live(&'a PodRevisionAddress),
    /// The head's generation carries a revocation tombstone. [`PtrRuntime::pod_head`] hides it,
    /// but it is still the committed head: a move names it as `expected`.
    Revoked(&'a PodRevisionAddress),
    /// The lifecycle authority holds another generation live for the pod (or none): the head
    /// describes a superseded or not yet live revision. [`PtrRuntime::pod_head`] hides it.
    Stale(&'a PodRevisionAddress),
}

/// The heads, derived from the committed history.
#[derive(Clone, Debug, Default)]
pub(crate) struct PodHeads {
    heads: BTreeMap<(ProjectId, PodIdentity), PodRevisionAddress>,
}

impl PodHeads {
    /// Rebuilds the heads from the `pod-head:` entries of a materialized state. Anything
    /// malformed fails closed.
    pub(crate) fn from_materialized(
        values: &BTreeMap<String, String>,
    ) -> Result<Self, RuntimeError> {
        let mut heads = BTreeMap::new();
        for (key, value) in values.range(POD_HEAD_PREFIX.to_owned()..) {
            let Some(rest) = key.strip_prefix(POD_HEAD_PREFIX) else {
                break;
            };
            let malformed = || {
                RuntimeError::PodHead(Box::new(PodHeadError::MalformedEntry { key: key.clone() }))
            };
            let address = PodRevisionAddress::parse(value).map_err(|_| malformed())?;
            let Some((project, identity)) = rest.split_once('/') else {
                return Err(malformed());
            };
            // The entry must be the one its own address belongs to.
            if project.is_empty() || identity.is_empty() || address.address.project.0 != project {
                return Err(malformed());
            }
            heads.insert(
                (
                    ProjectId(project.to_owned()),
                    PodIdentity(identity.to_owned()),
                ),
                address,
            );
        }
        Ok(Self { heads })
    }

    pub(crate) fn get(
        &self,
        project: &ProjectId,
        identity: &PodIdentity,
    ) -> Option<&PodRevisionAddress> {
        self.heads.get(&(project.clone(), identity.clone()))
    }

    pub(crate) fn set(
        &mut self,
        project: ProjectId,
        identity: PodIdentity,
        head: PodRevisionAddress,
    ) {
        self.heads.insert((project, identity), head);
    }

    pub(crate) fn iter(
        &self,
    ) -> impl Iterator<Item = (&ProjectId, &PodIdentity, &PodRevisionAddress)> {
        self.heads
            .iter()
            .map(|((project, identity), head)| (project, identity, head))
    }
}

/// The project a `pod:<project>/<namespace>/<pod>` lifecycle target names, or `None` when the
/// target is not of that form (three non-empty segments after `pod:`).
pub(crate) fn pod_capsule_project(target: &str) -> Option<&str> {
    let mut parts = target.strip_prefix("pod:")?.split('/');
    let (project, namespace, pod) = (parts.next()?, parts.next()?, parts.next()?);
    (parts.next().is_none() && !project.is_empty() && !namespace.is_empty() && !pod.is_empty())
        .then_some(project)
}

/// The lifecycle target a pod's generations are committed under: `pod:<project>/<namespace>/<pod>`.
/// A revocation of that target and generation revokes exactly that pod revision, and neural state
/// bound under it (a learned pod representation, for one) stops being admitted.
pub fn pod_lifecycle_target(pod: &PodRevisionAddress) -> String {
    format!(
        "pod:{}/{}/{}",
        pod.address.project, pod.address.namespace, pod.address.pod_id
    )
}

impl crate::directory::PodHeadSource for PtrRuntime {
    fn pod_head(&self, project: &ProjectId, identity: &PodIdentity) -> Option<&PodRevisionAddress> {
        PtrRuntime::pod_head(self, project, identity)
    }
}

impl PtrRuntime {
    /// Whether `head` may answer now: its generation is not revoked and is the one the lifecycle
    /// authority holds live for the pod, as neural-state admission requires of a bound
    /// generation. A generation that is not live (superseded, ahead of the lifecycle, or a pod
    /// unknown to it) fails closed.
    fn pod_head_is_current(&self, head: &PodRevisionAddress) -> bool {
        let target = pod_lifecycle_target(head);
        // The capsule must have been committed by the head's own project, whatever its id says.
        self.capsule_projects.get(&target).map(String::as_str)
            == Some(head.address.project.0.as_str())
            && !self
                .revoked_generations
                .contains(&(target.clone(), head.generation))
            && self.live_generations.get(&target) == Some(&head.generation)
    }

    /// The current head of `identity`, unless it may not answer.
    ///
    /// A head is hidden when its generation is revoked or is not the live generation of the pod
    /// (superseded, or never made live). Nothing falls back to an older revision: which older
    /// revision may answer is a decision for a new committed head.
    pub fn pod_head(
        &self,
        project: &ProjectId,
        identity: &PodIdentity,
    ) -> Option<&PodRevisionAddress> {
        let head = self.pod_heads.get(project, identity)?;
        self.pod_head_is_current(head).then_some(head)
    }

    /// The committed head of `identity` and why it may or may not answer. Unlike
    /// [`PtrRuntime::pod_head`] this shows a revoked or stale head, so a writer can name it as
    /// `expected` when it moves the identity to a new generation.
    pub fn pod_head_status(
        &self,
        project: &ProjectId,
        identity: &PodIdentity,
    ) -> Option<PodHeadStatus<'_>> {
        let head = self.pod_heads.get(project, identity)?;
        let target = pod_lifecycle_target(head);
        Some(
            if self
                .revoked_generations
                .contains(&(target.clone(), head.generation))
            {
                PodHeadStatus::Revoked(head)
            } else if self.live_generations.get(&target) == Some(&head.generation) {
                PodHeadStatus::Live(head)
            } else {
                PodHeadStatus::Stale(head)
            },
        )
    }

    /// Every head that may answer, ordered by project and identity.
    pub fn pod_heads(
        &self,
    ) -> impl Iterator<Item = (&ProjectId, &PodIdentity, &PodRevisionAddress)> {
        self.pod_heads
            .iter()
            .filter(|(_, _, head)| self.pod_head_is_current(head))
    }

    /// Moves the head of `request.identity` to `request.new` through the ledger.
    ///
    /// # Errors
    /// [`PodHeadError::Conflict`] when the committed head is not `request.expected`,
    /// [`PodHeadError::NotAdvancing`] when the generation does not rise,
    /// [`PodHeadError::RevokedGeneration`] when the new generation is revoked, and the ledger
    /// errors of [`PtrRuntime::commit`]. Nothing is appended on an error.
    pub fn advance_pod_head(
        &mut self,
        request: AdvancePodHead,
    ) -> Result<CommitIndex, RuntimeError> {
        self.commit(LedgerEvent::PodHeadAdvanced {
            project: request.project,
            identity: request.identity,
            old: request.expected,
            new: request.new,
        })
    }

    pub(crate) fn validate_pod_head_advance(
        &self,
        project: &ProjectId,
        identity: &PodIdentity,
        old: Option<&PodRevisionAddress>,
        new: &PodRevisionAddress,
    ) -> Result<(), RuntimeError> {
        let invalid = || {
            RuntimeError::PodHead(Box::new(PodHeadError::InvalidAddress {
                project: project.clone(),
                identity: identity.clone(),
            }))
        };
        new.validate().map_err(|_| invalid())?;
        // No pod revision has generation 0 (`PodSemanticManifest::validate` refuses it).
        if new.generation.0 == 0 {
            return Err(invalid());
        }
        for address in std::iter::once(new).chain(old) {
            if address.address.project != *project {
                return Err(RuntimeError::PodHead(Box::new(
                    PodHeadError::ProjectMismatch {
                        project: project.clone(),
                        identity: identity.clone(),
                        address_project: address.address.project.clone(),
                    },
                )));
            }
        }
        let current = self.pod_heads.get(project, identity);
        if current != old {
            return Err(RuntimeError::PodHead(Box::new(PodHeadError::Conflict {
                project: project.clone(),
                identity: identity.clone(),
                expected: old.cloned(),
                actual: current.cloned(),
            })));
        }
        if let Some(old) = old {
            if new.generation <= old.generation {
                return Err(RuntimeError::PodHead(Box::new(
                    PodHeadError::NotAdvancing {
                        project: project.clone(),
                        identity: identity.clone(),
                        current: old.generation,
                        requested: new.generation,
                    },
                )));
            }
        }
        let target = pod_lifecycle_target(new);
        if let Some(live) = self.live_generations.get(&target) {
            if new.generation < *live {
                return Err(RuntimeError::PodHead(Box::new(
                    PodHeadError::BelowLiveGeneration {
                        project: project.clone(),
                        identity: identity.clone(),
                        live: *live,
                        requested: new.generation,
                    },
                )));
            }
        }
        if self
            .revoked_generations
            .contains(&(target.clone(), new.generation))
        {
            return Err(RuntimeError::PodHead(Box::new(
                PodHeadError::RevokedGeneration {
                    project: project.clone(),
                    identity: identity.clone(),
                    target,
                    generation: new.generation,
                },
            )));
        }
        Ok(())
    }
}
