//! Head of a stable pod identity.
//!
//! A [`PodIdentity`] names a pod across artifacts and generations; its *head* is the
//! [`PodRevisionAddress`] that currently answers for it. The head moves only through a committed
//! [`ptr_ledger::LedgerEvent::PodHeadAdvanced`] record, which is a compare-and-swap: the writer
//! names the head it observed (`expected`) and the record is refused unless that is still the
//! head and the new generation is strictly above it.
//!
//! The heads are a projection of the `pod-head:` entries of the materialized state, so they
//! survive replay and a compacted snapshot without a second copy in the snapshot format. A head
//! whose generation carries a revocation tombstone is never returned.

use ptr_state::POD_HEAD_PREFIX;
use ptr_types::{Generation, PodIdentity, PodRevisionAddress};
use std::collections::BTreeMap;

use crate::{PtrRuntime, RuntimeError};
use ptr_ledger::LedgerEvent;
use ptr_types::CommitIndex;

/// A request to move the head of `identity` to `new`, provided it is still `expected`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdvancePodHead {
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
        identity: PodIdentity,
        expected: Option<PodRevisionAddress>,
        actual: Option<PodRevisionAddress>,
    },
    /// The new generation is below the generation the lifecycle authority holds live for the pod.
    BelowLiveGeneration {
        identity: PodIdentity,
        live: Generation,
        requested: Generation,
    },
    /// The new generation is not strictly above the current one.
    NotAdvancing {
        identity: PodIdentity,
        current: Generation,
        requested: Generation,
    },
    /// The new generation carries a revocation tombstone.
    RevokedGeneration {
        identity: PodIdentity,
        target: String,
        generation: Generation,
    },
    /// The address of a request is malformed.
    InvalidAddress { identity: PodIdentity },
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
    heads: BTreeMap<PodIdentity, PodRevisionAddress>,
}

impl PodHeads {
    /// Rebuilds the heads from the `pod-head:` entries of a materialized state. Anything
    /// malformed fails closed.
    pub(crate) fn from_materialized(
        values: &BTreeMap<String, String>,
    ) -> Result<Self, RuntimeError> {
        let mut heads = BTreeMap::new();
        for (key, value) in values.range(POD_HEAD_PREFIX.to_owned()..) {
            let Some(identity) = key.strip_prefix(POD_HEAD_PREFIX) else {
                break;
            };
            let malformed = || {
                RuntimeError::PodHead(Box::new(PodHeadError::MalformedEntry { key: key.clone() }))
            };
            let address = PodRevisionAddress::parse(value).map_err(|_| malformed())?;
            if identity.is_empty() {
                return Err(malformed());
            }
            heads.insert(PodIdentity(identity.to_owned()), address);
        }
        Ok(Self { heads })
    }

    pub(crate) fn get(&self, identity: &PodIdentity) -> Option<&PodRevisionAddress> {
        self.heads.get(identity)
    }

    pub(crate) fn set(&mut self, identity: PodIdentity, head: PodRevisionAddress) {
        self.heads.insert(identity, head);
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&PodIdentity, &PodRevisionAddress)> {
        self.heads.iter()
    }
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
    fn pod_head(&self, identity: &PodIdentity) -> Option<&PodRevisionAddress> {
        PtrRuntime::pod_head(self, identity)
    }
}

impl PtrRuntime {
    /// Whether `head` may answer now: its generation is not revoked and is the one the lifecycle
    /// authority holds live for the pod, as neural-state admission requires of a bound
    /// generation. A generation that is not live (superseded, ahead of the lifecycle, or a pod
    /// unknown to it) fails closed.
    fn pod_head_is_current(&self, head: &PodRevisionAddress) -> bool {
        let target = pod_lifecycle_target(head);
        !self
            .revoked_generations
            .contains(&(target.clone(), head.generation))
            && self.live_generations.get(&target) == Some(&head.generation)
    }

    /// The current head of `identity`, unless it may not answer.
    ///
    /// A head is hidden when its generation is revoked or is not the live generation of the pod
    /// (superseded, or never made live). Nothing falls back to an older revision: which older
    /// revision may answer is a decision for a new committed head.
    pub fn pod_head(&self, identity: &PodIdentity) -> Option<&PodRevisionAddress> {
        let head = self.pod_heads.get(identity)?;
        self.pod_head_is_current(head).then_some(head)
    }

    /// The committed head of `identity` and why it may or may not answer. Unlike
    /// [`PtrRuntime::pod_head`] this shows a revoked or stale head, so a writer can name it as
    /// `expected` when it moves the identity to a new generation.
    pub fn pod_head_status(&self, identity: &PodIdentity) -> Option<PodHeadStatus<'_>> {
        let head = self.pod_heads.get(identity)?;
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

    /// Every head that may answer, ordered by identity.
    pub fn pod_heads(&self) -> impl Iterator<Item = (&PodIdentity, &PodRevisionAddress)> {
        self.pod_heads
            .iter()
            .filter(|(_, head)| self.pod_head_is_current(head))
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
            identity: request.identity,
            old: request.expected,
            new: request.new,
        })
    }

    pub(crate) fn validate_pod_head_advance(
        &self,
        identity: &PodIdentity,
        old: Option<&PodRevisionAddress>,
        new: &PodRevisionAddress,
    ) -> Result<(), RuntimeError> {
        let invalid = || {
            RuntimeError::PodHead(Box::new(PodHeadError::InvalidAddress {
                identity: identity.clone(),
            }))
        };
        new.validate().map_err(|_| invalid())?;
        let current = self.pod_heads.get(identity);
        if current != old {
            return Err(RuntimeError::PodHead(Box::new(PodHeadError::Conflict {
                identity: identity.clone(),
                expected: old.cloned(),
                actual: current.cloned(),
            })));
        }
        if let Some(old) = old {
            if new.generation <= old.generation {
                return Err(RuntimeError::PodHead(Box::new(
                    PodHeadError::NotAdvancing {
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
                    identity: identity.clone(),
                    target,
                    generation: new.generation,
                },
            )));
        }
        Ok(())
    }
}
