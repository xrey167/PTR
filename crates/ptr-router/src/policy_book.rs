//! The active routing policy and the two host actions that change it.
//!
//! A [`FindingRouter`](crate::FindingRouter) is built with a scorer. To let a learned policy
//! replace the incumbent without rebuilding the router, the scorer is an [`ActiveScorer`]: a
//! cheap handle that reads whichever policy the [`PolicyBook`] currently holds. The two handles
//! are different types on purpose:
//!
//! * [`ActiveScorer`] implements [`AdmissibleScorer`] and can only be read from. The loop and
//!   the router hold this one. It has no method that changes the policy.
//! * [`PolicyBook`] is held by the host. [`PolicyBook::promote`] takes a [`LearnedPolicy`] and
//!   the [`PromotionEvidence`] the gate sealed for exactly that candidate and exactly the policy
//!   that is active now; [`PolicyBook::rollback`] returns to the previous one. Each returns a
//!   [`PolicyEvent`] for the host to record (`ptr-runtime` emits it as a runtime event).
//!
//! Nothing in the loop calls either action: promotion is never automatic.
//!
//! ```compile_fail
//! // A scorer that is not a LearnedPolicy cannot be promoted.
//! fn smuggle(book: &mut ptr_router::PolicyBook, scorer: ptr_router::UniformScorer,
//!            evidence: ptr_router::PromotionEvidence) {
//!     let _ = book.promote(scorer, evidence);
//! }
//! ```

use std::fmt;
use std::sync::{Arc, RwLock};

use sha2::{Digest as _, Sha256};

use crate::findings::{AdmissibleScorer, ScoringContext};
use crate::learning::LearnedPolicy;
use crate::ope::PromotionEvidence;
use crate::router::RouteError;
use ptr_types::PodId;

const UNIFORM_DOMAIN: &[u8] = b"ptr.router.policy.uniform.v1";
const MAX_VERSIONS: usize = 16;
const MAX_EVENTS: usize = 1024;

/// One installed policy.
#[derive(Clone, Debug)]
enum Installed {
    /// Equal weight for every admissible Pod (the baseline every learned policy starts from).
    Uniform {
        digest: [u8; 32],
    },
    Learned(LearnedPolicy),
}

impl Installed {
    fn digest(&self) -> [u8; 32] {
        match self {
            Self::Uniform { digest } => *digest,
            Self::Learned(policy) => policy.digest(),
        }
    }
}

/// The read-only handle the router scores with.
#[derive(Clone, Debug)]
pub struct ActiveScorer {
    current: Arc<RwLock<Arc<Installed>>>,
}

impl ActiveScorer {
    fn read(&self) -> Arc<Installed> {
        Arc::clone(
            &self
                .current
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
    }

    /// The digest of the policy that scores right now.
    pub fn digest(&self) -> [u8; 32] {
        self.read().digest()
    }
}

impl AdmissibleScorer for ActiveScorer {
    fn weights(&self, context: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
        match &*self.read() {
            Installed::Uniform { .. } => Ok(context
                .admissible
                .iter()
                .map(|pod| (pod.clone(), 1.0))
                .collect()),
            Installed::Learned(policy) => policy.weights(context),
        }
    }
}

/// What a host did to the active policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyAction {
    Promoted,
    RolledBack,
}

/// A recorded change of the active policy.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyEvent {
    /// 1-based position in the book's history.
    pub sequence: u64,
    pub action: PolicyAction,
    pub scope: String,
    pub from: [u8; 32],
    pub to: [u8; 32],
    /// The digest of the evidence a promotion rested on; `None` for a rollback.
    pub evidence: Option<[u8; 32]>,
}

/// Why a promotion or a rollback was refused. Nothing changed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PolicyBookError {
    /// The evidence was sealed for another candidate.
    EvidenceForAnotherCandidate,
    /// The evidence was sealed against a policy that is no longer the active one.
    StaleEvidence,
    /// The candidate was trained under another scope than the book's.
    ScopeMismatch,
    /// There is no earlier policy to return to.
    NothingToRollBack,
    /// The history is full.
    HistoryFull,
}

impl fmt::Display for PolicyBookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EvidenceForAnotherCandidate => {
                f.write_str("the evidence names another candidate")
            }
            Self::StaleEvidence => f.write_str("the evidence was sealed against another incumbent"),
            Self::ScopeMismatch => f.write_str("the candidate belongs to another scope"),
            Self::NothingToRollBack => f.write_str("no earlier policy to return to"),
            Self::HistoryFull => f.write_str("the policy history is full"),
        }
    }
}

impl std::error::Error for PolicyBookError {}

/// The host's owner of the active policy.
#[derive(Debug)]
pub struct PolicyBook {
    scope: String,
    active: ActiveScorer,
    /// Every policy installed so far, oldest first; the last is active unless rolled back.
    stack: Vec<Arc<Installed>>,
    events: Vec<PolicyEvent>,
}

impl PolicyBook {
    /// A book whose active policy is uniform.
    pub fn new(scope: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(UNIFORM_DOMAIN);
        hasher.update((scope.len() as u64).to_le_bytes());
        hasher.update(scope.as_bytes());
        let installed = Arc::new(Installed::Uniform {
            digest: hasher.finalize().into(),
        });
        Self {
            scope: scope.to_owned(),
            active: ActiveScorer {
                current: Arc::new(RwLock::new(Arc::clone(&installed))),
            },
            stack: vec![installed],
            events: Vec::new(),
        }
    }

    pub fn scope(&self) -> &str {
        &self.scope
    }

    /// The handle to give the router. All clones read the same active policy.
    pub fn scorer(&self) -> ActiveScorer {
        self.active.clone()
    }

    pub fn active_digest(&self) -> [u8; 32] {
        self.active.digest()
    }

    /// Whether the active policy is a learned one.
    pub fn is_learned(&self) -> bool {
        self.stack
            .last()
            .is_some_and(|installed| matches!(**installed, Installed::Learned(_)))
    }

    pub fn events(&self) -> &[PolicyEvent] {
        &self.events
    }

    fn install(&mut self, installed: Arc<Installed>) {
        *self
            .active
            .current
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = installed;
    }

    /// Makes `candidate` the active policy.
    ///
    /// # Errors
    /// [`PolicyBookError`] when the evidence is not for this candidate and the policy active now,
    /// the candidate is of another scope, or the history is full. Nothing changes.
    pub fn promote(
        &mut self,
        candidate: LearnedPolicy,
        evidence: PromotionEvidence,
    ) -> Result<PolicyEvent, PolicyBookError> {
        if candidate.scope() != self.scope {
            return Err(PolicyBookError::ScopeMismatch);
        }
        if *evidence.candidate() != candidate.digest() {
            return Err(PolicyBookError::EvidenceForAnotherCandidate);
        }
        let from = self.active.digest();
        if *evidence.incumbent() != from {
            return Err(PolicyBookError::StaleEvidence);
        }
        if self.stack.len() >= MAX_VERSIONS || self.events.len() >= MAX_EVENTS {
            return Err(PolicyBookError::HistoryFull);
        }
        let to = candidate.digest();
        let installed = Arc::new(Installed::Learned(candidate));
        self.stack.push(Arc::clone(&installed));
        self.install(installed);
        let event = PolicyEvent {
            sequence: self.events.len() as u64 + 1,
            action: PolicyAction::Promoted,
            scope: self.scope.clone(),
            from,
            to,
            evidence: Some(evidence.digest()),
        };
        self.events.push(event.clone());
        Ok(event)
    }

    /// Returns to the policy that was active before the last promotion.
    ///
    /// # Errors
    /// [`PolicyBookError::NothingToRollBack`] when only the initial policy remains.
    pub fn rollback(&mut self) -> Result<PolicyEvent, PolicyBookError> {
        if self.stack.len() < 2 {
            return Err(PolicyBookError::NothingToRollBack);
        }
        if self.events.len() >= MAX_EVENTS {
            return Err(PolicyBookError::HistoryFull);
        }
        let from = self.active.digest();
        self.stack.pop();
        let previous = self
            .stack
            .last()
            .cloned()
            .ok_or(PolicyBookError::NothingToRollBack)?;
        let to = previous.digest();
        self.install(previous);
        let event = PolicyEvent {
            sequence: self.events.len() as u64 + 1,
            action: PolicyAction::RolledBack,
            scope: self.scope.clone(),
            from,
            to,
            evidence: None,
        };
        self.events.push(event.clone());
        Ok(event)
    }
}
