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
//!     let mut sink: Vec<ptr_router::PolicyEvent> = Vec::new();
//!     let _ = book.promote(scorer, evidence, &mut sink);
//! }
//! ```
//!
//! ```compile_fail
//! // Evidence cannot be built by hand: its fields are private and it has no constructor.
//! let evidence = ptr_router::PromotionEvidence {
//!     candidate: [0; 32],
//!     incumbent: [0; 32],
//! };
//! ```

use std::fmt;
use std::sync::{Arc, RwLock};

use sha2::{Digest as _, Sha256};

use crate::findings::{AdmissibleScorer, Exploration, ScoringContext};
use crate::learning::LearnedPolicy;
use crate::ope::PromotionEvidence;
use crate::router::RouteError;
use ptr_types::PodId;

const UNIFORM_DOMAIN: &[u8] = b"ptr.router.policy.uniform.v1";
/// Most installed policies a book retains (the active one and its predecessors).
pub const MAX_VERSIONS: usize = 16;
/// Most events a book retains; older ones are dropped and counted.
pub const MAX_EVENTS: usize = 1024;

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
    exploration: Exploration,
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

    /// The exploration rate every router using this handle must have.
    pub fn exploration(&self) -> Exploration {
        self.exploration
    }
}

/// Where a [`PolicyBook`] hands every change of policy. Implemented by `PtrRuntime` (which
/// emits `RuntimeEvent::VcwPolicyChanged`) and by `Vec<PolicyEvent>`. It is observability: an
/// implementation cannot veto or alter the change, and the event is not an audit record (see
/// the module documentation).
pub trait PolicySink {
    fn record(&mut self, event: &PolicyEvent);
}

impl PolicySink for Vec<PolicyEvent> {
    fn record(&mut self, event: &PolicyEvent) {
        self.push(event.clone());
    }
}

impl AdmissibleScorer for ActiveScorer {
    fn bound_exploration(&self) -> Option<Exploration> {
        Some(self.exploration)
    }

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
    /// 1-based, monotonic across the book's whole life (it keeps counting when old events are
    /// dropped from the retained history).
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
    /// The candidate was trained under another exploration rate than the book's.
    ExplorationMismatch,
    /// There is no earlier policy to return to.
    NothingToRollBack,
    /// The book was asked to hold no exploration floor (a zero rate).
    NoFloor,
}

impl fmt::Display for PolicyBookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EvidenceForAnotherCandidate => {
                f.write_str("the evidence names another candidate")
            }
            Self::StaleEvidence => f.write_str("the evidence was sealed against another incumbent"),
            Self::ScopeMismatch => f.write_str("the candidate belongs to another scope"),
            Self::ExplorationMismatch => {
                f.write_str("the candidate was trained under another exploration rate")
            }
            Self::NothingToRollBack => f.write_str("no earlier policy to return to"),
            Self::NoFloor => f.write_str("a policy book needs a positive exploration rate"),
        }
    }
}

impl std::error::Error for PolicyBookError {}

/// The host's owner of the active policy.
#[derive(Debug)]
pub struct PolicyBook {
    scope: String,
    exploration: Exploration,
    active: ActiveScorer,
    /// The retained installed policies, oldest first (at most [`MAX_VERSIONS`]); the last is
    /// active.
    stack: Vec<Arc<Installed>>,
    events: Vec<PolicyEvent>,
    next_sequence: u64,
    dropped_events: u64,
}

impl PolicyBook {
    /// A book whose active policy is uniform, for routers that explore at `exploration`.
    ///
    /// # Errors
    /// [`PolicyBookError::NoFloor`] when `exploration` is zero: a learned policy needs the
    /// router's floor for positivity.
    pub fn new(scope: &str, exploration: Exploration) -> Result<Self, PolicyBookError> {
        if exploration.epsilon() <= 0.0 {
            return Err(PolicyBookError::NoFloor);
        }
        let mut hasher = Sha256::new();
        hasher.update(UNIFORM_DOMAIN);
        hasher.update((scope.len() as u64).to_le_bytes());
        hasher.update(scope.as_bytes());
        let installed = Arc::new(Installed::Uniform {
            digest: hasher.finalize().into(),
        });
        Ok(Self {
            scope: scope.to_owned(),
            exploration,
            active: ActiveScorer {
                current: Arc::new(RwLock::new(Arc::clone(&installed))),
                exploration,
            },
            stack: vec![installed],
            events: Vec::new(),
            next_sequence: 1,
            dropped_events: 0,
        })
    }

    /// The exploration rate of the routers this book serves.
    pub fn exploration(&self) -> Exploration {
        self.exploration
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

    /// The retained events, oldest first (the last [`MAX_EVENTS`]).
    pub fn events(&self) -> &[PolicyEvent] {
        &self.events
    }

    /// Events dropped from the front of [`Self::events`] because the history was full.
    pub fn dropped_events(&self) -> u64 {
        self.dropped_events
    }

    fn push_event(&mut self, mut event: PolicyEvent, sink: &mut dyn PolicySink) -> PolicyEvent {
        event.sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        if self.events.len() >= MAX_EVENTS {
            self.events.remove(0);
            self.dropped_events = self.dropped_events.saturating_add(1);
        }
        self.events.push(event.clone());
        sink.record(&event);
        event
    }

    fn install(&mut self, installed: Arc<Installed>) {
        *self
            .active
            .current
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = installed;
    }

    /// Makes `candidate` the active policy and hands the event to `sink`.
    ///
    /// # Errors
    /// [`PolicyBookError`] when the evidence is not for this candidate and the policy active now,
    /// or the candidate is of another scope or exploration rate. Nothing changes.
    pub fn promote(
        &mut self,
        candidate: LearnedPolicy,
        evidence: PromotionEvidence,
        sink: &mut dyn PolicySink,
    ) -> Result<PolicyEvent, PolicyBookError> {
        if candidate.scope() != self.scope {
            return Err(PolicyBookError::ScopeMismatch);
        }
        if candidate.exploration() != self.exploration {
            return Err(PolicyBookError::ExplorationMismatch);
        }
        if *evidence.candidate() != candidate.digest() {
            return Err(PolicyBookError::EvidenceForAnotherCandidate);
        }
        let from = self.active.digest();
        if *evidence.incumbent() != from {
            return Err(PolicyBookError::StaleEvidence);
        }
        let to = candidate.digest();
        let installed = Arc::new(Installed::Learned(candidate));
        if self.stack.len() >= MAX_VERSIONS {
            // The oldest retained policy falls out; the active one and its predecessor, which
            // is all a rollback needs, stay.
            self.stack.remove(0);
        }
        self.stack.push(Arc::clone(&installed));
        self.install(installed);
        let event = PolicyEvent {
            sequence: 0,
            action: PolicyAction::Promoted,
            scope: self.scope.clone(),
            from,
            to,
            evidence: Some(evidence.digest()),
        };
        Ok(self.push_event(event, sink))
    }

    /// Returns to the policy that was active before the last promotion and hands the event to
    /// `sink`. It never fails for want of room.
    ///
    /// # Errors
    /// [`PolicyBookError::NothingToRollBack`] when no earlier policy is retained.
    pub fn rollback(&mut self, sink: &mut dyn PolicySink) -> Result<PolicyEvent, PolicyBookError> {
        if self.stack.len() < 2 {
            return Err(PolicyBookError::NothingToRollBack);
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
            sequence: 0,
            action: PolicyAction::RolledBack,
            scope: self.scope.clone(),
            from,
            to,
            evidence: None,
        };
        Ok(self.push_event(event, sink))
    }
}
