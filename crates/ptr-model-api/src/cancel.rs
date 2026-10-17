//! Structured cancellation, deadlines and the typed [`GenerationError`].
//!
//! Everything here is `std`-only and synchronous. A [`CancellationToken`] is a shared flag
//! with optional parent links; a [`Deadline`] measures a budget against an injectable
//! [`Clock`] so tests never sleep.

use crate::{ModelError, TokenizeError};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug)]
struct TokenState {
    cancelled: AtomicBool,
    parent: Option<Arc<TokenState>>,
}

impl Drop for TokenState {
    /// Unlink the ancestor chain iteratively: the default recursive drop of `parent` would
    /// overflow the stack on a deep (but legal) chain of uniquely owned ancestors.
    /// `Arc::into_inner` hands the state to exactly one dropper, so a chain shared with a
    /// still-live token or racing drops stops (or continues) exactly once.
    fn drop(&mut self) {
        let mut next = self.parent.take();
        while let Some(parent) = next {
            next = match Arc::into_inner(parent) {
                Some(mut state) => state.parent.take(),
                None => None,
            };
        }
    }
}

impl TokenState {
    /// Walks the ancestor chain, so the cost is linear in the depth the caller built.
    fn is_cancelled(&self) -> bool {
        let mut node = self;
        loop {
            if node.cancelled.load(Ordering::SeqCst) {
                return true;
            }
            match node.parent.as_deref() {
                Some(parent) => node = parent,
                None => return false,
            }
        }
    }
}

/// Cooperative cancellation flag. Clones share one flag; a [`child`](Self::child) is
/// cancelled when it or any ancestor is cancelled, while cancelling a child never affects
/// its parent or siblings. Cancellation is sticky and cannot be undone.
#[derive(Clone, Debug)]
pub struct CancellationToken {
    state: Arc<TokenState>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            state: Arc::new(TokenState {
                cancelled: AtomicBool::new(false),
                parent: None,
            }),
        }
    }

    /// A token that is cancelled whenever `self` (or an ancestor of `self`) is.
    pub fn child(&self) -> Self {
        Self {
            state: Arc::new(TokenState {
                cancelled: AtomicBool::new(false),
                parent: Some(Arc::clone(&self.state)),
            }),
        }
    }

    pub fn cancel(&self) {
        self.state.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.state.is_cancelled()
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

/// Monotonic time source. `now` is the time since an arbitrary, fixed origin.
pub trait Clock: Send + Sync {
    fn now(&self) -> Duration;
}

/// [`Clock`] over [`std::time::Instant`], anchored at construction.
#[derive(Clone, Copy, Debug)]
pub struct SystemClock {
    origin: Instant,
}

impl SystemClock {
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SystemClock {
    fn now(&self) -> Duration {
        self.origin.elapsed()
    }
}

/// Test clock that only moves when told to. Clones share the same time.
#[derive(Clone, Debug, Default)]
pub struct ManualClock {
    nanos: Arc<AtomicU64>,
}

impl ManualClock {
    pub fn new() -> Self {
        Self::default()
    }

    /// Move time forward (saturating at `u64::MAX` nanoseconds).
    pub fn advance(&self, by: Duration) {
        let step = u64::try_from(by.as_nanos()).unwrap_or(u64::MAX);
        let _ = self
            .nanos
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |now| {
                Some(now.saturating_add(step))
            });
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Duration {
        Duration::from_nanos(self.nanos.load(Ordering::SeqCst))
    }
}

/// A time budget started at construction. Exceeded when `elapsed > budget`.
#[derive(Clone)]
pub struct Deadline {
    clock: Arc<dyn Clock>,
    started: Duration,
    budget: Duration,
}

impl fmt::Debug for Deadline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Deadline")
            .field("started", &self.started)
            .field("budget", &self.budget)
            .finish_non_exhaustive()
    }
}

impl Deadline {
    /// Start a deadline of `budget` on `clock` now.
    pub fn start(clock: Arc<dyn Clock>, budget: Duration) -> Self {
        let started = clock.now();
        Self {
            clock,
            started,
            budget,
        }
    }

    pub fn budget(&self) -> Duration {
        self.budget
    }

    pub fn elapsed(&self) -> Duration {
        self.clock.now().saturating_sub(self.started)
    }

    /// `Err(DeadlineExceeded)` once more than the budget has elapsed.
    pub fn check(&self) -> Result<(), GenerationError> {
        let elapsed = self.elapsed();
        if elapsed > self.budget {
            Err(GenerationError::DeadlineExceeded {
                elapsed,
                budget: self.budget,
            })
        } else {
            Ok(())
        }
    }
}

/// Typed failure of a (streaming) generation. Converts into the legacy [`ModelError`]
/// for call sites that still speak strings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GenerationError {
    /// The cancellation token fired after `at_token` generated tokens were delivered.
    Cancelled {
        at_token: usize,
    },
    DeadlineExceeded {
        elapsed: Duration,
        budget: Duration,
    },
    /// The backend does not offer `capability`.
    CapabilityUnsupported {
        backend: String,
        capability: &'static str,
    },
    /// Stream limits or options were rejected before the stream opened.
    InvalidLimits(&'static str),
    /// `next_event` was called after the stream already ended.
    StreamClosed,
    /// The source emitted more than `limit` non-text [`crate::ModelEvent`]s through one
    /// stream (see [`crate::MAX_STREAM_MODEL_EVENTS`]); the stream ended with this error.
    ModelEventLimit {
        limit: usize,
    },
    Tokenize(TokenizeError),
    Backend(ModelError),
}

impl fmt::Display for GenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled { at_token } => write!(f, "generation cancelled at token {at_token}"),
            Self::DeadlineExceeded { elapsed, budget } => {
                write!(f, "deadline exceeded: {elapsed:?} elapsed of {budget:?}")
            }
            Self::CapabilityUnsupported {
                backend,
                capability,
            } => write!(f, "backend {backend} does not support {capability}"),
            Self::InvalidLimits(why) => write!(f, "invalid stream limits: {why}"),
            Self::StreamClosed => write!(f, "stream already ended"),
            Self::ModelEventLimit { limit } => {
                write!(f, "stream emitted more than {limit} non-text model events")
            }
            Self::Tokenize(error) => write!(f, "tokenizer failure: {error}"),
            Self::Backend(error) => write!(f, "backend failure: {}", error.0),
        }
    }
}

impl std::error::Error for GenerationError {}

impl From<ModelError> for GenerationError {
    fn from(error: ModelError) -> Self {
        Self::Backend(error)
    }
}

impl From<TokenizeError> for GenerationError {
    fn from(error: TokenizeError) -> Self {
        Self::Tokenize(error)
    }
}

impl From<GenerationError> for ModelError {
    fn from(error: GenerationError) -> Self {
        match error {
            GenerationError::Backend(inner) => inner,
            other => ModelError(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_follows_parent_but_not_reverse() {
        let parent = CancellationToken::new();
        let child = parent.child();
        let grandchild = child.child();
        let sibling = parent.child();
        child.cancel();
        assert!(child.is_cancelled() && grandchild.is_cancelled());
        assert!(!parent.is_cancelled() && !sibling.is_cancelled());
        parent.cancel();
        assert!(sibling.is_cancelled());
    }

    #[test]
    fn deep_child_chain_drops_without_stack_overflow() {
        const DEPTH: usize = 1_000_000;
        let handle = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                let root = CancellationToken::new();
                let mut token = root.child();
                for _ in 0..DEPTH {
                    token = token.child();
                }
                root.cancel();
                assert!(
                    token.is_cancelled(),
                    "ancestor cancellation reaches the deepest token"
                );
                drop(root);
                drop(token);
            })
            .expect("spawn small-stack thread");
        handle.join().expect("deep chain dropped without overflow");
    }

    #[test]
    fn clones_share_the_flag() {
        let token = CancellationToken::new();
        let other = token.clone();
        other.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn deadline_uses_the_injected_clock() {
        let clock = ManualClock::new();
        clock.advance(Duration::from_secs(100));
        let deadline = Deadline::start(Arc::new(clock.clone()), Duration::from_secs(5));
        assert_eq!(deadline.check(), Ok(()));
        clock.advance(Duration::from_secs(5));
        assert_eq!(
            deadline.check(),
            Ok(()),
            "exactly the budget is not exceeded"
        );
        clock.advance(Duration::from_millis(1));
        assert!(matches!(
            deadline.check(),
            Err(GenerationError::DeadlineExceeded { budget, .. }) if budget == Duration::from_secs(5)
        ));
    }

    #[test]
    fn conversion_into_model_error() {
        let backend = GenerationError::Backend(ModelError("boom".into()));
        assert_eq!(ModelError::from(backend), ModelError("boom".into()));
        let cancelled = ModelError::from(GenerationError::Cancelled { at_token: 3 });
        assert!(cancelled.0.contains("cancelled at token 3"));
        let msg = GenerationError::CapabilityUnsupported {
            backend: "b".into(),
            capability: "streaming",
        }
        .to_string();
        assert!(msg.contains("streaming"));
    }

    #[test]
    fn system_clock_is_monotonic() {
        let clock = SystemClock::new();
        let a = clock.now();
        assert!(clock.now() >= a);
    }
}
