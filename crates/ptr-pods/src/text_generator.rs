//! A backend-independent, bounded text generator.
//!
//! Expert Pods that turn a prompt into text depend on this trait, never on a model backend. A
//! request carries a system prompt, a user message and a hard bound on the generated tokens;
//! the generator applies its own chat template. The bound is part of the type: a request above
//! [`MAX_NEW_TOKENS_LIMIT`] is refused, never clamped silently.

use std::fmt;

/// The largest `max_new_tokens` any generator accepts.
pub const MAX_NEW_TOKENS_LIMIT: u32 = 128;

/// One generation request.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct GenerationRequest<'a> {
    pub system: &'a str,
    pub user: &'a str,
    /// Hard bound on generated tokens, `1..=`[`MAX_NEW_TOKENS_LIMIT`].
    pub max_new_tokens: u32,
}

impl<'a> GenerationRequest<'a> {
    #[must_use]
    pub fn new(system: &'a str, user: &'a str, max_new_tokens: u32) -> Self {
        Self {
            system,
            user,
            max_new_tokens,
        }
    }

    /// Whether the request is within the global bounds.
    ///
    /// # Errors
    /// [`GenerationError::InvalidRequest`] for a zero or oversized token bound.
    pub fn validate(&self) -> Result<(), GenerationError> {
        if self.max_new_tokens == 0 || self.max_new_tokens > MAX_NEW_TOKENS_LIMIT {
            return Err(GenerationError::InvalidRequest(
                "max_new_tokens must be in 1..=128",
            ));
        }
        Ok(())
    }
}

/// Why generation stopped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum StopReason {
    /// The model ended its turn.
    EndOfTurn,
    /// The token bound was reached.
    MaxTokens,
}

/// What a generator produced.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct GenerationOutput {
    pub text: String,
    /// Tokens generated, the dropped special tokens included.
    pub new_tokens: u32,
    pub stop: StopReason,
    /// Generated special tokens that are in `new_tokens` but not in `text` (the text decoder
    /// drops them). Zero for a generator that does not drop any.
    pub dropped_special_tokens: u32,
}

impl GenerationOutput {
    #[must_use]
    pub fn new(text: String, new_tokens: u32, stop: StopReason) -> Self {
        Self {
            text,
            new_tokens,
            stop,
            dropped_special_tokens: 0,
        }
    }

    /// Record how many generated special tokens the text leaves out.
    #[must_use]
    pub fn with_dropped_special_tokens(mut self, dropped: u32) -> Self {
        self.dropped_special_tokens = dropped;
        self
    }
}

/// The SHA-256 digests (lowercase hex) of the files a generator was loaded from, by role
/// (`weights`, `config`, `tokenizer`, ...). It says which model a Pod ran; it is not a secret.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct ModelIdentity {
    /// `(role, sha256 hex)` pairs, sorted by role.
    pub files: Vec<(String, String)>,
}

impl ModelIdentity {
    #[must_use]
    pub fn new(mut files: Vec<(String, String)>) -> Self {
        files.sort();
        Self { files }
    }
}

/// Why a generator failed.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GenerationError {
    InvalidRequest(&'static str),
    /// The backend failed (message only; backend types do not cross this boundary).
    Backend(String),
}

impl fmt::Display for GenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(why) => write!(f, "invalid generation request: {why}"),
            Self::Backend(why) => write!(f, "generation backend: {why}"),
        }
    }
}

impl std::error::Error for GenerationError {}

/// Greedy-or-sampled text generation behind one bounded call.
pub trait TextGenerator {
    /// # Errors
    /// [`GenerationError`] for an invalid request or a backend failure.
    fn generate(
        &mut self,
        request: &GenerationRequest<'_>,
    ) -> Result<GenerationOutput, GenerationError>;

    /// The digests of the files this generator runs, when it knows them.
    fn identity(&self) -> Option<ModelIdentity> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_token_bound_is_enforced_not_clamped() {
        assert!(GenerationRequest::new("s", "u", 1).validate().is_ok());
        assert!(GenerationRequest::new("s", "u", 128).validate().is_ok());
        assert!(GenerationRequest::new("s", "u", 0).validate().is_err());
        assert!(GenerationRequest::new("s", "u", 129).validate().is_err());
    }
}
