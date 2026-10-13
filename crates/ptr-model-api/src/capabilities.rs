//! Backend capability descriptors and deterministic negotiation.
//!
//! [`negotiate`] filters a set of offered [`BackendCapabilities`] against the
//! [`CapabilityRequirements`] of a request and picks one backend. Every rejected backend
//! carries all of its typed [`RejectionReason`]s. The choice does not depend on the order
//! of the offers.
//!
//! # Tie-break among accepted backends (first difference wins)
//!
//! 1. the chosen dtype with the highest fidelity (declaration order of [`DType`]);
//! 2. the larger `max_context_tokens`;
//! 3. the larger `max_batch`;
//! 4. the lexicographically smaller `backend_id` (ids must be unique, so this is total).

use crate::{KvReuseMode, ModelRequest, SegmentedModelRequest, TokenizerIdentity};
use std::collections::BTreeSet;
use std::fmt;

/// Numeric format of model weights/activations, ordered from highest to lowest fidelity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DType {
    F32,
    Bf16,
    F16,
    Q8,
    Q4,
}

/// Which KV reuse modes a backend can serve.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KvReuseSupport {
    #[default]
    None,
    ExactOnly,
    ExactAndApproximate,
}

impl KvReuseSupport {
    pub fn supports(self, mode: KvReuseMode) -> bool {
        match (self, mode) {
            (Self::None, _) => false,
            (Self::ExactOnly, KvReuseMode::Exact) => true,
            (Self::ExactOnly, KvReuseMode::Approximate) => false,
            (Self::ExactAndApproximate, _) => true,
        }
    }
}

/// What one backend offers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendCapabilities {
    pub backend_id: String,
    pub max_context_tokens: usize,
    pub streaming: bool,
    pub resumable: bool,
    pub kv_reuse: KvReuseSupport,
    /// `None` means the backend does not expose a tokenizer identity.
    pub tokenizer: Option<TokenizerIdentity>,
    pub max_batch: usize,
    pub dtypes: Vec<DType>,
}

impl BackendCapabilities {
    /// A minimal descriptor: batch of 1, no optional features, no dtypes listed.
    pub fn new(backend_id: impl Into<String>, max_context_tokens: usize) -> Self {
        Self {
            backend_id: backend_id.into(),
            max_context_tokens,
            streaming: false,
            resumable: false,
            kv_reuse: KvReuseSupport::None,
            tokenizer: None,
            max_batch: 1,
            dtypes: Vec::new(),
        }
    }
}

/// What a request needs. Empty `acceptable_dtypes` means any dtype (or none) is fine.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CapabilityRequirements {
    pub context_tokens: usize,
    pub streaming: bool,
    pub resumable: bool,
    /// Weakest KV reuse mode the request needs the backend to be able to serve.
    pub kv_reuse: Option<KvReuseMode>,
    pub tokenizer: Option<TokenizerIdentity>,
    pub batch: usize,
    /// In order of preference.
    pub acceptable_dtypes: Vec<DType>,
}

impl CapabilityRequirements {
    /// Requirements of a plain request. The context need is estimated conservatively as
    /// the byte length of the raw text (an upper bound for byte-level and typical subword
    /// tokenizers); callers with a real token count should set `context_tokens` themselves.
    pub fn from_request(request: &ModelRequest) -> Self {
        Self {
            context_tokens: request.raw_text.len(),
            batch: 1,
            ..Self::default()
        }
    }

    /// Requirements of a segmented request: exact token count and tokenizer identity, and
    /// KV reuse support (Exact) since segments carry lineage.
    pub fn from_segmented(request: &SegmentedModelRequest) -> Self {
        Self {
            context_tokens: request.total_tokens(),
            kv_reuse: Some(KvReuseMode::Exact),
            tokenizer: Some(request.tokenizer().clone()),
            batch: 1,
            ..Self::default()
        }
    }

    pub fn with_streaming(mut self) -> Self {
        self.streaming = true;
        self
    }

    pub fn with_resumable(mut self) -> Self {
        self.resumable = true;
        self
    }

    pub fn with_kv_reuse(mut self, mode: KvReuseMode) -> Self {
        self.kv_reuse = Some(mode);
        self
    }

    pub fn with_batch(mut self, batch: usize) -> Self {
        self.batch = batch;
        self
    }

    pub fn with_dtypes(mut self, dtypes: Vec<DType>) -> Self {
        self.acceptable_dtypes = dtypes;
        self
    }
}

/// Why one backend cannot serve the requirements. Reasons are reported in the order of
/// this enum's variants.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RejectionReason {
    ContextTooSmall {
        required: usize,
        offered: usize,
    },
    StreamingUnsupported,
    ResumeUnsupported,
    KvReuseUnsupported {
        required: KvReuseMode,
        offered: KvReuseSupport,
    },
    TokenizerMismatch {
        required: TokenizerIdentity,
        offered: Option<TokenizerIdentity>,
    },
    BatchTooLarge {
        required: usize,
        offered: usize,
    },
    DTypeUnsupported {
        acceptable: Vec<DType>,
        offered: Vec<DType>,
    },
}

/// A backend that was not chosen, with every reason.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rejection {
    pub backend_id: String,
    pub reasons: Vec<RejectionReason>,
}

/// The outcome of a successful negotiation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NegotiatedPlan {
    /// Index into the `offered` slice passed to [`negotiate`].
    pub selected_index: usize,
    pub backend_id: String,
    /// The dtype the plan commits to, `None` only if nothing was required and the backend
    /// lists none.
    pub dtype: Option<DType>,
    /// Non-selected backends that did not satisfy the requirements, in `offered` order.
    /// Accepted-but-outranked backends are not listed.
    pub rejected: Vec<Rejection>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NegotiationError {
    NoCandidates,
    DuplicateBackendId(String),
    /// Every offered backend failed; one entry per backend, in `offered` order.
    NoBackendSatisfies {
        rejected: Vec<Rejection>,
    },
}

impl fmt::Display for NegotiationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoCandidates => write!(f, "no backend offered"),
            Self::DuplicateBackendId(id) => write!(f, "backend id {id} offered twice"),
            Self::NoBackendSatisfies { rejected } => {
                write!(
                    f,
                    "none of {} backends satisfies the requirements",
                    rejected.len()
                )
            }
        }
    }
}

impl std::error::Error for NegotiationError {}

fn reasons_for(
    requirements: &CapabilityRequirements,
    offered: &BackendCapabilities,
) -> Vec<RejectionReason> {
    let mut reasons = Vec::new();
    if requirements.context_tokens > offered.max_context_tokens {
        reasons.push(RejectionReason::ContextTooSmall {
            required: requirements.context_tokens,
            offered: offered.max_context_tokens,
        });
    }
    if requirements.streaming && !offered.streaming {
        reasons.push(RejectionReason::StreamingUnsupported);
    }
    if requirements.resumable && !offered.resumable {
        reasons.push(RejectionReason::ResumeUnsupported);
    }
    if let Some(mode) = requirements.kv_reuse {
        if !offered.kv_reuse.supports(mode) {
            reasons.push(RejectionReason::KvReuseUnsupported {
                required: mode,
                offered: offered.kv_reuse,
            });
        }
    }
    if let Some(required) = &requirements.tokenizer {
        if offered.tokenizer.as_ref() != Some(required) {
            reasons.push(RejectionReason::TokenizerMismatch {
                required: required.clone(),
                offered: offered.tokenizer.clone(),
            });
        }
    }
    if requirements.batch > offered.max_batch {
        reasons.push(RejectionReason::BatchTooLarge {
            required: requirements.batch,
            offered: offered.max_batch,
        });
    }
    if !requirements.acceptable_dtypes.is_empty()
        && !requirements
            .acceptable_dtypes
            .iter()
            .any(|d| offered.dtypes.contains(d))
    {
        reasons.push(RejectionReason::DTypeUnsupported {
            acceptable: requirements.acceptable_dtypes.clone(),
            offered: offered.dtypes.clone(),
        });
    }
    reasons
}

/// The dtype to commit to for an accepting backend: the first acceptable dtype the backend
/// offers (requirement order); with no preference, the backend's highest-fidelity dtype.
fn choose_dtype(
    requirements: &CapabilityRequirements,
    offered: &BackendCapabilities,
) -> Option<DType> {
    if requirements.acceptable_dtypes.is_empty() {
        offered.dtypes.iter().copied().min()
    } else {
        requirements
            .acceptable_dtypes
            .iter()
            .copied()
            .find(|d| offered.dtypes.contains(d))
    }
}

/// Choose the best of `offered` for `requirements`. Deterministic and independent of the
/// order of `offered` (see the module docs for the tie-break).
pub fn negotiate(
    requirements: &CapabilityRequirements,
    offered: &[BackendCapabilities],
) -> Result<NegotiatedPlan, NegotiationError> {
    if offered.is_empty() {
        return Err(NegotiationError::NoCandidates);
    }
    let mut seen = BTreeSet::new();
    for caps in offered {
        if !seen.insert(caps.backend_id.as_str()) {
            return Err(NegotiationError::DuplicateBackendId(
                caps.backend_id.clone(),
            ));
        }
    }
    let mut rejected = Vec::new();
    // (index, dtype); smaller sort key is better.
    let mut best: Option<(usize, Option<DType>)> = None;
    for (index, caps) in offered.iter().enumerate() {
        let reasons = reasons_for(requirements, caps);
        if !reasons.is_empty() {
            rejected.push(Rejection {
                backend_id: caps.backend_id.clone(),
                reasons,
            });
            continue;
        }
        let dtype = choose_dtype(requirements, caps);
        let better = match best {
            None => true,
            Some((best_index, best_dtype)) => {
                let current = &offered[best_index];
                // Fidelity: None sorts after any Some(dtype) by mapping to a rank.
                let rank = |d: Option<DType>| d.map_or(usize::MAX, |d| d as usize);
                (
                    rank(dtype),
                    std::cmp::Reverse(caps.max_context_tokens),
                    std::cmp::Reverse(caps.max_batch),
                    &caps.backend_id,
                ) < (
                    rank(best_dtype),
                    std::cmp::Reverse(current.max_context_tokens),
                    std::cmp::Reverse(current.max_batch),
                    &current.backend_id,
                )
            }
        };
        if better {
            best = Some((index, dtype));
        }
    }
    match best {
        Some((selected_index, dtype)) => Ok(NegotiatedPlan {
            selected_index,
            backend_id: offered[selected_index].backend_id.clone(),
            dtype,
            rejected,
        }),
        None => Err(NegotiationError::NoBackendSatisfies { rejected }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_reuse_support_matrix() {
        assert!(!KvReuseSupport::None.supports(KvReuseMode::Exact));
        assert!(KvReuseSupport::ExactOnly.supports(KvReuseMode::Exact));
        assert!(!KvReuseSupport::ExactOnly.supports(KvReuseMode::Approximate));
        assert!(KvReuseSupport::ExactAndApproximate.supports(KvReuseMode::Approximate));
    }

    #[test]
    fn dtype_choice_follows_requirement_order() {
        let mut caps = BackendCapabilities::new("a", 10);
        caps.dtypes = vec![DType::F16, DType::Q8];
        let req = CapabilityRequirements::default().with_dtypes(vec![DType::Q8, DType::F16]);
        assert_eq!(choose_dtype(&req, &caps), Some(DType::Q8));
        assert_eq!(
            choose_dtype(&CapabilityRequirements::default(), &caps),
            Some(DType::F16)
        );
    }
}
