//! Raw and typed ingress remain parallel. Typed interpretations never erase raw evidence.
//!
//! The crate offers a deterministic, std-only pipeline: bounded magic-byte sniffing
//! ([`sniff()`]), text normalization with a raw-offset map ([`normalize()`]), a structural
//! parser stack ([`parse()`]), a pluggable classifier contract with a rule-based reference
//! implementation ([`SemanticClassifier`], [`RuleClassifier`]), span-level provenance
//! ([`RawSpan`]) and a risk/disagreement verification policy ([`evaluate_policy`]).
//! [`IngressPipeline`] ties them together and always keeps the untouched raw input.

mod classify;
mod model;
mod normalize;
mod parse;
mod pipeline;
mod policy;
mod provenance;
mod sniff;

pub use classify::{
    ClassifierInput, ClassifierOutput, ClassifyError, LabelAxis, LabelEvidence, RuleClassifier,
    SemanticClassifier, StructureClassifier,
};
pub use model::{ArtifactKind, LabelScore, RawInput, SemanticProposal};
pub use normalize::{
    normalize, InvalidUtf8Policy, NormalizeError, NormalizeOptions, NormalizedText, NulPolicy,
    TextEncoding,
};
pub use parse::{
    parse, validate_json, CodeLanguage, DocumentFormat, JsonError, JsonErrorKind, ParseIssue,
    ParseOptions, ParsedDocument, Segment, SegmentKind,
};
pub use pipeline::{IngestHints, IngressConfig, IngressError, IngressPipeline, IngressRecord};
pub use policy::{
    cross_check, evaluate_policy, CrossCheckStatus, DisagreementReason, PolicyConfig, PolicyInput,
    VerificationDecision, HIGH_RISK_EFFECT_LABELS,
};
pub use provenance::{OffsetMap, RawSpan};
pub use sniff::{kind_for_extension, sniff, SniffReport, SNIFF_WINDOW};
