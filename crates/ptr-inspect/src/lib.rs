//! Structured runtime introspection: an owned [`InspectNode`] tree, a bounded [`walk`]
//! visitor, structural [`diff`], redaction wrappers (Invariant 13) and deterministic text,
//! compact and JSON renderers. No `Valuable`, tracing or `ptrctl` integration exists yet.

mod diff;
mod impls;
mod node;
mod path;
mod redact;
mod render;
mod visit;

pub use diff::{diff, Change, ChangeKind, MAX_DIFF_DEPTH};
pub use impls::InspectBytes;
pub use node::{
    BytesPreview, InspectNode, Inspectable, LengthClass, RedactionHint, MAX_BYTES_PREVIEW,
};
pub use path::{InspectPath, PathSegment};
pub use redact::{
    redact_keys, redact_keys_containing, redact_sensitive, PreviewPolicy, Private, RawEvidence,
    Secret, DEFAULT_SENSITIVE_KEYS, MAX_EVIDENCE_PREVIEW_BYTES,
};
pub use render::{
    render_compact, render_json, render_tree, RenderOptions, JSON_MAX_DEPTH, MAX_INDENT,
    MAX_RENDER_DEPTH, MAX_RENDER_ITEMS, MAX_TEXT_WIDTH,
};
pub use visit::{walk, Visitor, WalkError, WalkLimits, HARD_MAX_DEPTH, HARD_MAX_NODES};
