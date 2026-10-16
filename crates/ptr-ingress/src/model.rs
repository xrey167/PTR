//! Raw input and typed proposal data types.

use ptr_types::{ArtifactId, Probability, SemanticIssue};

#[derive(Clone, Debug, PartialEq)]
pub enum RawInput {
    Text(String),
    Bytes(Vec<u8>),
    Artifact(ArtifactId),
}

impl RawInput {
    /// The raw bytes when they are held inline; `None` for an unresolved artifact reference.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            RawInput::Text(text) => Some(text.as_bytes()),
            RawInput::Bytes(bytes) => Some(bytes),
            RawInput::Artifact(_) => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArtifactKind {
    Text,
    Image,
    Audio,
    Video,
    Pdf,
    Archive,
    Database,
    Unknown,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LabelScore {
    pub label: String,
    pub probability: Probability,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SemanticProposal {
    pub task_types: Vec<LabelScore>,
    pub reasoning_types: Vec<LabelScore>,
    pub effects: Vec<LabelScore>,
    pub issues: Vec<SemanticIssue>,
}
