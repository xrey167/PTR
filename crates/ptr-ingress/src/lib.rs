//! Raw and typed ingress remain parallel. Typed interpretations never erase raw evidence.

use ptr_types::{ArtifactId, Probability, SemanticIssue};

#[derive(Clone, Debug, PartialEq)]
pub enum RawInput {
    Text(String),
    Bytes(Vec<u8>),
    Artifact(ArtifactId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArtifactKind { Text, Image, Audio, Video, Pdf, Archive, Database, Unknown }

#[derive(Clone, Debug, PartialEq)]
pub struct LabelScore { pub label: String, pub probability: Probability }

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SemanticProposal {
    pub task_types: Vec<LabelScore>,
    pub reasoning_types: Vec<LabelScore>,
    pub effects: Vec<LabelScore>,
    pub issues: Vec<SemanticIssue>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CrossCheckStatus { Agree, SampleVerified, Disputed, NeedsFullCheck }

pub fn cross_check(raw_present: bool, hard_issue_count: usize) -> CrossCheckStatus {
    match (raw_present, hard_issue_count) {
        (false, _) => CrossCheckStatus::NeedsFullCheck,
        (_, 0) => CrossCheckStatus::SampleVerified,
        _ => CrossCheckStatus::Disputed,
    }
}
