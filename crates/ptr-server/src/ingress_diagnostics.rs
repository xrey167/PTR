//! Advisory ingress diagnostics for `POST /v1/requests`.
//!
//! The request text is ingested by `ptr_ingress::IngressPipeline` (rule-based reference
//! classifier) and summarised. The summary is a read-only report: it never blocks, rewrites or
//! authorizes anything, and a high-risk effect label appears here only as a diagnostic. The
//! runtime receives the original raw text unchanged.

use ptr_ingress::{
    ArtifactKind, CrossCheckStatus, DisagreementReason, IngestHints, IngressPipeline, LabelScore,
    RawInput,
};
use serde::Serialize;

#[derive(Debug)]
pub(crate) struct IngressDiagnosticsRunner {
    pipeline: IngressPipeline,
}

impl IngressDiagnosticsRunner {
    pub(crate) fn new() -> Self {
        Self {
            pipeline: IngressPipeline::default(),
        }
    }

    pub(crate) fn diagnose(&self, text: &str) -> IngressDiagnostics {
        match self
            .pipeline
            .ingest(RawInput::Text(text.to_owned()), &IngestHints::default())
        {
            Ok(record) => {
                let mut labels = Vec::new();
                for (axis, scores) in [
                    ("task", &record.proposal.task_types),
                    ("reasoning", &record.proposal.reasoning_types),
                    ("effect", &record.proposal.effects),
                ] {
                    labels.extend(scores.iter().map(|score| label(axis, score)));
                }
                IngressDiagnostics::Record(IngressRecordDiagnostics {
                    advisory: true,
                    kind: kind_name(record.kind),
                    labels,
                    verification: VerificationDiagnostics {
                        status: status_name(record.decision.status),
                        reasons: record.decision.reasons.iter().map(reason_code).collect(),
                    },
                })
            }
            Err(error) => IngressDiagnostics::Unavailable {
                advisory: true,
                error: error.to_string(),
            },
        }
    }
}

/// The value of the response field `ingress_diagnostics`.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub(crate) enum IngressDiagnostics {
    Record(IngressRecordDiagnostics),
    Unavailable { advisory: bool, error: String },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct IngressRecordDiagnostics {
    /// Always true: this report is advisory and is never an authorization.
    advisory: bool,
    kind: &'static str,
    labels: Vec<DiagnosticLabel>,
    verification: VerificationDiagnostics,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct DiagnosticLabel {
    axis: &'static str,
    label: String,
    probability: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct VerificationDiagnostics {
    status: &'static str,
    reasons: Vec<&'static str>,
}

fn label(axis: &'static str, score: &LabelScore) -> DiagnosticLabel {
    DiagnosticLabel {
        axis,
        label: score.label.clone(),
        probability: score.probability.get(),
    }
}

fn kind_name(kind: ArtifactKind) -> &'static str {
    match kind {
        ArtifactKind::Text => "text",
        ArtifactKind::Image => "image",
        ArtifactKind::Audio => "audio",
        ArtifactKind::Video => "video",
        ArtifactKind::Pdf => "pdf",
        ArtifactKind::Archive => "archive",
        ArtifactKind::Database => "database",
        ArtifactKind::Unknown => "unknown",
    }
}

fn status_name(status: CrossCheckStatus) -> &'static str {
    match status {
        CrossCheckStatus::Agree => "agree",
        CrossCheckStatus::SampleVerified => "sample_verified",
        CrossCheckStatus::Disputed => "disputed",
        CrossCheckStatus::NeedsFullCheck => "needs_full_check",
    }
}

fn reason_code(reason: &DisagreementReason) -> &'static str {
    match reason {
        DisagreementReason::NoRawEvidence => "no_raw_evidence",
        DisagreementReason::UnclassifiedArtifact => "unclassified_artifact",
        DisagreementReason::NormalizationRejected(_) => "normalization_rejected",
        DisagreementReason::LossyDecoding => "lossy_decoding",
        DisagreementReason::LowSniffConfidence => "low_sniff_confidence",
        DisagreementReason::DeclaredKindMismatch { .. } => "declared_kind_mismatch",
        DisagreementReason::ExtensionKindMismatch { .. } => "extension_kind_mismatch",
        DisagreementReason::JsonInvalid { .. } => "json_invalid",
        DisagreementReason::JsonTooDeep { .. } => "json_too_deep",
        DisagreementReason::ClassifierHardIssues { .. } => "classifier_hard_issues",
        DisagreementReason::ClassifierMismatch { .. } => "classifier_mismatch",
        DisagreementReason::ProbabilityDivergence { .. } => "probability_divergence",
        DisagreementReason::HighRiskEffect { .. } => "high_risk_effect",
    }
}

/// Body of a response that carries the diagnostics next to the unchanged response fields.
#[derive(Serialize)]
pub(crate) struct WithDiagnostics<'a> {
    #[serde(flatten)]
    pub(crate) response: &'a crate::ApiResponse,
    pub(crate) ingress_diagnostics: IngressDiagnostics,
}
