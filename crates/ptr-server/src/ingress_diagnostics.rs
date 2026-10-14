//! Advisory ingress diagnostics for `POST /v1/requests`.
//!
//! The request text is ingested by `ptr_ingress::IngressPipeline` (rule-based reference
//! classifier) and summarised. The summary is a read-only report: it never blocks, rewrites or
//! authorizes anything, and a high-risk effect label appears here only as a diagnostic. The
//! runtime receives the original raw text unchanged.
//!
//! Bounded (Invariant #8): the pipeline allocates per-character offset tables (about 56 bytes
//! per character), so at most `max_concurrent` analyses run at once (a request that finds no
//! free permit skips its diagnostics and never waits) and only a `max_analyzed_bytes` prefix
//! of the text is analysed.

use crate::options::{IngressDiagnosticsLimits, ServerOptionsError};
use ptr_ingress::{
    ArtifactKind, CrossCheckStatus, DisagreementReason, IngestHints, IngressConfig,
    IngressPipeline, LabelScore, RawInput,
};
use serde::Serialize;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
pub(crate) struct IngressDiagnosticsRunner {
    pipeline: IngressPipeline,
    max_concurrent: usize,
    max_analyzed_bytes: usize,
    in_flight: AtomicUsize,
}

/// One of the `max_concurrent` analysis permits; released on drop.
#[derive(Debug)]
pub(crate) struct DiagnosticsPermit<'a>(&'a AtomicUsize);

impl Drop for DiagnosticsPermit<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

impl IngressDiagnosticsRunner {
    pub(crate) fn new(limits: IngressDiagnosticsLimits) -> Result<Self, ServerOptionsError> {
        if limits.max_concurrent == 0 {
            return Err(ServerOptionsError::ZeroIngressConcurrency);
        }
        if limits.max_analyzed_bytes == 0 {
            return Err(ServerOptionsError::ZeroIngressAnalysisCap);
        }
        Ok(Self {
            // The pipeline refuses anything above the analysis cap itself as well.
            pipeline: IngressPipeline::new(IngressConfig {
                max_input_bytes: limits.max_analyzed_bytes,
                ..IngressConfig::default()
            }),
            max_concurrent: limits.max_concurrent,
            max_analyzed_bytes: limits.max_analyzed_bytes,
            in_flight: AtomicUsize::new(0),
        })
    }

    /// Take a permit without waiting; `None` when `max_concurrent` analyses are running.
    pub(crate) fn try_acquire(&self) -> Option<DiagnosticsPermit<'_>> {
        self.in_flight
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < self.max_concurrent).then_some(n + 1)
            })
            .ok()
            .map(|_| DiagnosticsPermit(&self.in_flight))
    }

    pub(crate) fn diagnose(&self, text: &str) -> IngressDiagnostics {
        let Some(_permit) = self.try_acquire() else {
            return IngressDiagnostics::Skipped {
                advisory: true,
                skipped: "busy",
            };
        };
        let mut end = text.len().min(self.max_analyzed_bytes);
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        let truncated = end < text.len();
        match self.pipeline.ingest(
            RawInput::Text(text[..end].to_owned()),
            &IngestHints::default(),
        ) {
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
                    truncated,
                    analyzed_bytes: end,
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
    Unavailable {
        advisory: bool,
        error: String,
    },
    /// No analysis permit was free; nothing was analysed and nothing waited.
    Skipped {
        advisory: bool,
        skipped: &'static str,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct IngressRecordDiagnostics {
    /// Always true: this report is advisory and is never an authorization.
    advisory: bool,
    /// True when the text was longer than the analysis cap and only a prefix was analysed.
    truncated: bool,
    /// Bytes of the request text that were analysed.
    analyzed_bytes: usize,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn runner(max_concurrent: usize, max_analyzed_bytes: usize) -> IngressDiagnosticsRunner {
        IngressDiagnosticsRunner::new(IngressDiagnosticsLimits {
            max_concurrent,
            max_analyzed_bytes,
        })
        .expect("valid limits")
    }

    fn json(diagnostics: &IngressDiagnostics) -> serde_json::Value {
        serde_json::to_value(diagnostics).expect("serializes")
    }

    #[test]
    fn permits_are_bounded_and_released_on_drop() {
        let runner = runner(2, 1024);
        let first = runner.try_acquire().expect("first");
        let second = runner.try_acquire().expect("second");
        assert!(runner.try_acquire().is_none());
        drop(first);
        let third = runner.try_acquire().expect("freed permit is reusable");
        assert!(runner.try_acquire().is_none());
        drop((second, third));
        assert_eq!(runner.in_flight.load(Ordering::Acquire), 0);
    }

    #[test]
    fn without_a_free_permit_the_analysis_is_skipped_not_waited_for() {
        let runner = runner(1, 1024);
        let held = runner.try_acquire().expect("permit");
        let skipped = json(&runner.diagnose("please delete the file"));
        assert_eq!(
            skipped,
            serde_json::json!({"advisory": true, "skipped": "busy"})
        );
        drop(held);
        let analysed = json(&runner.diagnose("please delete the file"));
        assert_eq!(analysed["kind"], "text");
        assert_eq!(analysed["truncated"], false);
        assert_eq!(runner.in_flight.load(Ordering::Acquire), 0);
    }

    #[test]
    fn text_beyond_the_cap_is_cut_at_a_char_boundary_and_reported() {
        let runner = runner(1, 5);
        // The accented e is two bytes (4..6) and straddles the cap of 5.
        let diagnostics = json(&runner.diagnose("abcd\u{e9}fgh"));
        assert_eq!(diagnostics["truncated"], true);
        assert_eq!(diagnostics["analyzed_bytes"], 4);
        let exact = json(&runner.diagnose("abcde"));
        assert_eq!(exact["truncated"], false);
        assert_eq!(exact["analyzed_bytes"], 5);
    }

    #[test]
    fn zero_limits_are_refused() {
        let refused = |max_concurrent, max_analyzed_bytes| {
            IngressDiagnosticsRunner::new(IngressDiagnosticsLimits {
                max_concurrent,
                max_analyzed_bytes,
            })
            .err()
        };
        assert!(matches!(
            refused(0, 1),
            Some(ServerOptionsError::ZeroIngressConcurrency)
        ));
        assert!(matches!(
            refused(1, 0),
            Some(ServerOptionsError::ZeroIngressAnalysisCap)
        ));
    }
}
