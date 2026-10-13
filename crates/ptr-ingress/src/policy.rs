//! Risk- and disagreement-driven verification policy: decides how much raw-versus-typed
//! verification a classified input needs, and says why.

use crate::classify::{ClassifierOutput, LabelAxis, UNSPECIFIED_LABEL};
use crate::model::{ArtifactKind, LabelScore};
use crate::normalize::{NormalizeError, NormalizedText};
use crate::parse::ParseIssue;
use crate::sniff::SniffReport;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CrossCheckStatus {
    Agree,
    SampleVerified,
    Disputed,
    NeedsFullCheck,
}

pub fn cross_check(raw_present: bool, hard_issue_count: usize) -> CrossCheckStatus {
    match (raw_present, hard_issue_count) {
        (false, _) => CrossCheckStatus::NeedsFullCheck,
        (_, 0) => CrossCheckStatus::SampleVerified,
        _ => CrossCheckStatus::Disputed,
    }
}

/// Effect labels that always force a full check.
pub const HIGH_RISK_EFFECT_LABELS: [&str; 3] = ["write", "network", "exec"];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyConfig {
    /// Probability at or above which a high-risk effect label triggers a full check.
    pub risk_threshold: f32,
    /// Per-label probability gap between two classifiers at or above which they are disputed.
    pub divergence_threshold: f32,
    /// Sniff confidence below which the artifact kind is treated as weakly supported.
    pub min_sniff_confidence: f32,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            risk_threshold: 0.2,
            divergence_threshold: 0.5,
            min_sniff_confidence: 0.6,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum DisagreementReason {
    NoRawEvidence,
    /// A non-text artifact: sniffed, but there is no typed interpretation to cross-check.
    UnclassifiedArtifact,
    NormalizationRejected(NormalizeError),
    LossyDecoding,
    LowSniffConfidence,
    DeclaredKindMismatch {
        declared: ArtifactKind,
        sniffed: ArtifactKind,
    },
    ExtensionKindMismatch {
        from_extension: ArtifactKind,
        sniffed: ArtifactKind,
    },
    JsonInvalid {
        offset: usize,
    },
    JsonTooDeep {
        limit: usize,
    },
    ClassifierHardIssues {
        count: usize,
    },
    ClassifierMismatch {
        axis: LabelAxis,
        primary: String,
        secondary: String,
    },
    ProbabilityDivergence {
        axis: LabelAxis,
        label: String,
        delta: f32,
    },
    HighRiskEffect {
        label: String,
        probability: f32,
    },
}

impl DisagreementReason {
    /// The minimum verification level this reason demands.
    pub fn severity(&self) -> CrossCheckStatus {
        match self {
            DisagreementReason::NoRawEvidence
            | DisagreementReason::UnclassifiedArtifact
            | DisagreementReason::NormalizationRejected(_)
            | DisagreementReason::HighRiskEffect { .. } => CrossCheckStatus::NeedsFullCheck,
            DisagreementReason::DeclaredKindMismatch { .. }
            | DisagreementReason::ExtensionKindMismatch { .. }
            | DisagreementReason::JsonInvalid { .. }
            | DisagreementReason::JsonTooDeep { .. }
            | DisagreementReason::ClassifierHardIssues { .. }
            | DisagreementReason::ClassifierMismatch { .. }
            | DisagreementReason::ProbabilityDivergence { .. } => CrossCheckStatus::Disputed,
            DisagreementReason::LossyDecoding | DisagreementReason::LowSniffConfidence => {
                CrossCheckStatus::SampleVerified
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerificationDecision {
    pub status: CrossCheckStatus,
    pub reasons: Vec<DisagreementReason>,
}

/// Everything the policy looks at. `primary` is `None` for artifacts that were not classified.
#[derive(Clone, Copy, Debug)]
pub struct PolicyInput<'a> {
    pub raw_present: bool,
    pub declared_kind: Option<ArtifactKind>,
    pub extension_kind: Option<ArtifactKind>,
    pub sniff: &'a SniffReport,
    pub normalized: Option<&'a NormalizedText>,
    pub normalize_error: Option<NormalizeError>,
    pub parse_issues: &'a [ParseIssue],
    pub primary: Option<&'a ClassifierOutput>,
    pub secondary: Option<&'a ClassifierOutput>,
}

fn rank(status: CrossCheckStatus) -> u8 {
    match status {
        CrossCheckStatus::Agree => 0,
        CrossCheckStatus::SampleVerified => 1,
        CrossCheckStatus::Disputed => 2,
        CrossCheckStatus::NeedsFullCheck => 3,
    }
}

fn top(scores: &[LabelScore]) -> Option<&LabelScore> {
    scores.first()
}

fn abstains(scores: &[LabelScore]) -> bool {
    top(scores).is_none_or(|s| s.label == UNSPECIFIED_LABEL)
}

fn probability_of(scores: &[LabelScore], label: &str) -> f32 {
    scores
        .iter()
        .find(|s| s.label == label)
        .map_or(0.0, |s| s.probability.get())
}

fn compare_axis(
    axis: LabelAxis,
    config: &PolicyConfig,
    a: &[LabelScore],
    b: &[LabelScore],
    reasons: &mut Vec<DisagreementReason>,
) {
    let (Some(ta), Some(tb)) = (top(a), top(b)) else {
        return;
    };
    if abstains(a) || abstains(b) {
        return;
    }
    if ta.label != tb.label {
        reasons.push(DisagreementReason::ClassifierMismatch {
            axis,
            primary: ta.label.clone(),
            secondary: tb.label.clone(),
        });
        return;
    }
    let widest = a
        .iter()
        .map(|s| s.label.as_str())
        .chain(b.iter().map(|s| s.label.as_str()))
        .map(|label| {
            (
                label,
                (probability_of(a, label) - probability_of(b, label)).abs(),
            )
        })
        .filter(|(_, delta)| *delta >= config.divergence_threshold)
        .max_by(|x, y| x.1.total_cmp(&y.1).then_with(|| y.0.cmp(x.0)));
    if let Some((label, delta)) = widest {
        reasons.push(DisagreementReason::ProbabilityDivergence {
            axis,
            label: label.to_owned(),
            delta,
        });
    }
}

/// Evaluates the policy. Status is the most severe reason; without any reason it is `Agree`
/// when a second classifier confirmed the first and `SampleVerified` otherwise.
pub fn evaluate_policy(config: &PolicyConfig, input: &PolicyInput<'_>) -> VerificationDecision {
    let mut reasons = Vec::new();
    if !input.raw_present {
        reasons.push(DisagreementReason::NoRawEvidence);
    }
    if let Some(declared) = input.declared_kind {
        if declared != ArtifactKind::Unknown && declared != input.sniff.kind {
            reasons.push(DisagreementReason::DeclaredKindMismatch {
                declared,
                sniffed: input.sniff.kind,
            });
        }
    }
    if let Some(from_extension) = input.extension_kind {
        if from_extension != input.sniff.kind {
            reasons.push(DisagreementReason::ExtensionKindMismatch {
                from_extension,
                sniffed: input.sniff.kind,
            });
        }
    }
    if input.sniff.confidence.get() < config.min_sniff_confidence {
        reasons.push(DisagreementReason::LowSniffConfidence);
    }
    if let Some(error) = input.normalize_error {
        reasons.push(DisagreementReason::NormalizationRejected(error));
    }
    if input.normalized.is_some_and(NormalizedText::is_lossy) {
        reasons.push(DisagreementReason::LossyDecoding);
    }
    for issue in input.parse_issues {
        match issue {
            ParseIssue::JsonInvalid { offset } => {
                reasons.push(DisagreementReason::JsonInvalid { offset: *offset });
            }
            ParseIssue::JsonTooDeep { limit } => {
                reasons.push(DisagreementReason::JsonTooDeep { limit: *limit });
            }
            ParseIssue::UnclosedCodeFence { .. } | ParseIssue::SegmentLimitReached { .. } => {}
        }
    }
    match input.primary {
        None => {
            if input.raw_present && input.normalize_error.is_none() {
                reasons.push(DisagreementReason::UnclassifiedArtifact);
            }
        }
        Some(primary) => {
            let hard = primary.proposal.issues.iter().filter(|i| i.hard).count();
            if hard > 0 {
                reasons.push(DisagreementReason::ClassifierHardIssues { count: hard });
            }
            for label in HIGH_RISK_EFFECT_LABELS {
                let p = [Some(primary), input.secondary]
                    .into_iter()
                    .flatten()
                    .map(|out| probability_of(&out.proposal.effects, label))
                    .fold(0.0_f32, f32::max);
                if p > 0.0 && p >= config.risk_threshold {
                    reasons.push(DisagreementReason::HighRiskEffect {
                        label: label.to_owned(),
                        probability: p,
                    });
                }
            }
            if let Some(second) = input.secondary {
                let (a, b) = (&primary.proposal, &second.proposal);
                compare_axis(
                    LabelAxis::Task,
                    config,
                    &a.task_types,
                    &b.task_types,
                    &mut reasons,
                );
                compare_axis(
                    LabelAxis::Reasoning,
                    config,
                    &a.reasoning_types,
                    &b.reasoning_types,
                    &mut reasons,
                );
                compare_axis(
                    LabelAxis::Effect,
                    config,
                    &a.effects,
                    &b.effects,
                    &mut reasons,
                );
            }
        }
    }
    let status = reasons
        .iter()
        .map(DisagreementReason::severity)
        .max_by_key(|s| rank(*s))
        .unwrap_or(if input.secondary.is_some() {
            CrossCheckStatus::Agree
        } else {
            CrossCheckStatus::SampleVerified
        });
    VerificationDecision { status, reasons }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify::{ClassifierInput, RuleClassifier, SemanticClassifier};
    use crate::normalize::{normalize, NormalizeOptions};
    use crate::parse::{parse, ParseOptions};
    use crate::sniff::sniff;
    use ptr_types::Probability;

    fn score(label: &str, p: f32) -> LabelScore {
        LabelScore {
            label: label.to_owned(),
            probability: Probability::new(p).expect("p"),
        }
    }

    fn output(task: &[(&str, f32)], effects: &[(&str, f32)]) -> ClassifierOutput {
        let mut out = ClassifierOutput::default();
        out.proposal.task_types = task.iter().map(|(l, p)| score(l, *p)).collect();
        out.proposal.effects = effects.iter().map(|(l, p)| score(l, *p)).collect();
        out
    }

    fn decide(
        text: &str,
        declared: Option<ArtifactKind>,
        second: Option<&ClassifierOutput>,
    ) -> VerificationDecision {
        let raw = text.as_bytes();
        let sn = sniff(raw);
        let n = normalize(raw, &NormalizeOptions::default()).expect("ok");
        let d = parse(&n, &ParseOptions::default());
        let primary = RuleClassifier
            .classify(&ClassifierInput {
                kind: sn.kind,
                text: &n,
                document: &d,
            })
            .expect("ok");
        evaluate_policy(
            &PolicyConfig::default(),
            &PolicyInput {
                raw_present: true,
                declared_kind: declared,
                extension_kind: None,
                sniff: &sn,
                normalized: Some(&n),
                normalize_error: None,
                parse_issues: &d.issues,
                primary: Some(&primary),
                secondary: second,
            },
        )
    }

    #[test]
    fn legacy_cross_check_is_unchanged() {
        assert_eq!(cross_check(false, 0), CrossCheckStatus::NeedsFullCheck);
        assert_eq!(cross_check(true, 0), CrossCheckStatus::SampleVerified);
        assert_eq!(cross_check(true, 2), CrossCheckStatus::Disputed);
    }

    #[test]
    fn clean_single_classifier_is_sample_verified_and_two_agreeing_is_agree() {
        let one = decide("What is two plus two?", None, None);
        assert_eq!(one.status, CrossCheckStatus::SampleVerified);
        assert!(one.reasons.is_empty());
        let second = output(&[("question", 1.0)], &[("read-only", 1.0)]);
        let two = decide("What is two plus two?", None, Some(&second));
        assert_eq!(two.status, CrossCheckStatus::Agree, "{:?}", two.reasons);
    }

    #[test]
    fn high_risk_effects_always_need_full_check_even_when_classifiers_agree() {
        for text in [
            "Delete the build directory",
            "curl http://example.com",
            "sudo reboot now",
        ] {
            let second = output(&[("unspecified", 1.0)], &[("write", 1.0), ("network", 1.0)]);
            let d = decide(text, None, Some(&second));
            assert_eq!(d.status, CrossCheckStatus::NeedsFullCheck, "{text}");
            assert!(d
                .reasons
                .iter()
                .any(|r| matches!(r, DisagreementReason::HighRiskEffect { .. })));
        }
    }

    #[test]
    fn second_classifier_alone_can_raise_risk() {
        let second = output(&[("question", 1.0)], &[("exec", 1.0)]);
        let d = decide("What is two plus two?", None, Some(&second));
        assert_eq!(d.status, CrossCheckStatus::NeedsFullCheck);
    }

    #[test]
    fn top_label_mismatch_is_disputed_but_abstention_is_not() {
        let other = output(&[("data-analysis", 1.0)], &[("read-only", 1.0)]);
        let d = decide("What is two plus two?", None, Some(&other));
        assert_eq!(d.status, CrossCheckStatus::Disputed);
        assert!(matches!(
            d.reasons.as_slice(),
            [DisagreementReason::ClassifierMismatch {
                axis: LabelAxis::Task,
                ..
            }]
        ));
        let abstain = output(&[("unspecified", 1.0)], &[("read-only", 1.0)]);
        assert_eq!(
            decide("What is two plus two?", None, Some(&abstain)).status,
            CrossCheckStatus::Agree
        );
    }

    #[test]
    fn same_top_label_with_large_probability_gap_is_divergence() {
        let a = [score("question", 0.9), score("instruction", 0.1)];
        let b = [score("instruction", 0.7), score("question", 0.3)];
        let mut reasons = Vec::new();
        compare_axis(
            LabelAxis::Task,
            &PolicyConfig::default(),
            &a,
            &a,
            &mut reasons,
        );
        assert!(reasons.is_empty());
        let c = [
            score("question", 0.35),
            score("instruction", 0.3),
            score("data-analysis", 0.35),
        ];
        compare_axis(
            LabelAxis::Task,
            &PolicyConfig::default(),
            &a,
            &c,
            &mut reasons,
        );
        assert!(matches!(
            reasons.as_slice(),
            [DisagreementReason::ProbabilityDivergence { .. }]
        ));
        let mut mism = Vec::new();
        compare_axis(LabelAxis::Task, &PolicyConfig::default(), &a, &b, &mut mism);
        assert!(matches!(
            mism.as_slice(),
            [DisagreementReason::ClassifierMismatch { .. }]
        ));
    }

    #[test]
    fn declared_kind_and_extension_mismatches_are_disputed() {
        let d = decide("%PDF-1.4 hello", Some(ArtifactKind::Text), None);
        assert_eq!(d.status, CrossCheckStatus::Disputed);
        assert!(matches!(
            d.reasons.first(),
            Some(DisagreementReason::DeclaredKindMismatch {
                declared: ArtifactKind::Text,
                sniffed: ArtifactKind::Pdf
            })
        ));
        let raw = b"plain text";
        let sn = sniff(raw);
        let decision = evaluate_policy(
            &PolicyConfig::default(),
            &PolicyInput {
                raw_present: true,
                declared_kind: None,
                extension_kind: Some(ArtifactKind::Image),
                sniff: &sn,
                normalized: None,
                normalize_error: None,
                parse_issues: &[],
                primary: Some(&output(&[("question", 1.0)], &[("read-only", 1.0)])),
                secondary: None,
            },
        );
        assert_eq!(decision.status, CrossCheckStatus::Disputed);
    }

    #[test]
    fn invalid_json_is_disputed() {
        let d = decide("{\"a\": [1, 2", None, None);
        assert_eq!(d.status, CrossCheckStatus::Disputed);
        assert!(d
            .reasons
            .iter()
            .any(|r| matches!(r, DisagreementReason::JsonInvalid { .. })));
    }

    #[test]
    fn missing_raw_and_unclassified_artifacts_need_full_check() {
        let sn = sniff(b"\x89PNG\r\n\x1a\n");
        let base = PolicyInput {
            raw_present: false,
            declared_kind: None,
            extension_kind: None,
            sniff: &sn,
            normalized: None,
            normalize_error: None,
            parse_issues: &[],
            primary: None,
            secondary: None,
        };
        let cfg = PolicyConfig::default();
        let d = evaluate_policy(&cfg, &base);
        assert_eq!(d.status, CrossCheckStatus::NeedsFullCheck);
        assert_eq!(d.reasons, vec![DisagreementReason::NoRawEvidence]);
        let d = evaluate_policy(
            &cfg,
            &PolicyInput {
                raw_present: true,
                ..base
            },
        );
        assert_eq!(d.reasons, vec![DisagreementReason::UnclassifiedArtifact]);
    }

    #[test]
    fn lossy_decoding_demotes_agree_to_sample_verified() {
        let raw = b"What is caf\xE9?";
        let sn = sniff(raw);
        let n = normalize(raw, &NormalizeOptions::default()).expect("ok");
        assert!(n.is_lossy());
        let primary = output(&[("question", 1.0)], &[("read-only", 1.0)]);
        let d = evaluate_policy(
            &PolicyConfig::default(),
            &PolicyInput {
                raw_present: true,
                declared_kind: None,
                extension_kind: None,
                sniff: &sn,
                normalized: Some(&n),
                normalize_error: None,
                parse_issues: &[],
                primary: Some(&primary),
                secondary: Some(&primary),
            },
        );
        assert_eq!(d.status, CrossCheckStatus::SampleVerified);
        assert!(d.reasons.contains(&DisagreementReason::LossyDecoding));
    }

    #[test]
    fn classifier_hard_issue_is_disputed() {
        let mut primary = output(&[("question", 1.0)], &[("read-only", 1.0)]);
        primary.proposal.issues.push(ptr_types::SemanticIssue {
            code: "x".to_owned(),
            message: "m".to_owned(),
            hard: true,
        });
        let sn = sniff(b"hello there");
        let d = evaluate_policy(
            &PolicyConfig::default(),
            &PolicyInput {
                raw_present: true,
                declared_kind: None,
                extension_kind: None,
                sniff: &sn,
                normalized: None,
                normalize_error: None,
                parse_issues: &[],
                primary: Some(&primary),
                secondary: None,
            },
        );
        assert_eq!(d.status, CrossCheckStatus::Disputed);
        assert_eq!(
            d.reasons,
            vec![DisagreementReason::ClassifierHardIssues { count: 1 }]
        );
    }
}
