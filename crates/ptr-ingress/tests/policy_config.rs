//! Policy thresholds are validated; an invalid configuration can never fail open.

use ptr_ingress::{
    evaluate_policy, ClassifierInput, ClassifierOutput, ClassifyError, CrossCheckStatus,
    DisagreementReason, IngestHints, IngressConfig, IngressPipeline, LabelScore, PolicyConfig,
    PolicyConfigError, PolicyInput, PolicyThreshold, SemanticClassifier, SemanticProposal,
};
use ptr_types::Probability;

struct Exec;

impl SemanticClassifier for Exec {
    fn name(&self) -> &'static str {
        "test/exec"
    }

    fn classify(&self, _input: &ClassifierInput<'_>) -> Result<ClassifierOutput, ClassifyError> {
        let one = |label: &str| LabelScore {
            label: label.to_owned(),
            probability: Probability::new(1.0).expect("valid"),
        };
        Ok(ClassifierOutput {
            proposal: SemanticProposal {
                task_types: vec![one("command")],
                reasoning_types: Vec::new(),
                effects: vec![one("exec")],
                issues: Vec::new(),
            },
            evidence: Vec::new(),
        })
    }
}

fn config_with(policy: PolicyConfig) -> IngressConfig {
    IngressConfig {
        policy,
        ..IngressConfig::default()
    }
}

fn bad_values() -> [f32; 5] {
    [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.5]
}

#[test]
fn new_never_fails_open_on_an_invalid_risk_threshold() {
    for bad in bad_values() {
        let config = config_with(PolicyConfig {
            risk_threshold: bad,
            ..PolicyConfig::default()
        });
        let pipeline = IngressPipeline::new(config)
            .with_primary(Box::new(Exec))
            .with_secondary(Box::new(Exec));
        let record = pipeline
            .ingest_text("run it", &IngestHints::default())
            .expect("ingests");
        assert_eq!(
            record.decision.status,
            CrossCheckStatus::NeedsFullCheck,
            "threshold {bad}"
        );
        assert!(record.decision.reasons.iter().any(
            |r| matches!(r, DisagreementReason::HighRiskEffect { label, .. } if label == "exec")
        ));
    }
}

#[test]
fn try_new_rejects_each_invalid_threshold_with_a_typed_error() {
    for bad in bad_values() {
        let cases = [
            (
                PolicyConfig {
                    risk_threshold: bad,
                    ..PolicyConfig::default()
                },
                PolicyThreshold::Risk,
            ),
            (
                PolicyConfig {
                    divergence_threshold: bad,
                    ..PolicyConfig::default()
                },
                PolicyThreshold::Divergence,
            ),
            (
                PolicyConfig {
                    min_sniff_confidence: bad,
                    ..PolicyConfig::default()
                },
                PolicyThreshold::MinSniffConfidence,
            ),
        ];
        for (policy, threshold) in cases {
            let err = IngressPipeline::try_new(config_with(policy)).expect_err("rejected");
            assert!(matches!(
                err,
                PolicyConfigError::OutOfRange { threshold: t, .. } if t == threshold
            ));
            assert!(err.to_string().contains("[0, 1]"));
        }
    }
}

#[test]
fn try_new_accepts_boundaries_and_defaults() {
    for value in [0.0_f32, 1.0] {
        let policy = PolicyConfig {
            risk_threshold: value,
            divergence_threshold: value,
            min_sniff_confidence: value,
        };
        assert!(IngressPipeline::try_new(config_with(policy)).is_ok());
    }
    assert!(IngressPipeline::try_new(IngressConfig::default()).is_ok());
}

#[test]
fn new_replaces_invalid_thresholds_with_the_strictest_valid_values() {
    let pipeline = IngressPipeline::new(config_with(PolicyConfig {
        risk_threshold: f32::NAN,
        divergence_threshold: f32::NAN,
        min_sniff_confidence: f32::NAN,
    }));
    let policy = pipeline.config().policy;
    assert_eq!(policy.risk_threshold, 0.0);
    assert_eq!(policy.divergence_threshold, 0.0);
    assert_eq!(policy.min_sniff_confidence, 1.0);
}

#[test]
fn evaluate_policy_directly_also_fails_closed() {
    let sniff = ptr_ingress::sniff(b"hello world");
    let output = ClassifierOutput {
        proposal: SemanticProposal {
            task_types: Vec::new(),
            reasoning_types: Vec::new(),
            effects: vec![LabelScore {
                label: "exec".to_owned(),
                probability: Probability::new(1.0).expect("valid"),
            }],
            issues: Vec::new(),
        },
        evidence: Vec::new(),
    };
    let input = PolicyInput {
        raw_present: true,
        declared_kind: None,
        extension_kind: None,
        sniff: &sniff,
        normalized: None,
        normalize_error: None,
        parse_issues: &[],
        primary: Some(&output),
        secondary: Some(&output),
    };
    let policy = PolicyConfig {
        risk_threshold: f32::NAN,
        ..PolicyConfig::default()
    };
    let decision = evaluate_policy(&policy, &input);
    assert_eq!(decision.status, CrossCheckStatus::NeedsFullCheck);
}
