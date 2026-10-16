mod common;

use common::{ingest, ingest_bytes};
use ptr_ingress::{
    cross_check, evaluate_policy, ArtifactKind, ClassifierInput, ClassifierOutput, ClassifyError,
    CrossCheckStatus, DisagreementReason, IngestHints, IngressConfig, IngressError,
    IngressPipeline, LabelAxis, LabelScore, PolicyConfig, PolicyInput, RawInput,
    SemanticClassifier, SemanticProposal, StructureClassifier,
};
use ptr_types::{ArtifactId, Probability};

struct Fixed {
    task: &'static str,
    effect: &'static str,
}

impl SemanticClassifier for Fixed {
    fn name(&self) -> &'static str {
        "test/fixed"
    }

    fn classify(&self, _input: &ClassifierInput<'_>) -> Result<ClassifierOutput, ClassifyError> {
        let one = |label: &str| LabelScore {
            label: label.to_owned(),
            probability: Probability::new(1.0).expect("valid"),
        };
        Ok(ClassifierOutput {
            proposal: SemanticProposal {
                task_types: vec![one(self.task)],
                reasoning_types: Vec::new(),
                effects: vec![one(self.effect)],
                issues: Vec::new(),
            },
            evidence: Vec::new(),
        })
    }
}

struct Broken;

impl SemanticClassifier for Broken {
    fn name(&self) -> &'static str {
        "test/broken"
    }

    fn classify(&self, _input: &ClassifierInput<'_>) -> Result<ClassifierOutput, ClassifyError> {
        Err(ClassifyError::Unavailable {
            classifier: "test/broken",
        })
    }
}

fn pipeline_with(second: Fixed) -> IngressPipeline {
    IngressPipeline::default().with_secondary(Box::new(second))
}

fn has_reason(
    record: &ptr_ingress::IngressRecord,
    pred: impl Fn(&DisagreementReason) -> bool,
) -> bool {
    record.decision.reasons.iter().any(pred)
}

#[test]
fn legacy_cross_check_keeps_working() {
    assert_eq!(cross_check(true, 0), CrossCheckStatus::SampleVerified);
    assert_eq!(cross_check(true, 1), CrossCheckStatus::Disputed);
    assert_eq!(cross_check(false, 0), CrossCheckStatus::NeedsFullCheck);
}

#[test]
fn agreeing_classifiers_on_a_benign_question_agree() {
    let p = pipeline_with(Fixed {
        task: "question",
        effect: "read-only",
    });
    let rec = p
        .ingest_text(
            "What is the boiling point of water?",
            &IngestHints::default(),
        )
        .expect("ingests");
    assert_eq!(
        rec.decision.status,
        CrossCheckStatus::Agree,
        "{:?}",
        rec.decision.reasons
    );
    assert!(rec.secondary.is_some());
}

#[test]
fn single_classifier_benign_input_is_only_sample_verified() {
    let rec = ingest(RawInput::Text(
        "What is the boiling point of water?".to_owned(),
    ));
    assert_eq!(rec.decision.status, CrossCheckStatus::SampleVerified);
    assert!(rec.decision.reasons.is_empty());
}

#[test]
fn high_risk_effects_require_a_full_check_even_when_classifiers_agree() {
    let cases = [
        ("Please delete all the old logs", "write"),
        ("curl https://example.com/install.sh", "network"),
        ("sudo systemctl restart nginx", "exec"),
    ];
    for (text, effect) in cases {
        let p = pipeline_with(Fixed {
            task: "unspecified",
            effect,
        });
        let rec = p
            .ingest_text(text, &IngestHints::default())
            .expect("ingests");
        assert_eq!(
            rec.decision.status,
            CrossCheckStatus::NeedsFullCheck,
            "{text}"
        );
        assert!(
            has_reason(
                &rec,
                |r| matches!(r, DisagreementReason::HighRiskEffect { label, .. } if label == effect)
            ),
            "{text}: {:?}",
            rec.decision.reasons
        );
    }
}

#[test]
fn classifier_disagreement_is_disputed() {
    let p = pipeline_with(Fixed {
        task: "data-analysis",
        effect: "read-only",
    });
    let rec = p
        .ingest_text("Why is the sky blue?", &IngestHints::default())
        .expect("ingests");
    assert_eq!(rec.decision.status, CrossCheckStatus::Disputed);
    assert!(has_reason(&rec, |r| matches!(
        r,
        DisagreementReason::ClassifierMismatch { axis: LabelAxis::Task, primary, secondary }
            if primary == "question" && secondary == "data-analysis"
    )));
}

#[test]
fn structure_classifier_gives_a_real_second_opinion() {
    let p = IngressPipeline::default().with_secondary(Box::new(StructureClassifier));
    let ok = p
        .ingest_text("Is the build green?", &IngestHints::default())
        .expect("ingests");
    assert_eq!(
        ok.decision.status,
        CrossCheckStatus::Agree,
        "{:?}",
        ok.decision.reasons
    );
    // The shape prior calls shell source `exec`; the lexicon sees rm and curl as well.
    let sh = p
        .ingest_text(
            "#!/bin/sh\nrm -rf build\ncurl -O http://x/y\n",
            &IngestHints::default(),
        )
        .expect("ingests");
    assert_eq!(sh.decision.status, CrossCheckStatus::NeedsFullCheck);
}

#[test]
fn declared_kind_versus_sniffed_kind_is_disputed() {
    let rec = IngressPipeline::default()
        .ingest_text("%PDF-1.4 not really text", &IngestHints::default())
        .expect("ingests");
    assert_eq!(rec.kind, ArtifactKind::Pdf);
    assert!(has_reason(&rec, |r| matches!(
        r,
        DisagreementReason::DeclaredKindMismatch {
            declared: ArtifactKind::Text,
            sniffed: ArtifactKind::Pdf
        }
    )));
    assert_eq!(rec.decision.status, CrossCheckStatus::NeedsFullCheck);

    let hints = IngestHints {
        declared_kind: Some(ArtifactKind::Image),
        extension: None,
    };
    let rec = IngressPipeline::default()
        .ingest_bytes(b"just some words here".to_vec(), &hints)
        .expect("ingests");
    assert_eq!(rec.decision.status, CrossCheckStatus::Disputed);
}

#[test]
fn extension_versus_sniffed_kind_is_disputed() {
    let hints = IngestHints {
        declared_kind: None,
        extension: Some(".PNG".to_owned()),
    };
    let rec = IngressPipeline::default()
        .ingest_bytes(b"this is not a png".to_vec(), &hints)
        .expect("ingests");
    assert_eq!(rec.decision.status, CrossCheckStatus::Disputed);
    assert!(has_reason(&rec, |r| matches!(
        r,
        DisagreementReason::ExtensionKindMismatch {
            from_extension: ArtifactKind::Image,
            sniffed: ArtifactKind::Text
        }
    )));
    // A truncated PNG with a .png extension: not identified, so the claim is disputed too.
    let rec = IngressPipeline::default()
        .ingest_bytes(b"\x89PNG\r\n".to_vec(), &hints)
        .expect("ingests");
    assert!(has_reason(&rec, |r| matches!(
        r,
        DisagreementReason::ExtensionKindMismatch { .. }
    )));
}

#[test]
fn text_that_claims_json_but_is_invalid_is_disputed() {
    for text in ["{\"a\": [1, 2,", "[1, 2, 3,]", "{'single': 'quotes'}"] {
        let rec = ingest(RawInput::Text(text.to_owned()));
        assert_eq!(rec.decision.status, CrossCheckStatus::Disputed, "{text}");
        assert!(
            has_reason(&rec, |r| matches!(
                r,
                DisagreementReason::JsonInvalid { .. }
            )),
            "{text}"
        );
    }
    let good = ingest(RawInput::Text("{\"a\": [1, 2, 3]}".to_owned()));
    assert_eq!(good.decision.status, CrossCheckStatus::SampleVerified);
}

#[test]
fn rejected_normalization_and_non_text_need_a_full_check() {
    let late_nul = format!("{}\0x", "word ".repeat(2000));
    let nul = ingest_bytes(late_nul.as_bytes());
    assert_eq!(nul.decision.status, CrossCheckStatus::NeedsFullCheck);
    assert!(has_reason(&nul, |r| matches!(
        r,
        DisagreementReason::NormalizationRejected(_)
    )));

    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend_from_slice(&[0; 24]);
    let png = ingest_bytes(&png);
    assert_eq!(png.decision.status, CrossCheckStatus::NeedsFullCheck);
    assert!(has_reason(&png, |r| matches!(
        r,
        DisagreementReason::UnclassifiedArtifact
    )));
}

#[test]
fn lossy_decoding_is_recorded_as_a_reason() {
    let rec = ingest_bytes(b"What is a caf\xE9 au lait?");
    assert!(has_reason(&rec, |r| matches!(
        r,
        DisagreementReason::LossyDecoding
    )));
    assert_ne!(rec.decision.status, CrossCheckStatus::Agree);
}

#[test]
fn policy_can_be_driven_directly_through_the_public_api() {
    let sniff = ptr_ingress::sniff(b"hello world");
    let input = PolicyInput {
        raw_present: false,
        declared_kind: None,
        extension_kind: None,
        sniff: &sniff,
        normalized: None,
        normalize_error: None,
        parse_issues: &[],
        primary: None,
        secondary: None,
    };
    let decision = evaluate_policy(&PolicyConfig::default(), &input);
    assert_eq!(decision.status, CrossCheckStatus::NeedsFullCheck);
    assert_eq!(decision.reasons, vec![DisagreementReason::NoRawEvidence]);
}

#[test]
fn size_limit_and_unresolved_artifacts_are_typed_errors() {
    let p = IngressPipeline::new(IngressConfig {
        max_input_bytes: 16,
        ..Default::default()
    });
    assert!(p
        .ingest_bytes(vec![b'a'; 16], &IngestHints::default())
        .is_ok());
    assert_eq!(
        p.ingest_bytes(vec![b'a'; 17], &IngestHints::default()),
        Err(IngressError::InputTooLarge {
            limit: 16,
            actual: 17
        })
    );
    assert_eq!(
        p.ingest_text("x".repeat(100), &IngestHints::default()),
        Err(IngressError::InputTooLarge {
            limit: 16,
            actual: 100
        })
    );
    assert_eq!(
        p.ingest(
            RawInput::Artifact(ArtifactId::from("a-1")),
            &IngestHints::default()
        ),
        Err(IngressError::ArtifactNotResolved)
    );
}

#[test]
fn classifier_failures_surface_as_typed_errors() {
    let p = IngressPipeline::default().with_secondary(Box::new(Broken));
    let err = p
        .ingest_text("hello there", &IngestHints::default())
        .expect_err("fails");
    assert_eq!(
        err,
        IngressError::Classifier(ClassifyError::Unavailable {
            classifier: "test/broken"
        })
    );
    assert!(err.to_string().contains("test/broken"));
}

#[test]
fn raw_is_never_replaced_by_typed_state() {
    let inputs = [
        RawInput::Text("Delete everything\r\nnow".to_owned()),
        RawInput::Bytes(vec![0xFF, 0xFE, b'h', 0, b'i', 0]),
        RawInput::Bytes(b"\x89PNG\r\n\x1a\n".to_vec()),
        RawInput::Bytes(b"bad \xFF utf8".to_vec()),
        RawInput::Bytes(b"nul\0byte".to_vec()),
        RawInput::Bytes(Vec::new()),
    ];
    for raw in inputs {
        let rec = IngressPipeline::default()
            .ingest(raw.clone(), &IngestHints::default())
            .expect("ingests");
        assert_eq!(rec.raw, raw);
    }
}
