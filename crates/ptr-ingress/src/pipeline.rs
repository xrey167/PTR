//! The ingress pipeline: raw input to a record that holds the untouched raw next to its
//! sniffed kind, normalized text, segments, semantic proposal and verification decision.

use std::fmt;

use ptr_types::SemanticIssue;

use crate::classify::{
    ClassifierInput, ClassifierOutput, ClassifyError, LabelEvidence, RuleClassifier,
    SemanticClassifier,
};
use crate::model::{ArtifactKind, RawInput, SemanticProposal};
use crate::normalize::{normalize, NormalizeError, NormalizeOptions, NormalizedText, TextEncoding};
use crate::parse::{parse, ParseOptions, ParsedDocument, Segment};
use crate::policy::{evaluate_policy, PolicyConfig, PolicyInput, VerificationDecision};
use crate::sniff::{kind_for_extension, sniff, SniffReport};

/// Default input bound: 4 MiB.
const DEFAULT_MAX_INPUT_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IngressConfig {
    /// Inputs longer than this are rejected with [`IngressError::InputTooLarge`].
    pub max_input_bytes: usize,
    pub normalize: NormalizeOptions,
    pub parse: ParseOptions,
    pub policy: PolicyConfig,
}

impl Default for IngressConfig {
    fn default() -> Self {
        Self {
            max_input_bytes: DEFAULT_MAX_INPUT_BYTES,
            normalize: NormalizeOptions::default(),
            parse: ParseOptions::default(),
            policy: PolicyConfig::default(),
        }
    }
}

/// Caller-supplied claims about the input; the policy compares them with what was sniffed.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct IngestHints {
    pub declared_kind: Option<ArtifactKind>,
    /// File extension, with or without the dot.
    pub extension: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IngressError {
    InputTooLarge {
        limit: usize,
        actual: usize,
    },
    /// An artifact reference carries no bytes; resolve it before ingesting.
    ArtifactNotResolved,
    Classifier(ClassifyError),
}

impl fmt::Display for IngressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IngressError::InputTooLarge { limit, actual } => {
                write!(f, "input of {actual} bytes exceeds the limit of {limit}")
            }
            IngressError::ArtifactNotResolved => {
                f.write_str("artifact reference has no inline bytes")
            }
            IngressError::Classifier(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for IngressError {}

impl From<ClassifyError> for IngressError {
    fn from(error: ClassifyError) -> Self {
        IngressError::Classifier(error)
    }
}

/// Result of ingesting one input. `raw` is exactly what was submitted (Invariant: raw is never
/// replaced by typed state); every other field is a derived, provisional interpretation.
#[derive(Clone, Debug, PartialEq)]
pub struct IngressRecord {
    pub raw: RawInput,
    pub sniff: SniffReport,
    pub kind: ArtifactKind,
    pub normalized: Option<NormalizedText>,
    /// Set when the input sniffed as text but normalization refused it.
    pub normalize_error: Option<NormalizeError>,
    pub document: Option<ParsedDocument>,
    pub proposal: SemanticProposal,
    pub evidence: Vec<LabelEvidence>,
    /// Output of the second classifier, when one is configured.
    pub secondary: Option<ClassifierOutput>,
    pub decision: VerificationDecision,
}

impl IngressRecord {
    pub fn raw_bytes(&self) -> Option<&[u8]> {
        self.raw.as_bytes()
    }

    pub fn segments(&self) -> &[Segment] {
        self.document
            .as_ref()
            .map_or(&[], |d| d.segments.as_slice())
    }
}

pub struct IngressPipeline {
    config: IngressConfig,
    primary: Box<dyn SemanticClassifier>,
    secondary: Option<Box<dyn SemanticClassifier>>,
}

impl fmt::Debug for IngressPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IngressPipeline")
            .field("config", &self.config)
            .field("primary", &self.primary.name())
            .field("secondary", &self.secondary.as_ref().map(|c| c.name()))
            .finish()
    }
}

impl Default for IngressPipeline {
    fn default() -> Self {
        Self::new(IngressConfig::default())
    }
}

fn issue(code: &str, message: String, hard: bool) -> SemanticIssue {
    SemanticIssue {
        code: code.to_owned(),
        message,
        hard,
    }
}

impl IngressPipeline {
    /// A pipeline with the rule-based reference classifier and no second opinion.
    pub fn new(config: IngressConfig) -> Self {
        Self {
            config,
            primary: Box::new(RuleClassifier),
            secondary: None,
        }
    }

    pub fn with_primary(mut self, classifier: Box<dyn SemanticClassifier>) -> Self {
        self.primary = classifier;
        self
    }

    pub fn with_secondary(mut self, classifier: Box<dyn SemanticClassifier>) -> Self {
        self.secondary = Some(classifier);
        self
    }

    pub fn config(&self) -> &IngressConfig {
        &self.config
    }

    pub fn ingest(
        &self,
        raw: RawInput,
        hints: &IngestHints,
    ) -> Result<IngressRecord, IngressError> {
        let bytes = raw.as_bytes().ok_or(IngressError::ArtifactNotResolved)?;
        if bytes.len() > self.config.max_input_bytes {
            return Err(IngressError::InputTooLarge {
                limit: self.config.max_input_bytes,
                actual: bytes.len(),
            });
        }
        let sniffed = sniff(bytes);
        let declared_kind = hints
            .declared_kind
            .or_else(|| matches!(raw, RawInput::Text(_)).then_some(ArtifactKind::Text));
        let extension_kind = hints.extension.as_deref().and_then(kind_for_extension);

        let mut options = self.config.normalize;
        if options.forced_encoding.is_none()
            && matches!(
                sniffed.text_encoding,
                Some(TextEncoding::Utf16Le | TextEncoding::Utf16Be)
            )
        {
            options.forced_encoding = sniffed.text_encoding;
        }

        let mut normalized = None;
        let mut normalize_error = None;
        let mut document = None;
        let mut primary = None;
        let mut secondary = None;
        if sniffed.kind == ArtifactKind::Text {
            match normalize(bytes, &options) {
                Ok(text) => {
                    let doc = parse(&text, &self.config.parse);
                    let input = ClassifierInput {
                        kind: sniffed.kind,
                        text: &text,
                        document: &doc,
                    };
                    primary = Some(self.primary.classify(&input)?);
                    if let Some(second) = &self.secondary {
                        secondary = Some(second.classify(&input)?);
                    }
                    document = Some(doc);
                    normalized = Some(text);
                }
                Err(error) => normalize_error = Some(error),
            }
        }

        let no_issues = Vec::new();
        let decision = evaluate_policy(
            &self.config.policy,
            &PolicyInput {
                raw_present: true,
                declared_kind,
                extension_kind,
                sniff: &sniffed,
                normalized: normalized.as_ref(),
                normalize_error,
                parse_issues: document.as_ref().map_or(&no_issues, |d| &d.issues),
                primary: primary.as_ref(),
                secondary: secondary.as_ref(),
            },
        );

        let mut proposal = SemanticProposal::default();
        let mut evidence = Vec::new();
        if let Some(out) = primary {
            proposal = out.proposal;
            evidence = out.evidence;
        }
        if let Some(error) = normalize_error {
            proposal
                .issues
                .push(issue("normalize.rejected", error.to_string(), true));
        } else if sniffed.kind != ArtifactKind::Text {
            proposal.issues.push(issue(
                "ingress.non_text_artifact",
                format!(
                    "{} is not text; no typed interpretation produced",
                    sniffed.mime
                ),
                false,
            ));
        }
        if normalized.as_ref().is_some_and(NormalizedText::is_lossy) {
            proposal.issues.push(issue(
                "normalize.lossy",
                "decoding was lossy or guessed".to_owned(),
                false,
            ));
        }
        if let Some(doc) = &document {
            proposal
                .issues
                .extend(doc.issues.iter().map(|i| i.to_semantic_issue()));
        }

        Ok(IngressRecord {
            raw,
            kind: sniffed.kind,
            sniff: sniffed,
            normalized,
            normalize_error,
            document,
            proposal,
            evidence,
            secondary,
            decision,
        })
    }

    pub fn ingest_bytes(
        &self,
        bytes: Vec<u8>,
        hints: &IngestHints,
    ) -> Result<IngressRecord, IngressError> {
        self.ingest(RawInput::Bytes(bytes), hints)
    }

    pub fn ingest_text(
        &self,
        text: impl Into<String>,
        hints: &IngestHints,
    ) -> Result<IngressRecord, IngressError> {
        self.ingest(RawInput::Text(text.into()), hints)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::CrossCheckStatus;
    use ptr_types::ArtifactId;

    #[test]
    fn size_limit_is_enforced_at_the_boundary() {
        let p = IngressPipeline::new(IngressConfig {
            max_input_bytes: 4,
            ..Default::default()
        });
        assert!(p
            .ingest_bytes(b"abcd".to_vec(), &IngestHints::default())
            .is_ok());
        assert_eq!(
            p.ingest_bytes(b"abcde".to_vec(), &IngestHints::default()),
            Err(IngressError::InputTooLarge {
                limit: 4,
                actual: 5
            })
        );
    }

    #[test]
    fn unresolved_artifact_is_a_typed_error() {
        let p = IngressPipeline::default();
        let err = p
            .ingest(
                RawInput::Artifact(ArtifactId::from("a1")),
                &IngestHints::default(),
            )
            .expect_err("no bytes");
        assert_eq!(err, IngressError::ArtifactNotResolved);
        assert!(err.to_string().contains("no inline bytes"));
    }

    #[test]
    fn raw_is_kept_untouched() {
        let p = IngressPipeline::default();
        let raw = RawInput::Bytes(b"a\r\nb".to_vec());
        let rec = p
            .ingest(raw.clone(), &IngestHints::default())
            .expect("ingests");
        assert_eq!(rec.raw, raw);
        assert_eq!(
            rec.normalized.as_ref().map(NormalizedText::text),
            Some("a\nb")
        );
        assert_eq!(rec.decision.status, CrossCheckStatus::SampleVerified);
    }
}
