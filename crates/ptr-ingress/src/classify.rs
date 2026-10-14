//! Pluggable semantic classifier contract plus two deterministic reference implementations: a
//! lexicon/rule scorer ([`RuleClassifier`]) and a document-shape prior ([`StructureClassifier`]).
//! Neither is a learned model; both are std-only stand-ins behind the same trait a learned
//! classifier would implement.

use std::collections::BTreeMap;
use std::fmt;

use ptr_types::{Probability, SemanticIssue};

use crate::model::{ArtifactKind, LabelScore, SemanticProposal};
use crate::normalize::NormalizedText;
use crate::parse::{CodeLanguage, DocumentFormat, ParsedDocument, SegmentKind};
use crate::provenance::RawSpan;

/// Label used when a classifier has no signal on an axis (an abstention, not a prediction).
pub const UNSPECIFIED_LABEL: &str = "unspecified";
const MAX_EVIDENCE: usize = 64;
const MAX_EVIDENCE_PER_LABEL: usize = 3;
const MAX_TERM_LEN: usize = 16;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LabelAxis {
    Task,
    Reasoning,
    Effect,
}

/// Why a label was proposed: the matched term and where it sits in the raw input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LabelEvidence {
    pub axis: LabelAxis,
    pub label: String,
    /// Lexicon term (or synthetic marker such as `<code>`) that fired.
    pub term: &'static str,
    pub span: RawSpan,
}

#[derive(Clone, Copy, Debug)]
pub struct ClassifierInput<'a> {
    pub kind: ArtifactKind,
    pub text: &'a NormalizedText,
    pub document: &'a ParsedDocument,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClassifierOutput {
    pub proposal: SemanticProposal,
    pub evidence: Vec<LabelEvidence>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClassifyError {
    Unavailable {
        classifier: &'static str,
    },
    Failed {
        classifier: &'static str,
        reason: &'static str,
    },
}

impl fmt::Display for ClassifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClassifyError::Unavailable { classifier } => {
                write!(f, "classifier {classifier} unavailable")
            }
            ClassifyError::Failed { classifier, reason } => {
                write!(f, "classifier {classifier} failed: {reason}")
            }
        }
    }
}

impl std::error::Error for ClassifyError {}

/// Contract for turning parsed text into a [`SemanticProposal`]. Implementations must be
/// deterministic for equal input and must tie every label to evidence spans where they can.
pub trait SemanticClassifier: Send + Sync {
    fn name(&self) -> &'static str;
    fn classify(&self, input: &ClassifierInput<'_>) -> Result<ClassifierOutput, ClassifyError>;
}

struct LexEntry {
    term: &'static str,
    axis: LabelAxis,
    label: &'static str,
    weight: f32,
}

const fn lex(term: &'static str, axis: LabelAxis, label: &'static str, weight: f32) -> LexEntry {
    LexEntry {
        term,
        axis,
        label,
        weight,
    }
}

use LabelAxis::{Effect as E, Reasoning as R, Task as T};

const LEXICON: &[LexEntry] = &[
    lex("what", T, "question", 0.5),
    lex("why", T, "question", 0.5),
    lex("how", T, "question", 0.5),
    lex("who", T, "question", 0.5),
    lex("when", T, "question", 0.5),
    lex("where", T, "question", 0.5),
    lex("which", T, "question", 0.5),
    lex("explain", T, "question", 0.5),
    lex("please", T, "instruction", 0.5),
    lex("ensure", T, "instruction", 1.0),
    lex("must", T, "instruction", 0.5),
    lex("need", T, "instruction", 0.3),
    lex("fix", T, "code-change", 1.0),
    lex("refactor", T, "code-change", 1.0),
    lex("implement", T, "code-change", 1.0),
    lex("patch", T, "code-change", 1.0),
    lex("bug", T, "code-change", 1.0),
    lex("rename", T, "code-change", 1.0),
    lex("compile", T, "code-change", 0.5),
    lex("function", T, "code-change", 0.5),
    lex("commit", T, "code-change", 0.5),
    lex("merge", T, "code-change", 0.5),
    lex("diff", T, "code-change", 0.5),
    lex("tests", T, "code-change", 0.3),
    lex("analyze", T, "data-analysis", 1.0),
    lex("analyse", T, "data-analysis", 1.0),
    lex("dataset", T, "data-analysis", 1.0),
    lex("average", T, "data-analysis", 0.5),
    lex("median", T, "data-analysis", 0.5),
    lex("correlation", T, "data-analysis", 1.0),
    lex("statistics", T, "data-analysis", 1.0),
    lex("plot", T, "data-analysis", 0.5),
    lex("aggregate", T, "data-analysis", 0.5),
    lex("csv", T, "data-analysis", 0.5),
    lex("regression", T, "data-analysis", 1.0),
    lex("trend", T, "data-analysis", 0.5),
    lex("therefore", R, "deductive", 1.0),
    lex("thus", R, "deductive", 1.0),
    lex("hence", R, "deductive", 1.0),
    lex("implies", R, "deductive", 1.0),
    lex("prove", R, "deductive", 1.0),
    lex("proof", R, "deductive", 1.0),
    lex("because", R, "causal", 1.0),
    lex("cause", R, "causal", 1.0),
    lex("causes", R, "causal", 1.0),
    lex("due", R, "causal", 0.5),
    lex("since", R, "causal", 0.5),
    lex("reason", R, "causal", 0.5),
    lex("similar", R, "analogical", 1.0),
    lex("analogous", R, "analogical", 1.0),
    lex("like", R, "analogical", 0.3),
    lex("compare", R, "analogical", 0.5),
    lex("plan", R, "planning", 1.0),
    lex("first", R, "planning", 0.5),
    lex("then", R, "planning", 0.5),
    lex("next", R, "planning", 0.5),
    lex("finally", R, "planning", 0.5),
    lex("step", R, "planning", 1.0),
    lex("steps", R, "planning", 1.0),
    lex("roadmap", R, "planning", 1.0),
    lex("calculate", R, "quantitative", 1.0),
    lex("compute", R, "quantitative", 1.0),
    lex("sum", R, "quantitative", 0.5),
    lex("percent", R, "quantitative", 1.0),
    lex("ratio", R, "quantitative", 1.0),
    lex("estimate", R, "quantitative", 0.5),
    lex("probability", R, "quantitative", 1.0),
    lex("suppose", R, "counterfactual", 1.0),
    lex("would", R, "counterfactual", 0.5),
    lex("if", R, "counterfactual", 0.3),
    lex("imagine", R, "counterfactual", 1.0),
    lex("otherwise", R, "counterfactual", 0.5),
    lex("read", E, "read-only", 0.5),
    lex("show", E, "read-only", 0.5),
    lex("list", E, "read-only", 0.5),
    lex("display", E, "read-only", 0.5),
    lex("describe", E, "read-only", 0.5),
    lex("summarize", E, "read-only", 0.5),
    lex("explain", E, "read-only", 0.5),
    lex("cat", E, "read-only", 0.5),
    lex("view", E, "read-only", 0.5),
    lex("ls", E, "read-only", 0.5),
    lex("grep", E, "read-only", 0.5),
    lex("inspect", E, "read-only", 0.5),
    lex("write", E, "write", 1.0),
    lex("delete", E, "write", 1.0),
    lex("remove", E, "write", 1.0),
    lex("rm", E, "write", 1.0),
    lex("overwrite", E, "write", 1.0),
    lex("modify", E, "write", 1.0),
    lex("edit", E, "write", 1.0),
    lex("save", E, "write", 1.0),
    lex("mv", E, "write", 1.0),
    lex("truncate", E, "write", 1.0),
    lex("drop", E, "write", 1.0),
    lex("create", E, "write", 1.0),
    lex("mkdir", E, "write", 1.0),
    lex("touch", E, "write", 1.0),
    lex("chmod", E, "write", 1.0),
    lex("update", E, "write", 1.0),
    lex("append", E, "write", 1.0),
    lex("rename", E, "write", 1.0),
    lex("commit", E, "write", 1.0),
    lex("insert", E, "write", 1.0),
    lex("patch", E, "write", 1.0),
    lex("fix", E, "write", 0.5),
    lex("implement", E, "write", 0.5),
    lex("refactor", E, "write", 0.5),
    lex("curl", E, "network", 1.0),
    lex("wget", E, "network", 1.0),
    lex("http", E, "network", 1.0),
    lex("https", E, "network", 1.0),
    lex("download", E, "network", 1.0),
    lex("upload", E, "network", 1.0),
    lex("fetch", E, "network", 1.0),
    lex("email", E, "network", 1.0),
    lex("ssh", E, "network", 1.0),
    lex("scp", E, "network", 1.0),
    lex("send", E, "network", 1.0),
    lex("push", E, "network", 1.0),
    lex("webhook", E, "network", 1.0),
    lex("ftp", E, "network", 1.0),
    lex("socket", E, "network", 1.0),
    lex("browse", E, "network", 1.0),
    lex("post", E, "network", 0.5),
    lex("request", E, "network", 0.5),
    lex("api", E, "network", 0.5),
    lex("url", E, "network", 0.5),
    lex("connect", E, "network", 0.5),
    lex("run", E, "exec", 1.0),
    lex("execute", E, "exec", 1.0),
    lex("exec", E, "exec", 1.0),
    lex("sudo", E, "exec", 1.0),
    lex("install", E, "exec", 1.0),
    lex("deploy", E, "exec", 1.0),
    lex("kill", E, "exec", 1.0),
    lex("bash", E, "exec", 1.0),
    lex("sh", E, "exec", 1.0),
    lex("cargo", E, "exec", 1.0),
    lex("npm", E, "exec", 1.0),
    lex("pip", E, "exec", 1.0),
    lex("eval", E, "exec", 1.0),
    lex("spawn", E, "exec", 1.0),
    lex("launch", E, "exec", 1.0),
    lex("restart", E, "exec", 1.0),
    lex("shutdown", E, "exec", 1.0),
    lex("reboot", E, "exec", 1.0),
    lex("docker", E, "exec", 1.0),
    lex("systemctl", E, "exec", 1.0),
    lex("invoke", E, "exec", 1.0),
    lex("start", E, "exec", 0.5),
    lex("stop", E, "exec", 0.5),
    lex("git", E, "exec", 0.5),
];

#[derive(Default)]
struct Accumulator {
    task: BTreeMap<&'static str, f32>,
    reasoning: BTreeMap<&'static str, f32>,
    effect: BTreeMap<&'static str, f32>,
    evidence: Vec<LabelEvidence>,
    per_label: BTreeMap<(LabelAxis, &'static str), usize>,
}

impl Accumulator {
    fn add(
        &mut self,
        axis: LabelAxis,
        label: &'static str,
        weight: f32,
        term: &'static str,
        span: Option<RawSpan>,
    ) {
        let map = match axis {
            LabelAxis::Task => &mut self.task,
            LabelAxis::Reasoning => &mut self.reasoning,
            LabelAxis::Effect => &mut self.effect,
        };
        *map.entry(label).or_insert(0.0) += weight;
        let seen = self.per_label.entry((axis, label)).or_insert(0);
        if let Some(span) = span {
            if *seen < MAX_EVIDENCE_PER_LABEL && self.evidence.len() < MAX_EVIDENCE {
                *seen += 1;
                self.evidence.push(LabelEvidence {
                    axis,
                    label: label.to_owned(),
                    term,
                    span,
                });
            }
        }
    }

    fn into_output(self) -> ClassifierOutput {
        let mut task_types = distribution(&self.task);
        let mut issues = Vec::new();
        if task_types.is_empty() {
            task_types = single(UNSPECIFIED_LABEL);
            issues.push(SemanticIssue {
                code: "classify.no_task_signal".to_owned(),
                message: "no task-type evidence found".to_owned(),
                hard: false,
            });
        }
        let mut effects = distribution(&self.effect);
        if effects.is_empty() {
            effects = single("read-only");
        }
        ClassifierOutput {
            proposal: SemanticProposal {
                task_types,
                reasoning_types: distribution(&self.reasoning),
                effects,
                issues,
            },
            evidence: self.evidence,
        }
    }
}

fn single(label: &str) -> Vec<LabelScore> {
    Probability::new(1.0)
        .map(|probability| LabelScore {
            label: label.to_owned(),
            probability,
        })
        .into_iter()
        .collect()
}

/// Normalizes raw label weights into a probability distribution (sum 1), sorted by probability
/// descending then label. Empty when there is no positive mass.
fn distribution(scores: &BTreeMap<&'static str, f32>) -> Vec<LabelScore> {
    let total: f32 = scores.values().sum();
    if total <= 0.0 || !total.is_finite() {
        return Vec::new();
    }
    let mut out: Vec<LabelScore> = scores
        .iter()
        .filter_map(|(label, score)| {
            Probability::new((score / total).clamp(0.0, 1.0)).map(|probability| LabelScore {
                label: (*label).to_owned(),
                probability,
            })
        })
        .collect();
    out.sort_by(|a, b| {
        b.probability
            .get()
            .total_cmp(&a.probability.get())
            .then_with(|| a.label.cmp(&b.label))
    });
    out
}

/// Lexicon/rule-based reference classifier. Matches ASCII words case-insensitively anywhere in
/// the text (so `rm`, `curl`, `https` or "delete" all count), which makes it deliberately
/// conservative on effects: a question that merely mentions `rm` still carries a write signal.
#[derive(Clone, Copy, Debug, Default)]
pub struct RuleClassifier;

impl RuleClassifier {
    const NAME: &'static str = "ptr-ingress/rule-lexicon-v1";
}

fn is_shell_fence(language: Option<&str>) -> bool {
    language.is_some_and(|l| {
        matches!(
            l.to_ascii_lowercase().as_str(),
            "sh" | "bash" | "shell" | "zsh" | "console"
        )
    })
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn scan_words(
    acc: &mut Accumulator,
    norm: &NormalizedText,
    base: usize,
    text: &str,
    natural: bool,
) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let start = i;
        if bytes[i] == b'?' {
            i += 1;
            if natural {
                acc.add(
                    LabelAxis::Task,
                    "question",
                    1.0,
                    "?",
                    norm.raw_span(base + start..base + i),
                );
            }
        } else if is_word_byte(bytes[i]) {
            while i < bytes.len() && is_word_byte(bytes[i]) {
                i += 1;
            }
            let Some(token) = text.get(start..i).filter(|t| t.len() <= MAX_TERM_LEN) else {
                continue;
            };
            let lower = token.to_ascii_lowercase();
            for entry in LEXICON.iter().filter(|e| e.term == lower) {
                if entry.axis != LabelAxis::Effect && !natural {
                    continue;
                }
                let span = norm.raw_span(base + start..base + i);
                acc.add(entry.axis, entry.label, entry.weight, entry.term, span);
            }
        } else {
            i += 1;
        }
    }
}

impl SemanticClassifier for RuleClassifier {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn classify(&self, input: &ClassifierInput<'_>) -> Result<ClassifierOutput, ClassifyError> {
        let mut acc = Accumulator::default();
        for seg in &input.document.segments {
            let span = Some(seg.span);
            match &seg.kind {
                SegmentKind::Prose
                | SegmentKind::MarkdownHeading { .. }
                | SegmentKind::KeyValue
                | SegmentKind::SectionHeader => {
                    scan_words(&mut acc, input.text, seg.range.start, &seg.text, true);
                }
                SegmentKind::CodeFence { language } => {
                    acc.add(LabelAxis::Task, "code-change", 1.0, "<code>", span);
                    if is_shell_fence(language.as_deref()) {
                        scan_words(&mut acc, input.text, seg.range.start, &seg.text, false);
                    }
                }
                SegmentKind::SourceCode(lang) => {
                    acc.add(LabelAxis::Task, "code-change", 1.0, "<code>", span);
                    if *lang == CodeLanguage::Shell {
                        scan_words(&mut acc, input.text, seg.range.start, &seg.text, false);
                    }
                }
                SegmentKind::Json | SegmentKind::CsvRow { .. } => {
                    acc.add(
                        LabelAxis::Task,
                        "data-analysis",
                        0.5,
                        "<structured-data>",
                        span,
                    );
                }
            }
        }
        Ok(acc.into_output())
    }
}

/// Document-shape prior: ignores wording and looks only at the parsed structure and question
/// marks. Independent enough of [`RuleClassifier`] to serve as a second opinion; it abstains
/// (`unspecified`) when the shape says nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct StructureClassifier;

impl SemanticClassifier for StructureClassifier {
    fn name(&self) -> &'static str {
        "ptr-ingress/structure-prior-v1"
    }

    fn classify(&self, input: &ClassifierInput<'_>) -> Result<ClassifierOutput, ClassifyError> {
        let doc = input.document;
        let first = doc.segments.first().map(|s| s.span);
        let (task, term): (&'static str, &'static str) = match doc.format {
            DocumentFormat::SourceCode(_) => ("code-change", "<format:code>"),
            DocumentFormat::Json | DocumentFormat::Csv | DocumentFormat::KeyValue => {
                ("data-analysis", "<format:data>")
            }
            DocumentFormat::Markdown | DocumentFormat::Prose if input.text.text().contains('?') => {
                ("question", "<punctuation:?>")
            }
            DocumentFormat::Markdown | DocumentFormat::Prose | DocumentFormat::Empty => {
                (UNSPECIFIED_LABEL, "")
            }
        };
        let effect = match doc.format {
            DocumentFormat::SourceCode(CodeLanguage::Shell) => "exec",
            _ => "read-only",
        };
        let mut evidence = Vec::new();
        if let Some(span) = first {
            if task != UNSPECIFIED_LABEL {
                evidence.push(LabelEvidence {
                    axis: LabelAxis::Task,
                    label: task.to_owned(),
                    term,
                    span,
                });
            }
            if effect == "exec" {
                evidence.push(LabelEvidence {
                    axis: LabelAxis::Effect,
                    label: effect.to_owned(),
                    term,
                    span,
                });
            }
        }
        Ok(ClassifierOutput {
            proposal: SemanticProposal {
                task_types: single(task),
                reasoning_types: Vec::new(),
                effects: single(effect),
                issues: Vec::new(),
            },
            evidence,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::{normalize, NormalizeOptions};
    use crate::parse::{parse, ParseOptions};

    fn run(classifier: &dyn SemanticClassifier, input: &str) -> ClassifierOutput {
        let n = normalize(input.as_bytes(), &NormalizeOptions::default()).expect("normalizes");
        let d = parse(&n, &ParseOptions::default());
        classifier
            .classify(&ClassifierInput {
                kind: ArtifactKind::Text,
                text: &n,
                document: &d,
            })
            .expect("classifies")
    }

    fn top(scores: &[LabelScore]) -> &str {
        scores.first().map_or("", |s| s.label.as_str())
    }

    fn sum(scores: &[LabelScore]) -> f32 {
        scores.iter().map(|s| s.probability.get()).sum()
    }

    #[test]
    fn distributions_are_normalized_sorted_and_deterministic() {
        let out = run(
            &RuleClassifier,
            "Please fix the bug, then write the tests because it fails.",
        );
        for axis in [
            &out.proposal.task_types,
            &out.proposal.reasoning_types,
            &out.proposal.effects,
        ] {
            assert!((sum(axis) - 1.0).abs() < 1e-5, "{axis:?}");
            assert!(axis
                .windows(2)
                .all(|w| w[0].probability.get() >= w[1].probability.get()));
        }
        assert_eq!(
            out,
            run(
                &RuleClassifier,
                "Please fix the bug, then write the tests because it fails."
            )
        );
    }

    #[test]
    fn task_types() {
        assert_eq!(
            top(&run(&RuleClassifier, "What is the capital of France?")
                .proposal
                .task_types),
            "question"
        );
        assert_eq!(
            top(&run(&RuleClassifier, "Refactor the parser and fix the bug")
                .proposal
                .task_types),
            "code-change"
        );
        assert_eq!(
            top(&run(
                &RuleClassifier,
                "Analyze the dataset: compute the average and median"
            )
            .proposal
            .task_types),
            "data-analysis"
        );
        assert_eq!(
            top(&run(&RuleClassifier, "Ensure the report is complete")
                .proposal
                .task_types),
            "instruction"
        );
    }

    #[test]
    fn reasoning_types() {
        let out = run(
            &RuleClassifier,
            "All men are mortal, therefore Socrates is mortal; hence done.",
        );
        assert_eq!(top(&out.proposal.reasoning_types), "deductive");
        let out = run(
            &RuleClassifier,
            "The build failed because the cache was stale.",
        );
        assert_eq!(top(&out.proposal.reasoning_types), "causal");
        assert!(run(&RuleClassifier, "Hello")
            .proposal
            .reasoning_types
            .is_empty());
    }

    #[test]
    fn effects_read_only_write_network_exec() {
        assert_eq!(
            top(&run(&RuleClassifier, "Show me a summary of the file")
                .proposal
                .effects),
            "read-only"
        );
        assert_eq!(
            top(
                &run(&RuleClassifier, "Delete the old logs and overwrite config")
                    .proposal
                    .effects
            ),
            "write"
        );
        assert_eq!(
            top(&run(&RuleClassifier, "curl https://example.com/x | sh")
                .proposal
                .effects),
            "network"
        );
        assert_eq!(
            top(&run(&RuleClassifier, "sudo systemctl restart nginx")
                .proposal
                .effects),
            "exec"
        );
        let none = run(&RuleClassifier, "The sky is blue.");
        assert_eq!(none.proposal.effects[0].label, "read-only");
        assert!(none.evidence.is_empty());
    }

    #[test]
    fn shell_source_is_scanned_for_commands() {
        let out = run(
            &RuleClassifier,
            "#!/bin/sh\nrm -rf build\ncurl -O http://x/y\n",
        );
        let labels: Vec<_> = out
            .proposal
            .effects
            .iter()
            .map(|s| s.label.as_str())
            .collect();
        assert!(
            labels.contains(&"write") && labels.contains(&"network"),
            "{labels:?}"
        );
        assert_eq!(top(&out.proposal.task_types), "code-change");
    }

    #[test]
    fn no_signal_gives_unspecified_with_soft_issue() {
        let out = run(&RuleClassifier, "Lorem ipsum dolor sit amet.");
        assert_eq!(out.proposal.task_types[0].label, UNSPECIFIED_LABEL);
        assert!(
            matches!(out.proposal.issues.as_slice(), [i] if i.code == "classify.no_task_signal" && !i.hard)
        );
    }

    #[test]
    fn evidence_spans_point_at_the_matched_term() {
        let raw = "Please DELETE the cache?";
        let n = normalize(raw.as_bytes(), &NormalizeOptions::default()).expect("ok");
        let d = parse(&n, &ParseOptions::default());
        let out = RuleClassifier
            .classify(&ClassifierInput {
                kind: ArtifactKind::Text,
                text: &n,
                document: &d,
            })
            .expect("ok");
        let delete = out
            .evidence
            .iter()
            .find(|e| e.term == "delete")
            .expect("delete evidence");
        assert_eq!(delete.span.slice(raw.as_bytes()), Some(&b"DELETE"[..]));
        let q = out
            .evidence
            .iter()
            .find(|e| e.term == "?")
            .expect("question evidence");
        assert_eq!(q.span.slice(raw.as_bytes()), Some(&b"?"[..]));
    }

    #[test]
    fn evidence_is_bounded() {
        let raw = "delete ".repeat(500);
        let out = run(&RuleClassifier, &raw);
        assert!(out.evidence.len() <= MAX_EVIDENCE_PER_LABEL);
    }

    #[test]
    fn structure_classifier_is_shape_only_and_abstains() {
        let code = run(&StructureClassifier, "fn main() {\n    let mut x = 1;\n}\n");
        assert_eq!(top(&code.proposal.task_types), "code-change");
        let q = run(&StructureClassifier, "Is this fine?");
        assert_eq!(top(&q.proposal.task_types), "question");
        let blank = run(&StructureClassifier, "Delete everything.");
        assert_eq!(top(&blank.proposal.task_types), UNSPECIFIED_LABEL);
        assert_eq!(top(&blank.proposal.effects), "read-only");
        let sh = run(&StructureClassifier, "#!/bin/sh\necho hi\n");
        assert_eq!(top(&sh.proposal.effects), "exec");
    }

    #[test]
    fn classify_error_displays() {
        let e = ClassifyError::Failed {
            classifier: "x",
            reason: "boom",
        };
        assert_eq!(e.to_string(), "classifier x failed: boom");
        assert_eq!(
            ClassifyError::Unavailable { classifier: "x" }.to_string(),
            "classifier x unavailable"
        );
    }
}
