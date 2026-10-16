//! `ptrctl ingest`: run one file through `ptr_ingress::IngressPipeline` and render the record
//! with `ptr_inspect`.
//!
//! The file is read with a hard byte bound. Raw bytes are never printed: the record shows the
//! raw length and an FNV-1a fingerprint, plus a bounded preview of each segment only under
//! `--preview`. The rendered tree passes through `ptr_inspect::redact_sensitive` before it is
//! printed. An ingest decision such as `NeedsFullCheck` is a result, not an error.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use ptr_ingress::{
    IngestHints, IngressConfig, IngressError, IngressPipeline, IngressRecord, LabelScore, RawInput,
    Segment,
};
use ptr_inspect::{
    redact_sensitive, render_json, render_tree, InspectNode, Inspectable, PreviewPolicy,
    RawEvidence, RenderOptions, MAX_EVIDENCE_PREVIEW_BYTES,
};

use crate::cli::{
    flag_number, flag_value, is_help, usage_error, EXIT_CODES_HELP, EXIT_INPUT, EXIT_OK,
};

/// Default read bound, equal to the pipeline's own default input bound.
const DEFAULT_MAX_BYTES: u64 = 4 * 1024 * 1024;

const USAGE: &str = "usage: ptrctl ingest <path> [--format tree|json] [--max-bytes N] [--preview]";

fn help() -> String {
    format!(
        "{USAGE}

Run a file through the ingress pipeline and print the record (kind, MIME, encoding and
lossy flag, segments with raw byte spans, proposal labels, verification decision and
reasons). A decision such as NeedsFullCheck is a result, not an error.

options:
  --format tree|json  rendering (default tree)
  --max-bytes N       refuse files longer than N bytes (default {DEFAULT_MAX_BYTES})
  --preview           also show a bounded text preview of each segment's raw bytes
                      (at most {MAX_EVIDENCE_PREVIEW_BYTES} bytes); off by default

{EXIT_CODES_HELP}"
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Format {
    Tree,
    Json,
}

struct Request {
    path: PathBuf,
    format: Format,
    max_bytes: u64,
    preview: bool,
}

pub(crate) fn command(args: Vec<String>) -> i32 {
    if args.first().is_some_and(|arg| is_help(arg)) {
        println!("{}", help());
        return EXIT_OK;
    }
    let request = match parse(args) {
        Ok(request) => request,
        Err(code) => return code,
    };
    let bytes = match read_bounded(&request.path, request.max_bytes) {
        Ok(bytes) => bytes,
        Err(message) => {
            eprintln!("ptrctl ingest: {message}");
            return EXIT_INPUT;
        }
    };
    let limit = usize::try_from(request.max_bytes).unwrap_or(usize::MAX);
    let pipeline = IngressPipeline::new(IngressConfig {
        max_input_bytes: limit,
        ..IngressConfig::default()
    });
    let hints = IngestHints {
        declared_kind: None,
        extension: request
            .path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_owned),
    };
    let record = match pipeline.ingest(RawInput::Bytes(bytes), &hints) {
        Ok(record) => record,
        Err(error) => return ingress_failure(&error),
    };
    let view = RecordView {
        record: &record,
        preview: request.preview,
    };
    let node = redact_sensitive(&view.inspect());
    match request.format {
        Format::Tree => println!("{}", render_tree(&node, &RenderOptions::default())),
        Format::Json => println!("{}", render_json(&node)),
    }
    EXIT_OK
}

fn ingress_failure(error: &IngressError) -> i32 {
    eprintln!("ptrctl ingest: {error}");
    EXIT_INPUT
}

fn parse(args: Vec<String>) -> Result<Request, i32> {
    let mut path = None;
    let mut format = Format::Tree;
    let mut max_bytes = DEFAULT_MAX_BYTES;
    let mut preview = false;
    let mut rest = args.into_iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--format" => {
                format = match flag_value(&mut rest, "ingest", "--format", USAGE)?.as_str() {
                    "tree" => Format::Tree,
                    "json" => Format::Json,
                    other => {
                        return Err(usage_error(
                            "ingest",
                            &format!("unknown format {other:?} (expected tree or json)"),
                            USAGE,
                        ))
                    }
                };
            }
            "--max-bytes" => max_bytes = flag_number(&mut rest, "ingest", "--max-bytes", USAGE)?,
            "--preview" => preview = true,
            flag if flag.starts_with("--") => {
                return Err(usage_error(
                    "ingest",
                    &format!("unknown option {flag}"),
                    USAGE,
                ))
            }
            _ => {
                if path.replace(PathBuf::from(&arg)).is_some() {
                    return Err(usage_error("ingest", "only one path is accepted", USAGE));
                }
            }
        }
    }
    match path {
        Some(path) => Ok(Request {
            path,
            format,
            max_bytes,
            preview,
        }),
        None => Err(usage_error("ingest", "a path is required", USAGE)),
    }
}

/// Read at most `max_bytes` bytes; a longer file is an error rather than a silent truncation.
fn read_bounded(path: &Path, max_bytes: u64) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let metadata = file
        .metadata()
        .map_err(|e| format!("cannot stat {}: {e}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!("{} is not a regular file", path.display()));
    }
    let mut bytes = Vec::new();
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if bytes.len() as u64 > max_bytes {
        return Err(format!(
            "{} is longer than --max-bytes {max_bytes}",
            path.display()
        ));
    }
    Ok(bytes)
}

/// ptrctl-local view of an [`IngressRecord`]. Raw content appears only as length and
/// fingerprint, or as a bounded preview when `preview` is set.
struct RecordView<'a> {
    record: &'a IngressRecord,
    preview: bool,
}

fn variant(name: &str, value: &impl std::fmt::Debug) -> InspectNode {
    InspectNode::unit_variant(name, &format!("{value:?}"))
}

/// An `f32` probability widened to `f64` and rounded to four places, so the f32 representation
/// error does not show up in the output.
fn rounded(value: f32) -> f64 {
    (f64::from(value) * 10_000.0).round() / 10_000.0
}

fn labels(scores: &[LabelScore]) -> InspectNode {
    InspectNode::List(
        scores
            .iter()
            .map(|score| {
                InspectNode::new_struct(
                    "Label",
                    vec![
                        ("label", InspectNode::text(score.label.clone())),
                        (
                            "probability",
                            InspectNode::Number(rounded(score.probability.get())),
                        ),
                    ],
                )
            })
            .collect(),
    )
}

impl RecordView<'_> {
    fn segment(&self, segment: &Segment, raw: &[u8]) -> InspectNode {
        let mut fields = vec![
            ("kind", variant("SegmentKind", &segment.kind)),
            (
                "raw_start",
                InspectNode::UInt(segment.span.byte_start as u128),
            ),
            ("raw_end", InspectNode::UInt(segment.span.byte_end as u128)),
        ];
        let bytes = segment.span.slice(raw).unwrap_or(&[]);
        let policy = if self.preview {
            PreviewPolicy::Allow {
                max_bytes: MAX_EVIDENCE_PREVIEW_BYTES,
            }
        } else {
            PreviewPolicy::Deny
        };
        fields.push(("raw", RawEvidence::with_policy(bytes, policy).inspect()));
        InspectNode::new_struct("Segment", fields)
    }
}

impl Inspectable for RecordView<'_> {
    fn inspect(&self) -> InspectNode {
        let record = self.record;
        let raw = record.raw_bytes().unwrap_or(&[]);
        let (encoding, lossy) = match &record.normalized {
            Some(text) => (
                InspectNode::text(text.encoding().label()),
                InspectNode::Bool(text.is_lossy()),
            ),
            None => (InspectNode::Null, InspectNode::Null),
        };
        let segments = record
            .segments()
            .iter()
            .map(|segment| self.segment(segment, raw))
            .collect();
        let issues = record
            .proposal
            .issues
            .iter()
            .map(|issue| {
                InspectNode::new_struct(
                    "Issue",
                    vec![
                        ("code", InspectNode::text(issue.code.clone())),
                        ("message", InspectNode::text(issue.message.clone())),
                        ("hard", InspectNode::Bool(issue.hard)),
                    ],
                )
            })
            .collect();
        let reasons = record
            .decision
            .reasons
            .iter()
            .map(|reason| InspectNode::text(format!("{reason:?}")))
            .collect();
        let normalize_error = record.normalize_error.map_or(InspectNode::Null, |error| {
            InspectNode::text(error.to_string())
        });
        InspectNode::new_struct(
            "IngressRecord",
            vec![
                ("kind", variant("ArtifactKind", &record.kind)),
                ("mime", InspectNode::text(record.sniff.mime)),
                (
                    "sniff_confidence",
                    InspectNode::Number(rounded(record.sniff.confidence.get())),
                ),
                ("encoding", encoding),
                ("lossy", lossy),
                ("normalize_error", normalize_error),
                ("raw", RawEvidence::new(raw).inspect()),
                ("segments", InspectNode::List(segments)),
                (
                    "proposal",
                    InspectNode::new_struct(
                        "Proposal",
                        vec![
                            ("task_types", labels(&record.proposal.task_types)),
                            ("reasoning_types", labels(&record.proposal.reasoning_types)),
                            ("effects", labels(&record.proposal.effects)),
                            ("issues", InspectNode::List(issues)),
                        ],
                    ),
                ),
                (
                    "decision",
                    InspectNode::new_struct(
                        "Decision",
                        vec![
                            (
                                "status",
                                variant("CrossCheckStatus", &record.decision.status),
                            ),
                            ("reasons", InspectNode::List(reasons)),
                        ],
                    ),
                ),
            ],
        )
    }
}
