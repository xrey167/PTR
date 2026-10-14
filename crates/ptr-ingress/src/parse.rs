//! Deterministic structural parser stack over normalized text: a bounded JSON recognizer plus
//! heuristic detection of Markdown, source code, key=value, CSV and prose. Every segment carries
//! the raw span it was decoded from. The heuristics are shallow by design; they say what a text
//! looks like, not what it means.

use std::ops::Range;

use ptr_types::SemanticIssue;

use crate::normalize::NormalizedText;
use crate::provenance::RawSpan;

/// Hard cap on JSON nesting regardless of configuration (keeps recursion stack-safe).
const JSON_DEPTH_HARD_CAP: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodeLanguage {
    Rust,
    Python,
    JavaScript,
    Shell,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DocumentFormat {
    Empty,
    Json,
    Markdown,
    SourceCode(CodeLanguage),
    KeyValue,
    Csv,
    Prose,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SegmentKind {
    Json,
    MarkdownHeading { level: u8 },
    CodeFence { language: Option<String> },
    SourceCode(CodeLanguage),
    KeyValue,
    SectionHeader,
    CsvRow { index: usize },
    Prose,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Segment {
    pub kind: SegmentKind,
    /// Byte range into [`NormalizedText::text`].
    pub range: Range<usize>,
    pub text: String,
    /// Bytes of the original raw input this segment was decoded from.
    pub span: RawSpan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JsonErrorKind {
    Syntax,
    TooDeep,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JsonError {
    pub kind: JsonErrorKind,
    /// Byte offset inside the validated string.
    pub offset: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParseIssue {
    /// The text looks like JSON (starts with `{` or a plausible `[`) but does not validate.
    JsonInvalid {
        offset: usize,
    },
    JsonTooDeep {
        limit: usize,
    },
    UnclosedCodeFence {
        offset: usize,
    },
    SegmentLimitReached {
        limit: usize,
    },
}

impl ParseIssue {
    pub fn code(&self) -> &'static str {
        match self {
            ParseIssue::JsonInvalid { .. } => "parse.json_invalid",
            ParseIssue::JsonTooDeep { .. } => "parse.json_too_deep",
            ParseIssue::UnclosedCodeFence { .. } => "parse.unclosed_code_fence",
            ParseIssue::SegmentLimitReached { .. } => "parse.segment_limit",
        }
    }

    /// Hard issues mean the declared or apparent structure could not be verified.
    pub fn is_hard(&self) -> bool {
        match self {
            ParseIssue::JsonInvalid { .. } | ParseIssue::JsonTooDeep { .. } => true,
            ParseIssue::UnclosedCodeFence { .. } | ParseIssue::SegmentLimitReached { .. } => false,
        }
    }

    pub fn to_semantic_issue(&self) -> SemanticIssue {
        let message = match self {
            ParseIssue::JsonInvalid { offset } => {
                format!("JSON-like text invalid at byte {offset}")
            }
            ParseIssue::JsonTooDeep { limit } => {
                format!("JSON nesting exceeds depth limit {limit}")
            }
            ParseIssue::UnclosedCodeFence { offset } => {
                format!("code fence at byte {offset} never closes")
            }
            ParseIssue::SegmentLimitReached { limit } => format!("segment limit {limit} reached"),
        };
        SemanticIssue {
            code: self.code().to_owned(),
            message,
            hard: self.is_hard(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParseOptions {
    pub max_json_depth: usize,
    pub max_segments: usize,
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self {
            max_json_depth: 64,
            max_segments: 4096,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsedDocument {
    pub format: DocumentFormat,
    pub segments: Vec<Segment>,
    pub issues: Vec<ParseIssue>,
}

struct Line<'a> {
    start: usize,
    end: usize,
    text: &'a str,
}

impl Line<'_> {
    fn trimmed_range(&self) -> Range<usize> {
        let lead = self.text.len() - self.text.trim_start().len();
        let start = self.start + lead;
        start..start + self.text.trim().len()
    }
}

fn split_lines(text: &str) -> Vec<Line<'_>> {
    let mut lines = Vec::new();
    let mut start = 0;
    for piece in text.split('\n') {
        lines.push(Line {
            start,
            end: start + piece.len(),
            text: piece,
        });
        start += piece.len() + 1;
    }
    if text.ends_with('\n') || text.is_empty() {
        lines.pop();
    }
    lines
}

struct Builder<'a> {
    norm: &'a NormalizedText,
    options: ParseOptions,
    segments: Vec<Segment>,
    issues: Vec<ParseIssue>,
}

impl Builder<'_> {
    fn push(&mut self, kind: SegmentKind, range: Range<usize>) {
        if self.segments.len() >= self.options.max_segments {
            let issue = ParseIssue::SegmentLimitReached {
                limit: self.options.max_segments,
            };
            if !self.issues.contains(&issue) {
                self.issues.push(issue);
            }
            return;
        }
        let (Some(text), Some(span)) = (
            self.norm.text().get(range.clone()),
            self.norm.raw_span(range.clone()),
        ) else {
            return;
        };
        self.segments.push(Segment {
            kind,
            range,
            text: text.to_owned(),
            span,
        });
    }

    fn finish(self, format: DocumentFormat) -> ParsedDocument {
        ParsedDocument {
            format,
            segments: self.segments,
            issues: self.issues,
        }
    }
}

/// Parses normalized text into a format guess plus typed, span-carrying segments.
pub fn parse(text: &NormalizedText, options: &ParseOptions) -> ParsedDocument {
    let s = text.text();
    let mut b = Builder {
        norm: text,
        options: *options,
        segments: Vec::new(),
        issues: Vec::new(),
    };
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return b.finish(DocumentFormat::Empty);
    }
    let lead = s.len() - s.trim_start().len();
    if looks_like_json(trimmed) {
        match validate_json(trimmed, options.max_json_depth) {
            Ok(_) => {
                b.push(SegmentKind::Json, lead..lead + trimmed.len());
                return b.finish(DocumentFormat::Json);
            }
            Err(JsonError {
                kind: JsonErrorKind::Syntax,
                offset,
            }) => {
                b.issues.push(ParseIssue::JsonInvalid {
                    offset: lead + offset,
                });
            }
            Err(JsonError {
                kind: JsonErrorKind::TooDeep,
                ..
            }) => {
                b.issues.push(ParseIssue::JsonTooDeep {
                    limit: options.max_json_depth,
                });
            }
        }
    }
    let lines = split_lines(s);
    let has_fence = lines
        .iter()
        .any(|l| fence_marker(l.text.trim_start()).is_some());
    if !has_fence {
        if let Some(lang) = detect_code(&lines) {
            b.push(SegmentKind::SourceCode(lang), lead..lead + trimmed.len());
            return b.finish(DocumentFormat::SourceCode(lang));
        }
    }
    let has_heading = lines
        .iter()
        .any(|l| heading_level(l.text.trim_start()).is_some());
    if has_fence {
        markdown_segments(&mut b, &lines);
        return b.finish(DocumentFormat::Markdown);
    }
    // Checked before headings: `# comment` lines of INI/TOML-like text look like headings.
    if is_key_value(&lines) {
        for line in &lines {
            let t = line.text.trim();
            match kv_line(t) {
                Some(KvLine::Section) => b.push(SegmentKind::SectionHeader, line.trimmed_range()),
                Some(KvLine::Assign) => b.push(SegmentKind::KeyValue, line.trimmed_range()),
                None => {}
            }
        }
        return b.finish(DocumentFormat::KeyValue);
    }
    if has_heading {
        markdown_segments(&mut b, &lines);
        return b.finish(DocumentFormat::Markdown);
    }
    let rows: Vec<&Line<'_>> = lines.iter().filter(|l| !l.text.trim().is_empty()).collect();
    if csv_delimiter(&rows).is_some() {
        for (index, line) in rows.iter().enumerate() {
            b.push(SegmentKind::CsvRow { index }, line.trimmed_range());
        }
        return b.finish(DocumentFormat::Csv);
    }
    let mut para = None;
    for line in &lines {
        extend_paragraph(&mut b, &mut para, line);
    }
    flush_paragraph(&mut b, &mut para);
    b.finish(DocumentFormat::Prose)
}

fn extend_paragraph(b: &mut Builder<'_>, para: &mut Option<Range<usize>>, line: &Line<'_>) {
    if line.text.trim().is_empty() {
        flush_paragraph(b, para);
        return;
    }
    let r = line.trimmed_range();
    *para = Some(match para.take() {
        Some(open) => open.start..r.end,
        None => r,
    });
}

fn flush_paragraph(b: &mut Builder<'_>, para: &mut Option<Range<usize>>) {
    if let Some(range) = para.take() {
        b.push(SegmentKind::Prose, range);
    }
}

fn fence_marker(t: &str) -> Option<&'static str> {
    if t.starts_with("```") {
        Some("```")
    } else if t.starts_with("~~~") {
        Some("~~~")
    } else {
        None
    }
}

fn heading_level(t: &str) -> Option<u8> {
    let hashes = t.bytes().take_while(|&c| c == b'#').count();
    let rest = t.get(hashes..)?;
    (1..=6).contains(&hashes).then_some(())?;
    (rest.starts_with(' ') && !rest.trim().is_empty()).then_some(u8::try_from(hashes).ok()?)
}

fn markdown_segments(b: &mut Builder<'_>, lines: &[Line<'_>]) {
    let mut para = None;
    let mut i = 0;
    while let Some(line) = lines.get(i) {
        let t = line.text.trim_start();
        if let Some(marker) = fence_marker(t) {
            flush_paragraph(b, &mut para);
            let info = t.get(3..).unwrap_or("").trim();
            let language = info.split_whitespace().next().map(str::to_owned);
            let close = (i + 1..lines.len()).find(|&j| {
                lines
                    .get(j)
                    .is_some_and(|l| l.text.trim_start().starts_with(marker))
            });
            let last = close.unwrap_or(lines.len() - 1);
            if close.is_none() {
                b.issues
                    .push(ParseIssue::UnclosedCodeFence { offset: line.start });
            }
            let end = lines.get(last).map_or(line.end, |l| l.end);
            b.push(SegmentKind::CodeFence { language }, line.start..end);
            i = last + 1;
            continue;
        }
        if let Some(level) = heading_level(t) {
            flush_paragraph(b, &mut para);
            b.push(SegmentKind::MarkdownHeading { level }, line.trimmed_range());
        } else {
            extend_paragraph(b, &mut para, line);
        }
        i += 1;
    }
    flush_paragraph(b, &mut para);
}

enum KvLine {
    Section,
    Assign,
}

fn is_comment(t: &str) -> bool {
    t.starts_with('#') || t.starts_with(';') || t.starts_with("//")
}

fn kv_line(t: &str) -> Option<KvLine> {
    if t.is_empty() || is_comment(t) {
        return None;
    }
    if t.starts_with('[') && t.ends_with(']') && t.len() > 2 {
        return Some(KvLine::Section);
    }
    let idx = t.find('=')?;
    if t.get(idx + 1..)?.starts_with('=') {
        return None;
    }
    let key = t.get(..idx)?.trim_end();
    let mut chars = key.chars();
    let first = chars.next()?;
    let valid = (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'));
    valid.then_some(KvLine::Assign)
}

fn is_key_value(lines: &[Line<'_>]) -> bool {
    let (mut content, mut matched, mut assigns) = (0, 0, 0);
    for line in lines {
        let t = line.text.trim();
        if t.is_empty() || is_comment(t) {
            continue;
        }
        content += 1;
        match kv_line(t) {
            Some(KvLine::Assign) => {
                matched += 1;
                assigns += 1;
            }
            Some(KvLine::Section) => matched += 1,
            None => {}
        }
    }
    content > 0 && matched == content && assigns > 0
}

fn csv_delimiter(rows: &[&Line<'_>]) -> Option<char> {
    if rows.len() < 2 {
        return None;
    }
    [',', '\t', ';'].into_iter().find(|&delim| {
        let counts: Vec<usize> = rows
            .iter()
            .map(|l| count_outside_quotes(l.text, delim))
            .collect();
        let first = counts.first().copied().unwrap_or(0);
        first >= 1 && counts.iter().all(|&c| c == first) && (first >= 2 || rows.len() >= 3)
    })
}

fn count_outside_quotes(line: &str, delim: char) -> usize {
    let mut quoted = false;
    let mut count = 0;
    for c in line.chars() {
        if c == '"' {
            quoted = !quoted;
        } else if c == delim && !quoted {
            count += 1;
        }
    }
    count
}

fn detect_code(lines: &[Line<'_>]) -> Option<CodeLanguage> {
    let first = lines
        .iter()
        .find(|l| !l.text.trim().is_empty())?
        .text
        .trim();
    if let Some(interp) = first.strip_prefix("#!") {
        let by_shebang = [
            ("python", CodeLanguage::Python),
            ("node", CodeLanguage::JavaScript),
            ("deno", CodeLanguage::JavaScript),
            ("rust-script", CodeLanguage::Rust),
            ("sh", CodeLanguage::Shell),
        ];
        if let Some((_, lang)) = by_shebang.iter().find(|(name, _)| interp.contains(name)) {
            return Some(*lang);
        }
    }
    let mut scores = [0usize; 4];
    for line in lines {
        let t = line.text.trim();
        scores[0] += usize::from(rust_hit(t));
        scores[1] += usize::from(python_hit(t));
        scores[2] += usize::from(js_hit(t));
        scores[3] += usize::from(shell_hit(t));
    }
    let order = [
        CodeLanguage::Rust,
        CodeLanguage::Python,
        CodeLanguage::JavaScript,
        CodeLanguage::Shell,
    ];
    let mut best: Option<(usize, CodeLanguage)> = None;
    for (score, lang) in scores.into_iter().zip(order) {
        if score >= 2 && best.is_none_or(|(top, _)| score > top) {
            best = Some((score, lang));
        }
    }
    best.map(|(_, lang)| lang)
}

fn starts_any(t: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| t.starts_with(p))
}

fn rust_hit(t: &str) -> bool {
    starts_any(
        t,
        &[
            "fn ",
            "pub fn ",
            "pub(crate) fn ",
            "pub struct ",
            "struct ",
            "impl ",
            "impl<",
            "use std::",
            "use crate::",
            "let mut ",
            "pub mod ",
            "pub enum ",
            "pub trait ",
            "#[derive",
            "#![",
            "async fn ",
            "pub async fn ",
            "extern crate",
        ],
    ) || ["println!(", "fn main(", "-> Self", "&mut ", "::new("]
        .iter()
        .any(|p| t.contains(p))
}

fn python_hit(t: &str) -> bool {
    starts_any(
        t,
        &[
            "def ",
            "async def ",
            "elif ",
            "@staticmethod",
            "@classmethod",
            "if __name__",
            "with open(",
        ],
    ) || (t.starts_with("class ") && t.ends_with(':'))
        || (t.starts_with("from ") && t.contains(" import "))
        || (t.starts_with("import ") && !t.contains(" from ") && !t.contains(['\'', '"', ';']))
        || (starts_any(t, &["if ", "for ", "while ", "try", "else"]) && t.ends_with(':'))
        || t.contains("self.")
        || (t.contains("print(") && !t.contains(';'))
}

fn js_hit(t: &str) -> bool {
    starts_any(
        t,
        &[
            "function ",
            "async function ",
            "var ",
            "module.exports",
            "document.",
            "window.",
        ],
    ) || (t.starts_with("let ") && !t.starts_with("let mut ") && !t.contains(": "))
        || (t.starts_with("const ") && t.contains(" = ") && !t.contains(": "))
        || (t.starts_with("import ") && (t.contains(" from '") || t.contains(" from \"")))
        || (t.starts_with("export ")
            && (t.contains("const ") || t.contains("function ") || t.contains("default")))
        || t.contains("=>")
        || t.contains("console.log(")
        || t.contains("require(")
        || t.contains("===")
}

fn shell_hit(t: &str) -> bool {
    matches!(t, "fi" | "then" | "done" | "do" | "esac")
        || starts_any(
            t,
            &[
                "echo ", "if [", "cd ", "grep ", "sudo ", "apt ", "apt-get ", "mkdir ", "chmod ",
                "rm ", "set -e", "case ", "source ", "curl ", "wget ",
            ],
        )
        || (t.starts_with("export ")
            && t.contains('=')
            && !t.contains("const ")
            && !t.contains("function ")
            && !t.contains("default"))
        || t.contains("$(")
        || t.contains("${")
}

fn looks_like_json(t: &str) -> bool {
    if t.starts_with('{') {
        return true;
    }
    t.strip_prefix('[').is_some_and(|rest| {
        matches!(
            rest.trim_start().chars().next(),
            Some('[' | '{' | '"' | '-' | ']' | '0'..='9')
        )
    })
}

/// Validates `text` as one strict JSON value (RFC 8259 grammar, no extensions). Returns the
/// maximum container nesting depth, or the first error. Nesting beyond `max_depth` (capped at
/// 256) yields [`JsonErrorKind::TooDeep`].
pub fn validate_json(text: &str, max_depth: usize) -> Result<usize, JsonError> {
    let mut p = JsonParser {
        b: text.as_bytes(),
        pos: 0,
        max_depth: max_depth.min(JSON_DEPTH_HARD_CAP),
    };
    let depth = p.value(0)?;
    p.skip_ws();
    if p.pos < p.b.len() {
        return Err(p.syntax());
    }
    Ok(depth)
}

struct JsonParser<'a> {
    b: &'a [u8],
    pos: usize,
    max_depth: usize,
}

impl JsonParser<'_> {
    fn syntax(&self) -> JsonError {
        JsonError {
            kind: JsonErrorKind::Syntax,
            offset: self.pos,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, byte: u8) -> Result<(), JsonError> {
        if self.peek() == Some(byte) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.syntax())
        }
    }

    /// `depth` is the nesting of the enclosing container; returns the deepest nesting seen.
    fn value(&mut self, depth: usize) -> Result<usize, JsonError> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.object(depth + 1),
            Some(b'[') => self.array(depth + 1),
            Some(b'"') => self.string().map(|()| depth),
            Some(b'-' | b'0'..=b'9') => self.number().map(|()| depth),
            Some(b't') => self.literal(b"true").map(|()| depth),
            Some(b'f') => self.literal(b"false").map(|()| depth),
            Some(b'n') => self.literal(b"null").map(|()| depth),
            _ => Err(self.syntax()),
        }
    }

    fn enter(&self, depth: usize) -> Result<(), JsonError> {
        if depth > self.max_depth {
            Err(JsonError {
                kind: JsonErrorKind::TooDeep,
                offset: self.pos,
            })
        } else {
            Ok(())
        }
    }

    fn object(&mut self, depth: usize) -> Result<usize, JsonError> {
        self.enter(depth)?;
        self.pos += 1;
        let mut deepest = depth;
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(deepest);
        }
        loop {
            self.skip_ws();
            self.string()?;
            self.skip_ws();
            self.expect(b':')?;
            deepest = deepest.max(self.value(depth)?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(deepest);
                }
                _ => return Err(self.syntax()),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<usize, JsonError> {
        self.enter(depth)?;
        self.pos += 1;
        let mut deepest = depth;
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(deepest);
        }
        loop {
            deepest = deepest.max(self.value(depth)?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(deepest);
                }
                _ => return Err(self.syntax()),
            }
        }
    }

    fn string(&mut self) -> Result<(), JsonError> {
        self.expect(b'"')?;
        loop {
            match self.peek() {
                None | Some(0..=0x1F) => return Err(self.syntax()),
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(());
                }
                Some(b'\\') => {
                    self.pos += 1;
                    match self.peek() {
                        Some(b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') => {
                            self.pos += 1
                        }
                        Some(b'u') => {
                            self.pos += 1;
                            for _ in 0..4 {
                                if !self.peek().is_some_and(|c| c.is_ascii_hexdigit()) {
                                    return Err(self.syntax());
                                }
                                self.pos += 1;
                            }
                        }
                        _ => return Err(self.syntax()),
                    }
                }
                Some(_) => self.pos += 1,
            }
        }
    }

    fn digits(&mut self) -> Result<(), JsonError> {
        let start = self.pos;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.pos == start {
            Err(self.syntax())
        } else {
            Ok(())
        }
    }

    fn number(&mut self) -> Result<(), JsonError> {
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => self.digits()?,
            _ => return Err(self.syntax()),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            self.digits()?;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            self.digits()?;
        }
        Ok(())
    }

    fn literal(&mut self, word: &[u8]) -> Result<(), JsonError> {
        if self.b.get(self.pos..self.pos + word.len()) == Some(word) {
            self.pos += word.len();
            Ok(())
        } else {
            Err(self.syntax())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::{normalize, NormalizeOptions};

    fn doc(input: &str) -> ParsedDocument {
        let n = normalize(input.as_bytes(), &NormalizeOptions::default()).expect("normalizes");
        parse(&n, &ParseOptions::default())
    }

    fn kinds(d: &ParsedDocument) -> Vec<SegmentKind> {
        d.segments.iter().map(|s| s.kind.clone()).collect()
    }

    #[test]
    fn json_accepts_valid_and_reports_depth() {
        assert_eq!(
            validate_json(r#"{"a":[1,2.5e-3,{"b":null}],"c":"\u00e9\n"}"#, 64),
            Ok(3)
        );
        assert_eq!(validate_json("[]", 4), Ok(1));
        assert_eq!(validate_json("  [ ]  ", 4), Ok(1));
    }

    #[test]
    fn json_rejects_invalid_forms() {
        for bad in [
            "{",
            "[1,]",
            "{\"a\":}",
            "{'a':1}",
            "[01]",
            "[1.]",
            "[.5]",
            "\"\\x\"",
            "[1] x",
            "{\"a\" 1}",
            "[tru]",
            "\"a\nb\"",
            "{\"a\":1,}",
            "[\"\\u12G4\"]",
        ] {
            let err = validate_json(bad, 64).expect_err(bad);
            assert_eq!(err.kind, JsonErrorKind::Syntax, "{bad}");
        }
    }

    #[test]
    fn json_depth_limit_is_enforced_and_hard_capped() {
        let deep = format!("{}1{}", "[".repeat(5), "]".repeat(5));
        assert_eq!(validate_json(&deep, 5), Ok(5));
        assert_eq!(
            validate_json(&deep, 4).expect_err("deep").kind,
            JsonErrorKind::TooDeep
        );
        let huge = format!("{}{}", "[".repeat(5000), "]".repeat(5000));
        assert_eq!(
            validate_json(&huge, usize::MAX).expect_err("cap").kind,
            JsonErrorKind::TooDeep
        );
    }

    #[test]
    fn valid_json_is_one_segment_with_trimmed_range() {
        let d = doc("  {\"a\": 1}\n");
        assert_eq!(d.format, DocumentFormat::Json);
        assert_eq!(d.segments.len(), 1);
        assert_eq!(d.segments[0].text, "{\"a\": 1}");
        assert_eq!(
            d.segments[0].span,
            RawSpan {
                byte_start: 2,
                byte_end: 10
            }
        );
    }

    #[test]
    fn invalid_and_too_deep_json_raise_issues_and_fall_back() {
        let d = doc("{\"a\": ");
        assert_eq!(d.format, DocumentFormat::Prose);
        assert!(matches!(
            d.issues.as_slice(),
            [ParseIssue::JsonInvalid { .. }]
        ));
        assert!(d.issues[0].to_semantic_issue().hard);
        let n = normalize("[[[[1]]]]".as_bytes(), &NormalizeOptions::default()).expect("ok");
        let d = parse(
            &n,
            &ParseOptions {
                max_json_depth: 2,
                ..Default::default()
            },
        );
        assert_eq!(d.issues, vec![ParseIssue::JsonTooDeep { limit: 2 }]);
    }

    #[test]
    fn markdown_headings_and_fences() {
        let d = doc("# Title\n\nIntro text\nmore\n\n```rust\nfn x() {}\n```\n## Sub\ntail\n");
        assert_eq!(d.format, DocumentFormat::Markdown);
        assert_eq!(
            kinds(&d),
            vec![
                SegmentKind::MarkdownHeading { level: 1 },
                SegmentKind::Prose,
                SegmentKind::CodeFence {
                    language: Some("rust".to_owned())
                },
                SegmentKind::MarkdownHeading { level: 2 },
                SegmentKind::Prose,
            ]
        );
        assert_eq!(d.segments[1].text, "Intro text\nmore");
        assert_eq!(d.segments[2].text, "```rust\nfn x() {}\n```");
    }

    #[test]
    fn unclosed_fence_is_flagged_softly() {
        let d = doc("```py\nprint(1)\n");
        assert_eq!(d.format, DocumentFormat::Markdown);
        assert!(matches!(
            d.issues.as_slice(),
            [ParseIssue::UnclosedCodeFence { offset: 0 }]
        ));
        assert!(!d.issues[0].is_hard());
        assert_eq!(d.segments[0].text, "```py\nprint(1)");
    }

    #[test]
    fn source_code_languages() {
        let cases = [
            (
                "fn main() {\n    let mut x = 1;\n    println!(\"{x}\");\n}\n",
                CodeLanguage::Rust,
            ),
            (
                "import os\n\ndef run(x):\n    return os.getcwd()\n",
                CodeLanguage::Python,
            ),
            (
                "const a = 1;\nfunction f(x) { return x => x; }\nconsole.log(a);\n",
                CodeLanguage::JavaScript,
            ),
            ("#!/bin/bash\necho hi\n", CodeLanguage::Shell),
            ("#!/usr/bin/env python3\nx = 1\n", CodeLanguage::Python),
            ("cd /tmp\nmkdir -p x\necho done\n", CodeLanguage::Shell),
        ];
        for (src, lang) in cases {
            let d = doc(src);
            assert_eq!(d.format, DocumentFormat::SourceCode(lang), "{src}");
            assert_eq!(d.segments.len(), 1);
        }
    }

    #[test]
    fn python_comment_is_not_a_markdown_heading() {
        let d = doc("# helper\nimport sys\ndef f():\n    return sys.argv\n");
        assert_eq!(d.format, DocumentFormat::SourceCode(CodeLanguage::Python));
    }

    #[test]
    fn key_value_and_toml_like() {
        let d = doc("# comment\n[server]\nhost = \"a\"\nport=8080\n");
        assert_eq!(d.format, DocumentFormat::KeyValue);
        assert_eq!(
            kinds(&d),
            vec![
                SegmentKind::SectionHeader,
                SegmentKind::KeyValue,
                SegmentKind::KeyValue
            ]
        );
        assert_eq!(doc("a == b").format, DocumentFormat::Prose);
    }

    #[test]
    fn csv_detection_needs_consistent_columns() {
        let d = doc("name,age,city\nann,31,rome\nbob,42,oslo\n");
        assert_eq!(d.format, DocumentFormat::Csv);
        assert_eq!(d.segments.len(), 3);
        assert_eq!(d.segments[1].kind, SegmentKind::CsvRow { index: 1 });
        assert_eq!(doc("a,b,c\nd,e\n").format, DocumentFormat::Prose);
        assert_eq!(
            doc("Yes, sure.\nNo, thanks.\n").format,
            DocumentFormat::Prose
        );
        assert_eq!(
            doc("\"a,b\",c\n\"d,e\",f\n\"g,h\",i\n").format,
            DocumentFormat::Csv
        );
    }

    #[test]
    fn prose_paragraphs_and_empty() {
        let d = doc("First para\nstill first.\n\n\nSecond.\n");
        assert_eq!(d.format, DocumentFormat::Prose);
        assert_eq!(d.segments.len(), 2);
        assert_eq!(doc("  \n\n").format, DocumentFormat::Empty);
        assert!(doc("").segments.is_empty());
    }

    #[test]
    fn segment_limit_is_bounded() {
        let n = normalize(b"a\n\nb\n\nc\n", &NormalizeOptions::default()).expect("ok");
        let d = parse(
            &n,
            &ParseOptions {
                max_segments: 2,
                ..Default::default()
            },
        );
        assert_eq!(d.segments.len(), 2);
        assert_eq!(d.issues, vec![ParseIssue::SegmentLimitReached { limit: 2 }]);
    }
}
