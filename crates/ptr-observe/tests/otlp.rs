use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use ptr_observe::{
    fields, ManualClock, OtlpFileConfig, OtlpJsonFileExporter, SpanStatus, TraceEvent, TraceLevel,
    Tracer, TracerConfig, REDACTED,
};

/// Minimal test-only JSON parser (serde_json is not a dev-dependency of this crate).
#[derive(Debug, Clone, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

struct Parser<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.chars.peek().is_some_and(|c| c.is_whitespace()) {
            self.chars.next();
        }
    }

    fn expect(&mut self, want: char) {
        assert_eq!(self.chars.next(), Some(want));
    }

    fn value(&mut self) -> Json {
        self.ws();
        match self.chars.peek().copied().expect("value") {
            '{' => {
                self.chars.next();
                let mut map = BTreeMap::new();
                loop {
                    self.ws();
                    if self.chars.peek() == Some(&'}') {
                        self.chars.next();
                        return Json::Obj(map);
                    }
                    let Json::Str(key) = self.value() else {
                        panic!("key must be string")
                    };
                    self.ws();
                    self.expect(':');
                    let value = self.value();
                    assert!(map.insert(key, value).is_none(), "duplicate key");
                    self.ws();
                    if self.chars.peek() == Some(&',') {
                        self.chars.next();
                    }
                }
            }
            '[' => {
                self.chars.next();
                let mut items = Vec::new();
                loop {
                    self.ws();
                    if self.chars.peek() == Some(&']') {
                        self.chars.next();
                        return Json::Arr(items);
                    }
                    items.push(self.value());
                    self.ws();
                    if self.chars.peek() == Some(&',') {
                        self.chars.next();
                    }
                }
            }
            '"' => {
                self.chars.next();
                let mut out = String::new();
                loop {
                    match self.chars.next().expect("unterminated string") {
                        '"' => return Json::Str(out),
                        '\\' => match self.chars.next().expect("escape") {
                            'n' => out.push('\n'),
                            'r' => out.push('\r'),
                            't' => out.push('\t'),
                            'b' => out.push('\u{08}'),
                            'f' => out.push('\u{0c}'),
                            '"' => out.push('"'),
                            '\\' => out.push('\\'),
                            '/' => out.push('/'),
                            'u' => {
                                let hex: String =
                                    (0..4).filter_map(|_| self.chars.next()).collect();
                                let code = u32::from_str_radix(&hex, 16).expect("hex");
                                out.push(char::from_u32(code).expect("scalar"));
                            }
                            other => panic!("bad escape {other}"),
                        },
                        c => {
                            assert!(c as u32 >= 0x20, "raw control char in JSON string");
                            out.push(c);
                        }
                    }
                }
            }
            't' | 'f' | 'n' => {
                let mut word = String::new();
                while self.chars.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
                    word.push(self.chars.next().expect("char"));
                }
                match word.as_str() {
                    "true" => Json::Bool(true),
                    "false" => Json::Bool(false),
                    "null" => Json::Null,
                    other => panic!("bad literal {other}"),
                }
            }
            _ => {
                let mut num = String::new();
                while self
                    .chars
                    .peek()
                    .is_some_and(|c| c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E'))
                {
                    num.push(self.chars.next().expect("char"));
                }
                assert!(!num.is_empty(), "unexpected character");
                Json::Num(num)
            }
        }
    }
}

fn parse(text: &str) -> Json {
    let mut parser = Parser {
        chars: text.chars().peekable(),
    };
    let value = parser.value();
    parser.ws();
    assert!(parser.chars.next().is_none(), "trailing input");
    value
}

impl Json {
    fn get(&self, key: &str) -> &Json {
        match self {
            Json::Obj(map) => map.get(key).unwrap_or_else(|| panic!("missing key {key}")),
            other => panic!("not an object: {other:?}"),
        }
    }

    fn has(&self, key: &str) -> bool {
        matches!(self, Json::Obj(map) if map.contains_key(key))
    }

    fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(items) => items,
            other => panic!("not an array: {other:?}"),
        }
    }

    fn str(&self) -> &str {
        match self {
            Json::Str(text) => text,
            other => panic!("not a string: {other:?}"),
        }
    }
}

fn temp_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "ptr-observe-otlp-{label}-{}-{:?}.jsonl",
        std::process::id(),
        std::thread::current().id()
    ))
}

fn sample_spans() -> Vec<ptr_observe::Span> {
    let clock = Arc::new(ManualClock::new(1_000));
    let tracer = Tracer::new(
        TracerConfig {
            id_seed: 3,
            ..TracerConfig::default()
        },
        clock.clone(),
    );
    let root = tracer.root_span("req \"quoted\"\n").expect("root");
    root.set_field(fields::OPERATION, "plan");
    root.set_field("count", 7u64);
    root.set_field("huge", u64::MAX);
    root.set_field("flag", true);
    clock.advance(50);
    let child = root.child("child").expect("child");
    child.event(TraceEvent::new(TraceLevel::Warn, "slow").with_field("path", "C:\\tmp\\\u{1}x"));
    clock.advance(25);
    child.fail("boom");
    clock.advance(25);
    root.finish();
    tracer.drain_finished()
}

#[test]
fn exported_line_has_otlp_json_trace_layout() {
    let path = temp_path("layout");
    let _ = std::fs::remove_file(&path);
    let exporter = OtlpJsonFileExporter::new(&path, OtlpFileConfig::default());
    let spans = sample_spans();
    assert_eq!(exporter.export(&spans).expect("export"), 2);
    assert_eq!(exporter.export(&[]).expect("noop"), 0);
    assert_eq!(exporter.export(&spans[..1]).expect("second batch"), 1);

    let text = std::fs::read_to_string(&path).expect("read");
    let _ = std::fs::remove_file(&path);
    let lines: Vec<_> = text.lines().collect();
    assert_eq!(lines.len(), 2, "one JSON document per export call");

    let doc = parse(lines[0]);
    let resource_spans = doc.get("resourceSpans").arr();
    assert_eq!(resource_spans.len(), 1);
    let service = &resource_spans[0].get("resource").get("attributes").arr()[0];
    assert_eq!(service.get("key").str(), "service.name");
    assert_eq!(service.get("value").get("stringValue").str(), "ptr");
    let scope_spans = resource_spans[0].get("scopeSpans").arr();
    assert_eq!(scope_spans[0].get("scope").get("name").str(), "ptr-observe");
    let json_spans = scope_spans[0].get("spans").arr();
    assert_eq!(json_spans.len(), 2);

    let child = &json_spans[0];
    let root = &json_spans[1];
    assert_eq!(child.get("traceId").str(), root.get("traceId").str());
    assert_eq!(child.get("traceId").str().len(), 32);
    assert_eq!(child.get("spanId").str().len(), 16);
    assert_eq!(child.get("parentSpanId").str(), root.get("spanId").str());
    assert!(!root.has("parentSpanId"));
    assert_eq!(root.get("name").str(), "req \"quoted\"\n");
    assert_eq!(root.get("startTimeUnixNano"), &Json::Str("1000".into()));
    assert_eq!(root.get("endTimeUnixNano"), &Json::Str("1100".into()));
    assert_eq!(child.get("startTimeUnixNano"), &Json::Str("1050".into()));
    assert_eq!(root.get("status").get("code"), &Json::Num("1".into()));
    assert_eq!(child.get("status").get("code"), &Json::Num("2".into()));
    assert_eq!(child.get("status").get("message").str(), "boom");

    let attrs: BTreeMap<_, _> = root
        .get("attributes")
        .arr()
        .iter()
        .map(|a| (a.get("key").str().to_owned(), a.get("value").clone()))
        .collect();
    assert_eq!(attrs["count"].get("intValue"), &Json::Str("7".into()));
    assert_eq!(
        attrs["huge"].get("stringValue"),
        &Json::Str(u64::MAX.to_string())
    );
    assert_eq!(attrs["flag"].get("boolValue"), &Json::Bool(true));
    assert_eq!(attrs[fields::OPERATION].get("stringValue").str(), "plan");

    let event = &child.get("events").arr()[0];
    assert_eq!(event.get("name").str(), "slow");
    assert_eq!(event.get("timeUnixNano"), &Json::Str("1050".into()));
    let event_attrs: BTreeMap<_, _> = event
        .get("attributes")
        .arr()
        .iter()
        .map(|a| (a.get("key").str().to_owned(), a.get("value").clone()))
        .collect();
    assert_eq!(
        event_attrs["path"].get("stringValue").str(),
        "C:\\tmp\\\u{1}x"
    );
    assert_eq!(event_attrs["ptr.level"].get("stringValue").str(), "warn");
}

#[test]
fn epoch_offset_and_service_name_are_configurable() {
    let config = OtlpFileConfig {
        service_name: "svc \"x\"".into(),
        unix_epoch_offset_ns: 1_700_000_000_000_000_000,
        ..OtlpFileConfig::default()
    };
    let exporter = OtlpJsonFileExporter::new(temp_path("unused"), config);
    let doc = parse(&exporter.render(&sample_spans()));
    let rs = &doc.get("resourceSpans").arr()[0];
    assert_eq!(
        rs.get("resource").get("attributes").arr()[0]
            .get("value")
            .get("stringValue")
            .str(),
        "svc \"x\""
    );
    let span = &rs.get("scopeSpans").arr()[0].get("spans").arr()[1];
    assert_eq!(
        span.get("startTimeUnixNano"),
        &Json::Str("1700000000000001000".into())
    );
}

#[test]
fn exporter_redacts_secrets_before_writing() {
    let clock = Arc::new(ManualClock::new(0));
    let tracer = Tracer::new(TracerConfig::default(), clock);
    let span = tracer.root_span("login").expect("root");
    span.set_field("db.password", "hunter2hunter2");
    span.set_field("url", "postgres://u:pw-secret-9@db/x");
    span.event(
        TraceEvent::new(TraceLevel::Info, "hdr").with_field("Authorization", "Bearer abcdefghijkl"),
    );
    span.fail("failed with Bearer zzzzzzzzzzzz");
    let spans = tracer.drain_finished();

    let exporter = OtlpJsonFileExporter::new(temp_path("redact"), OtlpFileConfig::default());
    let rendered = exporter.render(&spans);
    for plaintext in ["hunter2", "pw-secret-9", "abcdefghijkl", "zzzzzzzzzzzz"] {
        assert!(!rendered.contains(plaintext), "leaked {plaintext}");
    }
    assert!(rendered.contains(REDACTED));
    assert_eq!(
        spans[0].status,
        SpanStatus::Error("failed with Bearer zzzzzzzzzzzz".into())
    );
}

#[test]
fn unwritable_path_is_typed_export_error() {
    let dir = std::env::temp_dir()
        .join("ptr-observe-no-such-dir-x")
        .join("f.jsonl");
    let exporter = OtlpJsonFileExporter::new(dir, OtlpFileConfig::default());
    let error = exporter.export(&sample_spans()).expect_err("must fail");
    assert_eq!(error.code(), "export");
}

#[test]
fn exporter_redacts_secret_looking_request_id() {
    let clock = Arc::new(ManualClock::new(0));
    let tracer = Tracer::new(TracerConfig::default(), clock);
    let span = tracer.root_span("req").expect("root");
    span.set_field(
        "client.request",
        ptr_types::RequestId::from("Bearer abcdef0123456789"),
    );
    span.event(TraceEvent::request(
        TraceLevel::Info,
        "inner",
        ptr_types::RequestId::from("q8Zr3xK9LmP2vB7nT5yW1cD6hJ4sA0eFgUiOpXzRtYu"),
        ptr_types::Revision(1),
    ));
    span.finish();
    let spans = tracer.drain_finished();

    let path = temp_path("reqid");
    let exporter = OtlpJsonFileExporter::new(path.clone(), OtlpFileConfig::default());
    exporter.export(&spans).expect("export");
    let written = std::fs::read_to_string(&path).expect("read");
    let _ = std::fs::remove_file(&path);
    for plaintext in [
        "abcdef0123456789",
        "q8Zr3xK9LmP2vB7nT5yW1cD6hJ4sA0eFgUiOpXzRtYu",
    ] {
        assert!(!written.contains(plaintext), "leaked {plaintext}");
    }
    assert!(written.contains(REDACTED));
}
