//! A stdin/stdout JSON-lines bridge from a harness to a Python retrieval driver.
//!
//! The drivers (`research/baselines/rag_reference/q003_driver.py` and
//! `research/baselines/strong_retrieval/driver.py`) speak protocol 1: one JSON object per line
//! in, one JSON object per line out, in order. A reply is `{"ok":true,...}` or
//! `{"ok":false,"error":"..."}`; a malformed or invalid request is answered with an error and
//! the driver keeps running; end of input (or `quit`) ends it. The digest-pinned baseline
//! implementation is what runs: the bridge adds no retrieval logic.
//!
//! The API is small on purpose, so another harness can copy this file as it is (it needs only
//! `super::json::json_string`):
//!
//! ```text
//! let mut bridge = Bridge::spawn("python3", &[script], &Expect::driver("rag_reference"))?;
//! bridge.upsert_many(&[Doc { id, text, embedding }])?;
//! let hits = bridge.search(query_text, query_embedding, 10)?;   // best first
//! bridge.delete(id)?;  bridge.reset()?;  bridge.size()?;
//! ```
//!
//! Every failure is a [`BridgeError`] and is a harness-integrity failure (the harness exits 2):
//! the child cannot be spawned, dies, answers late or with something that is not JSON or not the
//! protocol, or answers a request the harness built correctly with `ok: false`. None of them is
//! a finding about the product under test. A bridge that is dropped kills its child.

// A harness uses a few of the operations; the rest are the bridge's surface for the next one.
#![allow(dead_code)]

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::Duration;

/// The protocol version the bridge speaks.
pub const PROTOCOL: i64 = 1;

/// How long a reply may take before the child counts as hung. The slowest request (indexing a
/// few thousand documents, or a reranked search of the strong baseline) stays far below it.
pub const REPLY_TIMEOUT: Duration = Duration::from_secs(3600);

/// Why the bridge failed. Every variant is the harness's, never the product's.
#[derive(Debug, PartialEq)]
pub enum BridgeError {
    /// The child process could not be started.
    Spawn(String),
    /// The pipes to the child broke, it exited or it did not answer in time.
    Io(String),
    /// The child answered with something that is not a protocol-1 reply of the kind asked for.
    Protocol(String),
    /// The child understood the request and refused it (`ok: false`).
    Driver(String),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spawn(text) => write!(formatter, "bridge: cannot spawn the driver: {text}"),
            Self::Io(text) => write!(formatter, "bridge: i/o with the driver failed: {text}"),
            Self::Protocol(text) => write!(formatter, "bridge: protocol mismatch: {text}"),
            Self::Driver(text) => write!(formatter, "bridge: the driver refused a request: {text}"),
        }
    }
}

/// What the harness expects the driver to say about itself in its `hello` reply.
pub struct Expect<'a> {
    /// The `driver` field (`"rag_reference"`, ...).
    pub driver: &'a str,
    /// The `embedding_model` field, when the driver reports one and the harness pins it.
    pub embedding_model: Option<&'a str>,
}

impl<'a> Expect<'a> {
    pub fn driver(driver: &'a str) -> Self {
        Self {
            driver,
            embedding_model: None,
        }
    }
}

/// One document to index.
pub struct Doc<'a> {
    pub id: &'a str,
    pub text: &'a str,
    pub embedding: &'a [f32],
}

/// One search hit, best first. The score is the driver's (fused rank score), only its order is
/// meaningful.
#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub id: String,
    pub score: f64,
}

/// A running driver.
pub struct Bridge {
    child: Child,
    stdin: Option<ChildStdin>,
    replies: Receiver<Result<String, String>>,
}

impl std::fmt::Debug for Bridge {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Bridge(pid {})", self.child.id())
    }
}

impl Bridge {
    /// Start `program args...` (standard input and output piped, standard error inherited),
    /// say `hello` and check the reply against `expect`.
    pub fn spawn(program: &str, args: &[&str], expect: &Expect) -> Result<Self, BridgeError> {
        let mut child = Command::new(program)
            .args(args)
            // A driver must leave nothing in the baseline directory it runs from (its digest is
            // pinned): no bytecode caches.
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| BridgeError::Spawn(format!("{program}: {error}")))?;
        let stdin = child.stdin.take();
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| BridgeError::Spawn("no standard output pipe".to_string()))?;
        let (sender, replies) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let line = line.map_err(|error| error.to_string());
                let failed = line.is_err();
                if sender.send(line).is_err() || failed {
                    return;
                }
            }
        });
        let mut bridge = Self {
            child,
            stdin,
            replies,
        };
        let hello = bridge.call("{\"op\":\"hello\"}")?;
        let driver = hello.get("driver").and_then(Value::as_str);
        if driver != Some(expect.driver) {
            return Err(BridgeError::Protocol(format!(
                "hello names the driver {driver:?}, expected {:?}",
                expect.driver
            )));
        }
        if hello.get("protocol").and_then(Value::as_i64) != Some(PROTOCOL) {
            return Err(BridgeError::Protocol(format!(
                "hello names the protocol {:?}, expected {PROTOCOL}",
                hello.get("protocol")
            )));
        }
        if let Some(model) = expect.embedding_model {
            let reported = hello.get("embedding_model").and_then(Value::as_str);
            if reported != Some(model) {
                return Err(BridgeError::Protocol(format!(
                    "hello names the embedding model {reported:?}, expected {model:?}"
                )));
            }
        }
        Ok(bridge)
    }

    /// Send one request line and read its reply. `ok: false` is [`BridgeError::Driver`].
    fn call(&mut self, request: &str) -> Result<Value, BridgeError> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| BridgeError::Io("the driver was already asked to quit".to_string()))?;
        stdin
            .write_all(request.as_bytes())
            .and_then(|()| stdin.write_all(b"\n"))
            .and_then(|()| stdin.flush())
            .map_err(|error| BridgeError::Io(format!("write: {error}")))?;
        let line = match self.replies.recv_timeout(REPLY_TIMEOUT) {
            Ok(Ok(line)) => line,
            Ok(Err(error)) => return Err(BridgeError::Io(format!("read: {error}"))),
            Err(RecvTimeoutError::Timeout) => {
                return Err(BridgeError::Io("no reply in time".to_string()))
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(BridgeError::Io(
                    "the driver closed its output (it exited)".to_string(),
                ))
            }
        };
        let reply = parse(&line).map_err(|error| {
            BridgeError::Protocol(format!("reply is not JSON ({error}): {line}"))
        })?;
        match reply.get("ok") {
            Some(Value::Bool(true)) => Ok(reply),
            Some(Value::Bool(false)) => Err(BridgeError::Driver(
                reply
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("no error text")
                    .to_string(),
            )),
            _ => Err(BridgeError::Protocol(format!(
                "reply has no boolean ok: {line}"
            ))),
        }
    }

    /// Drop every document.
    pub fn reset(&mut self) -> Result<(), BridgeError> {
        self.call("{\"op\":\"reset\"}").map(|_| ())
    }

    /// Index (or replace) one document.
    pub fn upsert(&mut self, doc: &Doc) -> Result<(), BridgeError> {
        let request = format!("{{\"op\":\"upsert\",{}}}", doc_fields(doc)?);
        self.call(&request).map(|_| ())
    }

    /// Index (or replace) documents in one request; the driver's count is checked.
    pub fn upsert_many(&mut self, docs: &[Doc]) -> Result<(), BridgeError> {
        let mut parts = Vec::with_capacity(docs.len());
        for doc in docs {
            parts.push(format!("{{{}}}", doc_fields(doc)?));
        }
        let request = format!("{{\"op\":\"upsert_many\",\"docs\":[{}]}}", parts.join(","));
        let reply = self.call(&request)?;
        match reply.get("upserted").and_then(Value::as_i64) {
            Some(count) if count == docs.len() as i64 => Ok(()),
            other => Err(BridgeError::Protocol(format!(
                "upsert_many of {} documents was acknowledged as {other:?}",
                docs.len()
            ))),
        }
    }

    /// Remove a document (removing an unknown id is the driver's business).
    pub fn delete(&mut self, id: &str) -> Result<(), BridgeError> {
        let request = format!(
            "{{\"op\":\"delete\",\"id\":{}}}",
            super::json::json_string(id)
        );
        self.call(&request).map(|_| ())
    }

    /// How many documents the driver holds.
    pub fn size(&mut self) -> Result<usize, BridgeError> {
        let reply = self.call("{\"op\":\"size\"}")?;
        match reply.get("documents").and_then(Value::as_i64) {
            Some(count) if count >= 0 => Ok(count as usize),
            other => Err(BridgeError::Protocol(format!("size reply holds {other:?}"))),
        }
    }

    /// The best `limit` documents for the query, best first.
    pub fn search(
        &mut self,
        query: &str,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<Hit>, BridgeError> {
        let request = format!(
            "{{\"op\":\"search\",\"query\":{},\"embedding\":{},\"limit\":{limit}}}",
            super::json::json_string(query),
            embedding_json(embedding)?
        );
        let reply = self.call(&request)?;
        let Some(Value::Arr(hits)) = reply.get("hits") else {
            return Err(BridgeError::Protocol(
                "search reply has no hits array".to_string(),
            ));
        };
        if hits.len() > limit {
            return Err(BridgeError::Protocol(format!(
                "search returned {} hits for a limit of {limit}",
                hits.len()
            )));
        }
        hits.iter()
            .map(|hit| {
                let id = hit.get("id").and_then(Value::as_str);
                let score = hit.get("score").and_then(Value::as_f64);
                match (id, score) {
                    (Some(id), Some(score)) => Ok(Hit {
                        id: id.to_string(),
                        score,
                    }),
                    _ => Err(BridgeError::Protocol(format!("malformed hit {hit:?}"))),
                }
            })
            .collect()
    }

    /// Ask the driver to end and wait for it. A driver that exits non-zero is an error.
    pub fn quit(mut self) -> Result<(), BridgeError> {
        self.call("{\"op\":\"quit\"}")?;
        self.stdin = None;
        let status = self
            .child
            .wait()
            .map_err(|error| BridgeError::Io(format!("wait: {error}")))?;
        if status.success() {
            Ok(())
        } else {
            Err(BridgeError::Io(format!("the driver exited with {status}")))
        }
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        self.stdin = None;
        // Best effort: a driver that has already exited has nothing to kill.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn embedding_json(embedding: &[f32]) -> Result<String, BridgeError> {
    let mut parts = Vec::with_capacity(embedding.len());
    for value in embedding {
        if !value.is_finite() {
            return Err(BridgeError::Protocol(
                "an embedding holds a non-finite number".to_string(),
            ));
        }
        // Widened first: the decimal text of the f64 is the f32's exact value to f64 precision,
        // so the driver parses the number the harness holds.
        parts.push(format!("{}", f64::from(*value)));
    }
    Ok(format!("[{}]", parts.join(",")))
}

fn doc_fields(doc: &Doc) -> Result<String, BridgeError> {
    Ok(format!(
        "\"id\":{},\"text\":{},\"embedding\":{}",
        super::json::json_string(doc.id),
        super::json::json_string(doc.text),
        embedding_json(doc.embedding)?
    ))
}

/// A parsed JSON value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    /// Every number, integer or not, as the f64 it parses to.
    Num(f64),
    Str(String),
    Arr(Vec<Value>),
    Obj(Vec<(String, Value)>),
}

impl Value {
    /// The member `key` of an object.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Self::Obj(pairs) => pairs.iter().find(|(name, _)| name == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(text) => Some(text),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Num(number) => Some(*number),
            _ => None,
        }
    }

    /// The number when it is an integer in the exact range of an f64.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Num(number) if number.fract() == 0.0 && number.abs() < 9.0e15 => {
                Some(*number as i64)
            }
            _ => None,
        }
    }
}

/// Parse one JSON document (surrounding whitespace allowed, nothing after it).
pub fn parse(text: &str) -> Result<Value, String> {
    let mut parser = Parser {
        bytes: text.as_bytes(),
        at: 0,
        depth: 0,
    };
    parser.space();
    let value = parser.value()?;
    parser.space();
    if parser.at != parser.bytes.len() {
        return Err(format!("trailing characters at byte {}", parser.at));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
    depth: usize,
}

const MAX_DEPTH: usize = 32;

impl Parser<'_> {
    fn space(&mut self) {
        while matches!(self.bytes.get(self.at), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
        }
    }

    fn eat(&mut self, byte: u8) -> Result<(), String> {
        if self.bytes.get(self.at) == Some(&byte) {
            self.at += 1;
            Ok(())
        } else {
            Err(format!("expected {:?} at byte {}", byte as char, self.at))
        }
    }

    fn keyword(&mut self, word: &str, value: Value) -> Result<Value, String> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(value)
        } else {
            Err(format!("bad literal at byte {}", self.at))
        }
    }

    fn value(&mut self) -> Result<Value, String> {
        match self.bytes.get(self.at) {
            None => Err("unexpected end".to_string()),
            Some(b'n') => self.keyword("null", Value::Null),
            Some(b't') => self.keyword("true", Value::Bool(true)),
            Some(b'f') => self.keyword("false", Value::Bool(false)),
            Some(b'"') => self.string().map(Value::Str),
            Some(b'[') => self.nested(Self::array),
            Some(b'{') => self.nested(Self::object),
            Some(_) => self.number(),
        }
    }

    fn nested(&mut self, inner: fn(&mut Self) -> Result<Value, String>) -> Result<Value, String> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err("nested too deeply".to_string());
        }
        let value = inner(self);
        self.depth -= 1;
        value
    }

    fn array(&mut self) -> Result<Value, String> {
        self.eat(b'[')?;
        let mut items = Vec::new();
        self.space();
        if self.bytes.get(self.at) == Some(&b']') {
            self.at += 1;
            return Ok(Value::Arr(items));
        }
        loop {
            self.space();
            items.push(self.value()?);
            self.space();
            match self.bytes.get(self.at) {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Value::Arr(items));
                }
                _ => return Err(format!("expected , or ] at byte {}", self.at)),
            }
        }
    }

    fn object(&mut self) -> Result<Value, String> {
        self.eat(b'{')?;
        let mut pairs = Vec::new();
        self.space();
        if self.bytes.get(self.at) == Some(&b'}') {
            self.at += 1;
            return Ok(Value::Obj(pairs));
        }
        loop {
            self.space();
            let key = self.string()?;
            self.space();
            self.eat(b':')?;
            self.space();
            let value = self.value()?;
            pairs.push((key, value));
            self.space();
            match self.bytes.get(self.at) {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Value::Obj(pairs));
                }
                _ => return Err(format!("expected , or }} at byte {}", self.at)),
            }
        }
    }

    fn number(&mut self) -> Result<Value, String> {
        let start = self.at;
        while matches!(
            self.bytes.get(self.at),
            Some(b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
        ) {
            self.at += 1;
        }
        let text = std::str::from_utf8(&self.bytes[start..self.at]).map_err(|e| e.to_string())?;
        match text.parse::<f64>() {
            Ok(number) if number.is_finite() => Ok(Value::Num(number)),
            _ => Err(format!("bad number {text:?} at byte {start}")),
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let digits = self
            .bytes
            .get(self.at..self.at + 4)
            .and_then(|slice| std::str::from_utf8(slice).ok())
            .and_then(|slice| u32::from_str_radix(slice, 16).ok())
            .ok_or_else(|| format!("bad \\u escape at byte {}", self.at))?;
        self.at += 4;
        Ok(digits)
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            let start = self.at;
            while !matches!(self.bytes.get(self.at), Some(b'"' | b'\\') | None) {
                self.at += 1;
            }
            out.push_str(
                std::str::from_utf8(&self.bytes[start..self.at]).map_err(|e| e.to_string())?,
            );
            match self.bytes.get(self.at) {
                None => return Err("unterminated string".to_string()),
                Some(b'"') => {
                    self.at += 1;
                    return Ok(out);
                }
                _ => {
                    self.at += 1;
                    let escape = *self.bytes.get(self.at).ok_or("unterminated escape")?;
                    self.at += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let first = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&first) {
                                if self.bytes.get(self.at..self.at + 2) != Some(b"\\u") {
                                    return Err("lone high surrogate".to_string());
                                }
                                self.at += 2;
                                let second = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&second) {
                                    return Err("bad low surrogate".to_string());
                                }
                                0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                            } else {
                                first
                            };
                            out.push(char::from_u32(code).ok_or("bad code point")?);
                        }
                        other => return Err(format!("bad escape \\{}", other as char)),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn driver_script() -> String {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        root.join("research/baselines/rag_reference/q003_driver.py")
            .to_string_lossy()
            .into_owned()
    }

    fn spawn_real() -> Bridge {
        Bridge::spawn(
            "python3",
            &[&driver_script()],
            &Expect::driver("rag_reference"),
        )
        .expect("the pinned reference driver starts")
    }

    #[test]
    fn the_parser_reads_documents_and_refuses_the_malformed() {
        let value =
            parse(r#" {"a":[1,-2.5e1,"x\n\u00e9\ud83d\ude00"],"b":{"c":null,"d":true}} "#).unwrap();
        assert_eq!(
            value.get("a"),
            Some(&Value::Arr(vec![
                Value::Num(1.0),
                Value::Num(-25.0),
                Value::Str("x\né😀".to_string())
            ]))
        );
        assert_eq!(
            value.get("b").and_then(|b| b.get("d")),
            Some(&Value::Bool(true))
        );
        assert_eq!(Value::Num(3.0).as_i64(), Some(3));
        assert_eq!(Value::Num(3.5).as_i64(), None);
        for bad in [
            "",
            "{",
            "[1,]",
            "{\"a\" 1}",
            "tru",
            "\"x",
            "1 2",
            "[1 2]",
            "\"\\ud83d\"",
            "nan",
            "\"\\q\"",
        ] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
        let deep = format!("{}{}", "[".repeat(40), "]".repeat(40));
        assert!(parse(&deep).is_err(), "nesting is bounded");
    }

    #[test]
    fn documents_round_trip_through_the_real_driver() {
        let mut bridge = spawn_real();
        let one = [1.0_f32, 0.0, 0.0];
        let two = [0.0_f32, 1.0, 0.0];
        bridge
            .upsert_many(&[
                Doc {
                    id: "a",
                    text: "Renata lives in Lisbon",
                    embedding: &one,
                },
                Doc {
                    id: "b",
                    text: "Corin works for Brightwell Labs",
                    embedding: &two,
                },
            ])
            .unwrap();
        assert_eq!(bridge.size().unwrap(), 2);
        let hits = bridge.search("where does Renata live", &one, 5).unwrap();
        assert_eq!(hits.first().map(|hit| hit.id.as_str()), Some("a"));
        assert!(hits.windows(2).all(|pair| pair[0].score >= pair[1].score));
        bridge.delete("a").unwrap();
        let hits = bridge.search("where does Renata live", &one, 5).unwrap();
        assert!(hits.iter().all(|hit| hit.id != "a"));
        bridge.reset().unwrap();
        assert_eq!(bridge.size().unwrap(), 0);
        bridge.quit().unwrap();
    }

    #[test]
    fn a_request_the_driver_refuses_is_a_driver_error() {
        let mut bridge = spawn_real();
        bridge
            .upsert(&Doc {
                id: "a",
                text: "x",
                embedding: &[1.0, 0.0],
            })
            .unwrap();
        // The index uses two dimensions: a three-dimensional embedding is refused.
        let refused = bridge.upsert(&Doc {
            id: "b",
            text: "y",
            embedding: &[1.0, 0.0, 0.0],
        });
        assert!(
            matches!(refused, Err(BridgeError::Driver(_))),
            "{refused:?}"
        );
        // The driver keeps running after a refusal.
        assert_eq!(bridge.size().unwrap(), 1);
        let nonfinite = bridge.upsert(&Doc {
            id: "c",
            text: "z",
            embedding: &[f32::NAN, 0.0],
        });
        assert!(matches!(nonfinite, Err(BridgeError::Protocol(_))));
        let too_many = bridge.search("x", &[1.0, 0.0], 0);
        assert!(
            matches!(too_many, Err(BridgeError::Driver(_))),
            "{too_many:?}"
        );
    }

    #[test]
    fn a_missing_program_a_wrong_driver_and_a_silent_child_are_bridge_errors() {
        let missing = Bridge::spawn(
            "/nonexistent/interpreter",
            &[],
            &Expect::driver("rag_reference"),
        );
        assert!(matches!(missing, Err(BridgeError::Spawn(_))));

        let wrong_driver = Bridge::spawn(
            "python3",
            &[&driver_script()],
            &Expect::driver("strong_retrieval"),
        );
        assert!(matches!(wrong_driver, Err(BridgeError::Protocol(_))));

        let wrong_model = Bridge::spawn(
            "python3",
            &[&driver_script()],
            &Expect {
                driver: "rag_reference",
                embedding_model: Some("another/model"),
            },
        );
        assert!(matches!(wrong_model, Err(BridgeError::Protocol(_))));

        let exits = Bridge::spawn("python3", &["-c", "pass"], &Expect::driver("rag_reference"));
        assert!(matches!(exits, Err(BridgeError::Io(_))), "{exits:?}");

        let garbage = Bridge::spawn(
            "python3",
            &["-c", "print('not json', flush=True); input()"],
            &Expect::driver("rag_reference"),
        );
        assert!(
            matches!(garbage, Err(BridgeError::Protocol(_))),
            "{garbage:?}"
        );

        let old_protocol = Bridge::spawn(
            "python3",
            &[
                "-c",
                "print('{\"ok\":true,\"driver\":\"rag_reference\",\"protocol\":2}', flush=True); input()",
            ],
            &Expect::driver("rag_reference"),
        );
        assert!(matches!(old_protocol, Err(BridgeError::Protocol(_))));
    }
}
