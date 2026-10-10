//! A JSON-lines bridge to a child process, and the small JSON reader its replies need.
//!
//! The reference arm (`research/baselines/rag_reference/q003_driver.py`), the strong arm
//! (`research/baselines/strong_retrieval/driver.py`) and the BM25-only helper
//! (`experiments/retrieval/Q003-postgres-hybrid/bm25_driver.py`) speak one protocol: one JSON
//! object per line in, one per line out, `{"ok": true, ...}` or `{"ok": false, "error": ...}`.
//! None of them is code under test, so every failure of the bridge (a process that does not start,
//! a reply that does not parse, a refusal, a timeout, an end of file) is the harness's: the caller
//! turns it into exit status 2.
//!
//! The workspace has no JSON parser the harness may depend on, so [`parse_json`] reads the replies
//! (and the pool file) into the emitter's own [`Json`] value.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::Duration;

use crate::experiments::layered_kv::emit::{resolve, Json};

/// Parse one JSON document. Integers without a fraction or exponent that fit an `i64` are
/// [`Json::Int`], every other number is [`Json::Float`].
pub fn parse_json(text: &str) -> Result<Json, String> {
    let mut reader = Reader {
        bytes: text.as_bytes(),
        at: 0,
    };
    reader.skip_space();
    let value = reader.value(0)?;
    reader.skip_space();
    if reader.at != reader.bytes.len() {
        return Err(format!("trailing characters at byte {}", reader.at));
    }
    Ok(value)
}

const MAX_DEPTH: usize = 32;

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn skip_space(&mut self) {
        while matches!(self.bytes.get(self.at), Some(b' ' | b'\t' | b'\r' | b'\n')) {
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

    fn literal(&mut self, word: &str, value: Json) -> Result<Json, String> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(value)
        } else {
            Err(format!("unexpected token at byte {}", self.at))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > MAX_DEPTH {
            return Err("nesting is too deep".to_string());
        }
        match self.bytes.get(self.at) {
            None => Err("unexpected end of input".to_string()),
            Some(b'n') => self.literal("null", Json::Null),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'"') => self.string().map(Json::Str),
            Some(b'[') => {
                self.at += 1;
                let mut items = Vec::new();
                self.skip_space();
                if self.bytes.get(self.at) == Some(&b']') {
                    self.at += 1;
                    return Ok(Json::Arr(items));
                }
                loop {
                    self.skip_space();
                    items.push(self.value(depth + 1)?);
                    self.skip_space();
                    match self.bytes.get(self.at) {
                        Some(b',') => self.at += 1,
                        Some(b']') => {
                            self.at += 1;
                            return Ok(Json::Arr(items));
                        }
                        _ => return Err(format!("expected , or ] at byte {}", self.at)),
                    }
                }
            }
            Some(b'{') => {
                self.at += 1;
                let mut pairs = Vec::new();
                self.skip_space();
                if self.bytes.get(self.at) == Some(&b'}') {
                    self.at += 1;
                    return Ok(Json::Obj(pairs));
                }
                loop {
                    self.skip_space();
                    let key = self.string()?;
                    self.skip_space();
                    self.eat(b':')?;
                    self.skip_space();
                    pairs.push((key, self.value(depth + 1)?));
                    self.skip_space();
                    match self.bytes.get(self.at) {
                        Some(b',') => self.at += 1,
                        Some(b'}') => {
                            self.at += 1;
                            return Ok(Json::Obj(pairs));
                        }
                        _ => return Err(format!("expected , or }} at byte {}", self.at)),
                    }
                }
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(other) => Err(format!(
                "unexpected character {:?} at byte {}",
                *other as char, self.at
            )),
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.at;
        let mut fractional = false;
        while let Some(byte) = self.bytes.get(self.at) {
            match byte {
                b'0'..=b'9' | b'-' | b'+' => {}
                b'.' | b'e' | b'E' => fractional = true,
                _ => break,
            }
            self.at += 1;
        }
        let text = std::str::from_utf8(&self.bytes[start..self.at])
            .map_err(|_| "a number is not UTF-8".to_string())?;
        if !fractional {
            if let Ok(number) = text.parse::<i64>() {
                return Ok(Json::Int(number));
            }
        }
        text.parse::<f64>()
            .ok()
            .filter(|number| number.is_finite())
            .map(Json::Float)
            .ok_or_else(|| format!("{text:?} is not a number"))
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let digits = self
            .bytes
            .get(self.at..self.at + 4)
            .ok_or_else(|| "a \\u escape is cut short".to_string())?;
        let text = std::str::from_utf8(digits).map_err(|_| "a \\u escape is not ASCII")?;
        self.at += 4;
        u32::from_str_radix(text, 16).map_err(|_| format!("{text:?} is not hexadecimal"))
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let byte = *self
                .bytes
                .get(self.at)
                .ok_or_else(|| "a string is not closed".to_string())?;
            self.at += 1;
            match byte {
                b'"' => break,
                b'\\' => {
                    let escape = *self
                        .bytes
                        .get(self.at)
                        .ok_or_else(|| "a string ends in a backslash".to_string())?;
                    self.at += 1;
                    let character = match escape {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let high = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&high) {
                                if self.bytes.get(self.at..self.at + 2) != Some(b"\\u") {
                                    return Err("a high surrogate has no partner".to_string());
                                }
                                self.at += 2;
                                let low = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&low) {
                                    return Err("a high surrogate has a bad partner".to_string());
                                }
                                0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
                            } else {
                                high
                            };
                            char::from_u32(code).ok_or_else(|| "a lone surrogate".to_string())?
                        }
                        other => return Err(format!("unknown escape \\{}", other as char)),
                    };
                    let mut buffer = [0u8; 4];
                    out.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
                }
                byte if byte < 0x20 => return Err("a control character in a string".to_string()),
                byte => out.push(byte),
            }
        }
        String::from_utf8(out).map_err(|_| "a string is not UTF-8".to_string())
    }
}

/// A float in the shortest form that reads back as the same `f32`, which is how an `f32` embedding
/// is sent: every value is a half-precision number widened to `f32`, so the decimal text is short.
pub fn push_f32(out: &mut String, value: f32) {
    use std::fmt::Write as _;
    let _ = write!(out, "{value}");
}

/// `[v0,v1,...]` for an embedding.
pub fn push_embedding(out: &mut String, values: &[f32]) {
    out.push('[');
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        push_f32(out, *value);
    }
    out.push(']');
}

/// One document of an `upsert_many` request.
pub struct DocRef<'a> {
    pub id: &'a str,
    pub text: &'a str,
    pub embedding: &'a [f32],
}

pub fn upsert_many_line(docs: &[DocRef<'_>]) -> String {
    let mut out = String::from("{\"op\":\"upsert_many\",\"docs\":[");
    for (index, doc) in docs.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str("{\"id\":");
        out.push_str(&crate::experiments::json::json_string(doc.id));
        out.push_str(",\"text\":");
        out.push_str(&crate::experiments::json::json_string(doc.text));
        out.push_str(",\"embedding\":");
        push_embedding(&mut out, doc.embedding);
        out.push('}');
    }
    out.push_str("]}");
    out
}

pub fn search_line(query: &str, embedding: &[f32], limit: usize) -> String {
    let mut out = String::from("{\"op\":\"search\",\"query\":");
    out.push_str(&crate::experiments::json::json_string(query));
    out.push_str(",\"embedding\":");
    push_embedding(&mut out, embedding);
    out.push_str(&format!(",\"limit\":{limit}}}"));
    out
}

/// One ranked list a driver returned, best first.
#[derive(Clone, Debug, PartialEq)]
pub struct Reply {
    pub hits: Vec<String>,
    /// `reranked` of a strong-retrieval reply; `None` for drivers without a reranker.
    pub reranked: Option<bool>,
    pub rerank_seconds: f64,
}

/// A running driver process.
pub struct Bridge {
    name: String,
    child: Child,
    stdin: Option<ChildStdin>,
    replies: Receiver<Result<String, String>>,
    timeout: Duration,
    /// Requests sent and answered.
    pub requests: u64,
}

impl Bridge {
    /// Start `program args...` with `cwd` as its working directory. Its standard error is passed
    /// through, so a Python traceback reaches the operator.
    pub fn spawn(
        name: &str,
        program: &str,
        args: &[String],
        cwd: &Path,
        timeout: Duration,
    ) -> Result<Self, String> {
        let mut child = Command::new(program)
            .args(args)
            .current_dir(cwd)
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| format!("{name}: cannot start {program}: {error}"))?;
        let stdin = child.stdin.take();
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| format!("{name}: no standard output"))?;
        let (sender, replies) = mpsc::channel();
        std::thread::spawn(move || {
            let mut lines = BufReader::new(stdout).lines();
            loop {
                match lines.next() {
                    Some(Ok(line)) => {
                        if sender.send(Ok(line)).is_err() {
                            return;
                        }
                    }
                    Some(Err(error)) => {
                        let _ = sender.send(Err(format!("read failed: {error}")));
                        return;
                    }
                    None => {
                        let _ = sender.send(Err("the driver closed its output".to_string()));
                        return;
                    }
                }
            }
        });
        Ok(Self {
            name: name.to_string(),
            child,
            stdin,
            replies,
            timeout,
            requests: 0,
        })
    }

    /// Send one request line and read one reply object; a reply with `ok` other than `true` is an
    /// error carrying the driver's message.
    pub fn request(&mut self, line: &str) -> Result<Json, String> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| format!("{}: already closed", self.name))?;
        stdin
            .write_all(line.as_bytes())
            .and_then(|()| stdin.write_all(b"\n"))
            .and_then(|()| stdin.flush())
            .map_err(|error| format!("{}: write failed: {error}", self.name))?;
        let text = match self.replies.recv_timeout(self.timeout) {
            Ok(Ok(text)) => text,
            Ok(Err(error)) => return Err(format!("{}: {error}", self.name)),
            Err(RecvTimeoutError::Timeout) => {
                return Err(format!(
                    "{}: no reply within {} s",
                    self.name,
                    self.timeout.as_secs()
                ))
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(format!("{}: the reader stopped", self.name))
            }
        };
        let reply =
            parse_json(&text).map_err(|error| format!("{}: bad reply: {error}", self.name))?;
        self.requests += 1;
        match resolve(&reply, "ok") {
            Some(Json::Bool(true)) => Ok(reply),
            _ => Err(format!(
                "{}: the driver refused a request: {}",
                self.name,
                match resolve(&reply, "error") {
                    Some(Json::Str(message)) => message.clone(),
                    _ => text,
                }
            )),
        }
    }

    pub fn hello(&mut self) -> Result<Json, String> {
        self.request("{\"op\":\"hello\"}")
    }

    pub fn reset(&mut self) -> Result<(), String> {
        self.request("{\"op\":\"reset\"}").map(|_| ())
    }

    /// The number of documents the driver holds.
    pub fn size(&mut self) -> Result<i64, String> {
        match resolve(&self.request("{\"op\":\"size\"}")?, "documents") {
            Some(Json::Int(count)) => Ok(*count),
            other => Err(format!("{}: size is {other:?}", self.name)),
        }
    }

    /// Replace the driver's index by `docs`, in batches that keep a request line small.
    pub fn load(&mut self, docs: &[DocRef<'_>]) -> Result<(), String> {
        self.reset()?;
        for batch in docs.chunks(200) {
            self.request(&upsert_many_line(batch))?;
        }
        Ok(())
    }

    pub fn search(
        &mut self,
        query: &str,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Reply, String> {
        let reply = self.request(&search_line(query, embedding, limit))?;
        let Some(Json::Arr(items)) = resolve(&reply, "hits") else {
            return Err(format!("{}: the reply has no hits list", self.name));
        };
        let mut hits = Vec::with_capacity(items.len());
        for item in items {
            match resolve(item, "id") {
                Some(Json::Str(id)) => hits.push(id.clone()),
                other => return Err(format!("{}: a hit has id {other:?}", self.name)),
            }
        }
        let reranked = match resolve(&reply, "reranked") {
            Some(Json::Bool(flag)) => Some(*flag),
            _ => None,
        };
        let rerank_seconds = match resolve(&reply, "rerank_seconds") {
            Some(Json::Float(seconds)) => *seconds,
            Some(Json::Int(seconds)) => *seconds as f64,
            _ => 0.0,
        };
        Ok(Reply {
            hits,
            reranked,
            rerank_seconds,
        })
    }

    /// Ask the driver to end and wait for it, killing it if it does not.
    pub fn close(mut self) {
        let _ = self.request("{\"op\":\"quit\"}");
        self.stdin = None;
        for _ in 0..50 {
            if matches!(self.child.try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        // A bridge dropped on an error path must not leave a Python process behind.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reader_reads_replies_and_pool_lines() {
        let reply = parse_json(
            r#"{"ok":true,"hits":[{"id":"a@1","score":0.5},{"id":"b@2","score":1e-3}],"n":-7,"s":"x\"\u00e9\ud83d\ude00\n","t":null,"f":false}"#,
        )
        .unwrap();
        assert_eq!(resolve(&reply, "n"), Some(&Json::Int(-7)));
        assert_eq!(
            resolve(&reply, "s"),
            Some(&Json::Str("x\"\u{e9}\u{1f600}\n".to_string()))
        );
        let Some(Json::Arr(hits)) = resolve(&reply, "hits") else {
            panic!("hits");
        };
        assert_eq!(hits.len(), 2);
        assert_eq!(resolve(&hits[1], "score"), Some(&Json::Float(0.001)));
        assert_eq!(resolve(&reply, "t"), Some(&Json::Null));
    }

    #[test]
    fn the_reader_refuses_what_is_not_json() {
        for bad in [
            "",
            "{",
            "{\"a\":}",
            "[1,]",
            "{\"a\":1,}",
            "\"abc",
            "tru",
            "{\"a\":1} x",
            "\"\\ud83d\"",
            "01x",
            "\"a\u{1}b\"",
            "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]",
        ] {
            assert!(parse_json(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn requests_are_one_line_json_that_reads_back() {
        let docs = [DocRef {
            id: "cap-0001@2",
            text: "He said \"hi\"\n",
            embedding: &[0.5, -0.25, 1.0],
        }];
        let line = upsert_many_line(&docs);
        assert!(!line.contains('\n'));
        let parsed = parse_json(&line).unwrap();
        let Some(Json::Arr(items)) = resolve(&parsed, "docs") else {
            panic!("docs");
        };
        assert_eq!(
            resolve(&items[0], "id"),
            Some(&Json::Str("cap-0001@2".into()))
        );
        assert_eq!(
            resolve(&items[0], "text"),
            Some(&Json::Str("He said \"hi\"\n".into()))
        );
        let Some(Json::Arr(vector)) = resolve(&items[0], "embedding") else {
            panic!("embedding");
        };
        assert_eq!(vector.len(), 3);
        assert_eq!(vector[1], Json::Float(-0.25));
        let search = parse_json(&search_line("q?", &[1.0, 2.0], 10)).unwrap();
        assert_eq!(resolve(&search, "limit"), Some(&Json::Int(10)));
    }

    #[test]
    fn a_driver_that_exits_or_stays_silent_is_an_error_not_a_hang() {
        let here = std::env::current_dir().unwrap();
        let mut gone = Bridge::spawn(
            "gone",
            "sh",
            &["-c".to_string(), "exit 0".to_string()],
            &here,
            Duration::from_secs(5),
        )
        .unwrap();
        // Either the write finds the pipe closed or the reader finds the end of the output.
        let error = gone.hello().unwrap_err();
        assert!(
            error.contains("closed") || error.contains("write failed"),
            "{error}"
        );
        let mut silent = Bridge::spawn(
            "silent",
            "sh",
            &["-c".to_string(), "sleep 30".to_string()],
            &here,
            Duration::from_millis(300),
        )
        .unwrap();
        assert!(silent.hello().unwrap_err().contains("no reply"));
        let mut echo = Bridge::spawn(
            "refusing",
            "sh",
            &[
                "-c".to_string(),
                "while read line; do echo '{\"ok\":false,\"error\":\"nope\"}'; done".to_string(),
            ],
            &here,
            Duration::from_secs(5),
        )
        .unwrap();
        assert!(echo.hello().unwrap_err().contains("nope"));
        assert!(Bridge::spawn(
            "missing",
            "no-such-program-q003",
            &[],
            &here,
            Duration::from_secs(1)
        )
        .is_err());
    }
}
