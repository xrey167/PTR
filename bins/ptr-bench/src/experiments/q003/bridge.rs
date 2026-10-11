//! A JSON-lines bridge to a child process, and the small JSON reader its replies need.
//!
//! The reference arm (`research/baselines/rag_reference/q003_driver.py`), the strong arm
//! (`research/baselines/strong_retrieval/driver.py`) and the BM25-only helper
//! (`experiments/retrieval/Q003-v3-postgres-hybrid/bm25_driver.py`) speak one protocol: one JSON
//! object per line in, one per line out, `{"ok": true, ...}` or `{"ok": false, "error": ...}`.
//! None of them is code under test, so every failure of the bridge (a process that does not start,
//! a reply that does not parse, a refusal, a timeout, an end of file) is the harness's: the caller
//! turns it into exit status 2.
//!
//! The workspace has no JSON parser the harness may depend on, so [`parse_json`] reads the replies
//! (and the pool file) into the emitter's own [`Json`] value.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use super::watch::{monitor, Stop, StopReason};
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

/// What the threads around a driver tell the caller of [`Bridge::request`].
enum Event {
    Line(String),
    /// The driver closed its output.
    Closed,
    ReadFailed(String),
    WriteFailed(String),
}

/// How long a driver gets, after its output closed, to show its exit status.
const EXIT_STATUS_WAIT: Duration = Duration::from_secs(5);
/// How long a driver that was told to quit gets to end before it is killed.
const QUIT_GRACE: Duration = Duration::from_secs(120);

/// A running driver process.
///
/// The framing is one request per line in, one reply per line out. Three ways this can stall
/// are bounded here: the line is written and flushed by a thread of its own, so a driver
/// that does not read its input (a full pipe) is bounded by the same per-request timeout as one
/// that does not answer; a driver that exits, closes its output or refuses is a typed
/// [`Stop`]; and a request that times out kills the driver, so a late reply can never be read as
/// the answer to the next request.
pub struct Bridge {
    name: String,
    child: Child,
    lines: Option<mpsc::Sender<String>>,
    events: Receiver<Event>,
    timeout: Duration,
    /// Requests sent and answered.
    pub requests: u64,
}

/// The `op` of a request line, for the heartbeat (the requests are built by this module).
fn op_of(line: &str) -> &str {
    line.split("\"op\":\"")
        .nth(1)
        .and_then(|rest| rest.split('"').next())
        .unwrap_or("?")
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
    ) -> Result<Self, Stop> {
        let mut child = Command::new(program)
            .args(args)
            .current_dir(cwd)
            .env("PYTHONDONTWRITEBYTECODE", "1")
            // Under the runner no Python is unbuffered; an operator's own setting must not make a
            // run (or a mutation check) differ from it, so a driver that does not flush fails here too.
            .env_remove("PYTHONUNBUFFERED")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| {
                Stop::new(
                    StopReason::SpawnFailed,
                    format!("{name}: cannot start {program}: {error}"),
                )
            })?;
        let (Some(mut stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(Stop::new(
                StopReason::SpawnFailed,
                format!("{name}: no standard input or output"),
            ));
        };
        let (events_in, events) = mpsc::channel();
        let reader = events_in.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let event = match line {
                    Ok(line) => Event::Line(line),
                    Err(error) => {
                        let _ = reader.send(Event::ReadFailed(error.to_string()));
                        return;
                    }
                };
                if reader.send(event).is_err() {
                    return;
                }
            }
            let _ = reader.send(Event::Closed);
        });
        let (lines, queue) = mpsc::channel::<String>();
        std::thread::spawn(move || {
            // The thread ends, and with it the driver's input, when the bridge drops its sender.
            for line in queue {
                let written = stdin
                    .write_all(line.as_bytes())
                    .and_then(|()| stdin.write_all(b"\n"))
                    .and_then(|()| stdin.flush());
                if let Err(error) = written {
                    let _ = events_in.send(Event::WriteFailed(error.to_string()));
                    return;
                }
            }
        });
        monitor().child_started(name, child.id());
        Ok(Self {
            name: name.to_string(),
            child,
            lines: Some(lines),
            events,
            timeout,
            requests: 0,
        })
    }

    fn stop(&self, reason: StopReason, detail: impl std::fmt::Display) -> Stop {
        Stop::new(reason, format!("{}: {detail}", self.name))
    }

    /// The driver's exit status, given a short while to show: `exited with status N`, or that it
    /// is still running (then it is killed).
    fn exit_detail(&mut self) -> String {
        let began = Instant::now();
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => return format!("the driver exited with {status}"),
                Ok(None) if began.elapsed() < EXIT_STATUS_WAIT => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Ok(None) => {
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    return "the driver closed its output but kept running; it was killed"
                        .to_string();
                }
                Err(error) => return format!("the driver's status is unknown ({error})"),
            }
        }
    }

    /// Kill the driver; nothing it says later can be taken for a reply.
    fn kill(&mut self) {
        self.lines = None;
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    /// Send one request line and read one reply object; a reply with `ok` other than `true` is an
    /// error carrying the driver's message. The whole exchange (write, flush, reply) is bounded
    /// by the bridge's timeout.
    pub fn request(&mut self, line: &str) -> Result<Json, Stop> {
        if line.contains('\n') || line.contains('\r') {
            return Err(self.stop(
                StopReason::BadRequest,
                "a request must be one line (no line break inside it)",
            ));
        }
        monitor().request_begin(&self.name, op_of(line));
        let outcome = self.exchange(line);
        monitor().request_end(&self.name);
        if outcome.is_err() {
            self.kill();
        }
        outcome
    }

    fn exchange(&mut self, line: &str) -> Result<Json, Stop> {
        let Some(lines) = self.lines.as_ref() else {
            return Err(self.stop(StopReason::WriteFailed, "already closed or stopped"));
        };
        // A send that fails means the writer thread has ended; its event says why.
        let _ = lines.send(line.to_string());
        let text = match self.events.recv_timeout(self.timeout) {
            Ok(Event::Line(text)) => text,
            Ok(Event::Closed) => {
                let detail = self.exit_detail();
                return Err(self.stop(StopReason::DriverExited, detail));
            }
            Ok(Event::ReadFailed(error)) => {
                return Err(self.stop(StopReason::ReaderStopped, format!("read failed: {error}")))
            }
            Ok(Event::WriteFailed(error)) => {
                let detail = self.exit_detail();
                return Err(self.stop(
                    StopReason::WriteFailed,
                    format!("write failed ({error}); {detail}"),
                ));
            }
            Err(RecvTimeoutError::Timeout) => {
                return Err(self.stop(
                    StopReason::RequestTimeout,
                    format!(
                        "no reply to {} within {} s; the driver was killed",
                        op_of(line),
                        self.timeout.as_secs()
                    ),
                ))
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(self.stop(StopReason::ReaderStopped, "the reader stopped"))
            }
        };
        let reply = parse_json(&text)
            .map_err(|error| self.stop(StopReason::BadReply, format!("bad reply: {error}")))?;
        self.requests += 1;
        match resolve(&reply, "ok") {
            Some(Json::Bool(true)) => Ok(reply),
            _ => Err(self.stop(
                StopReason::DriverRefused,
                format!(
                    "the driver refused a request: {}",
                    match resolve(&reply, "error") {
                        Some(Json::Str(message)) => message.clone(),
                        _ => text,
                    }
                ),
            )),
        }
    }

    pub fn hello(&mut self) -> Result<Json, Stop> {
        self.request("{\"op\":\"hello\"}")
    }

    pub fn reset(&mut self) -> Result<(), Stop> {
        self.request("{\"op\":\"reset\"}").map(|_| ())
    }

    /// The number of documents the driver holds.
    pub fn size(&mut self) -> Result<i64, Stop> {
        match resolve(&self.request("{\"op\":\"size\"}")?, "documents") {
            Some(Json::Int(count)) => Ok(*count),
            other => Err(self.stop(StopReason::BadReply, format!("size is {other:?}"))),
        }
    }

    /// Replace the driver's index by `docs`, in batches that keep a request line small.
    pub fn load(&mut self, docs: &[DocRef<'_>]) -> Result<(), Stop> {
        self.reset()?;
        for batch in docs.chunks(200) {
            self.request(&upsert_many_line(batch))?;
        }
        Ok(())
    }

    pub fn search(&mut self, query: &str, embedding: &[f32], limit: usize) -> Result<Reply, Stop> {
        let reply = self.request(&search_line(query, embedding, limit))?;
        let Some(Json::Arr(items)) = resolve(&reply, "hits") else {
            return Err(self.stop(StopReason::BadReply, "the reply has no hits list"));
        };
        let mut hits = Vec::with_capacity(items.len());
        for item in items {
            match resolve(item, "id") {
                Some(Json::Str(id)) => hits.push(id.clone()),
                other => {
                    return Err(self.stop(StopReason::BadReply, format!("a hit has id {other:?}")))
                }
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

    /// Ask the driver to end and wait for it. A driver that answers the quit and then exits with
    /// a status other than 0, or that dies before answering, is a [`Stop`]: its failure must not
    /// pass as a finished run. One that does not end within a grace period is killed.
    pub fn close(mut self) -> Result<(), Stop> {
        self.request("{\"op\":\"quit\"}")?;
        self.lines = None;
        let began = Instant::now();
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) if status.success() => return Ok(()),
                Ok(Some(status)) => {
                    return Err(self.stop(
                        StopReason::DriverExited,
                        format!("the driver exited with {status} after quit"),
                    ))
                }
                Ok(None) if began.elapsed() < QUIT_GRACE => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Ok(None) => {
                    eprintln!(
                        "{}: {} did not end {} s after quit; it was killed",
                        super::params::BENCHMARK,
                        self.name,
                        QUIT_GRACE.as_secs()
                    );
                    return Ok(());
                }
                Err(error) => {
                    return Err(self.stop(
                        StopReason::DriverExited,
                        format!("the driver's status is unknown ({error})"),
                    ))
                }
            }
        }
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        // A bridge dropped on an error path must not leave a Python process behind.
        let _ = self.child.kill();
        let _ = self.child.wait();
        monitor().child_gone(self.child.id());
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

    fn sh(script: &str, timeout: Duration) -> Bridge {
        Bridge::spawn(
            "driver",
            "sh",
            &["-c".to_string(), script.to_string()],
            &std::env::current_dir().unwrap(),
            timeout,
        )
        .unwrap()
    }

    fn python(script: &str, timeout: Duration) -> Bridge {
        Bridge::spawn(
            "driver",
            "python3",
            &["-c".to_string(), script.to_string()],
            &std::env::current_dir().unwrap(),
            timeout,
        )
        .unwrap()
    }

    #[test]
    fn a_driver_that_exits_stays_silent_or_refuses_is_a_typed_stop_not_a_hang() {
        let mut gone = sh("exit 3", Duration::from_secs(20));
        // Either the write finds the pipe closed or the reader finds the end of the output;
        // both show the exit status.
        let stop = gone.hello().unwrap_err();
        assert!(
            matches!(
                stop.reason,
                StopReason::DriverExited | StopReason::WriteFailed
            ),
            "{stop}"
        );
        assert!(stop.detail.contains("exit status: 3"), "{stop}");
        let mut silent = sh("exec sleep 30", Duration::from_millis(300));
        let stop = silent.hello().unwrap_err();
        assert_eq!(stop.reason, StopReason::RequestTimeout, "{stop}");
        assert!(stop
            .to_string()
            .starts_with("harness stop [request-timeout]"));
        let mut refusing = sh(
            "while read line; do echo '{\"ok\":false,\"error\":\"nope\"}'; done",
            Duration::from_secs(20),
        );
        let stop = refusing.hello().unwrap_err();
        assert_eq!(stop.reason, StopReason::DriverRefused);
        assert!(stop.detail.contains("nope"));
        let mut garbled = sh(
            "while read line; do echo 'not json'; done",
            Duration::from_secs(20),
        );
        assert_eq!(garbled.hello().unwrap_err().reason, StopReason::BadReply);
        let missing = Bridge::spawn(
            "missing",
            "no-such-program-q003",
            &[],
            &std::env::current_dir().unwrap(),
            Duration::from_secs(1),
        );
        assert_eq!(missing.err().unwrap().reason, StopReason::SpawnFailed);
    }

    /// The pipe case: a driver that never reads its input while a large request (more than any
    /// pipe holds) is written. Before the writer thread, `write_all` blocked for as long as the
    /// driver lived and the per-request timeout never started; now the exchange as a whole is
    /// bounded, and the test fails by its elapsed time on the old code.
    #[test]
    fn a_driver_that_does_not_read_its_input_is_bounded_by_the_request_timeout() {
        let mut deaf = sh("exec sleep 60", Duration::from_millis(500));
        let big = format!(
            "{{\"op\":\"upsert_many\",\"pad\":\"{}\"}}",
            "x".repeat(4 << 20)
        );
        let began = Instant::now();
        let stop = deaf.request(&big).unwrap_err();
        assert_eq!(stop.reason, StopReason::RequestTimeout, "{stop}");
        assert!(
            began.elapsed() < Duration::from_secs(20),
            "{:?}",
            began.elapsed()
        );
        // The driver was killed with the stop: the bridge is spent, not left half-open.
        assert_eq!(deaf.hello().unwrap_err().reason, StopReason::WriteFailed);
    }

    /// The case Q003-v2's diagnosis suspected: a driver that answers into a block-buffered pipe
    /// and never flushes. Its reply never arrives, which from outside looks like a deadlock (the
    /// harness reads, the driver reads); the harness ends it with a typed timeout, and the
    /// flushing twin answers.
    #[test]
    fn a_driver_that_does_not_flush_its_replies_is_a_timeout_and_the_flushing_one_answers() {
        let reply =
            "import sys\nfor line in sys.stdin:\n    sys.stdout.write('{\"ok\":true}\\n')\n";
        let mut unflushed = python(reply, Duration::from_millis(1500));
        assert_eq!(
            unflushed.hello().unwrap_err().reason,
            StopReason::RequestTimeout
        );
        let mut flushed = python(
            &format!("{reply}    sys.stdout.flush()\n"),
            Duration::from_secs(60),
        );
        flushed.hello().unwrap();
        flushed.hello().unwrap();
        assert_eq!(flushed.requests, 2);
    }

    #[test]
    fn a_request_that_is_not_one_line_is_refused_before_it_is_sent() {
        let mut echo = sh(
            "while read line; do echo '{\"ok\":true}'; done",
            Duration::from_secs(20),
        );
        for bad in ["{\"op\":\"a\"}\n{\"op\":\"b\"}", "{\"op\":\"a\"}\r"] {
            assert_eq!(
                echo.request(bad).unwrap_err().reason,
                StopReason::BadRequest
            );
        }
        // The refusal sent nothing and spent nothing: the bridge still works.
        echo.hello().unwrap();
        assert_eq!(echo.requests, 1);
    }

    #[test]
    fn a_driver_that_fails_when_it_ends_is_a_stop_and_one_that_ends_well_is_not() {
        let well = sh(
            "while read line; do echo '{\"ok\":true}'; done",
            Duration::from_secs(20),
        );
        well.close().unwrap();
        let failing = sh(
            "while read line; do echo '{\"ok\":true}'; exit 7; done",
            Duration::from_secs(20),
        );
        let stop = failing.close().unwrap_err();
        assert_eq!(stop.reason, StopReason::DriverExited, "{stop}");
        assert!(stop.detail.contains("exit status: 7"), "{stop}");
        let dead = sh("exit 0", Duration::from_secs(20));
        assert!(dead.close().is_err());
    }

    #[test]
    fn the_heartbeat_names_the_request_in_flight() {
        let mut slow = sh(
            "while read line; do sleep 1; echo '{\"ok\":true}'; done",
            Duration::from_secs(60),
        );
        let began = Instant::now();
        slow.hello().unwrap();
        assert!(began.elapsed() >= Duration::from_secs(1));
        assert!(monitor().describe().contains("slowest request"));
    }
}
