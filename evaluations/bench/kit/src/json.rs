//! A small JSON value with a serializer and a parser. Evidence records are
//! built from [`Value`]; the parser exists so tests (and `ptr-evalkit
//! validate`) can read a record back instead of trusting the writer.

use std::fmt::Write as _;

/// A JSON value. Objects keep insertion order, as Python dicts do.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `null`
    Null,
    /// `true` / `false`
    Bool(bool),
    /// An integer (written without a fraction).
    Int(i64),
    /// A finite float; a non-finite float serializes as `null`.
    Float(f64),
    /// A string.
    Str(String),
    /// An array.
    Arr(Vec<Value>),
    /// An object in insertion order.
    Obj(Vec<(String, Value)>),
}

impl Value {
    /// An object from `(key, value)` pairs.
    pub fn obj<K: Into<String>>(pairs: impl IntoIterator<Item = (K, Value)>) -> Self {
        Self::Obj(pairs.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    /// A string value.
    pub fn str(s: impl Into<String>) -> Self {
        Self::Str(s.into())
    }

    /// `Some(s)` as a string, otherwise `null`.
    pub fn opt_str(s: Option<impl Into<String>>) -> Self {
        s.map_or(Self::Null, |s| Self::Str(s.into()))
    }

    /// Field lookup on an object.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Self::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The string, if this is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Serialize with 2-space indentation (Python's `json.dump(indent=2)`).
    pub fn to_pretty(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, 0);
        out
    }

    fn write(&self, out: &mut String, depth: usize) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Self::Int(i) => {
                let _ = write!(out, "{i}");
            }
            Self::Float(f) if f.is_finite() => {
                let _ = write!(out, "{f:?}");
            }
            Self::Float(_) => out.push_str("null"),
            Self::Str(s) => escape_into(out, s),
            Self::Arr(items) if items.is_empty() => out.push_str("[]"),
            Self::Arr(items) => {
                out.push_str("[\n");
                for (i, item) in items.iter().enumerate() {
                    indent(out, depth + 1);
                    item.write(out, depth + 1);
                    out.push_str(if i + 1 < items.len() { ",\n" } else { "\n" });
                }
                indent(out, depth);
                out.push(']');
            }
            Self::Obj(pairs) if pairs.is_empty() => out.push_str("{}"),
            Self::Obj(pairs) => {
                out.push_str("{\n");
                for (i, (k, v)) in pairs.iter().enumerate() {
                    indent(out, depth + 1);
                    escape_into(out, k);
                    out.push_str(": ");
                    v.write(out, depth + 1);
                    out.push_str(if i + 1 < pairs.len() { ",\n" } else { "\n" });
                }
                indent(out, depth);
                out.push('}');
            }
        }
    }

    /// Parse a complete JSON document.
    pub fn parse(text: &str) -> Result<Value, String> {
        let mut parser = Parser {
            bytes: text.as_bytes(),
            pos: 0,
        };
        let value = parser.value(0)?;
        parser.skip_ws();
        if parser.pos != parser.bytes.len() {
            return Err(format!("trailing characters at byte {}", parser.pos));
        }
        Ok(value)
    }
}

fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("  ");
    }
}

fn escape_into(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

const MAX_DEPTH: usize = 64;

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn skip_ws(&mut self) {
        while matches!(self.bytes.get(self.pos), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.pos += 1;
        }
    }

    fn eat(&mut self, literal: &str) -> bool {
        if self.bytes[self.pos..].starts_with(literal.as_bytes()) {
            self.pos += literal.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, String> {
        if depth > MAX_DEPTH {
            return Err("nesting too deep".into());
        }
        self.skip_ws();
        match self.bytes.get(self.pos) {
            None => Err("unexpected end".into()),
            Some(b'n') if self.eat("null") => Ok(Value::Null),
            Some(b't') if self.eat("true") => Ok(Value::Bool(true)),
            Some(b'f') if self.eat("false") => Ok(Value::Bool(false)),
            Some(b'"') => self.string().map(Value::Str),
            Some(b'[') => {
                self.pos += 1;
                let mut items = Vec::new();
                self.skip_ws();
                if self.bytes.get(self.pos) == Some(&b']') {
                    self.pos += 1;
                    return Ok(Value::Arr(items));
                }
                loop {
                    items.push(self.value(depth + 1)?);
                    self.skip_ws();
                    match self.bytes.get(self.pos) {
                        Some(b',') => self.pos += 1,
                        Some(b']') => {
                            self.pos += 1;
                            return Ok(Value::Arr(items));
                        }
                        _ => return Err(format!("expected , or ] at byte {}", self.pos)),
                    }
                }
            }
            Some(b'{') => {
                self.pos += 1;
                let mut pairs = Vec::new();
                self.skip_ws();
                if self.bytes.get(self.pos) == Some(&b'}') {
                    self.pos += 1;
                    return Ok(Value::Obj(pairs));
                }
                loop {
                    self.skip_ws();
                    if self.bytes.get(self.pos) != Some(&b'"') {
                        return Err(format!("expected key at byte {}", self.pos));
                    }
                    let key = self.string()?;
                    self.skip_ws();
                    if self.bytes.get(self.pos) != Some(&b':') {
                        return Err(format!("expected : at byte {}", self.pos));
                    }
                    self.pos += 1;
                    pairs.push((key, self.value(depth + 1)?));
                    self.skip_ws();
                    match self.bytes.get(self.pos) {
                        Some(b',') => self.pos += 1,
                        Some(b'}') => {
                            self.pos += 1;
                            return Ok(Value::Obj(pairs));
                        }
                        _ => return Err(format!("expected , or }} at byte {}", self.pos)),
                    }
                }
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(format!("unexpected character at byte {}", self.pos)),
        }
    }

    fn number(&mut self) -> Result<Value, String> {
        let start = self.pos;
        while matches!(
            self.bytes.get(self.pos),
            Some(b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
        ) {
            self.pos += 1;
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|e| e.to_string())?;
        if let Ok(i) = text.parse::<i64>() {
            return Ok(Value::Int(i));
        }
        text.parse::<f64>()
            .map(Value::Float)
            .map_err(|_| format!("bad number {text:?}"))
    }

    fn string(&mut self) -> Result<String, String> {
        self.pos += 1; // opening quote
        let mut out = String::new();
        loop {
            let rest = &self.bytes[self.pos..];
            let Some(&byte) = rest.first() else {
                return Err("unterminated string".into());
            };
            match byte {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    let escape = *rest.get(1).ok_or("dangling backslash")?;
                    self.pos += 2;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'u' => {
                            let hex = self
                                .bytes
                                .get(self.pos..self.pos + 4)
                                .and_then(|h| std::str::from_utf8(h).ok())
                                .ok_or("short \\u escape")?;
                            let code = u32::from_str_radix(hex, 16).map_err(|e| e.to_string())?;
                            self.pos += 4;
                            out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                        }
                        other => return Err(format!("bad escape \\{}", other as char)),
                    }
                }
                _ => {
                    let width = utf8_width(byte);
                    let chunk = rest.get(..width).ok_or("truncated utf-8")?;
                    out.push_str(std::str::from_utf8(chunk).map_err(|e| e.to_string())?);
                    self.pos += width;
                }
            }
        }
    }
}

fn utf8_width(first: u8) -> usize {
    match first {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_scalars_and_matches_python_indent_two() {
        let v = Value::obj([
            ("a", Value::Int(1)),
            ("b", Value::Arr(vec![Value::Bool(true), Value::Null])),
            ("c", Value::obj::<&str>([])),
            ("d", Value::Arr(vec![])),
        ]);
        assert_eq!(
            v.to_pretty(),
            "{\n  \"a\": 1,\n  \"b\": [\n    true,\n    null\n  ],\n  \"c\": {},\n  \"d\": []\n}"
        );
    }

    #[test]
    fn floats_print_round_trip_and_non_finite_become_null() {
        assert_eq!(Value::Float(1.5).to_pretty(), "1.5");
        assert_eq!(Value::Float(1.0).to_pretty(), "1.0");
        assert_eq!(Value::Float(f64::NAN).to_pretty(), "null");
        assert_eq!(Value::Float(f64::INFINITY).to_pretty(), "null");
        assert_eq!(
            Value::parse(&Value::Float(0.1).to_pretty()).unwrap(),
            Value::Float(0.1)
        );
    }

    #[test]
    fn strings_escape_and_round_trip() {
        let nasty = "quote\" back\\ nl\n tab\t ctl\u{1} snow\u{2603} astral\u{1F600}";
        let text = Value::str(nasty).to_pretty();
        assert!(text.contains("\\u0001") && text.contains("\\n") && text.contains("\\\""));
        assert_eq!(Value::parse(&text).unwrap(), Value::str(nasty));
    }

    #[test]
    fn parse_round_trips_nested_documents() {
        let v = Value::obj([
            ("n", Value::Int(-7)),
            ("f", Value::Float(2.5e-3)),
            (
                "list",
                Value::Arr(vec![
                    Value::Int(1),
                    Value::str("x"),
                    Value::obj([("k", Value::Null)]),
                ]),
            ),
        ]);
        assert_eq!(Value::parse(&v.to_pretty()).unwrap(), v);
        assert_eq!(v.get("n"), Some(&Value::Int(-7)));
        assert_eq!(v.get("missing"), None);
        assert_eq!(Value::Int(1).get("n"), None);
    }

    #[test]
    fn parse_accepts_unicode_escapes_and_whitespace() {
        assert_eq!(
            Value::parse(" \t\n{ \"a\" : \"\\u00e9\" } ")
                .unwrap()
                .get("a"),
            Some(&Value::str("é"))
        );
    }

    #[test]
    fn parse_rejects_malformed_input() {
        for bad in [
            "",
            "{",
            "[1,",
            "{\"a\" 1}",
            "{\"a\":1,}",
            "[1 2]",
            "\"abc",
            "nul",
            "1 2",
            "{1:2}",
            "\"\\x\"",
            "\"\\u12\"",
            "-",
            "[1,]",
        ] {
            assert!(Value::parse(bad).is_err(), "{bad:?} should not parse");
        }
    }

    #[test]
    fn parse_bounds_nesting_depth() {
        let deep = "[".repeat(200) + &"]".repeat(200);
        assert_eq!(Value::parse(&deep), Err("nesting too deep".to_string()));
        let ok = "[".repeat(10) + &"]".repeat(10);
        assert!(Value::parse(&ok).is_ok());
    }

    #[test]
    fn accessors() {
        assert_eq!(Value::str("a").as_str(), Some("a"));
        assert_eq!(Value::Int(1).as_str(), None);
        assert_eq!(Value::opt_str(Some("x")), Value::str("x"));
        assert_eq!(Value::opt_str(None::<String>), Value::Null);
    }
}
