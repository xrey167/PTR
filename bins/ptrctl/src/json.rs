//! A minimal, bounded JSON parser, just enough for the span files `ptrctl trace` reads.
//!
//! Numbers are kept as their source text (only non-negative integers are readable through
//! [`Json::as_u64`]); nesting is limited to [`MAX_DEPTH`].

use std::fmt;

/// Deepest nesting accepted.
const MAX_DEPTH: usize = 32;

#[derive(Debug, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    Number(String),
    Str(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

#[derive(Debug, PartialEq)]
pub(crate) struct ParseError {
    offset: usize,
    what: &'static str,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid JSON at byte {}: {}", self.offset, self.what)
    }
}

impl Json {
    pub(crate) fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(text) => Some(text),
            _ => None,
        }
    }

    pub(crate) fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(items) => Some(items),
            _ => None,
        }
    }

    pub(crate) fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Number(text) => text.parse().ok(),
            _ => None,
        }
    }
}

pub(crate) fn parse(text: &str) -> Result<Json, ParseError> {
    let mut parser = Parser {
        bytes: text.as_bytes(),
        at: 0,
    };
    let value = parser.value(0)?;
    parser.skip_space();
    if parser.at != parser.bytes.len() {
        return Err(parser.error("trailing characters"));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn error(&self, what: &'static str) -> ParseError {
        ParseError {
            offset: self.at,
            what,
        }
    }

    fn skip_space(&mut self) {
        while matches!(self.bytes.get(self.at), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.at += 1;
        }
    }

    fn eat(&mut self, byte: u8) -> bool {
        if self.bytes.get(self.at) == Some(&byte) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn literal(&mut self, word: &'static str, value: Json) -> Result<Json, ParseError> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(value)
        } else {
            Err(self.error("unknown literal"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, ParseError> {
        if depth > MAX_DEPTH {
            return Err(self.error("nesting too deep"));
        }
        self.skip_space();
        match self.bytes.get(self.at) {
            None => Err(self.error("unexpected end")),
            Some(b'n') => self.literal("null", Json::Null),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'"') => self.string().map(Json::Str),
            Some(b'[') => self.array(depth),
            Some(b'{') => self.object(depth),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.error("unexpected character")),
        }
    }

    fn number(&mut self) -> Result<Json, ParseError> {
        let start = self.at;
        while matches!(
            self.bytes.get(self.at),
            Some(b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
        ) {
            self.at += 1;
        }
        let text = std::str::from_utf8(&self.bytes[start..self.at])
            .map_err(|_| self.error("invalid number"))?;
        Ok(Json::Number(text.to_owned()))
    }

    fn hex4(&mut self) -> Result<u32, ParseError> {
        let digits = self
            .bytes
            .get(self.at..self.at + 4)
            .and_then(|d| std::str::from_utf8(d).ok())
            .and_then(|d| u32::from_str_radix(d, 16).ok())
            .ok_or_else(|| self.error("bad \\u escape"))?;
        self.at += 4;
        Ok(digits)
    }

    fn string(&mut self) -> Result<String, ParseError> {
        self.at += 1; // opening quote
        let mut out = String::new();
        loop {
            let rest = &self.bytes[self.at..];
            let Some(&byte) = rest.first() else {
                return Err(self.error("unterminated string"));
            };
            match byte {
                b'"' => {
                    self.at += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.at += 1;
                    let Some(&escape) = self.bytes.get(self.at) else {
                        return Err(self.error("unterminated escape"));
                    };
                    self.at += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let first = self.hex4()?;
                            let code = if (0xd800..0xdc00).contains(&first)
                                && self.bytes[self.at..].starts_with(b"\\u")
                            {
                                self.at += 2;
                                let second = self.hex4()?;
                                if (0xdc00..0xe000).contains(&second) {
                                    0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00)
                                } else {
                                    return Err(self.error("unpaired surrogate"));
                                }
                            } else {
                                first
                            };
                            out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                        }
                        _ => return Err(self.error("unknown escape")),
                    }
                }
                0x00..=0x1f => return Err(self.error("control character in string")),
                _ => {
                    // The input is a &str, so a run up to the next quote or backslash is UTF-8.
                    let run = rest
                        .iter()
                        .position(|&b| b == b'"' || b == b'\\' || b < 0x20)
                        .unwrap_or(rest.len());
                    let text = std::str::from_utf8(&rest[..run])
                        .map_err(|_| self.error("invalid UTF-8"))?;
                    out.push_str(text);
                    self.at += run;
                }
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, ParseError> {
        self.at += 1;
        let mut items = Vec::new();
        self.skip_space();
        if self.eat(b']') {
            return Ok(Json::Array(items));
        }
        loop {
            items.push(self.value(depth + 1)?);
            self.skip_space();
            if self.eat(b',') {
                continue;
            }
            if self.eat(b']') {
                return Ok(Json::Array(items));
            }
            return Err(self.error("expected ',' or ']'"));
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, ParseError> {
        self.at += 1;
        let mut fields = Vec::new();
        self.skip_space();
        if self.eat(b'}') {
            return Ok(Json::Object(fields));
        }
        loop {
            self.skip_space();
            if self.bytes.get(self.at) != Some(&b'"') {
                return Err(self.error("expected a string key"));
            }
            let key = self.string()?;
            self.skip_space();
            if !self.eat(b':') {
                return Err(self.error("expected ':'"));
            }
            fields.push((key, self.value(depth + 1)?));
            self.skip_space();
            if self.eat(b',') {
                continue;
            }
            if self.eat(b'}') {
                return Ok(Json::Object(fields));
            }
            return Err(self.error("expected ',' or '}'"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_values_and_escapes() {
        let value = parse(r#"{"a":[1,"x\né😀",null,true],"b":{"c":"7"}}"#).expect("parses");
        let items = value.get("a").and_then(Json::as_array).expect("array");
        assert_eq!(items[0].as_u64(), Some(1));
        assert_eq!(items[1].as_str(), Some("x\n\u{e9}\u{1f600}"));
        assert_eq!(
            value
                .get("b")
                .and_then(|b| b.get("c"))
                .and_then(Json::as_str),
            Some("7")
        );
    }

    #[test]
    fn rejects_malformed_and_too_deep_input() {
        for bad in [
            "",
            "{",
            "[1,]",
            r#"{"a" 1}"#,
            r#""\x""#,
            "tru",
            "1 2",
            "\"a\u{1}\"",
        ] {
            assert!(parse(bad).is_err(), "{bad:?} must fail");
        }
        let deep = "[".repeat(MAX_DEPTH + 2);
        assert!(parse(&deep).is_err());
    }
}
