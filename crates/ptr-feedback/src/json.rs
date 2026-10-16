//! A minimal JSON reader and string writer for the trajectory record schema only.

use std::fmt::Write as _;

/// Deepest nesting the reader accepts.
const MAX_DEPTH: usize = 6;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum JsonError {
    Syntax { offset: usize },
    TooDeep,
}

/// Parse exactly one JSON value; trailing non-space input is a syntax error.
pub(crate) fn parse(input: &str) -> Result<Json, JsonError> {
    let mut parser = Parser {
        bytes: input.as_bytes(),
        input,
        pos: 0,
    };
    parser.skip_space();
    let value = parser.value(0)?;
    parser.skip_space();
    if parser.pos != parser.bytes.len() {
        return Err(parser.error());
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    input: &'a str,
    pos: usize,
}

impl Parser<'_> {
    fn error(&self) -> JsonError {
        JsonError::Syntax { offset: self.pos }
    }

    fn skip_space(&mut self) {
        while matches!(self.bytes.get(self.pos), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.pos += 1;
        }
    }

    fn eat(&mut self, byte: u8) -> Result<(), JsonError> {
        if self.bytes.get(self.pos) == Some(&byte) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.error())
        }
    }

    fn literal(&mut self, word: &str, value: Json) -> Result<Json, JsonError> {
        if self.input[self.pos..].starts_with(word) {
            self.pos += word.len();
            Ok(value)
        } else {
            Err(self.error())
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, JsonError> {
        match self.bytes.get(self.pos) {
            Some(b'n') => self.literal("null", Json::Null),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'"') => self.string().map(Json::Str),
            Some(b'[') => self.array(depth),
            Some(b'{') => self.object(depth),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(self.error()),
        }
    }

    fn number(&mut self) -> Result<Json, JsonError> {
        let start = self.pos;
        while matches!(
            self.bytes.get(self.pos),
            Some(b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
        ) {
            self.pos += 1;
        }
        match self.input[start..self.pos].parse::<f64>() {
            Ok(number) if number.is_finite() => Ok(Json::Num(number)),
            _ => {
                self.pos = start;
                Err(self.error())
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, JsonError> {
        let digits = self
            .input
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| self.error())?;
        let value = u32::from_str_radix(digits, 16).map_err(|_| self.error())?;
        self.pos += 4;
        Ok(value)
    }

    fn string(&mut self) -> Result<String, JsonError> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            let rest = &self.input[self.pos..];
            let Some(c) = rest.chars().next() else {
                return Err(self.error());
            };
            self.pos += c.len_utf8();
            match c {
                '"' => return Ok(out),
                '\\' => {
                    let Some(&escape) = self.bytes.get(self.pos) else {
                        return Err(self.error());
                    };
                    self.pos += 1;
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
                                self.eat(b'\\')?;
                                self.eat(b'u')?;
                                let second = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&second) {
                                    return Err(self.error());
                                }
                                0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                            } else {
                                first
                            };
                            out.push(char::from_u32(code).ok_or_else(|| self.error())?);
                        }
                        _ => return Err(self.error()),
                    }
                }
                c if (c as u32) < 0x20 => return Err(self.error()),
                c => out.push(c),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, JsonError> {
        if depth >= MAX_DEPTH {
            return Err(JsonError::TooDeep);
        }
        self.eat(b'[')?;
        self.skip_space();
        let mut items = Vec::new();
        if self.bytes.get(self.pos) == Some(&b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            self.skip_space();
            items.push(self.value(depth + 1)?);
            self.skip_space();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(self.error()),
            }
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, JsonError> {
        if depth >= MAX_DEPTH {
            return Err(JsonError::TooDeep);
        }
        self.eat(b'{')?;
        self.skip_space();
        let mut fields = Vec::new();
        if self.bytes.get(self.pos) == Some(&b'}') {
            self.pos += 1;
            return Ok(Json::Obj(fields));
        }
        loop {
            self.skip_space();
            let key = self.string()?;
            self.skip_space();
            self.eat(b':')?;
            self.skip_space();
            fields.push((key, self.value(depth + 1)?));
            self.skip_space();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Obj(fields));
                }
                _ => return Err(self.error()),
            }
        }
    }
}

/// Append `text` as a JSON string literal.
pub(crate) fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scalars_arrays_and_objects() {
        let value =
            parse(r#" {"a": [1, -2.5e1, true, null], "b": "x\n\u00e9\ud83d\ude00"} "#).unwrap();
        let Json::Obj(fields) = value else {
            panic!("object")
        };
        assert_eq!(fields[0].0, "a");
        assert_eq!(
            fields[0].1,
            Json::Arr(vec![
                Json::Num(1.0),
                Json::Num(-25.0),
                Json::Bool(true),
                Json::Null
            ])
        );
        assert_eq!(fields[1].1, Json::Str("x\né😀".into()));
    }

    #[test]
    fn rejects_malformed_input() {
        for bad in [
            "",
            "{",
            "[1,]",
            r#"{"a" 1}"#,
            "\"abc",
            "tru",
            "1 2",
            "\"\\ud800\"",
            "\"\u{1}\"",
            "-",
        ] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
        assert_eq!(parse("[[[[[[[[1]]]]]]]]"), Err(JsonError::TooDeep));
    }

    #[test]
    fn written_strings_round_trip() {
        let original = "quote \" back \\ nl \n tab \t bell \u{7} é 😀";
        let mut out = String::new();
        write_string(&mut out, original);
        assert_eq!(parse(&out).unwrap(), Json::Str(original.into()));
    }
}
