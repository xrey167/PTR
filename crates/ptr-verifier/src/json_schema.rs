//! JSON Schema verification over a pinned subset, implemented in-house (no dependency).
//!
//! # The pinned subset
//!
//! A schema is a JSON object with any of these keywords, and nothing else (`title` and
//! `description` are accepted and ignored):
//!
//! | keyword | applies to | value |
//! |---------|------------|-------|
//! | `type` | any | one of `null boolean integer number string array object` |
//! | `required` | objects | array of property names |
//! | `properties` | objects | map of name to sub-schema |
//! | `additionalProperties` | objects | `false` (no other property) or `true` |
//! | `enum` | any | array of 1 to [`MAX_ENUM_MEMBERS`] JSON values (numbers compare by exact value) |
//! | `minimum`, `maximum` | numbers | a number (inclusive) |
//! | `minLength`, `maxLength` | strings | non-negative integer, counted in Unicode scalar values |
//! | `items` | arrays | a sub-schema applied to every element |
//!
//! Numbers are exact decimals, never floats: `9007199254740993` is not `9007199254740992.0`, a
//! 60-digit integer compares exactly, and `1e-400` is a positive number (strictly above
//! `"maximum": 0`), not zero. Exponents beyond [`MAX_EXPONENT`] in magnitude are a `json.parse`
//! failure (so a candidate cannot ask for an absurd magnitude). `1.0` is an `integer`
//! (draft 6 and later) and `-0` equals `0`.
//!
//! A schema with another keyword, a malformed keyword value, more than [`MAX_SCHEMA_BYTES`] bytes
//! or nesting beyond [`MAX_DEPTH`] is not guessed at: every report is `Unknown` with the soft
//! code `json.schema`.
//!
//! # The candidate
//!
//! Candidate text is parsed strictly (RFC 8259: no comments, no trailing commas, no duplicate
//! keys, no NaN/Infinity), with a depth bound and a byte bound; each violation is one finding
//! whose message starts with the JSON pointer (RFC 6901) of the offending value. Findings come in
//! a deterministic order (schema order within a node, array index order) and are capped at
//! [`MAX_FINDINGS`] violations. Only if a further violation actually exists does one soft
//! `json.truncated` finding follow, so a report holds at most `MAX_FINDINGS + 1` findings and
//! exactly `MAX_FINDINGS` violations carry no marker. Validation stops at the first violation
//! beyond the cap. Pointers shown in messages are cut at [`MAX_SHOWN_POINTER_BYTES`].
//!
//! Cost: object lookups are `O(log n)` (an index per object), enum equality is bounded by
//! [`MAX_ENUM_MEMBERS`] times the compared size, and the whole check is linear in the candidate
//! (at most 64 KiB) times the schema size (at most 16 KiB) in the worst case. See [`crate::codes`].

use std::cmp::Ordering;
use std::collections::BTreeMap;

use ptr_types::{Probability, VerificationLevel};

use crate::codes::finding;
use crate::{Finding, NamedVerifier, VerificationReport, VerificationStatus, Verifier};

/// Longest candidate document, in bytes.
pub const MAX_CANDIDATE_BYTES: usize = 64 * 1024;
/// Longest schema, in bytes.
pub const MAX_SCHEMA_BYTES: usize = 16 * 1024;
/// Deepest nesting of a candidate or of a schema.
pub const MAX_DEPTH: usize = 32;
/// Most violation findings in one report. A report holds at most this many plus one soft
/// `json.truncated` marker, and the marker is present only if a further violation exists.
pub const MAX_FINDINGS: usize = 32;
/// Most members of an `enum` (a larger one is outside the pinned subset: `Unknown`).
pub const MAX_ENUM_MEMBERS: usize = 256;
/// Largest absolute exponent of a number literal (`1e100000` is accepted, `1e100001` is not).
pub const MAX_EXPONENT: i64 = 100_000;
/// Longest JSON pointer shown in a finding message, in bytes.
pub const MAX_SHOWN_POINTER_BYTES: usize = 256;

/// An exact decimal: `(-1)^neg * digits * 10^exp`, with `digits` free of leading and trailing
/// zeros (zero is the empty digit string and never negative). No float is involved, so two
/// numbers compare exactly whatever their size or precision, and `1e-400` is not zero.
#[derive(Clone, Debug)]
struct Number {
    neg: bool,
    digits: Vec<u8>,
    exp: i64,
}

impl Number {
    fn from_parts(neg: bool, int: &[u8], frac: &[u8], exp10: i64) -> Self {
        let mut digits: Vec<u8> = int.iter().chain(frac).map(|b| b - b'0').collect();
        let leading = digits.iter().take_while(|d| **d == 0).count();
        digits.drain(..leading);
        let mut exp = exp10 - frac.len() as i64;
        while digits.last() == Some(&0) {
            digits.pop();
            exp += 1;
        }
        Self {
            neg: neg && !digits.is_empty(),
            digits,
            exp,
        }
    }

    fn sign(&self) -> i8 {
        if self.digits.is_empty() {
            0
        } else if self.neg {
            -1
        } else {
            1
        }
    }

    /// Position of the most significant digit.
    fn top(&self) -> i64 {
        self.exp + self.digits.len() as i64
    }

    fn is_integer(&self) -> bool {
        self.digits.is_empty() || self.exp >= 0
    }

    fn compare(&self, other: &Self) -> Ordering {
        let (a, b) = (self.sign(), other.sign());
        if a != b {
            return a.cmp(&b);
        }
        if a == 0 {
            return Ordering::Equal;
        }
        let magnitude = self
            .top()
            .cmp(&other.top())
            .then_with(|| self.digits.cmp(&other.digits));
        if a < 0 {
            magnitude.reverse()
        } else {
            magnitude
        }
    }

    /// The value as a `u32`, if it is a non-negative integer that small.
    fn to_u32(&self) -> Option<u32> {
        if self.sign() < 0 {
            return None;
        }
        if self.digits.is_empty() {
            return Some(0);
        }
        if self.exp < 0 || self.top() > 10 {
            return None;
        }
        let mut value: u128 = 0;
        for digit in &self.digits {
            value = value * 10 + u128::from(*digit);
        }
        value *= 10_u128.pow(u32::try_from(self.exp).ok()?);
        u32::try_from(value).ok()
    }
}

#[derive(Clone, Debug)]
struct Object {
    members: Vec<(String, Json)>,
    index: BTreeMap<String, usize>,
}

impl Object {
    fn get(&self, key: &str) -> Option<&Json> {
        self.index.get(key).map(|i| &self.members[*i].1)
    }
}

#[derive(Clone, Debug)]
enum Json {
    Null,
    Bool(bool),
    Num(Number),
    Str(String),
    Arr(Vec<Json>),
    Obj(Object),
}

impl Json {
    fn type_name(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Bool(_) => "boolean",
            Self::Num(n) if n.is_integer() => "integer",
            Self::Num(_) => "number",
            Self::Str(_) => "string",
            Self::Arr(_) => "array",
            Self::Obj(_) => "object",
        }
    }

    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Self::Obj(object) => object.get(key),
            _ => None,
        }
    }
}

fn json_eq(a: &Json, b: &Json) -> bool {
    match (a, b) {
        (Json::Null, Json::Null) => true,
        (Json::Bool(x), Json::Bool(y)) => x == y,
        (Json::Num(x), Json::Num(y)) => x.compare(y) == Ordering::Equal,
        (Json::Str(x), Json::Str(y)) => x == y,
        (Json::Arr(x), Json::Arr(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(x, y)| json_eq(x, y))
        }
        (Json::Obj(x), Json::Obj(y)) => {
            x.members.len() == y.members.len()
                && x.members
                    .iter()
                    .all(|(k, v)| y.get(k).is_some_and(|v2| json_eq(v, v2)))
        }
        _ => false,
    }
}

#[derive(Debug)]
enum ParseFailure {
    Depth,
    Syntax(usize, &'static str),
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn skip(&mut self) {
        while matches!(self.bytes.get(self.pos), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn fail<T>(&self, why: &'static str) -> Result<T, ParseFailure> {
        Err(ParseFailure::Syntax(self.pos, why))
    }

    fn expect(&mut self, literal: &[u8], why: &'static str) -> Result<(), ParseFailure> {
        if self.bytes[self.pos..].starts_with(literal) {
            self.pos += literal.len();
            Ok(())
        } else {
            self.fail(why)
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, ParseFailure> {
        if depth >= MAX_DEPTH {
            return Err(ParseFailure::Depth);
        }
        self.skip();
        match self.bytes.get(self.pos) {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self
                .expect(b"true", "bad literal")
                .map(|()| Json::Bool(true)),
            Some(b'f') => self
                .expect(b"false", "bad literal")
                .map(|()| Json::Bool(false)),
            Some(b'n') => self.expect(b"null", "bad literal").map(|()| Json::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => self.fail("unexpected character"),
            None => self.fail("unexpected end"),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, ParseFailure> {
        self.pos += 1;
        let mut object = Object {
            members: Vec::new(),
            index: BTreeMap::new(),
        };
        self.skip();
        if self.bytes.get(self.pos) == Some(&b'}') {
            self.pos += 1;
            return Ok(Json::Obj(object));
        }
        loop {
            self.skip();
            if self.bytes.get(self.pos) != Some(&b'"') {
                return self.fail("expected a property name");
            }
            let key = self.string()?;
            if object.index.contains_key(&key) {
                return self.fail("duplicate property name");
            }
            self.skip();
            if self.bytes.get(self.pos) != Some(&b':') {
                return self.fail("expected ':'");
            }
            self.pos += 1;
            let value = self.value(depth + 1)?;
            object.index.insert(key.clone(), object.members.len());
            object.members.push((key, value));
            self.skip();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Obj(object));
                }
                _ => return self.fail("expected ',' or '}'"),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, ParseFailure> {
        self.pos += 1;
        let mut items = Vec::new();
        self.skip();
        if self.bytes.get(self.pos) == Some(&b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.value(depth + 1)?);
            self.skip();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return self.fail("expected ',' or ']'"),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, ParseFailure> {
        let digits = self
            .bytes
            .get(self.pos..self.pos + 4)
            .ok_or(ParseFailure::Syntax(self.pos, "short \\u escape"))?;
        let mut value = 0;
        for digit in digits {
            let d = (*digit as char)
                .to_digit(16)
                .ok_or(ParseFailure::Syntax(self.pos, "bad \\u escape"))?;
            value = value * 16 + d;
        }
        self.pos += 4;
        Ok(value)
    }

    fn string(&mut self) -> Result<String, ParseFailure> {
        self.pos += 1;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let Some(&byte) = self.bytes.get(self.pos) else {
                return self.fail("unterminated string");
            };
            self.pos += 1;
            match byte {
                b'"' => break,
                b'\\' => {
                    let Some(&escape) = self.bytes.get(self.pos) else {
                        return self.fail("unterminated escape");
                    };
                    self.pos += 1;
                    let ch = match escape {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let first = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&first) {
                                if self.bytes.get(self.pos..self.pos + 2) != Some(b"\\u") {
                                    return self.fail("lone surrogate");
                                }
                                self.pos += 2;
                                let second = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&second) {
                                    return self.fail("lone surrogate");
                                }
                                0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                            } else {
                                first
                            };
                            match char::from_u32(code) {
                                Some(ch) => ch,
                                None => return self.fail("lone surrogate"),
                            }
                        }
                        _ => return self.fail("bad escape"),
                    };
                    let mut buffer = [0; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buffer).as_bytes());
                }
                0..=0x1f => return self.fail("control character in string"),
                other => out.push(other),
            }
        }
        String::from_utf8(out).map_err(|_| ParseFailure::Syntax(self.pos, "invalid UTF-8"))
    }

    fn number(&mut self) -> Result<Json, ParseFailure> {
        let bytes = self.bytes;
        let negative = bytes.get(self.pos) == Some(&b'-');
        if negative {
            self.pos += 1;
        }
        let int_start = self.pos;
        match bytes.get(self.pos) {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => self.digits(),
            _ => return self.fail("bad number"),
        }
        let int_end = self.pos;
        let (mut frac_start, mut frac_end) = (self.pos, self.pos);
        if bytes.get(self.pos) == Some(&b'.') {
            self.pos += 1;
            if !matches!(bytes.get(self.pos), Some(b'0'..=b'9')) {
                return self.fail("bad number");
            }
            frac_start = self.pos;
            self.digits();
            frac_end = self.pos;
        }
        let mut exponent: i64 = 0;
        if matches!(bytes.get(self.pos), Some(b'e' | b'E')) {
            self.pos += 1;
            let exponent_negative = bytes.get(self.pos) == Some(&b'-');
            if matches!(bytes.get(self.pos), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(bytes.get(self.pos), Some(b'0'..=b'9')) {
                return self.fail("bad number");
            }
            while let Some(digit @ b'0'..=b'9') = bytes.get(self.pos).copied() {
                exponent = (exponent * 10 + i64::from(digit - b'0')).min(MAX_EXPONENT + 1);
                self.pos += 1;
            }
            if exponent > MAX_EXPONENT {
                return self.fail("exponent out of range");
            }
            if exponent_negative {
                exponent = -exponent;
            }
        }
        Ok(Json::Num(Number::from_parts(
            negative,
            &bytes[int_start..int_end],
            &bytes[frac_start..frac_end],
            exponent,
        )))
    }

    fn digits(&mut self) {
        while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
    }
}

fn parse_json(text: &str) -> Result<Json, ParseFailure> {
    let mut parser = Parser {
        bytes: text.as_bytes(),
        pos: 0,
    };
    let value = parser.value(0)?;
    parser.skip();
    if parser.pos != parser.bytes.len() {
        return parser.fail("trailing characters");
    }
    Ok(value)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum JsonType {
    Null,
    Boolean,
    Integer,
    Number,
    String,
    Array,
    Object,
}

impl JsonType {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "null" => Self::Null,
            "boolean" => Self::Boolean,
            "integer" => Self::Integer,
            "number" => Self::Number,
            "string" => Self::String,
            "array" => Self::Array,
            "object" => Self::Object,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Boolean => "boolean",
            Self::Integer => "integer",
            Self::Number => "number",
            Self::String => "string",
            Self::Array => "array",
            Self::Object => "object",
        }
    }

    fn admits(self, value: &Json) -> bool {
        match (self, value) {
            (Self::Null, Json::Null)
            | (Self::Boolean, Json::Bool(_))
            | (Self::String, Json::Str(_))
            | (Self::Array, Json::Arr(_))
            | (Self::Object, Json::Obj(_))
            | (Self::Number, Json::Num(_)) => true,
            (Self::Integer, Json::Num(n)) => n.is_integer(),
            _ => false,
        }
    }
}

#[derive(Debug, Default)]
struct Schema {
    ty: Option<JsonType>,
    required: Vec<String>,
    properties: Vec<(String, Schema)>,
    property_index: BTreeMap<String, usize>,
    forbid_extra: bool,
    members: Option<Vec<Json>>,
    minimum: Option<Number>,
    maximum: Option<Number>,
    min_length: Option<usize>,
    max_length: Option<usize>,
    items: Option<Box<Schema>>,
}

fn length_keyword(value: &Json) -> Result<usize, String> {
    match value {
        Json::Num(n) => n
            .to_u32()
            .map(|v| v as usize)
            .ok_or_else(|| "a length keyword needs a non-negative integer".into()),
        _ => Err("a length keyword needs a non-negative integer".into()),
    }
}

fn compile(value: &Json, depth: usize) -> Result<Schema, String> {
    if depth > MAX_DEPTH {
        return Err(format!("schema is nested deeper than {MAX_DEPTH}"));
    }
    let Json::Obj(keywords) = value else {
        return Err("a schema must be a JSON object".into());
    };
    let mut schema = Schema::default();
    for (keyword, value) in &keywords.members {
        match (keyword.as_str(), value) {
            ("title" | "description", _) => {}
            ("type", Json::Str(name)) => {
                schema.ty =
                    Some(JsonType::parse(name).ok_or_else(|| format!("unknown type {name:?}"))?);
            }
            ("required", Json::Arr(names)) => {
                for name in names {
                    let Json::Str(name) = name else {
                        return Err("required must list property names".into());
                    };
                    schema.required.push(name.clone());
                }
            }
            ("properties", Json::Obj(properties)) => {
                for (name, sub) in &properties.members {
                    schema
                        .property_index
                        .insert(name.clone(), schema.properties.len());
                    schema
                        .properties
                        .push((name.clone(), compile(sub, depth + 1)?));
                }
            }
            ("additionalProperties", Json::Bool(allowed)) => schema.forbid_extra = !allowed,
            ("enum", Json::Arr(members))
                if !members.is_empty() && members.len() <= MAX_ENUM_MEMBERS =>
            {
                schema.members = Some(members.clone());
            }
            ("minimum", Json::Num(n)) => schema.minimum = Some(n.clone()),
            ("maximum", Json::Num(n)) => schema.maximum = Some(n.clone()),
            ("minLength", v) => schema.min_length = Some(length_keyword(v)?),
            ("maxLength", v) => schema.max_length = Some(length_keyword(v)?),
            ("items", sub @ Json::Obj(_)) => {
                schema.items = Some(Box::new(compile(sub, depth + 1)?));
            }
            (
                "type"
                | "required"
                | "properties"
                | "additionalProperties"
                | "enum"
                | "minimum"
                | "maximum"
                | "items",
                _,
            ) => return Err(format!("keyword {keyword:?} has an unsupported value")),
            (other, _) => return Err(format!("keyword {other:?} is outside the pinned subset")),
        }
    }
    Ok(schema)
}

/// RFC 6901 pointer of a child.
fn child(pointer: &str, token: &str) -> String {
    let mut out = String::with_capacity(pointer.len() + token.len() + 1);
    out.push_str(pointer);
    out.push('/');
    for ch in token.chars() {
        match ch {
            '~' => out.push_str("~0"),
            '/' => out.push_str("~1"),
            other => out.push(other),
        }
    }
    out
}

/// The pointer as shown in a message: `(root)` for the document, and at most
/// [`MAX_SHOWN_POINTER_BYTES`] bytes (cut at a character boundary, then `...`) so a key chosen by
/// the candidate cannot inflate the report, its trace or its digest.
fn shown(pointer: &str) -> String {
    if pointer.is_empty() {
        return "(root)".into();
    }
    if pointer.len() <= MAX_SHOWN_POINTER_BYTES {
        return pointer.into();
    }
    let mut end = MAX_SHOWN_POINTER_BYTES;
    while !pointer.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &pointer[..end])
}

struct Collector {
    findings: Vec<Finding>,
    truncated: bool,
}

impl Collector {
    fn full(&self) -> bool {
        self.findings.len() >= MAX_FINDINGS
    }

    /// Records a violation. Once the report is full, the first further violation only sets
    /// `truncated` (and validation stops: see the checks in [`validate`]).
    fn push(&mut self, code: &'static str, pointer: &str, why: String) {
        if self.truncated {
            return;
        }
        if self.full() {
            self.truncated = true;
        } else {
            self.findings
                .push(finding(code, format!("{}: {why}", shown(pointer))));
        }
    }
}

fn validate(schema: &Schema, value: &Json, pointer: &str, out: &mut Collector) {
    if out.truncated {
        return;
    }
    if let Some(ty) = schema.ty {
        if !ty.admits(value) {
            out.push(
                "json.type",
                pointer,
                format!("expected {}, found {}", ty.name(), value.type_name()),
            );
        }
    }
    if let Some(members) = &schema.members {
        if !members.iter().any(|member| json_eq(member, value)) {
            out.push("json.enum", pointer, "value is not an enum member".into());
        }
    }
    match value {
        Json::Num(n) => {
            if let Some(min) = &schema.minimum {
                if n.compare(min) == Ordering::Less {
                    out.push("json.range", pointer, "value is below minimum".into());
                }
            }
            if let Some(max) = &schema.maximum {
                if n.compare(max) == Ordering::Greater {
                    out.push("json.range", pointer, "value is above maximum".into());
                }
            }
        }
        Json::Str(text) => {
            let length = text.chars().count();
            if schema.min_length.is_some_and(|min| length < min) {
                out.push(
                    "json.length",
                    pointer,
                    "string is shorter than minLength".into(),
                );
            }
            if schema.max_length.is_some_and(|max| length > max) {
                out.push(
                    "json.length",
                    pointer,
                    "string is longer than maxLength".into(),
                );
            }
        }
        Json::Arr(items) => {
            if let Some(item_schema) = &schema.items {
                for (index, item) in items.iter().enumerate() {
                    if out.truncated {
                        break;
                    }
                    validate(item_schema, item, &child(pointer, &index.to_string()), out);
                }
            }
        }
        Json::Obj(object) => {
            for name in &schema.required {
                if out.truncated {
                    break;
                }
                if value.get(name).is_none() {
                    out.push(
                        "json.required",
                        pointer,
                        format!("missing required property {name:?}"),
                    );
                }
            }
            for (name, sub) in &schema.properties {
                if out.truncated {
                    break;
                }
                if let Some(present) = value.get(name) {
                    validate(sub, present, &child(pointer, name), out);
                }
            }
            if schema.forbid_extra {
                for (name, _) in &object.members {
                    if out.truncated {
                        break;
                    }
                    if !schema.property_index.contains_key(name) {
                        out.push(
                            "json.extra",
                            &child(pointer, name),
                            "property is not allowed".into(),
                        );
                    }
                }
            }
        }
        Json::Null | Json::Bool(_) => {}
    }
}

/// Verifies a JSON document (the candidate text) against one schema of the pinned subset.
#[derive(Debug)]
pub struct JsonSchemaVerifier {
    schema: Result<Schema, String>,
}

impl JsonSchemaVerifier {
    /// A verifier for the schema text `schema`. A schema outside the pinned subset makes every
    /// report `Unknown` (`json.schema`).
    pub fn new(schema: &str) -> Self {
        let compiled = if schema.len() > MAX_SCHEMA_BYTES {
            Err(format!("schema is longer than {MAX_SCHEMA_BYTES} bytes"))
        } else {
            match parse_json(schema) {
                Ok(value) => compile(&value, 0),
                Err(ParseFailure::Depth) => Err("schema is nested too deeply".into()),
                Err(ParseFailure::Syntax(at, why)) => {
                    Err(format!("schema is not JSON at byte {at}: {why}"))
                }
            }
        };
        Self { schema: compiled }
    }
}

fn report(status: VerificationStatus, findings: Vec<Finding>) -> VerificationReport {
    let score = if status == VerificationStatus::Pass {
        1.0
    } else {
        0.0
    };
    VerificationReport {
        status,
        level: VerificationLevel::Deterministic,
        score: Probability::new(score).expect("bounded score"),
        findings,
    }
}

impl Verifier<String> for JsonSchemaVerifier {
    fn verify(&self, candidate: &String) -> VerificationReport {
        let schema = match &self.schema {
            Ok(schema) => schema,
            Err(why) => {
                return report(
                    VerificationStatus::Unknown,
                    vec![finding("json.schema", why.clone())],
                );
            }
        };
        if candidate.len() > MAX_CANDIDATE_BYTES {
            return report(
                VerificationStatus::Fail,
                vec![finding(
                    "json.oversize",
                    format!(
                        "document is {} bytes, the bound is {MAX_CANDIDATE_BYTES}",
                        candidate.len()
                    ),
                )],
            );
        }
        let value = match parse_json(candidate) {
            Ok(value) => value,
            Err(ParseFailure::Depth) => {
                return report(
                    VerificationStatus::Fail,
                    vec![finding(
                        "json.depth",
                        format!("document is nested deeper than {MAX_DEPTH}"),
                    )],
                );
            }
            Err(ParseFailure::Syntax(at, why)) => {
                return report(
                    VerificationStatus::Fail,
                    vec![finding(
                        "json.parse",
                        format!("not valid JSON at byte {at}: {why}"),
                    )],
                );
            }
        };
        let mut out = Collector {
            findings: Vec::new(),
            truncated: false,
        };
        validate(schema, &value, "", &mut out);
        if out.truncated {
            out.findings.push(finding(
                "json.truncated",
                format!("more than {MAX_FINDINGS} violations; the rest are not listed"),
            ));
        }
        let status = if out.findings.is_empty() {
            VerificationStatus::Pass
        } else {
            VerificationStatus::Fail
        };
        report(status, out.findings)
    }
}

impl NamedVerifier<String> for JsonSchemaVerifier {
    fn name(&self) -> &'static str {
        "json-schema"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PERSON: &str = r#"{
        "type": "object",
        "required": ["name", "age"],
        "additionalProperties": false,
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 5},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "role": {"enum": ["admin", "user", 3]},
            "tags": {"type": "array", "items": {"type": "string"}}
        }
    }"#;

    fn check(schema: &str, doc: &str) -> VerificationReport {
        JsonSchemaVerifier::new(schema).verify(&doc.to_owned())
    }

    fn codes(report: &VerificationReport) -> Vec<&str> {
        report.findings.iter().map(|f| f.code.as_str()).collect()
    }

    #[test]
    fn a_conforming_document_passes() {
        let report = check(
            PERSON,
            r#"{"name":"Ada","age":36,"role":"admin","tags":["a","b"]}"#,
        );
        assert_eq!(report.status, VerificationStatus::Pass);
        assert_eq!(report.level, VerificationLevel::Deterministic);
        assert_eq!(report.score.get(), 1.0);
        assert!(report.findings.is_empty());
        assert_eq!(
            check(PERSON, r#"{"name":"A","age":36.0,"role":3.0}"#).status,
            VerificationStatus::Pass
        );
    }

    #[test]
    fn every_keyword_violation_has_its_code_and_a_pointer() {
        let cases: [(&str, &str, &str, &str); 9] = [
            (
                r#"{"name":"Ada","age":"x"}"#,
                "json.type",
                "/age",
                "expected integer",
            ),
            (
                r#"{"name":"Ada","age":1.5}"#,
                "json.type",
                "/age",
                "found number",
            ),
            (r#"{"name":"Ada"}"#, "json.required", "(root)", "\"age\""),
            (
                r#"{"name":"Ada","age":1,"role":"root"}"#,
                "json.enum",
                "/role",
                "enum",
            ),
            (
                r#"{"name":"Ada","age":-1}"#,
                "json.range",
                "/age",
                "below minimum",
            ),
            (
                r#"{"name":"Ada","age":151}"#,
                "json.range",
                "/age",
                "above maximum",
            ),
            (r#"{"name":"","age":1}"#, "json.length", "/name", "shorter"),
            (
                r#"{"name":"Adalbert","age":1}"#,
                "json.length",
                "/name",
                "longer",
            ),
            (
                r#"{"name":"Ada","age":1,"x":1}"#,
                "json.extra",
                "/x",
                "not allowed",
            ),
        ];
        for (doc, code, pointer, text) in cases {
            let report = check(PERSON, doc);
            assert_eq!(report.status, VerificationStatus::Fail, "{doc}");
            assert_eq!(codes(&report), [code], "{doc}");
            assert!(report.findings[0].hard);
            assert!(report.findings[0].message.starts_with(pointer), "{doc}");
            assert!(report.findings[0].message.contains(text), "{doc}");
        }
    }

    #[test]
    fn items_are_checked_with_index_pointers_and_pointers_are_escaped() {
        let report = check(PERSON, r#"{"name":"Ada","age":1,"tags":["a",2,"c",null]}"#);
        let pointers: Vec<&str> = report
            .findings
            .iter()
            .map(|f| f.message.split(':').next().unwrap())
            .collect();
        assert_eq!(pointers, ["/tags/1", "/tags/3"]);
        assert_eq!(codes(&report), ["json.type", "json.type"]);
        let report = check(r#"{"additionalProperties": false}"#, r#"{"a/b":1,"c~d":2}"#);
        assert!(report.findings[0].message.starts_with("/a~1b:"));
        assert!(report.findings[1].message.starts_with("/c~0d:"));
        let report = check(
            r#"{"type":"array","items":{"type":"array","items":{"type":"null"}}}"#,
            "[[null],[1]]",
        );
        assert!(report.findings[0].message.starts_with("/1/0:"));
    }

    #[test]
    fn several_violations_are_all_reported_in_a_fixed_order() {
        let report = check(PERSON, r#"{"age":200,"zzz":1,"name":""}"#);
        assert_eq!(codes(&report), ["json.length", "json.range", "json.extra"]);
    }

    #[test]
    fn invalid_json_is_a_parse_finding() {
        for doc in [
            "",
            "{",
            "{\"a\":}",
            "[1,]",
            "{\"a\":1,}",
            "'a'",
            "{\"a\":1}{",
            "01",
            "1.",
            "-",
            "NaN",
            "Infinity",
            "1e100001",
            "1e99999999999999999999",
            "{\"a\":1,\"a\":2}",
            "\"\\x\"",
            "\"\\ud800\"",
            "\"a\nb\"",
            "tru",
            "[1 2]",
            "{\"a\" 1}",
            "// c\n1",
            "\"\\u12\"",
        ] {
            let report = check("{}", doc);
            assert_eq!(report.status, VerificationStatus::Fail, "{doc:?}");
            assert_eq!(codes(&report), ["json.parse"], "{doc:?}");
            assert!(report.findings[0].hard);
        }
        assert_eq!(
            check("{}", "  [1, {\"a\": \"\\u00e9\\ud83d\\ude00\"}]  ").status,
            VerificationStatus::Pass
        );
    }

    #[test]
    fn adversarial_nesting_and_size_are_findings_not_panics() {
        let deep = format!("{}{}", "[".repeat(100_000), "]".repeat(100_000));
        let report = check("{}", &deep);
        assert_eq!(codes(&report), ["json.oversize"]);
        let deep = format!("{}{}", "[".repeat(10_000), "]".repeat(10_000));
        let report = check("{}", &deep);
        assert_eq!(codes(&report), ["json.depth"]);
        let ok_depth = format!("{}{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
        assert_eq!(check("{}", &ok_depth).status, VerificationStatus::Pass);
        let over = format!("{}{}", "[".repeat(MAX_DEPTH + 1), "]".repeat(MAX_DEPTH + 1));
        assert_eq!(codes(&check("{}", &over)), ["json.depth"]);
        let big = format!("\"{}\"", "a".repeat(MAX_CANDIDATE_BYTES));
        assert_eq!(codes(&check("{}", &big)), ["json.oversize"]);
        // Huge numbers are exact decimals; they never overflow.
        let huge = "9".repeat(400);
        assert_eq!(check("{}", &huge).status, VerificationStatus::Pass);
        assert_eq!(
            check(r#"{"type":"number"}"#, &"9".repeat(60)).status,
            VerificationStatus::Pass
        );
        assert_eq!(
            codes(&check(r#"{"maximum":10}"#, &"9".repeat(60))),
            ["json.range"]
        );
        assert_eq!(
            check(r#"{"type":"integer"}"#, "1e3").status,
            VerificationStatus::Pass
        );
    }

    #[test]
    fn violations_beyond_the_cap_are_marked_truncated() {
        let doc = format!("[{}]", vec!["1"; 100].join(","));
        let report = check(r#"{"items":{"type":"string"}}"#, &doc);
        assert_eq!(report.status, VerificationStatus::Fail);
        assert_eq!(report.findings.len(), MAX_FINDINGS + 1);
        let last = report.findings.last().unwrap();
        assert_eq!(last.code, "json.truncated");
        assert!(!last.hard);
    }

    #[test]
    fn the_truncation_marker_needs_a_further_violation() {
        let schema = r#"{"items":{"type":"string"}}"#;
        let doc = |bad: usize| {
            let mut items = vec!["1"; bad];
            items.extend(vec!["\"ok\""; 40]);
            format!("[{}]", items.join(","))
        };
        // Exactly 32 violations: all listed, no marker, however many valid items follow.
        let report = check(schema, &doc(MAX_FINDINGS));
        assert_eq!(report.findings.len(), MAX_FINDINGS);
        assert!(report.findings.iter().all(|f| f.code == "json.type"));
        // 33 violations: 32 listed plus the marker.
        let report = check(schema, &doc(MAX_FINDINGS + 1));
        assert_eq!(report.findings.len(), MAX_FINDINGS + 1);
        assert_eq!(report.findings[MAX_FINDINGS].code, "json.truncated");
        assert!(report.findings[..MAX_FINDINGS]
            .iter()
            .all(|f| f.code == "json.type"));
        // Objects: 32 extra properties against 33.
        let extras = |n: usize| {
            let members: Vec<String> = (0..n).map(|i| format!("\"k{i}\":1")).collect();
            format!("{{{}}}", members.join(","))
        };
        let strict = r#"{"additionalProperties":false}"#;
        assert_eq!(
            check(strict, &extras(MAX_FINDINGS)).findings.len(),
            MAX_FINDINGS
        );
        let report = check(strict, &extras(MAX_FINDINGS + 1));
        assert_eq!(report.findings.len(), MAX_FINDINGS + 1);
        assert_eq!(report.findings[MAX_FINDINGS].code, "json.truncated");
    }

    #[test]
    fn shown_pointers_are_truncated() {
        let key = "k".repeat(10_000);
        let doc = format!("{{\"{key}\":1}}");
        let report = check(r#"{"additionalProperties":false}"#, &doc);
        assert_eq!(codes(&report), ["json.extra"]);
        let message = &report.findings[0].message;
        assert!(
            message.len() < MAX_SHOWN_POINTER_BYTES + 64,
            "{}",
            message.len()
        );
        assert!(message.contains("..."));
        // A multi-byte key is cut at a character boundary.
        let key = "é".repeat(500);
        let doc = format!("{{\"{key}\":1}}");
        let report = check(r#"{"additionalProperties":false}"#, &doc);
        assert!(report.findings[0].message.len() < MAX_SHOWN_POINTER_BYTES + 64);
        // A short pointer is shown whole.
        let report = check(r#"{"additionalProperties":false}"#, r#"{"abc":1}"#);
        assert!(report.findings[0].message.starts_with("/abc:"));
    }

    #[test]
    fn numbers_compare_exactly_beyond_f64() {
        let max = |bound: &str, doc: &str| check(&format!(r#"{{"maximum":{bound}}}"#), doc).status;
        // 2^53 + 1 is above the float 2^53.
        assert_eq!(
            max("9007199254740992.0", "9007199254740993"),
            VerificationStatus::Fail
        );
        assert_eq!(
            max("9007199254740992.0", "9007199254740992"),
            VerificationStatus::Pass
        );
        assert_eq!(
            max("9007199254740993", "9007199254740992.0"),
            VerificationStatus::Pass
        );
        // Sixty-digit integers differing in the last digit.
        let big = format!("{}1", "9".repeat(59));
        let bigger = format!("{}2", "9".repeat(59));
        assert_eq!(max(&big, &bigger), VerificationStatus::Fail);
        assert_eq!(max(&bigger, &big), VerificationStatus::Pass);
        assert_eq!(max(&big, &big), VerificationStatus::Pass);
        // Underflow is not zero.
        assert_eq!(max("0", "1e-400"), VerificationStatus::Fail);
        assert_eq!(max("0", "-1e-400"), VerificationStatus::Pass);
        assert_eq!(max("0", "0.000"), VerificationStatus::Pass);
        assert_eq!(max("0", "-0"), VerificationStatus::Pass);
        assert_eq!(max("1e-400", "1e-401"), VerificationStatus::Pass);
        assert_eq!(
            check(r#"{"minimum":1e-400}"#, "0")
                .findings
                .first()
                .map(|f| f.code.as_str()),
            Some("json.range")
        );
        // Sign, magnitude and trailing zeros.
        assert_eq!(
            max("-5", "-4.9999999999999999999999"),
            VerificationStatus::Fail
        );
        assert_eq!(max("-5", "-5.0000"), VerificationStatus::Pass);
        assert_eq!(max("100", "1e2"), VerificationStatus::Pass);
        assert_eq!(max("100", "1.00001e2"), VerificationStatus::Fail);
        assert_eq!(max("1e100000", "1e99999"), VerificationStatus::Pass);
    }

    #[test]
    fn enum_equality_is_exact_for_large_numbers() {
        let schema = r#"{"enum":[9007199254740992.0]}"#;
        assert_eq!(
            check(schema, "9007199254740992").status,
            VerificationStatus::Pass
        );
        assert_eq!(codes(&check(schema, "9007199254740993")), ["json.enum"]);
        let schema = r#"{"enum":[1e-400]}"#;
        assert_eq!(codes(&check(schema, "0")), ["json.enum"]);
        assert_eq!(check(schema, "0.0e-399").status, VerificationStatus::Fail);
        assert_eq!(check(schema, "10e-401").status, VerificationStatus::Pass);
        assert_eq!(
            check(r#"{"enum":[0]}"#, "-0.0").status,
            VerificationStatus::Pass
        );
    }

    #[test]
    fn enum_size_is_capped_and_equality_is_bounded() {
        let members = |n: usize| (0..n).map(|i| i.to_string()).collect::<Vec<_>>().join(",");
        let at_cap = format!(r#"{{"enum":[{}]}}"#, members(MAX_ENUM_MEMBERS));
        assert_eq!(check(&at_cap, "255").status, VerificationStatus::Pass);
        let over = format!(r#"{{"enum":[{}]}}"#, members(MAX_ENUM_MEMBERS + 1));
        let report = check(&over, "1");
        assert_eq!(report.status, VerificationStatus::Unknown);
        assert_eq!(codes(&report), ["json.schema"]);
        // A wide object compared against a wide enum member stays fast and correct.
        let wide: Vec<String> = (0..800).map(|i| format!("\"k{i}\":{i}")).collect();
        let wide = format!("{{{}}}", wide.join(","));
        let schema = format!(r#"{{"enum":[{wide}]}}"#);
        assert_eq!(check(&schema, &wide).status, VerificationStatus::Pass);
        assert_eq!(
            check(&schema, &wide.replace("\"k799\":799", "\"k799\":0")).status,
            VerificationStatus::Fail
        );
    }

    #[test]
    fn duplicate_keys_are_found_in_wide_objects() {
        let members: Vec<String> = (0..3000).map(|i| format!("\"k{i}\":1")).collect();
        let ok = format!("{{{}}}", members.join(","));
        assert_eq!(check("{}", &ok).status, VerificationStatus::Pass);
        let dup = format!("{{{},\"k0\":2}}", members.join(","));
        assert_eq!(codes(&check("{}", &dup)), ["json.parse"]);
        assert!(check("{}", &dup).findings[0].message.contains("duplicate"));
    }

    #[test]
    fn a_schema_outside_the_subset_is_unknown() {
        for schema in [
            "",
            "[]",
            "{",
            r#"{"pattern":"a+"}"#,
            r##"{"$ref":"#"}"##,
            r#"{"oneOf":[]}"#,
            r#"{"type":"float"}"#,
            r#"{"type":["string","null"]}"#,
            r#"{"required":[1]}"#,
            r#"{"enum":[]}"#,
            r#"{"minLength":-1}"#,
            r#"{"minLength":1.5}"#,
            r#"{"additionalProperties":{}}"#,
            r#"{"items":[]}"#,
            r#"{"properties":{"a":1}}"#,
        ] {
            let report = check(schema, "1");
            assert_eq!(report.status, VerificationStatus::Unknown, "{schema}");
            assert_eq!(codes(&report), ["json.schema"], "{schema}");
            assert!(!report.findings[0].hard);
        }
        let deep = format!(
            "{}{}",
            r#"{"items":"#.repeat(40),
            "{}".to_owned() + &"}".repeat(40)
        );
        assert_eq!(check(&deep, "1").status, VerificationStatus::Unknown);
        let big = format!(r#"{{"title":"{}"}}"#, "x".repeat(MAX_SCHEMA_BYTES));
        assert_eq!(check(&big, "1").status, VerificationStatus::Unknown);
        assert_eq!(
            check(r#"{"title":"t","description":"d"}"#, "1").status,
            VerificationStatus::Pass
        );
    }

    #[test]
    fn enum_compares_numbers_by_value_and_objects_regardless_of_order() {
        let schema = r#"{"enum":[1,{"a":1,"b":[true]},null]}"#;
        assert_eq!(check(schema, "1.0").status, VerificationStatus::Pass);
        assert_eq!(
            check(schema, r#"{"b":[true],"a":1}"#).status,
            VerificationStatus::Pass
        );
        assert_eq!(check(schema, "null").status, VerificationStatus::Pass);
        assert_eq!(codes(&check(schema, r#"{"a":1}"#)), ["json.enum"]);
        assert_eq!(codes(&check(schema, "false")), ["json.enum"]);
    }

    #[test]
    fn verification_is_deterministic() {
        let verifier = JsonSchemaVerifier::new(PERSON);
        for doc in [r#"{"age":200,"zzz":1}"#, "[", r#"{"name":"A","age":1}"#] {
            let doc = doc.to_owned();
            assert_eq!(verifier.verify(&doc), verifier.verify(&doc));
        }
        assert_eq!(NamedVerifier::<String>::name(&verifier), "json-schema");
    }
}
