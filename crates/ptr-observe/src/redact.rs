//! Redaction middleware (Invariant 13: secrets are never logged or stored).
//!
//! Two independent mechanisms; either one redacts a field value:
//!
//! 1. **Key deny-list.** The field key is normalized (ASCII lowercase, `-`, `.` and space mapped
//!    to `_`) and redacted when it *contains* a normalized deny entry, so `x-api-key` matches
//!    `api_key` and `db.password_hash` matches `password`. Substring matching deliberately
//!    over-redacts (`token_count` matches `token`); keys on the exact-match allow-list are exempt.
//!    The key name itself is not redacted.
//! 2. **Value heuristics** on string values (and on event names / span status messages). A string
//!    is secret-like when any of these hold:
//!    - an auth scheme: `Bearer <token>` or `Basic <token>` with a token of at least 8
//!      base64-alphabet characters;
//!    - a URL whose authority carries a password (`scheme://user:pass@host`);
//!    - a PEM private-key header (`-----BEGIN ... PRIVATE KEY-----`);
//!    - a JWT-shaped word (`eyJ...` with two dots);
//!    - a word with a well-known credential prefix (`sk-`, `ghp_`, `gho_`, `github_pat_`,
//!      `glpat-`, `xoxb-`, `xoxp-`, `AKIA` + 16 uppercase/digits);
//!    - a *token* (maximal run of `[A-Za-z0-9+/_-]`) of at least `min_secret_len` characters that
//!      is either purely hexadecimal, or mixes at least two of lowercase/uppercase/digit and has
//!      Shannon entropy of at least `entropy_bits_per_char`.
//!
//!    These are heuristics: long hex digests (for example SHA-256) are redacted as false
//!    positives, and low-entropy or short secrets are not caught by the value path; use the key
//!    deny-list for those.
//!
//! Redaction replaces the whole value with [`REDACTED`] (as a string, whatever the original type).

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{Span, SpanEvent, SpanStatus, TraceError, TraceEvent, TraceSink, TraceValue};

pub const REDACTED: &str = "[REDACTED]";

/// Hard maximum of [`RedactionPolicy::min_secret_len`]: a larger threshold would let the value
/// heuristics miss every realistic secret, so [`Redactor`] clamps it to this length.
pub const MAX_MIN_SECRET_LEN: usize = 256;

#[derive(Clone, Debug, PartialEq)]
pub struct RedactionPolicy {
    deny_keys: Vec<String>,
    allow_keys: Vec<String>,
    /// Shortest token the value heuristics treat as a secret; clamped to
    /// [`MAX_MIN_SECRET_LEN`] when applied.
    pub min_secret_len: usize,
    pub entropy_bits_per_char: f64,
}

fn normalize_key(key: &str) -> String {
    key.chars()
        .map(|c| match c {
            '-' | '.' | ' ' => '_',
            other => other.to_ascii_lowercase(),
        })
        .collect()
}

impl Default for RedactionPolicy {
    fn default() -> Self {
        Self {
            deny_keys: [
                "password",
                "passwd",
                "secret",
                "token",
                "api_key",
                "apikey",
                "authorization",
                "cookie",
                "dsn",
                "private_key",
            ]
            .iter()
            .map(|s| (*s).to_owned())
            .collect(),
            allow_keys: Vec::new(),
            min_secret_len: 32,
            entropy_bits_per_char: 4.2,
        }
    }
}

impl RedactionPolicy {
    /// Adds a deny-list entry (matched as a normalized substring of the field key).
    pub fn with_deny_key(mut self, key: &str) -> Self {
        self.deny_keys.push(normalize_key(key));
        self
    }

    /// Exempts an exact (normalized) key from the key deny-list. Value heuristics still apply.
    pub fn with_allow_key(mut self, key: &str) -> Self {
        self.allow_keys.push(normalize_key(key));
        self
    }
}

#[derive(Clone, Debug, Default)]
pub struct Redactor {
    policy: RedactionPolicy,
}

fn is_token_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '_' | '-')
}

fn entropy_bits_per_char(token: &str) -> f64 {
    let mut counts = [0usize; 256];
    for byte in token.bytes() {
        counts[usize::from(byte)] += 1;
    }
    let len = token.len() as f64;
    counts
        .iter()
        .filter(|count| **count > 0)
        .map(|count| {
            let p = *count as f64 / len;
            -p * p.log2()
        })
        .sum()
}

fn is_auth_scheme(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    ["bearer ", "basic "].iter().any(|scheme| {
        lower.match_indices(scheme).any(|(at, _)| {
            let boundary = at == 0
                || !lower[..at]
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_ascii_alphanumeric());
            let rest = &lower[at + scheme.len()..];
            boundary
                && rest
                    .chars()
                    .take_while(|c| is_token_char(*c) || *c == '=')
                    .count()
                    >= 8
        })
    })
}

fn is_url_with_password(word: &str) -> bool {
    word.match_indices("://").any(|(at, _)| {
        let rest = &word[at + 3..];
        let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
        authority
            .rsplit_once('@')
            .is_some_and(|(userinfo, _)| userinfo.contains(':'))
    })
}

fn has_known_prefix(word: &str) -> bool {
    const PREFIXES: [&str; 7] = [
        "sk-",
        "ghp_",
        "gho_",
        "github_pat_",
        "glpat-",
        "xoxb-",
        "xoxp-",
    ];
    let word = word.trim_matches(|c: char| !c.is_ascii_alphanumeric());
    if PREFIXES
        .iter()
        .any(|prefix| word.starts_with(prefix) && word.len() >= prefix.len() + 8)
    {
        return true;
    }
    word.len() >= 20
        && word.starts_with("AKIA")
        && word
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
}

fn is_jwt(word: &str) -> bool {
    word.starts_with("eyJ") && word.matches('.').count() == 2 && word.len() >= 20
}

impl Redactor {
    pub fn new(policy: RedactionPolicy) -> Self {
        Self { policy }
    }

    pub fn policy(&self) -> &RedactionPolicy {
        &self.policy
    }

    pub fn is_secret_key(&self, key: &str) -> bool {
        let key = normalize_key(key);
        if self.policy.allow_keys.contains(&key) {
            return false;
        }
        self.policy
            .deny_keys
            .iter()
            .any(|deny| !deny.is_empty() && key.contains(deny.as_str()))
    }

    pub fn looks_like_secret(&self, text: &str) -> bool {
        if text.contains("PRIVATE KEY-----") || is_auth_scheme(text) {
            return true;
        }
        if text
            .split_whitespace()
            .any(|word| is_url_with_password(word) || is_jwt(word) || has_known_prefix(word))
        {
            return true;
        }
        text.split(|c: char| !is_token_char(c))
            .any(|token| self.is_high_entropy(token))
    }

    fn is_high_entropy(&self, token: &str) -> bool {
        if token.len() < self.policy.min_secret_len.min(MAX_MIN_SECRET_LEN) {
            return false;
        }
        if token.bytes().all(|b| b.is_ascii_hexdigit()) {
            return true;
        }
        let classes = [
            token.chars().any(|c| c.is_ascii_lowercase()),
            token.chars().any(|c| c.is_ascii_uppercase()),
            token.chars().any(|c| c.is_ascii_digit()),
        ]
        .iter()
        .filter(|present| **present)
        .count();
        classes >= 2 && entropy_bits_per_char(token) >= self.policy.entropy_bits_per_char
    }

    /// Returns the text unchanged, or [`REDACTED`] when it looks like a secret.
    pub fn scrub_text<'a>(&self, text: &'a str) -> Cow<'a, str> {
        if self.looks_like_secret(text) {
            Cow::Borrowed(REDACTED)
        } else {
            Cow::Borrowed(text)
        }
    }

    /// Redacts a field map in place and returns how many values were replaced.
    pub fn redact_fields(&self, fields: &mut BTreeMap<String, TraceValue>) -> usize {
        let mut redacted = 0;
        for (key, value) in fields.iter_mut() {
            // Every variant that carries caller-supplied text is inspected: `String` and
            // `RequestId` (a client may supply any string as a request id). The numeric and
            // boolean variants (`U64`, `Bool`, `Revision`, `Generation`, `CommitIndex`) hold no
            // text and cannot encode a credential.
            let text_secret = match value {
                TraceValue::String(text) => self.looks_like_secret(text),
                TraceValue::RequestId(id) => self.looks_like_secret(&id.0),
                TraceValue::U64(_)
                | TraceValue::Bool(_)
                | TraceValue::Revision(_)
                | TraceValue::Generation(_)
                | TraceValue::CommitIndex(_) => false,
            };
            let secret = self.is_secret_key(key) || text_secret;
            if secret && *value != TraceValue::String(REDACTED.to_owned()) {
                *value = TraceValue::String(REDACTED.to_owned());
                redacted += 1;
            }
        }
        redacted
    }

    /// Returns a redacted copy of the event and the number of replaced values.
    pub fn redact_event(&self, event: &TraceEvent) -> (TraceEvent, usize) {
        let mut copy = event.clone();
        let mut redacted = self.redact_fields(&mut copy.fields);
        if self.looks_like_secret(&copy.name) {
            copy.name = REDACTED.to_owned();
            redacted += 1;
        }
        (copy, redacted)
    }

    /// Returns a redacted copy of a finished span (fields, events, name and status message).
    pub fn redact_span(&self, span: &Span) -> (Span, usize) {
        let mut copy = span.clone();
        let mut redacted = self.redact_fields(&mut copy.fields);
        if self.looks_like_secret(&copy.name) {
            copy.name = REDACTED.to_owned();
            redacted += 1;
        }
        if let SpanStatus::Error(message) = &mut copy.status {
            if self.looks_like_secret(message) {
                *message = REDACTED.to_owned();
                redacted += 1;
            }
        }
        copy.events = span
            .events
            .iter()
            .map(|entry| {
                let (event, count) = self.redact_event(&entry.event);
                redacted += count;
                SpanEvent {
                    tick: entry.tick,
                    event,
                }
            })
            .collect();
        (copy, redacted)
    }
}

/// Wraps a sink so that the inner sink only ever sees redacted events.
pub struct RedactingSink<S: TraceSink> {
    inner: S,
    redactor: Redactor,
    redacted_values: AtomicU64,
}

impl<S: TraceSink> fmt::Debug for RedactingSink<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RedactingSink")
            .field("redacted_values", &self.redacted_values())
            .finish_non_exhaustive()
    }
}

impl<S: TraceSink> RedactingSink<S> {
    pub fn new(inner: S) -> Self {
        Self::with_redactor(inner, Redactor::default())
    }

    pub fn with_redactor(inner: S, redactor: Redactor) -> Self {
        Self {
            inner,
            redactor,
            redacted_values: AtomicU64::new(0),
        }
    }

    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Total number of values replaced so far.
    pub fn redacted_values(&self) -> u64 {
        self.redacted_values.load(Ordering::Relaxed)
    }
}

impl<S: TraceSink> TraceSink for RedactingSink<S> {
    fn emit(&self, event: &TraceEvent) -> Result<(), TraceError> {
        let (clean, count) = self.redactor.redact_event(event);
        self.redacted_values
            .fetch_add(count as u64, Ordering::Relaxed);
        self.inner.emit(&clean)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn an_unbounded_secret_length_threshold_is_clamped() {
        let policy = RedactionPolicy {
            min_secret_len: usize::MAX,
            ..RedactionPolicy::default()
        };
        let redactor = Redactor::new(policy);
        // Built at runtime from a non-repeating pseudo-random sequence so no secret-like literal is
        // committed (push protection would flag a high-entropy string constant).
        let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        let mut state = 0x2545_f491_u32;
        let secret: String = (0..MAX_MIN_SECRET_LEN + 64)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                char::from(alphabet[state as usize % alphabet.len()])
            })
            .collect();
        assert!(secret.len() > MAX_MIN_SECRET_LEN);
        assert!(redactor.looks_like_secret(&secret));
    }

    use super::*;

    const RANDOM_B64: &str = "q8Zr3xK9LmP2vB7nT5yW1cD6hJ4sA0eFgUiOpXzRtYu";

    #[test]
    fn key_matching_is_case_insensitive_substring_with_separator_normalization() {
        let r = Redactor::default();
        for key in [
            "password",
            "PASSWORD",
            "x-api-key",
            "X-Api-Key",
            "db.private_key",
            "Authorization",
            "set-cookie",
            "pg.dsn",
            "client_secret",
            "refresh_token",
        ] {
            assert!(r.is_secret_key(key), "{key}");
        }
        for key in ["ptr.operation", "error.code", "name"] {
            assert!(!r.is_secret_key(key), "{key}");
        }
    }

    #[test]
    fn allow_list_exempts_exact_key_and_custom_deny_applies() {
        let policy = RedactionPolicy::default()
            .with_allow_key("ptr.token_count")
            .with_deny_key("Session-Id");
        let r = Redactor::new(policy);
        assert!(!r.is_secret_key("ptr.token_count"));
        assert!(r.is_secret_key("ptr.token_counts"));
        assert!(r.is_secret_key("x.session.id"));
    }

    #[test]
    fn value_heuristics_detect_secret_shapes() {
        let r = Redactor::default();
        let hex = "0123456789abcdef0123456789abcdef";
        let jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl";
        for text in [
            "Bearer abcdefgh12345".to_owned(),
            "authorization failed for bearer abcdefghij".to_owned(),
            "Basic dXNlcjpwYXNzd29yZA==".to_owned(),
            "postgres://user:pass@db.local:5432/x".to_owned(),
            "-----BEGIN RSA PRIVATE KEY-----".to_owned(),
            jwt.to_owned(),
            "sk-abcdefghij1234".to_owned(),
            "ghp_abcdefghij1234".to_owned(),
            "AKIAABCDEFGHIJKLMNOP".to_owned(),
            hex.to_owned(),
            format!("key={RANDOM_B64}"),
        ] {
            assert!(r.looks_like_secret(&text), "{text}");
        }
    }

    #[test]
    fn value_heuristics_leave_ordinary_text_alone() {
        let r = Redactor::default();
        for text in [
            "plain message",
            "basic usage of the planner",
            "bearer",
            "postgres://user@host/db",
            "https://example.com/a/b",
            "550e8400-e29b-41d4-a716-446655440000",
            "request-handler-timeout-30s-config-value",
            "/usr/local/lib/some/very/long/path/without/digits",
            "abc123",
        ] {
            assert!(!r.looks_like_secret(text), "{text}");
        }
    }

    #[test]
    fn redact_event_replaces_key_and_value_matches_of_any_type() {
        let r = Redactor::default();
        let event = TraceEvent::new(crate::TraceLevel::Info, "login")
            .with_field("password", 12345u64)
            .with_field("note", "Bearer abcdefgh12345")
            .with_field("ok", "fine");
        let (clean, count) = r.redact_event(&event);
        assert_eq!(count, 2);
        assert_eq!(
            clean.fields["password"],
            TraceValue::String(REDACTED.into())
        );
        assert_eq!(clean.fields["note"], TraceValue::String(REDACTED.into()));
        assert_eq!(clean.fields["ok"], TraceValue::String("fine".into()));
        assert!(!format!("{clean:?}").contains("abcdefgh12345"));
    }
}
