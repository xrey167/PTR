//! Deny-list redaction of secret-looking text before it is persisted (Invariant 13).
//!
//! This is a heuristic pass over whitespace-separated words, not a guarantee: it removes values
//! that follow a key-like name (`password=...`, `"api_key": "..."`, `Authorization: Bearer ...`),
//! well-known credential prefixes (`sk-`, `ghp_`, `AKIA...`, `xox?-`) and PEM private key
//! blocks. A secret in free prose that matches none of these is not detected.

/// What replaces a removed secret.
pub const REDACTED: &str = "[REDACTED]";

const KEY_NAMES: [&str; 12] = [
    "api_key",
    "apikey",
    "api-key",
    "secret",
    "password",
    "passwd",
    "passphrase",
    "token",
    "authorization",
    "private_key",
    "credential",
    "access_key",
];

const SECRET_PREFIXES: [(&str, usize); 9] = [
    ("sk-", 16),
    ("ghp_", 20),
    ("gho_", 20),
    ("ghs_", 20),
    ("github_pat_", 20),
    ("xoxb-", 10),
    ("xoxp-", 10),
    ("AKIA", 16),
    ("AIza", 30),
];

/// `text` with secret-looking values replaced by [`REDACTED`]. Idempotent.
pub fn redact(text: &str) -> String {
    let without_keys = redact_private_key_blocks(text);
    let mut out = String::with_capacity(without_keys.len());
    let mut redact_next_word = false;
    let mut rest = without_keys.as_str();
    while !rest.is_empty() {
        let is_space = rest.starts_with(char::is_whitespace);
        let end = rest
            .find(|c: char| c.is_whitespace() != is_space)
            .unwrap_or(rest.len());
        let (piece, tail) = rest.split_at(end);
        rest = tail;
        if is_space {
            out.push_str(piece);
            continue;
        }
        if redact_next_word && is_bearer_scheme(piece) {
            out.push_str(piece);
            continue;
        }
        if redact_next_word {
            redact_next_word = false;
            if let Some(kept) = redact_word_value(piece) {
                out.push_str(&kept);
                continue;
            }
        }
        match redact_word(piece) {
            WordOutcome::Keep => out.push_str(piece),
            WordOutcome::Replace(replacement) => out.push_str(&replacement),
            WordOutcome::ReplaceAndRedactNext(replacement) => {
                out.push_str(&replacement);
                redact_next_word = true;
            }
        }
        if is_bearer_scheme(piece) {
            redact_next_word = true;
        }
    }
    out
}

enum WordOutcome {
    Keep,
    Replace(String),
    ReplaceAndRedactNext(String),
}

fn redact_word(word: &str) -> WordOutcome {
    let core = word.trim_matches(is_wrapper);
    if has_secret_prefix(core) {
        return WordOutcome::Replace(rewrap(word, REDACTED));
    }
    if let Some(split) = word.find([':', '=']) {
        let (name, value) = word.split_at(split + 1);
        if is_key_like(name) {
            if value.trim_matches(is_wrapper).is_empty() {
                return WordOutcome::ReplaceAndRedactNext(word.to_owned());
            }
            return WordOutcome::Replace(format!("{name}{}", rewrap(value, REDACTED)));
        }
    }
    WordOutcome::Keep
}

/// The word that follows a bare key such as `"password":`, replaced.
fn redact_word_value(word: &str) -> Option<String> {
    if word.trim_matches(is_wrapper).is_empty() {
        return None;
    }
    Some(rewrap(word, REDACTED))
}

fn is_bearer_scheme(word: &str) -> bool {
    word.trim_matches(is_wrapper).eq_ignore_ascii_case("bearer")
}

fn is_key_like(name: &str) -> bool {
    let lowered = name.to_ascii_lowercase();
    KEY_NAMES.iter().any(|key| lowered.contains(key))
}

fn has_secret_prefix(core: &str) -> bool {
    SECRET_PREFIXES.iter().any(|(prefix, min_len)| {
        core.starts_with(prefix)
            && core.len() >= *min_len
            && core[prefix.len()..]
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    })
}

fn is_wrapper(c: char) -> bool {
    matches!(
        c,
        '"' | '\'' | '`' | ',' | ';' | '(' | ')' | '[' | ']' | '{' | '}'
    )
}

/// Replace the core of `word` with `replacement`, keeping surrounding quotes and punctuation.
fn rewrap(word: &str, replacement: &str) -> String {
    if word.contains(REDACTED) {
        return word.to_owned();
    }
    let start = word.len() - word.trim_start_matches(is_wrapper).len();
    let end = word.trim_end_matches(is_wrapper).len().max(start);
    format!("{}{}{}", &word[..start], replacement, &word[end..])
}

fn redact_private_key_blocks(text: &str) -> String {
    const BEGIN: &str = "-----BEGIN ";
    const END_MARK: &str = "-----END ";
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(BEGIN) {
        let after = &rest[start..];
        let header_end = after[BEGIN.len()..].find("-----").map(|p| p + BEGIN.len());
        let is_private = header_end.is_some_and(|end| after[..end].contains("PRIVATE KEY"));
        if !is_private {
            out.push_str(&rest[..start + BEGIN.len()]);
            rest = &rest[start + BEGIN.len()..];
            continue;
        }
        out.push_str(&rest[..start]);
        out.push_str(REDACTED);
        rest = match after.find(END_MARK) {
            Some(end) => {
                let tail = &after[end + END_MARK.len()..];
                match tail.find("-----") {
                    Some(close) => &tail[close + 5..],
                    None => "",
                }
            }
            None => "",
        };
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_value_pairs_lose_their_value() {
        assert_eq!(redact("password=hunter2 ok"), "password=[REDACTED] ok");
        assert_eq!(redact("API_KEY: abc123"), "API_KEY: [REDACTED]");
        assert_eq!(
            redact(r#"{"password": "hunter2", "name": "x"}"#),
            r#"{"password": "[REDACTED]", "name": "x"}"#
        );
        assert_eq!(redact("token=abc,"), "token=[REDACTED],");
    }

    #[test]
    fn bearer_values_and_credential_prefixes_are_removed() {
        assert_eq!(
            redact("Authorization: Bearer abcdef123456"),
            "Authorization: Bearer [REDACTED]"
        );
        assert_eq!(redact("use Bearer zzz now"), "use Bearer [REDACTED] now");
        let sk = "key sk-abcdefghijklmnopqrstuvwx here";
        assert_eq!(redact(sk), "key [REDACTED] here");
        assert_eq!(redact("AKIAABCDEFGHIJKLMNOP"), "[REDACTED]");
        assert_eq!(
            redact("(ghp_abcdefghijklmnopqrstuvwxyz0123)"),
            "([REDACTED])"
        );
    }

    #[test]
    fn private_key_blocks_are_removed_through_the_end_marker() {
        let text =
            "before\n-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----\nafter";
        assert_eq!(redact(text), "before\n[REDACTED]\nafter");
        let unterminated = "x -----BEGIN PRIVATE KEY-----\nabc";
        assert_eq!(redact(unterminated), "x [REDACTED]");
        let public = "-----BEGIN CERTIFICATE-----\nabc\n-----END CERTIFICATE-----";
        assert!(redact(public).contains("BEGIN CERTIFICATE"));
    }

    #[test]
    fn ordinary_text_is_untouched_and_redaction_is_idempotent() {
        let plain = "Move the file; the token count is high. sk-short\n\tdone";
        assert_eq!(redact(plain), plain);
        let secret = "password=abc sk-abcdefghijklmnopqrstuvwx";
        let once = redact(secret);
        assert_eq!(redact(&once), once);
    }
}
