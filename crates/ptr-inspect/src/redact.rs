//! Wrappers and passes that keep secret, private and raw content out of inspection output
//! (Invariant 13).

use crate::node::{
    fingerprint_label, fnv1a64, InspectNode, Inspectable, LengthClass, RedactionHint,
};

/// Nesting beyond which [`redact_keys`] replaces a subtree with `Redacted` (fail closed).
const MAX_REDACT_DEPTH: usize = 256;

/// Hard cap on a [`RawEvidence`] preview, whatever the policy asks for.
pub const MAX_EVIDENCE_PREVIEW_BYTES: usize = 256;

/// Key names [`redact_sensitive`] treats as sensitive when they appear inside a field or map
/// key name (case-insensitive substring match).
pub const DEFAULT_SENSITIVE_KEYS: &[&str] = &[
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "private_key",
    "credential",
    "authorization",
    "cookie",
    "session_key",
];

/// A value that never renders: `inspect` yields [`InspectNode::Redacted`] and
/// `Debug` prints `Secret([redacted])` whatever `T` is, so a secret nested in a
/// derived `Debug` or a tracing field cannot print its plaintext. The field is
/// private, so the value is reached only through [`Secret::expose`] or
/// [`Secret::into_inner`], calls a reviewer can search for.
#[derive(Clone, Eq, PartialEq)]
pub struct Secret<T>(T);

impl<T> Secret<T> {
    /// Wrap a value so inspection and debug formatting redact it.
    pub fn new(value: T) -> Self {
        Self(value)
    }

    /// Borrow the wrapped value without redaction.
    pub fn expose(&self) -> &T {
        &self.0
    }

    /// Consume the wrapper and return the value without redaction.
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> std::fmt::Debug for Secret<T> {
    /// Write `Secret([redacted])`, propagating any formatter error.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Secret([redacted])")
    }
}

impl<T> Inspectable for Secret<T> {
    fn inspect(&self) -> InspectNode {
        InspectNode::Redacted
    }
}

/// A private value (user data, not a credential). Inspection yields a redaction marker that
/// names the type and, if built with [`Private::with_length_class`], a coarse size class;
/// never the content. `Debug` prints `Private([redacted])`.
#[derive(Clone, Eq, PartialEq)]
pub struct Private<T> {
    value: T,
    length_class: Option<LengthClass>,
}

impl<T> Private<T> {
    /// Wrap a value; the marker carries the type name only.
    pub fn new(value: T) -> Self {
        Self {
            value,
            length_class: None,
        }
    }

    /// Borrow the wrapped value without redaction.
    pub fn expose(&self) -> &T {
        &self.value
    }

    /// Consume the wrapper and return the value without redaction.
    pub fn into_inner(self) -> T {
        self.value
    }
}

impl<T: AsRef<[u8]>> Private<T> {
    /// Wrap a value and record its size class, fixed at construction.
    pub fn with_length_class(value: T) -> Self {
        let class = LengthClass::from_len(value.as_ref().len());
        Self {
            value,
            length_class: Some(class),
        }
    }
}

impl<T> std::fmt::Debug for Private<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Private([redacted])")
    }
}

impl<T> Inspectable for Private<T> {
    fn inspect(&self) -> InspectNode {
        InspectNode::RedactedWith(RedactionHint::new(
            std::any::type_name::<T>(),
            self.length_class,
        ))
    }
}

/// Whether a [`RawEvidence`] preview of its content may be shown.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PreviewPolicy {
    /// Show length and fingerprint only.
    #[default]
    Deny,
    /// Also show up to `max_bytes` leading bytes (at most [`MAX_EVIDENCE_PREVIEW_BYTES`]) as
    /// lossy UTF-8.
    Allow { max_bytes: usize },
}

/// Raw source bytes (Invariant 1) as seen by inspection: length and an FNV-1a 64 fingerprint,
/// plus a truncated preview only under [`PreviewPolicy::Allow`].
///
/// The fingerprint is not a cryptographic digest and does not hide low-entropy content from
/// guessing; evidence that is itself a credential belongs in [`Secret`]. `Debug` prints
/// the length only.
#[derive(Clone, Eq, PartialEq)]
pub struct RawEvidence<T> {
    value: T,
    policy: PreviewPolicy,
}

impl<T> RawEvidence<T> {
    /// Wrap evidence with previews denied.
    pub fn new(value: T) -> Self {
        Self::with_policy(value, PreviewPolicy::Deny)
    }

    /// Wrap evidence under an explicit preview policy.
    pub fn with_policy(value: T, policy: PreviewPolicy) -> Self {
        Self { value, policy }
    }

    /// Borrow the wrapped evidence.
    pub fn expose(&self) -> &T {
        &self.value
    }

    /// Consume the wrapper and return the evidence.
    pub fn into_inner(self) -> T {
        self.value
    }

    /// The preview policy in force.
    pub fn policy(&self) -> PreviewPolicy {
        self.policy
    }
}

impl<T: AsRef<[u8]>> std::fmt::Debug for RawEvidence<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RawEvidence {{ len: {}, content: [withheld] }}",
            self.value.as_ref().len()
        )
    }
}

impl<T: AsRef<[u8]>> Inspectable for RawEvidence<T> {
    fn inspect(&self) -> InspectNode {
        let bytes = self.value.as_ref();
        let mut fields = vec![
            ("len", InspectNode::UInt(bytes.len() as u128)),
            (
                "fingerprint",
                InspectNode::text(fingerprint_label(fnv1a64(bytes))),
            ),
        ];
        if let PreviewPolicy::Allow { max_bytes } = self.policy {
            let keep = max_bytes.min(MAX_EVIDENCE_PREVIEW_BYTES).min(bytes.len());
            fields.push((
                "preview",
                InspectNode::text(String::from_utf8_lossy(&bytes[..keep]).into_owned()),
            ));
            fields.push(("truncated", InspectNode::Bool(keep < bytes.len())));
        }
        InspectNode::new_struct("RawEvidence", fields)
    }
}

/// Replace the value of every struct/enum field and text-keyed map entry whose name equals
/// one of `keys` (case-insensitive) with [`InspectNode::Redacted`]. Names stay visible.
/// Subtrees nested deeper than 256 levels are redacted whole.
pub fn redact_keys(node: &InspectNode, keys: &[&str]) -> InspectNode {
    let wanted: Vec<String> = keys.iter().map(|k| k.to_lowercase()).collect();
    redact(
        node,
        &|name| wanted.iter().any(|k| *k == name.to_lowercase()),
        0,
    )
}

/// Like [`redact_keys`], but a name matches when it contains any of `fragments`
/// (case-insensitive), so `db_password` matches `password`.
pub fn redact_keys_containing(node: &InspectNode, fragments: &[&str]) -> InspectNode {
    let wanted: Vec<String> = fragments
        .iter()
        .filter(|f| !f.is_empty())
        .map(|f| f.to_lowercase())
        .collect();
    redact(
        node,
        &|name| {
            let name = name.to_lowercase();
            wanted.iter().any(|f| name.contains(f.as_str()))
        },
        0,
    )
}

/// [`redact_keys_containing`] with [`DEFAULT_SENSITIVE_KEYS`].
pub fn redact_sensitive(node: &InspectNode) -> InspectNode {
    redact_keys_containing(node, DEFAULT_SENSITIVE_KEYS)
}

fn redact(node: &InspectNode, hit: &dyn Fn(&str) -> bool, depth: usize) -> InspectNode {
    if depth > MAX_REDACT_DEPTH {
        return InspectNode::Redacted;
    }
    let fields = |fields: &[(String, InspectNode)]| -> Vec<(String, InspectNode)> {
        fields
            .iter()
            .map(|(name, value)| {
                let value = if hit(name) {
                    InspectNode::Redacted
                } else {
                    redact(value, hit, depth + 1)
                };
                (name.clone(), value)
            })
            .collect()
    };
    match node {
        InspectNode::Struct { name, fields: f } => InspectNode::Struct {
            name: name.clone(),
            fields: fields(f),
        },
        InspectNode::Enum {
            name,
            variant,
            fields: f,
        } => InspectNode::Enum {
            name: name.clone(),
            variant: variant.clone(),
            fields: fields(f),
        },
        InspectNode::List(items) => {
            InspectNode::List(items.iter().map(|i| redact(i, hit, depth + 1)).collect())
        }
        InspectNode::Map(entries) => InspectNode::Map(
            entries
                .iter()
                .map(|(key, value)| {
                    let value = match key {
                        InspectNode::Text(name) if hit(name) => InspectNode::Redacted,
                        _ => redact(value, hit, depth + 1),
                    };
                    (redact(key, hit, depth + 1), value)
                })
                .collect(),
        ),
        InspectNode::Null
        | InspectNode::Bool(_)
        | InspectNode::Number(_)
        | InspectNode::Int(_)
        | InspectNode::UInt(_)
        | InspectNode::Text(_)
        | InspectNode::Bytes(_)
        | InspectNode::Redacted
        | InspectNode::RedactedWith(_) => node.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_marks_type_and_size_class_but_not_content() {
        let private = Private::with_length_class(String::from("alice@example.com"));
        let InspectNode::RedactedWith(hint) = private.inspect() else {
            panic!("expected hint");
        };
        assert!(hint.type_name().contains("String"));
        assert_eq!(hint.length_class(), Some(LengthClass::Medium));
        assert_eq!(Private::new(1u8).inspect(), Private::new(2u8).inspect());
        assert_eq!(format!("{private:?}"), "Private([redacted])");
        assert_eq!(format!("{private:#?}"), "Private([redacted])");
    }

    #[test]
    fn raw_evidence_denies_preview_by_default() {
        let evidence = RawEvidence::new(b"top secret bytes".to_vec());
        let rendered = format!("{:?}", evidence.inspect());
        assert!(!rendered.contains("top secret"));
        assert!(rendered.contains("fnv1a64:"));
        assert_eq!(
            format!("{evidence:?}"),
            "RawEvidence { len: 16, content: [withheld] }"
        );
    }

    #[test]
    fn raw_evidence_preview_is_truncated_when_allowed() {
        let evidence =
            RawEvidence::with_policy("hello world", PreviewPolicy::Allow { max_bytes: 5 });
        let node = evidence.inspect();
        let InspectNode::Struct { fields, .. } = node else {
            panic!("expected struct");
        };
        assert!(fields.contains(&("preview".to_owned(), InspectNode::text("hello"))));
        assert!(fields.contains(&("truncated".to_owned(), InspectNode::Bool(true))));
    }

    #[test]
    fn raw_evidence_preview_is_capped_whatever_the_policy() {
        let big = vec![b'x'; 1000];
        let evidence = RawEvidence::with_policy(
            big,
            PreviewPolicy::Allow {
                max_bytes: usize::MAX,
            },
        );
        let InspectNode::Struct { fields, .. } = evidence.inspect() else {
            panic!("expected struct");
        };
        let preview = fields.iter().find(|(n, _)| n == "preview").map(|(_, v)| v);
        assert_eq!(
            preview,
            Some(&InspectNode::text("x".repeat(MAX_EVIDENCE_PREVIEW_BYTES)))
        );
    }

    #[test]
    fn redact_keys_is_case_insensitive_and_reaches_maps_and_enums() {
        let tree = InspectNode::new_struct(
            "Cfg",
            vec![
                ("User", InspectNode::text("bob")),
                ("PASSWORD", InspectNode::text("pw")),
                (
                    "headers",
                    InspectNode::Map(vec![
                        (InspectNode::text("Token"), InspectNode::text("t")),
                        (InspectNode::text("accept"), InspectNode::text("*/*")),
                    ]),
                ),
                (
                    "auth",
                    InspectNode::variant("Auth", "Basic", vec![("secret", InspectNode::text("s"))]),
                ),
            ],
        );
        let cleaned = redact_keys(&tree, &["password", "token", "secret"]);
        let dump = format!("{cleaned:?}");
        assert!(dump.contains("bob") && dump.contains("*/*"));
        for leaked in ["\"pw\"", "\"t\"", "\"s\""] {
            assert!(!dump.contains(leaked), "{dump}");
        }
    }

    #[test]
    fn redact_sensitive_matches_fragments() {
        let tree = InspectNode::new_struct(
            "Db",
            vec![
                ("db_password", InspectNode::text("pw")),
                ("host", InspectNode::text("h")),
            ],
        );
        let cleaned = redact_sensitive(&tree);
        assert_eq!(
            cleaned,
            InspectNode::new_struct(
                "Db",
                vec![
                    ("db_password", InspectNode::Redacted),
                    ("host", InspectNode::text("h"))
                ]
            )
        );
        assert_eq!(redact_keys(&tree, &["password"]), tree);
    }

    #[test]
    fn overly_deep_trees_are_redacted_whole() {
        let mut node = InspectNode::text("deep secret");
        for _ in 0..(MAX_REDACT_DEPTH + 5) {
            node = InspectNode::List(vec![node]);
        }
        let cleaned = redact_keys(&node, &[]);
        assert!(!format!("{cleaned:?}").contains("deep secret"));
    }
}
