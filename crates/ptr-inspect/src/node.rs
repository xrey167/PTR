//! The generic inspection tree and the trait that produces it.

use crate::render;

/// Longest byte prefix a [`BytesPreview`] keeps.
pub const MAX_BYTES_PREVIEW: usize = 32;

/// A typed, owned snapshot of a value's shape, safe to render or diff.
///
/// Integers are kept lossless (`Int` holds every `i64`/`i128`, `UInt` every `u64`/`u128`);
/// `Number` is for floating point only. `Redacted` and `RedactedWith` never carry content.
#[derive(Clone, Debug, PartialEq)]
pub enum InspectNode {
    Null,
    Bool(bool),
    /// A floating-point value. Non-finite values are allowed and render as strings in JSON.
    Number(f64),
    /// A signed integer, lossless for every `i64` and `i128`.
    Int(i128),
    /// An unsigned integer, lossless for every `u64` and `u128`.
    UInt(u128),
    Text(String),
    /// Byte string kept as length, fingerprint and a bounded prefix.
    Bytes(BytesPreview),
    Struct {
        name: String,
        fields: Vec<(String, InspectNode)>,
    },
    List(Vec<InspectNode>),
    /// Key/value pairs; keys are nodes. Order is the producer's order.
    Map(Vec<(InspectNode, InspectNode)>),
    /// One variant of an enum. Tuple variants use the field names `0`, `1`, ...
    Enum {
        name: String,
        variant: String,
        fields: Vec<(String, InspectNode)>,
    },
    Redacted,
    /// A redacted value that still says what kind of value was withheld.
    RedactedWith(RedactionHint),
}

/// Length, fingerprint and a bounded prefix of a byte string.
///
/// The fingerprint is FNV-1a 64: deterministic and not a cryptographic digest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BytesPreview {
    len: usize,
    fingerprint: u64,
    head: Vec<u8>,
}

impl BytesPreview {
    /// Summarise `bytes`, keeping at most [`MAX_BYTES_PREVIEW`] leading bytes.
    pub fn new(bytes: &[u8]) -> Self {
        let keep = bytes.len().min(MAX_BYTES_PREVIEW);
        Self {
            len: bytes.len(),
            fingerprint: fnv1a64(bytes),
            head: bytes[..keep].to_vec(),
        }
    }

    /// Full length of the summarised bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the summarised byte string was empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// FNV-1a 64 fingerprint of the full byte string.
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// The kept prefix.
    pub fn head(&self) -> &[u8] {
        &self.head
    }

    /// Whether the prefix is shorter than the full byte string.
    pub fn is_truncated(&self) -> bool {
        self.head.len() < self.len
    }

    /// Lowercase hex of the kept prefix.
    pub fn head_hex(&self) -> String {
        hex(&self.head)
    }
}

/// Coarse, stable size class of a withheld value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LengthClass {
    Empty,
    /// 1 to 15 bytes.
    Short,
    /// 16 to 255 bytes.
    Medium,
    /// 256 bytes or more.
    Long,
}

impl LengthClass {
    /// Classify a byte length.
    pub fn from_len(len: usize) -> Self {
        match len {
            0 => Self::Empty,
            1..=15 => Self::Short,
            16..=255 => Self::Medium,
            _ => Self::Long,
        }
    }

    /// Stable label such as `1-15`.
    pub fn label(self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::Short => "1-15",
            Self::Medium => "16-255",
            Self::Long => "256+",
        }
    }
}

/// What a redacted node may say about the value it replaced.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RedactionHint {
    type_name: &'static str,
    length_class: Option<LengthClass>,
}

impl RedactionHint {
    /// A hint naming the withheld value's type and, optionally, its size class.
    pub fn new(type_name: &'static str, length_class: Option<LengthClass>) -> Self {
        Self {
            type_name,
            length_class,
        }
    }

    /// The Rust type name of the withheld value.
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// The size class, if the wrapper recorded one.
    pub fn length_class(&self) -> Option<LengthClass> {
        self.length_class
    }
}

impl InspectNode {
    /// A struct node from borrowed field names.
    pub fn new_struct(name: &str, fields: Vec<(&str, InspectNode)>) -> Self {
        Self::Struct {
            name: name.to_owned(),
            fields: fields.into_iter().map(|(n, v)| (n.to_owned(), v)).collect(),
        }
    }

    /// An enum variant with fields.
    pub fn variant(name: &str, variant: &str, fields: Vec<(&str, InspectNode)>) -> Self {
        Self::Enum {
            name: name.to_owned(),
            variant: variant.to_owned(),
            fields: fields.into_iter().map(|(n, v)| (n.to_owned(), v)).collect(),
        }
    }

    /// An enum variant without fields.
    pub fn unit_variant(name: &str, variant: &str) -> Self {
        Self::variant(name, variant, Vec::new())
    }

    /// A byte-string node (length, fingerprint, bounded prefix).
    pub fn bytes(bytes: &[u8]) -> Self {
        Self::Bytes(BytesPreview::new(bytes))
    }

    /// A text node.
    pub fn text(value: impl Into<String>) -> Self {
        Self::Text(value.into())
    }

    /// A map node whose entries are sorted by the rendered key, so unordered sources
    /// (for example a `HashMap`) produce a deterministic tree. The sort is stable.
    pub fn map_sorted(mut entries: Vec<(InspectNode, InspectNode)>) -> Self {
        entries.sort_by_cached_key(|(k, _)| render::key_label(k));
        Self::Map(entries)
    }

    /// Whether this node is a redaction marker of either kind.
    pub fn is_redacted(&self) -> bool {
        matches!(self, Self::Redacted | Self::RedactedWith(_))
    }

    /// Short kind name, for example `struct` or `uint`.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Bool(_) => "bool",
            Self::Number(_) => "number",
            Self::Int(_) => "int",
            Self::UInt(_) => "uint",
            Self::Text(_) => "text",
            Self::Bytes(_) => "bytes",
            Self::Struct { .. } => "struct",
            Self::List(_) => "list",
            Self::Map(_) => "map",
            Self::Enum { .. } => "enum",
            Self::Redacted | Self::RedactedWith(_) => "redacted",
        }
    }
}

/// Produces an [`InspectNode`] for a value. Implementations must not put secret or private
/// content into the node (Invariant 13); wrap such values in `Secret`, `Private` or
/// `RawEvidence`.
pub trait Inspectable {
    fn inspect(&self) -> InspectNode;
}

/// FNV-1a 64-bit hash. Deterministic, std-only, not cryptographic.
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Fingerprint label, `fnv1a64:` plus 16 hex digits.
pub(crate) fn fingerprint_label(fingerprint: u64) -> String {
    format!("fnv1a64:{fingerprint:016x}")
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from(DIGITS[usize::from(byte >> 4)]));
        out.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_matches_published_vectors() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn bytes_preview_is_bounded_and_fingerprints_everything() {
        let data = vec![0xabu8; 100];
        let preview = BytesPreview::new(&data);
        assert_eq!(preview.len(), 100);
        assert_eq!(preview.head().len(), MAX_BYTES_PREVIEW);
        assert!(preview.is_truncated());
        assert_eq!(preview.fingerprint(), fnv1a64(&data));
        assert_eq!(&preview.head_hex()[..4], "abab");
        assert!(!BytesPreview::new(b"ab").is_truncated());
        assert!(BytesPreview::new(b"").is_empty());
    }

    #[test]
    fn length_classes_have_stable_boundaries() {
        assert_eq!(LengthClass::from_len(0), LengthClass::Empty);
        assert_eq!(LengthClass::from_len(15), LengthClass::Short);
        assert_eq!(LengthClass::from_len(16), LengthClass::Medium);
        assert_eq!(LengthClass::from_len(255), LengthClass::Medium);
        assert_eq!(LengthClass::from_len(256), LengthClass::Long);
        assert_eq!(LengthClass::Short.label(), "1-15");
    }

    #[test]
    fn map_sorted_orders_by_rendered_key() {
        let node = InspectNode::map_sorted(vec![
            (InspectNode::text("b"), InspectNode::Int(2)),
            (InspectNode::text("a"), InspectNode::Int(1)),
        ]);
        assert_eq!(
            node,
            InspectNode::Map(vec![
                (InspectNode::text("a"), InspectNode::Int(1)),
                (InspectNode::text("b"), InspectNode::Int(2)),
            ])
        );
    }
}
