#[derive(Clone, Debug, PartialEq)]
pub enum InspectNode {
    Null,
    Bool(bool),
    Number(f64),
    Text(String),
    Struct {
        name: String,
        fields: Vec<(String, InspectNode)>,
    },
    List(Vec<InspectNode>),
    Redacted,
}

pub trait Inspectable {
    fn inspect(&self) -> InspectNode;
}

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
