//! Locations inside an inspection tree.

use std::fmt;

/// One step from a container to a child.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PathSegment {
    /// A struct or enum field name.
    Field(String),
    /// A list position.
    Index(usize),
    /// A map entry, identified by the rendered key.
    Key(String),
}

/// A location in an inspection tree, rendered as `$.field[3]{"key"}`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct InspectPath {
    segments: Vec<PathSegment>,
}

impl InspectPath {
    /// The empty path naming the root.
    pub fn root() -> Self {
        Self::default()
    }

    /// The steps from the root.
    pub fn segments(&self) -> &[PathSegment] {
        &self.segments
    }

    /// Number of steps from the root.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Whether the path has no steps (same as [`InspectPath::is_root`]).
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Whether this is the root path.
    pub fn is_root(&self) -> bool {
        self.segments.is_empty()
    }

    /// A copy extended by one step.
    pub fn child(&self, segment: PathSegment) -> Self {
        let mut next = self.clone();
        next.segments.push(segment);
        next
    }

    pub(crate) fn push(&mut self, segment: PathSegment) {
        self.segments.push(segment);
    }

    pub(crate) fn pop(&mut self) {
        self.segments.pop();
    }
}

impl fmt::Display for InspectPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("$")?;
        for segment in &self.segments {
            match segment {
                PathSegment::Field(name) => write!(f, ".{name}")?,
                PathSegment::Index(index) => write!(f, "[{index}]")?,
                PathSegment::Key(key) => write!(f, "{{{key}}}")?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_covers_every_segment_kind() {
        let path = InspectPath::root()
            .child(PathSegment::Field("a".into()))
            .child(PathSegment::Index(3))
            .child(PathSegment::Key("\"k\"".into()));
        assert_eq!(path.to_string(), "$.a[3]{\"k\"}");
        assert_eq!(InspectPath::root().to_string(), "$");
        assert!(InspectPath::root().is_root());
        assert_eq!(path.len(), 3);
    }
}
