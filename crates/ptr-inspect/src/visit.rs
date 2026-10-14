//! Bounded, ordered traversal of an inspection tree.

use std::fmt;

use crate::node::InspectNode;
use crate::path::{InspectPath, PathSegment};
use crate::render::key_label;

/// Deepest nesting `walk` accepts whatever the caller asks for, so the recursion stays
/// within the stack.
pub const HARD_MAX_DEPTH: usize = 512;

/// Most nodes `walk` visits whatever the caller asks for.
pub const HARD_MAX_NODES: usize = 1 << 24;

/// Bounds on one traversal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WalkLimits {
    /// Deepest allowed nesting; the root is depth 0. Clamped to [`HARD_MAX_DEPTH`].
    pub max_depth: usize,
    /// Most nodes visited, counting the root and containers. Clamped to [`HARD_MAX_NODES`].
    pub max_nodes: usize,
}

impl Default for WalkLimits {
    fn default() -> Self {
        Self {
            max_depth: 64,
            max_nodes: 100_000,
        }
    }
}

/// A traversal exceeded one of its limits and was abandoned.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WalkError {
    DepthExceeded { limit: usize, path: InspectPath },
    NodeLimitExceeded { limit: usize, path: InspectPath },
}

impl fmt::Display for WalkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DepthExceeded { limit, path } => {
                write!(f, "inspection depth limit {limit} exceeded at {path}")
            }
            Self::NodeLimitExceeded { limit, path } => {
                write!(f, "inspection node limit {limit} exceeded at {path}")
            }
        }
    }
}

impl std::error::Error for WalkError {}

/// Callbacks for [`walk`]. Every method defaults to doing nothing.
///
/// Order is pre-order: `enter_*`, then the children in tree order, then `leave_*`. Leaves
/// (`Null`, `Bool`, `Number`, `Int`, `UInt`, `Text`, `Bytes`, `Redacted`, `RedactedWith`) get
/// a single `leaf` call. A map's children are its values, at `PathSegment::Key(rendered key)`;
/// keys are not visited separately. When `walk` fails it stops at once and does not call the
/// `leave_*` methods of the containers it was inside.
pub trait Visitor {
    fn enter_struct(&mut self, _path: &InspectPath, _name: &str) {}
    fn leave_struct(&mut self, _path: &InspectPath, _name: &str) {}
    fn enter_list(&mut self, _path: &InspectPath, _len: usize) {}
    fn leave_list(&mut self, _path: &InspectPath, _len: usize) {}
    fn enter_map(&mut self, _path: &InspectPath, _len: usize) {}
    fn leave_map(&mut self, _path: &InspectPath, _len: usize) {}
    fn enter_enum(&mut self, _path: &InspectPath, _name: &str, _variant: &str) {}
    fn leave_enum(&mut self, _path: &InspectPath, _name: &str, _variant: &str) {}
    fn leaf(&mut self, _path: &InspectPath, _node: &InspectNode) {}
}

/// Walk `root` in pre-order, calling `visitor`, within `limits`.
///
/// Returns the number of nodes visited.
pub fn walk(
    root: &InspectNode,
    visitor: &mut dyn Visitor,
    limits: &WalkLimits,
) -> Result<usize, WalkError> {
    let mut state = State {
        max_depth: limits.max_depth.min(HARD_MAX_DEPTH),
        max_nodes: limits.max_nodes.min(HARD_MAX_NODES),
        visited: 0,
        path: InspectPath::root(),
    };
    state.visit(root, visitor, 0)?;
    Ok(state.visited)
}

struct State {
    max_depth: usize,
    max_nodes: usize,
    visited: usize,
    path: InspectPath,
}

impl State {
    fn visit(
        &mut self,
        node: &InspectNode,
        visitor: &mut dyn Visitor,
        depth: usize,
    ) -> Result<(), WalkError> {
        if depth > self.max_depth {
            return Err(WalkError::DepthExceeded {
                limit: self.max_depth,
                path: self.path.clone(),
            });
        }
        if self.visited >= self.max_nodes {
            return Err(WalkError::NodeLimitExceeded {
                limit: self.max_nodes,
                path: self.path.clone(),
            });
        }
        self.visited += 1;
        match node {
            InspectNode::Struct { name, fields } => {
                visitor.enter_struct(&self.path, name);
                self.fields(fields, visitor, depth)?;
                visitor.leave_struct(&self.path, name);
            }
            InspectNode::Enum {
                name,
                variant,
                fields,
            } => {
                visitor.enter_enum(&self.path, name, variant);
                self.fields(fields, visitor, depth)?;
                visitor.leave_enum(&self.path, name, variant);
            }
            InspectNode::List(items) => {
                visitor.enter_list(&self.path, items.len());
                for (index, item) in items.iter().enumerate() {
                    self.path.push(PathSegment::Index(index));
                    self.visit(item, visitor, depth + 1)?;
                    self.path.pop();
                }
                visitor.leave_list(&self.path, items.len());
            }
            InspectNode::Map(entries) => {
                visitor.enter_map(&self.path, entries.len());
                for (key, value) in entries {
                    self.path.push(PathSegment::Key(key_label(key)));
                    self.visit(value, visitor, depth + 1)?;
                    self.path.pop();
                }
                visitor.leave_map(&self.path, entries.len());
            }
            InspectNode::Null
            | InspectNode::Bool(_)
            | InspectNode::Number(_)
            | InspectNode::Int(_)
            | InspectNode::UInt(_)
            | InspectNode::Text(_)
            | InspectNode::Bytes(_)
            | InspectNode::Redacted
            | InspectNode::RedactedWith(_) => visitor.leaf(&self.path, node),
        }
        Ok(())
    }

    fn fields(
        &mut self,
        fields: &[(String, InspectNode)],
        visitor: &mut dyn Visitor,
        depth: usize,
    ) -> Result<(), WalkError> {
        for (name, value) in fields {
            self.path.push(PathSegment::Field(name.clone()));
            self.visit(value, visitor, depth + 1)?;
            self.path.pop();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Trace(Vec<String>);

    impl Visitor for Trace {
        fn enter_struct(&mut self, path: &InspectPath, name: &str) {
            self.0.push(format!("enter {name} {path}"));
        }
        fn leave_struct(&mut self, path: &InspectPath, name: &str) {
            self.0.push(format!("leave {name} {path}"));
        }
        fn enter_list(&mut self, path: &InspectPath, len: usize) {
            self.0.push(format!("enter list({len}) {path}"));
        }
        fn leave_list(&mut self, path: &InspectPath, len: usize) {
            self.0.push(format!("leave list({len}) {path}"));
        }
        fn leaf(&mut self, path: &InspectPath, node: &InspectNode) {
            self.0.push(format!("leaf {} {path}", node.kind()));
        }
    }

    fn sample() -> InspectNode {
        InspectNode::new_struct(
            "S",
            vec![
                ("a", InspectNode::Int(1)),
                (
                    "l",
                    InspectNode::List(vec![InspectNode::Null, InspectNode::Bool(true)]),
                ),
            ],
        )
    }

    #[test]
    fn visits_in_pre_order_with_paths() {
        let mut trace = Trace::default();
        let count = walk(&sample(), &mut trace, &WalkLimits::default()).unwrap();
        assert_eq!(count, 5);
        assert_eq!(
            trace.0,
            vec![
                "enter S $",
                "leaf int $.a",
                "enter list(2) $.l",
                "leaf null $.l[0]",
                "leaf bool $.l[1]",
                "leave list(2) $.l",
                "leave S $",
            ]
        );
    }

    #[test]
    fn depth_limit_is_typed_and_names_the_path() {
        let limits = WalkLimits {
            max_depth: 1,
            max_nodes: 100,
        };
        let err = walk(&sample(), &mut Trace::default(), &limits).unwrap_err();
        assert_eq!(
            err,
            WalkError::DepthExceeded {
                limit: 1,
                path: InspectPath::root()
                    .child(PathSegment::Field("l".into()))
                    .child(PathSegment::Index(0)),
            }
        );
        assert!(err.to_string().contains("$.l[0]"));
    }

    #[test]
    fn node_limit_is_typed() {
        let limits = WalkLimits {
            max_depth: 10,
            max_nodes: 3,
        };
        let err = walk(&sample(), &mut Trace::default(), &limits).unwrap_err();
        assert!(matches!(err, WalkError::NodeLimitExceeded { limit: 3, .. }));
    }
}
