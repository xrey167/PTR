//! Structural difference between two inspection trees.
//!
//! Structs and enum variants are matched field by field by name, maps entry by entry by
//! key (rendered label, label collisions resolved by exact structural comparison), and lists
//! index by index: a list that grew reports `Added` for the new positions, one that shrank
//! reports `Removed` (no LCS alignment, so an insertion near the
//! front reports every later position as `Changed`). Trees nested deeper than
//! [`MAX_DIFF_DEPTH`] are compared as a whole and reported as one opaque change.

use std::collections::HashMap;

use crate::node::InspectNode;
use crate::path::{InspectPath, PathSegment};
use crate::render::key_label;

/// Nesting beyond which `diff` stops recursing.
pub const MAX_DIFF_DEPTH: usize = 128;

/// One difference between the two trees.
#[derive(Clone, Debug, PartialEq)]
pub struct Change {
    pub path: InspectPath,
    pub kind: ChangeKind,
}

/// What differs at a [`Change::path`].
#[derive(Clone, Debug, PartialEq)]
pub enum ChangeKind {
    /// Present only in the new tree.
    Added(InspectNode),
    /// Present only in the old tree.
    Removed(InspectNode),
    /// Same kind of node, different value (or a different enum variant). When either side is
    /// a redaction marker and the other is not, both sides are reported as `Redacted`, so a
    /// plaintext value that was protected on the other side is not exposed by the diff.
    Changed { from: InspectNode, to: InspectNode },
    /// The node kind, or the struct/enum name, differs.
    TypeChanged { from: InspectNode, to: InspectNode },
}

/// Differences that turn `old` into `new`, in a deterministic order: for each container,
/// matched children in the old tree's order, then children only in the new tree.
///
/// Two redaction markers compare equal whatever they hide. Callers who want key-based
/// redaction apply `redact_keys` to both trees first.
pub fn diff(old: &InspectNode, new: &InspectNode) -> Vec<Change> {
    let mut changes = Vec::new();
    let mut path = InspectPath::root();
    diff_at(old, new, &mut path, 0, &mut changes);
    changes
}

fn push(changes: &mut Vec<Change>, path: &InspectPath, kind: ChangeKind) {
    changes.push(Change {
        path: path.clone(),
        kind,
    });
}

fn scalar_equal(a: &InspectNode, b: &InspectNode) -> bool {
    match (a, b) {
        (InspectNode::Number(x), InspectNode::Number(y)) => {
            x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()) || x == y
        }
        _ => a == b,
    }
}

fn diff_at(
    old: &InspectNode,
    new: &InspectNode,
    path: &mut InspectPath,
    depth: usize,
    changes: &mut Vec<Change>,
) {
    if old.is_redacted() || new.is_redacted() {
        if !(old.is_redacted() && new.is_redacted()) {
            let kind = ChangeKind::Changed {
                from: InspectNode::Redacted,
                to: InspectNode::Redacted,
            };
            push(changes, path, kind);
        }
        return;
    }
    if depth >= MAX_DIFF_DEPTH {
        if old != new {
            let note = InspectNode::text("<subtree beyond diff depth limit>");
            let kind = ChangeKind::Changed {
                from: note.clone(),
                to: note,
            };
            push(changes, path, kind);
        }
        return;
    }
    match (old, new) {
        (
            InspectNode::Struct {
                name: old_name,
                fields: old_fields,
            },
            InspectNode::Struct {
                name: new_name,
                fields: new_fields,
            },
        ) if old_name == new_name => diff_fields(old_fields, new_fields, path, depth, changes),
        (
            InspectNode::Enum {
                name: old_name,
                variant: old_variant,
                fields: old_fields,
            },
            InspectNode::Enum {
                name: new_name,
                variant: new_variant,
                fields: new_fields,
            },
        ) if old_name == new_name => {
            if old_variant == new_variant {
                diff_fields(old_fields, new_fields, path, depth, changes);
            } else {
                let kind = ChangeKind::Changed {
                    from: old.clone(),
                    to: new.clone(),
                };
                push(changes, path, kind);
            }
        }
        (InspectNode::List(old_items), InspectNode::List(new_items)) => {
            for index in 0..old_items.len().max(new_items.len()) {
                path.push(PathSegment::Index(index));
                match (old_items.get(index), new_items.get(index)) {
                    (Some(a), Some(b)) => diff_at(a, b, path, depth + 1, changes),
                    (Some(a), None) => push(changes, path, ChangeKind::Removed(a.clone())),
                    (None, Some(b)) => push(changes, path, ChangeKind::Added(b.clone())),
                    (None, None) => {}
                }
                path.pop();
            }
        }
        (InspectNode::Map(old_entries), InspectNode::Map(new_entries)) => {
            diff_map(old_entries, new_entries, path, depth, changes);
        }
        _ if std::mem::discriminant(old) != std::mem::discriminant(new)
            || matches!(old, InspectNode::Struct { .. } | InspectNode::Enum { .. }) =>
        {
            let kind = ChangeKind::TypeChanged {
                from: old.clone(),
                to: new.clone(),
            };
            push(changes, path, kind);
        }
        _ => {
            if !scalar_equal(old, new) {
                let kind = ChangeKind::Changed {
                    from: old.clone(),
                    to: new.clone(),
                };
                push(changes, path, kind);
            }
        }
    }
}

fn diff_fields(
    old_fields: &[(String, InspectNode)],
    new_fields: &[(String, InspectNode)],
    path: &mut InspectPath,
    depth: usize,
    changes: &mut Vec<Change>,
) {
    for (name, old_value) in old_fields {
        path.push(PathSegment::Field(name.clone()));
        match new_fields.iter().find(|(n, _)| n == name) {
            Some((_, new_value)) => diff_at(old_value, new_value, path, depth + 1, changes),
            None => push(changes, path, ChangeKind::Removed(old_value.clone())),
        }
        path.pop();
    }
    for (name, new_value) in new_fields {
        if !old_fields.iter().any(|(n, _)| n == name) {
            path.push(PathSegment::Field(name.clone()));
            push(changes, path, ChangeKind::Added(new_value.clone()));
            path.pop();
        }
    }
}

fn diff_map(
    old_entries: &[(InspectNode, InspectNode)],
    new_entries: &[(InspectNode, InspectNode)],
    path: &mut InspectPath,
    depth: usize,
    changes: &mut Vec<Change>,
) {
    // `key_label` is bounded, so distinct keys can share a label. Bucket by label, then
    // resolve collisions by exact structural comparison; each new entry is matched once.
    let mut buckets: HashMap<String, Vec<usize>> = HashMap::new();
    for (index, (key, _)) in new_entries.iter().enumerate() {
        buckets.entry(key_label(key)).or_default().push(index);
    }
    let mut matched = vec![false; new_entries.len()];
    for (key, old_value) in old_entries {
        let label = key_label(key);
        let found = buckets.get(&label).and_then(|indexes| {
            indexes
                .iter()
                .copied()
                .find(|&i| !matched[i] && key_equal(key, &new_entries[i].0))
        });
        path.push(PathSegment::Key(label));
        match found {
            Some(i) => {
                matched[i] = true;
                diff_at(old_value, &new_entries[i].1, path, depth + 1, changes);
            }
            None => push(changes, path, ChangeKind::Removed(old_value.clone())),
        }
        path.pop();
    }
    for (index, (key, new_value)) in new_entries.iter().enumerate() {
        if !matched[index] {
            path.push(PathSegment::Key(key_label(key)));
            push(changes, path, ChangeKind::Added(new_value.clone()));
            path.pop();
        }
    }
}

/// Exact structural equality of two map keys (NaN equals NaN, as in `scalar_equal`). Two
/// redaction markers are equal, so redacted keys never reveal or compare hidden content.
fn key_equal(a: &InspectNode, b: &InspectNode) -> bool {
    match (a, b) {
        (InspectNode::List(x), InspectNode::List(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| key_equal(p, q))
        }
        (InspectNode::Map(x), InspectNode::Map(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y)
                    .all(|((k1, v1), (k2, v2))| key_equal(k1, k2) && key_equal(v1, v2))
        }
        (
            InspectNode::Struct {
                name: n1,
                fields: f1,
            },
            InspectNode::Struct {
                name: n2,
                fields: f2,
            },
        ) => n1 == n2 && fields_equal(f1, f2),
        (
            InspectNode::Enum {
                name: n1,
                variant: v1,
                fields: f1,
            },
            InspectNode::Enum {
                name: n2,
                variant: v2,
                fields: f2,
            },
        ) => n1 == n2 && v1 == v2 && fields_equal(f1, f2),
        _ => scalar_equal(a, b),
    }
}

fn fields_equal(a: &[(String, InspectNode)], b: &[(String, InspectNode)]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|((n1, v1), (n2, v2))| n1 == n2 && key_equal(v1, v2))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(fields: Vec<(&str, InspectNode)>) -> InspectNode {
        InspectNode::new_struct("S", fields)
    }

    #[test]
    fn equal_trees_have_no_changes() {
        let a = s(vec![
            ("x", InspectNode::Number(f64::NAN)),
            ("y", InspectNode::Redacted),
        ]);
        assert!(diff(&a, &a.clone()).is_empty());
    }

    #[test]
    fn field_changes_additions_and_removals() {
        let old = s(vec![
            ("a", InspectNode::Int(1)),
            ("gone", InspectNode::Null),
        ]);
        let new = s(vec![
            ("a", InspectNode::Int(2)),
            ("new", InspectNode::Bool(true)),
        ]);
        let changes = diff(&old, &new);
        let paths: Vec<String> = changes.iter().map(|c| c.path.to_string()).collect();
        assert_eq!(paths, vec!["$.a", "$.gone", "$.new"]);
        assert!(matches!(changes[0].kind, ChangeKind::Changed { .. }));
        assert!(matches!(changes[1].kind, ChangeKind::Removed(_)));
        assert!(matches!(changes[2].kind, ChangeKind::Added(_)));
    }

    #[test]
    fn redacted_vs_plaintext_never_exposes_the_plaintext() {
        let old = s(vec![("k", InspectNode::Redacted)]);
        let new = s(vec![("k", InspectNode::text("hunter2"))]);
        let changes = diff(&old, &new);
        assert_eq!(changes.len(), 1);
        assert!(!format!("{changes:?}").contains("hunter2"));
        assert!(diff(&old, &s(vec![("k", InspectNode::Redacted)])).is_empty());
    }

    #[test]
    fn depth_cutoff_is_opaque() {
        let mut a = InspectNode::text("p");
        let mut b = InspectNode::text("q");
        for _ in 0..(MAX_DIFF_DEPTH + 2) {
            a = InspectNode::List(vec![a]);
            b = InspectNode::List(vec![b]);
        }
        let changes = diff(&a, &b);
        assert_eq!(changes.len(), 1);
        assert!(!format!("{changes:?}").contains("\"p\""));
    }

    #[test]
    fn map_keys_sharing_a_truncated_label_are_distinct() {
        let key = |tail: &str| InspectNode::text(format!("{}{tail}", "a".repeat(5000)));
        assert_eq!(
            crate::render::key_label(&key("X")),
            crate::render::key_label(&key("Y"))
        );
        let old = InspectNode::Map(vec![
            (key("X"), InspectNode::Int(1)),
            (key("Y"), InspectNode::Int(2)),
        ]);
        let new = InspectNode::Map(vec![
            (key("X"), InspectNode::Int(1)),
            (key("Z"), InspectNode::Int(9)),
        ]);
        let changes = diff(&old, &new);
        assert_eq!(changes.len(), 2, "{changes:?}");
        assert!(matches!(
            changes[0].kind,
            ChangeKind::Removed(InspectNode::Int(2))
        ));
        assert!(matches!(
            changes[1].kind,
            ChangeKind::Added(InspectNode::Int(9))
        ));
        let changed = diff(
            &old,
            &InspectNode::Map(vec![
                (key("Y"), InspectNode::Int(2)),
                (key("X"), InspectNode::Int(5)),
            ]),
        );
        assert_eq!(changed.len(), 1, "{changed:?}");
        assert!(matches!(changed[0].kind, ChangeKind::Changed { .. }));
        assert!(diff(&old, &old.clone()).is_empty());
    }
}
