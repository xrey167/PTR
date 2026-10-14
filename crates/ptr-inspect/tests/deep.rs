//! Attacker-deep trees must not overflow the stack in diff, clone, equality or drop.

use ptr_inspect::{diff, render_compact, render_json, InspectNode, RenderOptions};

const DEPTH: usize = 100_000;

fn deep(leaf: InspectNode) -> InspectNode {
    let mut node = leaf;
    for _ in 0..DEPTH {
        node = InspectNode::List(vec![node]);
    }
    node
}

/// Tear a tree down iteratively: dropping a 100_000-deep tree recursively would overflow.
fn dismantle(node: InspectNode) {
    let mut pending = vec![node];
    while let Some(node) = pending.pop() {
        if let InspectNode::List(items) = node {
            pending.extend(items);
        } else if let InspectNode::Map(entries) = node {
            for (key, value) in entries {
                pending.push(key);
                pending.push(value);
            }
        }
    }
}

fn on_small_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(1024 * 1024)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("deep tree must not overflow the stack");
}

#[test]
fn diff_of_deep_trees_sharing_the_prefix_does_not_overflow() {
    on_small_stack(|| {
        let old = deep(InspectNode::text("p"));
        let new = deep(InspectNode::text("q"));
        let changes = diff(&old, &new);
        assert_eq!(changes.len(), 1);
        let same = deep(InspectNode::text("p"));
        assert!(diff(&old, &same).is_empty());
        assert!(old == same && old != new);
        for tree in [old, new, same] {
            dismantle(tree);
        }
    });
}

#[test]
fn deep_map_keys_and_removed_subtrees_do_not_overflow() {
    on_small_stack(|| {
        let old = InspectNode::Map(vec![(deep(InspectNode::Int(1)), InspectNode::Int(1))]);
        let new = InspectNode::Map(vec![(deep(InspectNode::Int(1)), InspectNode::Int(2))]);
        assert_eq!(diff(&old, &new).len(), 1);
        let removed = InspectNode::List(vec![deep(InspectNode::Null)]);
        assert_eq!(diff(&removed, &InspectNode::List(Vec::new())).len(), 1);
        for tree in [old, new, removed] {
            dismantle(tree);
        }
    });
}

#[test]
fn renderers_stay_bounded_on_deep_trees() {
    on_small_stack(|| {
        let node = deep(InspectNode::Null);
        assert!(!render_json(&node).is_empty());
        assert!(!render_compact(&node, &RenderOptions::default()).is_empty());
        dismantle(node);
    });
}
