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

/// Dropping a 100_000-deep tree recursively would overflow; `InspectNode` tears down iteratively.
fn dismantle(node: InspectNode) {
    drop(node);
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

fn on_tiny_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("renderer must not overflow the stack");
}

#[test]
fn unbounded_max_depth_is_clamped_in_tree_and_compact_renderers() {
    on_tiny_stack(|| {
        let node = deep(InspectNode::Null);
        let options = RenderOptions {
            max_depth: usize::MAX,
            ..RenderOptions::default()
        };
        let tree = ptr_inspect::render_tree(&node, &options);
        assert!(tree.contains('…'));
        assert!(render_compact(&node, &options).contains('…'));
        dismantle(node);
    });
}

#[test]
fn map_sorted_is_deterministic_for_keys_sharing_a_truncated_label() {
    let prefix = "k".repeat(5000);
    let keys: Vec<InspectNode> = ["a", "b", "c", "d"]
        .iter()
        .map(|s| InspectNode::text(format!("{prefix}{s}")))
        .collect();
    let build = |order: &[usize]| {
        let entries = order
            .iter()
            .map(|&i| (keys[i].clone(), InspectNode::Int(i as i128)));
        InspectNode::map_sorted(entries.collect())
    };
    let first = build(&[0, 1, 2, 3]);
    for order in [[3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]] {
        assert_eq!(build(&order), first);
    }
}

#[test]
fn hash_set_inspection_is_deterministic_for_items_sharing_a_truncated_label() {
    use ptr_inspect::Inspectable;
    let prefix = "k".repeat(5000);
    let items: Vec<String> = ["a", "b", "c", "d", "e", "f"]
        .iter()
        .map(|s| format!("{prefix}{s}"))
        .collect();
    // Every HashSet gets its own RandomState, so iteration order differs between instances.
    let first = items
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .inspect();
    for _ in 0..32 {
        let set: std::collections::HashSet<String> = items.iter().rev().cloned().collect();
        assert_eq!(set.inspect(), first);
    }
}
