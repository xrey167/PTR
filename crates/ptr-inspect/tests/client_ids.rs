//! Client-controlled identifiers must not reach any rendering verbatim (Invariant 13).

use ptr_inspect::{
    render_compact, render_json, render_tree, InspectNode, Inspectable, RenderOptions,
};
use ptr_types::{
    EvidenceId, InvitationId, ProvenanceRef, RequestId, RetrievalKey, SessionId, TraceId,
};

const TOKEN: &str = "sk-live-9f8e7d6c5b4a39281706";

fn renders(node: &InspectNode) -> Vec<String> {
    let options = RenderOptions::default();
    vec![
        render_tree(node, &options),
        render_compact(node, &options),
        render_json(node),
        format!("{node:?}"),
    ]
}

fn assert_hidden(node: &InspectNode) {
    for text in renders(node) {
        assert!(!text.contains(TOKEN), "token leaked: {text}");
    }
}

#[test]
fn session_id_is_a_stable_digest_in_every_rendering() {
    let a = SessionId::from(TOKEN).inspect();
    assert_hidden(&a);
    assert_eq!(a, SessionId::from(TOKEN).inspect());
    assert_ne!(a, SessionId::from("sk-live-other").inspect());
    assert!(render_compact(&a, &RenderOptions::default()).contains("session:fnv1a64:"));
}

#[test]
fn other_client_controlled_strings_are_digests() {
    assert_hidden(&RequestId::from(TOKEN).inspect());
    assert_hidden(&TraceId::from(TOKEN).inspect());
    assert_hidden(&InvitationId::from(TOKEN).inspect());
    assert_hidden(&RetrievalKey::from(TOKEN).inspect());
    let provenance = ProvenanceRef {
        source: EvidenceId::from("ev-1"),
        note: Some(format!("header Authorization: {TOKEN}")),
    };
    assert_hidden(&provenance.inspect());
    assert_ne!(
        RequestId::from("a").inspect(),
        RequestId::from("b").inspect()
    );
}

#[test]
fn system_assigned_ids_stay_readable() {
    let text = render_compact(
        &EvidenceId::from("ev-1").inspect(),
        &RenderOptions::default(),
    );
    assert!(text.contains("ev-1"), "{text}");
}
