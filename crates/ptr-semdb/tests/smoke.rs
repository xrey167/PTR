use ptr_semdb::{SemanticDelta, SemanticHost};
#[test]
fn snapshot_tracks_revision() {
    let mut h = SemanticHost::default();
    let mut d = SemanticDelta::default();
    d.upserts.insert("x".into(), "1".into());
    let (r, _) = h.apply_delta(d).unwrap();
    assert_eq!(h.snapshot().revision, r);
}
