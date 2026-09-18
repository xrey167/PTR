use ptr_semdb::{SemanticDelta, SemanticHost};
fn main() {
    let mut host = SemanticHost::default();
    let mut delta = SemanticDelta::default();
    delta.upserts.insert("system.status".into(), "ready".into());
    let (revision, _) = host.apply_delta(delta);
    println!("ptrd scaffold ready at semantic revision {}", revision.0);
}
