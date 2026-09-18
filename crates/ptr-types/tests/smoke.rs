use ptr_types::{Effect, Probability};
#[test]
fn public_types_work() {
    assert_eq!(Probability::new(1.0).unwrap().get(), 1.0);
    assert_eq!(Effect::Read, Effect::Read);
}
