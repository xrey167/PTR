use ptr_inspect::{InspectNode, Inspectable, Secret};
#[test]
fn secret_is_redacted() {
    assert_eq!(Secret("token").inspect(), InspectNode::Redacted);
}
