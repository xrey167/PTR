use ptr_inspect::{Inspectable,InspectNode,Secret};
#[test] fn secret_is_redacted(){ assert_eq!(Secret("token").inspect(),InspectNode::Redacted); }
