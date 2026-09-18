#[derive(Clone, Debug, PartialEq)]
pub enum InspectNode {
    Null,
    Bool(bool),
    Number(f64),
    Text(String),
    Struct { name: String, fields: Vec<(String, InspectNode)> },
    List(Vec<InspectNode>),
    Redacted,
}

pub trait Inspectable { fn inspect(&self) -> InspectNode; }

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Secret<T>(pub T);
impl<T> Inspectable for Secret<T> { fn inspect(&self) -> InspectNode { InspectNode::Redacted } }
