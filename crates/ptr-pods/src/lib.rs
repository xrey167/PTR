use ptr_types::{CapabilityId, Effect, PodId, TypeId};
use std::marker::PhantomData;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodManifest {
    pub id: PodId,
    pub capabilities: Vec<CapabilityId>,
    pub accepts: Vec<TypeId>,
    pub produces: Vec<TypeId>,
    pub effects: Vec<Effect>,
    pub protocol_version: u32,
}

pub trait Pod {
    type Input;
    type Output;
    type Error;
    fn manifest(&self) -> &PodManifest;
    fn invoke(&self, input: Self::Input) -> Result<Self::Output, Self::Error>;
}

pub struct Ready;
pub struct Revoked;
pub struct PodLease<S> { pub pod_id: PodId, _state: PhantomData<S> }
impl PodLease<Ready> {
    pub fn new(pod_id: PodId) -> Self { Self { pod_id, _state: PhantomData } }
    pub fn revoke(self) -> PodLease<Revoked> { PodLease { pod_id: self.pod_id, _state: PhantomData } }
    pub fn can_invoke(&self) -> bool { true }
}
impl PodLease<Revoked> { pub fn can_invoke(&self) -> bool { false } }
