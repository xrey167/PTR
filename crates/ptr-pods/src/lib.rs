use ptr_protocol::TypedPayload;
use ptr_types::{CapabilityId, Effect, PodId, ProjectId, TypeId};
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodManifest {
    /// The one project this Pod serves. A Pod that served every project would
    /// make the project boundary advisory, since resolution is the only thing
    /// standing between a request and a Pod's data.
    pub project: ProjectId,
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

pub trait DynPod: Send + Sync {
    fn manifest(&self) -> &PodManifest;
    fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String>;
}

/// Pods addressed by project *and* id.
///
/// The project is part of the key rather than a filter applied afterwards, so
/// two projects may each register an `echo` Pod without one shadowing the
/// other, and no lookup path exists that forgets to check it.
#[derive(Default)]
pub struct PodRegistry {
    pods: BTreeMap<(ProjectId, PodId), Arc<dyn DynPod>>,
}

impl PodRegistry {
    pub fn register(&mut self, pod: Arc<dyn DynPod>) -> Option<Arc<dyn DynPod>> {
        let manifest = pod.manifest();
        let key = (manifest.project.clone(), manifest.id.clone());
        self.pods.insert(key, pod)
    }

    pub fn get(&self, project: &ProjectId, id: &PodId) -> Option<Arc<dyn DynPod>> {
        self.pods.get(&(project.clone(), id.clone())).cloned()
    }

    /// Resolve within one project only.
    ///
    /// A Pod belonging to another project is not a worse match here, it is not a
    /// match at all: the caller learns nothing about whether it exists.
    pub fn resolve(
        &self,
        project: &ProjectId,
        capability: &CapabilityId,
        input_type: &TypeId,
    ) -> Option<Arc<dyn DynPod>> {
        self.pods
            .iter()
            .find(|((pod_project, _), pod)| {
                let manifest = pod.manifest();
                pod_project == project
                    && manifest.capabilities.contains(capability)
                    && manifest.accepts.contains(input_type)
            })
            .map(|(_, pod)| pod.clone())
    }

    pub fn len(&self) -> usize {
        self.pods.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pods.is_empty()
    }
}

pub struct Ready;
pub struct Revoked;
pub struct PodLease<S> {
    pub pod_id: PodId,
    _state: PhantomData<S>,
}

impl PodLease<Ready> {
    pub fn new(pod_id: PodId) -> Self {
        Self {
            pod_id,
            _state: PhantomData,
        }
    }

    pub fn revoke(self) -> PodLease<Revoked> {
        PodLease {
            pod_id: self.pod_id,
            _state: PhantomData,
        }
    }

    pub fn can_invoke(&self) -> bool {
        true
    }
}

impl PodLease<Revoked> {
    pub fn can_invoke(&self) -> bool {
        false
    }
}

#[derive(Debug, PartialEq)]
pub enum LeaseInvokeError<E> {
    WrongPod,
    Pod(E),
}

/// Invoke a Pod only with a lease whose typestate is Ready.
///
/// A revoked lease cannot be passed to this function.
///
/// ~~~compile_fail
/// use ptr_pods::{invoke_with_lease, Pod, PodLease, PodManifest, Ready};
/// use ptr_types::{PodId, ProjectId};
///
/// struct Echo { manifest: PodManifest }
/// impl Pod for Echo {
///     type Input = String;
///     type Output = String;
///     type Error = ();
///     fn manifest(&self) -> &PodManifest { &self.manifest }
///     fn invoke(&self, input: String) -> Result<String, ()> { Ok(input) }
/// }
///
/// # let manifest = PodManifest {
/// #   project: ProjectId::from("p"), id: PodId::from("echo"),
/// #   capabilities: vec![], accepts: vec![],
/// #   produces: vec![], effects: vec![], protocol_version: 1
/// # };
/// let pod = Echo { manifest };
/// let ready: PodLease<Ready> = PodLease::new(PodId::from("echo"));
/// let revoked = ready.revoke();
/// let _ = invoke_with_lease(&pod, &revoked, "x".to_string());
/// ~~~
pub fn invoke_with_lease<P: Pod>(
    pod: &P,
    lease: &PodLease<Ready>,
    input: P::Input,
) -> Result<P::Output, LeaseInvokeError<P::Error>> {
    if lease.pod_id != pod.manifest().id {
        return Err(LeaseInvokeError::WrongPod);
    }
    pod.invoke(input).map_err(LeaseInvokeError::Pod)
}
