use super::{
    CapabilityId, Digest, Generation, NamespaceId, NodeId, PeerId, PodId, ProjectId, RegionId,
    RequestId, ScopeId, SessionId, TraceId, TypeId, ZoneId,
};
use std::fmt;

const SCHEME: &str = "ptr://";

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PodAddress {
    pub project: ProjectId,
    pub namespace: NamespaceId,
    pub pod_id: PodId,
}

impl PodAddress {
    pub fn new(
        project: ProjectId,
        namespace: NamespaceId,
        pod_id: PodId,
    ) -> Result<Self, ResolveError> {
        let address = Self {
            project,
            namespace,
            pod_id,
        };
        address.validate()?;
        Ok(address)
    }

    pub fn validate(&self) -> Result<(), ResolveError> {
        for (field, value) in [
            ("project", &self.project.0),
            ("namespace", &self.namespace.0),
            ("pod_id", &self.pod_id.0),
        ] {
            if value.is_empty() || !value.bytes().all(is_segment_byte) {
                return Err(ResolveError::InvalidAddress { field });
            }
        }
        Ok(())
    }

    pub fn canonical_uri(&self) -> Result<String, ResolveError> {
        self.validate()?;
        Ok(format!(
            "{SCHEME}{}/{}/{}",
            self.project, self.namespace, self.pod_id
        ))
    }

    pub fn parse(value: &str) -> Result<Self, ResolveError> {
        let rest = value
            .strip_prefix(SCHEME)
            .ok_or(ResolveError::InvalidScheme)?;
        let mut parts = rest.split('/');
        let project = parts.next().unwrap_or_default();
        let namespace = parts.next().unwrap_or_default();
        let pod_id = parts.next().unwrap_or_default();
        if parts.next().is_some() || project.is_empty() || namespace.is_empty() || pod_id.is_empty()
        {
            return Err(ResolveError::InvalidAddress { field: "address" });
        }
        Self::new(project.into(), namespace.into(), pod_id.into())
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PodRevisionAddress {
    pub address: PodAddress,
    pub semantic_revision: Digest,
    pub generation: Generation,
}

impl PodRevisionAddress {
    pub fn validate(&self) -> Result<(), ResolveError> {
        self.address.validate()?;
        if self.semantic_revision == [0; 32] {
            return Err(ResolveError::MissingBinding {
                field: "semantic_revision",
            });
        }
        Ok(())
    }

    pub fn canonical_uri(&self) -> Result<String, ResolveError> {
        self.validate()?;
        Ok(format!(
            "{}@revision:{}#generation:{}",
            self.address.canonical_uri()?,
            hex(&self.semantic_revision),
            self.generation.0
        ))
    }

    pub fn parse(value: &str) -> Result<Self, ResolveError> {
        let (base, fragment) =
            value
                .split_once("#generation:")
                .ok_or(ResolveError::InvalidAddress {
                    field: "generation",
                })?;
        let generation = fragment
            .parse::<u64>()
            .map_err(|_| ResolveError::InvalidAddress {
                field: "generation",
            })?;
        let (base, revision) =
            base.split_once("@revision:")
                .ok_or(ResolveError::InvalidAddress {
                    field: "semantic_revision",
                })?;
        let semantic_revision = parse_hex_digest(revision)?;
        let address = PodAddress::parse(base)?;
        let parsed = Self {
            address,
            semantic_revision,
            generation: Generation(generation),
        };
        parsed.validate()?;
        Ok(parsed)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PodTransport {
    InProcess,
    LocalWorker,
    IrohQuic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeshRouteKind {
    Direct,
    Relay,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshEndpointBinding {
    pub network_id: super::NetworkId,
    pub peer_id: PeerId,
    pub route: MeshRouteKind,
    pub relay_id: Option<PeerId>,
    pub membership_generation: Generation,
    pub membership_revision: super::Revision,
}

impl MeshEndpointBinding {
    pub fn validate(&self) -> Result<(), ResolveError> {
        if self.network_id.0.is_empty() || self.peer_id.0.is_empty() {
            return Err(ResolveError::MissingBinding {
                field: "mesh_identity",
            });
        }
        if self.membership_generation.0 == 0 || self.membership_revision.0 == 0 {
            return Err(ResolveError::StaleRoute);
        }
        match (&self.route, &self.relay_id) {
            (MeshRouteKind::Direct, None) => Ok(()),
            (MeshRouteKind::Direct, Some(_)) => Err(ResolveError::ConflictingEndpoint),
            (MeshRouteKind::Relay, Some(relay)) if !relay.0.is_empty() => Ok(()),
            (MeshRouteKind::Relay, _) => Err(ResolveError::MissingBinding { field: "relay_id" }),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodEndpoint {
    pub pod: PodRevisionAddress,
    pub artifact_id: super::ArtifactId,
    pub artifact_active: bool,
    pub node_id: NodeId,
    pub device_id: Option<super::DeviceId>,
    pub region: RegionId,
    pub zone: ZoneId,
    pub peer_id: PeerId,
    pub transport: PodTransport,
    pub endpoint_epoch: u64,
    pub fencing_token: u128,
    pub available_vram_bytes: u64,
    pub healthy: bool,
    pub load_bps: u16,
    pub capabilities: Vec<CapabilityId>,
    pub accepts: Vec<TypeId>,
    pub produces: Vec<TypeId>,
    pub mesh: Option<MeshEndpointBinding>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RouteConstraints {
    pub artifact_id: Option<super::ArtifactId>,
    pub capability: Option<CapabilityId>,
    pub input_type: Option<TypeId>,
    pub output_type: Option<TypeId>,
    pub region: Option<RegionId>,
    pub zone: Option<ZoneId>,
    pub required_vram_bytes: Option<u64>,
    pub min_epoch: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodRoute {
    pub source: Option<PodAddress>,
    pub destination: PodRevisionAddress,
    pub artifact_id: super::ArtifactId,
    pub endpoint: PodEndpoint,
    pub capability: CapabilityId,
    pub input_type: TypeId,
    pub output_type: TypeId,
    pub trace_id: TraceId,
    pub request_id: RequestId,
    pub deadline: super::Timestamp,
    pub hop_limit: u16,
    pub visited: Vec<PodAddress>,
    pub placement_epoch: u64,
    pub fencing_token: u128,
}

impl PodRoute {
    pub fn validate(&self) -> Result<(), ResolveError> {
        self.destination.validate()?;
        if self.hop_limit == 0 {
            return Err(ResolveError::HopLimitExceeded);
        }
        if self.visited.contains(&self.destination.address) {
            return Err(ResolveError::RouteCycle);
        }
        if let Some(source) = &self.source {
            if source.project != self.destination.address.project
                || source.namespace != self.destination.address.namespace
            {
                return Err(ResolveError::ScopeMismatch);
            }
        }
        if self.endpoint.pod != self.destination {
            return Err(ResolveError::EndpointMismatch);
        }
        if self.artifact_id != self.endpoint.artifact_id || !self.endpoint.artifact_active {
            return Err(ResolveError::StaleRoute);
        }
        if self.placement_epoch != self.endpoint.endpoint_epoch {
            return Err(ResolveError::StaleRoute);
        }
        if self.fencing_token == 0 || self.fencing_token != self.endpoint.fencing_token {
            return Err(ResolveError::StaleRoute);
        }
        if let Some(mesh) = &self.endpoint.mesh {
            mesh.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameKind {
    Request,
    Answer,
    Refusal,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodHeader {
    pub protocol_version: u16,
    pub frame_kind: FrameKind,
    pub source: Option<PodAddress>,
    pub destination: PodRevisionAddress,
    pub capability: CapabilityId,
    pub input_type: TypeId,
    pub expected_output_type: TypeId,
    pub execution_manifest: Digest,
    pub artifact: Digest,
    pub session_id: SessionId,
    pub scope_id: ScopeId,
    pub request_id: RequestId,
    pub placement_epoch: u64,
    pub fencing_token: u128,
    pub trace_id: TraceId,
    pub deadline: super::Timestamp,
    pub hop_limit: u16,
}

impl PodHeader {
    pub fn validate(&self) -> Result<(), ResolveError> {
        if self.protocol_version == 0 {
            return Err(ResolveError::InvalidProtocolVersion);
        }
        self.destination.validate()?;
        if self.execution_manifest == [0; 32] {
            return Err(ResolveError::MissingBinding {
                field: "execution_manifest",
            });
        }
        if self.artifact == [0; 32] {
            return Err(ResolveError::MissingBinding { field: "artifact" });
        }
        if self.session_id.0.is_empty()
            || self.scope_id.0.is_empty()
            || self.request_id.0.is_empty()
            || self.trace_id.0.is_empty()
        {
            return Err(ResolveError::MissingBinding {
                field: "runtime_ids",
            });
        }
        if let Some(source) = &self.source {
            if source.project != self.destination.address.project
                || source.namespace != self.destination.address.namespace
            {
                return Err(ResolveError::ScopeMismatch);
            }
        }
        if self.capability.0.is_empty()
            || self.input_type.0.is_empty()
            || self.expected_output_type.0.is_empty()
        {
            return Err(ResolveError::MissingBinding {
                field: "typed_contract",
            });
        }
        if self.hop_limit == 0 {
            return Err(ResolveError::HopLimitExceeded);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResolveError {
    InvalidScheme,
    InvalidAddress { field: &'static str },
    MissingBinding { field: &'static str },
    InvalidProtocolVersion,
    HopLimitExceeded,
    RouteCycle,
    EndpointMismatch,
    StaleRoute,
    ConflictingEndpoint,
    ScopeMismatch,
    NotFound,
    ConstraintMismatch { field: &'static str },
}

impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ResolveError {}

fn is_segment_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'~' | b'-')
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn parse_hex_digest(value: &str) -> Result<Digest, ResolveError> {
    if value.len() != 64 {
        return Err(ResolveError::InvalidAddress {
            field: "semantic_revision",
        });
    }
    let mut digest = [0; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).map_err(|_| {
            ResolveError::InvalidAddress {
                field: "semantic_revision",
            }
        })?;
    }
    Ok(digest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn address() -> PodAddress {
        PodAddress::new("project".into(), "namespace".into(), "pod".into()).unwrap()
    }

    fn revision() -> PodRevisionAddress {
        PodRevisionAddress {
            address: address(),
            semantic_revision: [7; 32],
            generation: Generation(3),
        }
    }

    #[test]
    fn address_round_trips_canonically() {
        let address = address();
        let uri = address.canonical_uri().unwrap();
        assert_eq!(uri, "ptr://project/namespace/pod");
        assert_eq!(PodAddress::parse(&uri).unwrap(), address);
    }

    #[test]
    fn address_rejects_ambiguous_segments() {
        assert!(PodAddress::parse("ptr://project/ns/pod/extra").is_err());
        assert!(PodAddress::parse("http://project/ns/pod").is_err());
        assert!(PodAddress::new("project/ns".into(), "ns".into(), "pod".into()).is_err());
    }

    #[test]
    fn revision_uri_binds_generation_and_digest() {
        let revision = revision();
        let uri = revision.canonical_uri().unwrap();
        assert!(uri.contains("@revision:"));
        assert!(uri.ends_with("#generation:3"));
        assert_eq!(PodRevisionAddress::parse(&uri).unwrap(), revision);
    }

    #[test]
    fn route_rejects_zero_hops_and_endpoint_mismatch() {
        let destination = revision();
        let endpoint = PodEndpoint {
            pod: destination.clone(),
            artifact_id: "artifact".into(),
            artifact_active: true,
            node_id: "node".into(),
            device_id: None,
            region: "eu".into(),
            zone: "a".into(),
            peer_id: "peer".into(),
            transport: PodTransport::InProcess,
            endpoint_epoch: 1,
            fencing_token: 1,
            available_vram_bytes: 1024,
            healthy: true,
            load_bps: 0,
            capabilities: vec!["infer".into()],
            accepts: vec!["input".into()],
            produces: vec!["output".into()],
            mesh: None,
        };
        let mut route = PodRoute {
            source: None,
            destination,
            artifact_id: "artifact".into(),
            endpoint,
            capability: "infer".into(),
            input_type: "input".into(),
            output_type: "output".into(),
            trace_id: "trace".into(),
            request_id: "request".into(),
            deadline: super::super::Timestamp(10),
            hop_limit: 0,
            visited: Vec::new(),
            placement_epoch: 1,
            fencing_token: 1,
        };
        assert_eq!(route.validate(), Err(ResolveError::HopLimitExceeded));
        route.hop_limit = 1;
        route.visited.push(route.destination.address.clone());
        assert_eq!(route.validate(), Err(ResolveError::RouteCycle));
    }
}
