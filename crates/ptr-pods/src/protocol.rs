use ptr_protocol::TypedPayload;
use ptr_types::{CapabilityId, Digest, PodAddress, PodRevisionAddress, PrincipalId, Timestamp};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProtocolBinding {
    InProcess,
    LocalWorker,
    IrohQuic,
    Tcp,
    Udp,
    Ssh,
    Wireguard,
    Mqtt,
    Http,
    WebSocket,
    Grpc,
    Mcp,
    Cuda,
    Moq,
    Rfb,
    Ipc,
    Mavlink,
    Zmtp,
    Socks5,
    HttpConnect,
    WebRtc,
    JsonRpc,
    Custom(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MessagePattern {
    RequestResponse,
    PublishSubscribe,
    Pipeline,
    Duplex,
    Stream,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeliveryMode {
    ReliableOrdered,
    UnreliableDatagram,
    BestEffort,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConnectionScope {
    PerRequest,
    Session,
    Persistent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProtocolProfile {
    pub pattern: MessagePattern,
    pub delivery: DeliveryMode,
    pub connection: ConnectionScope,
    pub multiplexed: bool,
    pub signed: bool,
    pub max_frame_bytes: usize,
}

impl ProtocolBinding {
    pub fn profile(&self) -> ProtocolProfile {
        match self {
            Self::Tcp | Self::Ssh | Self::Http | Self::HttpConnect | Self::Socks5 => {
                ProtocolProfile {
                    pattern: MessagePattern::Stream,
                    delivery: DeliveryMode::ReliableOrdered,
                    connection: ConnectionScope::Session,
                    multiplexed: false,
                    signed: false,
                    max_frame_bytes: 1 << 20,
                }
            }
            Self::Udp => ProtocolProfile {
                pattern: MessagePattern::RequestResponse,
                delivery: DeliveryMode::UnreliableDatagram,
                connection: ConnectionScope::PerRequest,
                multiplexed: false,
                signed: false,
                max_frame_bytes: 65_507,
            },
            Self::IrohQuic | Self::Moq | Self::WebSocket | Self::Grpc | Self::WebRtc => {
                ProtocolProfile {
                    pattern: MessagePattern::Duplex,
                    delivery: DeliveryMode::ReliableOrdered,
                    connection: ConnectionScope::Session,
                    multiplexed: true,
                    signed: false,
                    max_frame_bytes: 1 << 20,
                }
            }
            Self::Mqtt => ProtocolProfile {
                pattern: MessagePattern::PublishSubscribe,
                delivery: DeliveryMode::BestEffort,
                connection: ConnectionScope::Persistent,
                multiplexed: true,
                signed: false,
                max_frame_bytes: 1 << 20,
            },
            Self::JsonRpc => ProtocolProfile {
                pattern: MessagePattern::RequestResponse,
                delivery: DeliveryMode::ReliableOrdered,
                connection: ConnectionScope::Session,
                multiplexed: true,
                signed: false,
                max_frame_bytes: 1 << 20,
            },
            Self::Zmtp => ProtocolProfile {
                pattern: MessagePattern::RequestResponse,
                delivery: DeliveryMode::ReliableOrdered,
                connection: ConnectionScope::Session,
                multiplexed: true,
                signed: false,
                max_frame_bytes: 1 << 20,
            },
            Self::Ipc | Self::LocalWorker => ProtocolProfile {
                pattern: MessagePattern::Pipeline,
                delivery: DeliveryMode::ReliableOrdered,
                connection: ConnectionScope::Persistent,
                multiplexed: true,
                signed: false,
                max_frame_bytes: 1 << 20,
            },
            Self::Mavlink => ProtocolProfile {
                pattern: MessagePattern::Stream,
                delivery: DeliveryMode::ReliableOrdered,
                connection: ConnectionScope::Session,
                multiplexed: false,
                signed: true,
                max_frame_bytes: 280,
            },
            Self::Rfb => ProtocolProfile {
                pattern: MessagePattern::Duplex,
                delivery: DeliveryMode::ReliableOrdered,
                connection: ConnectionScope::Session,
                multiplexed: false,
                signed: false,
                max_frame_bytes: 1 << 20,
            },
            Self::InProcess | Self::Cuda | Self::Wireguard | Self::Mcp | Self::Custom(_) => {
                ProtocolProfile {
                    pattern: MessagePattern::RequestResponse,
                    delivery: DeliveryMode::ReliableOrdered,
                    connection: ConnectionScope::PerRequest,
                    multiplexed: false,
                    signed: false,
                    max_frame_bytes: 1 << 20,
                }
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeProtocolRequest {
    pub link: PodLink,
    pub input: TypedPayload,
    pub request_id: ptr_types::RequestId,
    pub endpoint: NetworkEndpoint,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NetworkEndpoint {
    pub host: String,
    pub port: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeProtocolResponse {
    pub output: TypedPayload,
    pub request_id: ptr_types::RequestId,
    pub protocol: ProtocolBinding,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProtocolError {
    Unsupported,
    Backpressure,
    Deadline,
    Cancelled,
    PeerMismatch,
    StaleBinding,
    Transport(String),
    InvalidEndpoint,
    FrameTooLarge,
}

pub trait NativeProtocolExecutor: Send + Sync {
    fn protocol(&self) -> ProtocolBinding;
    fn execute(
        &self,
        request: NativeProtocolRequest,
    ) -> Result<NativeProtocolResponse, ProtocolError>;
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EgressPolicy {
    pub topics: Vec<String>,
    pub hosts: Vec<String>,
    pub ports: Vec<u16>,
}

impl EgressPolicy {
    pub fn validate(&self) -> Result<(), PodLinkError> {
        if self.topics.iter().any(|value| value.is_empty())
            || self.hosts.iter().any(|value| value.is_empty())
            || self.ports.contains(&0)
        {
            return Err(PodLinkError::InvalidEgressPolicy);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodLink {
    pub trace_id: String,
    pub source: PodAddress,
    pub target: PodRevisionAddress,
    pub artifact_id: ptr_types::ArtifactId,
    pub protocol: ProtocolBinding,
    pub capability: CapabilityId,
    pub acl: Vec<PrincipalId>,
    pub deadline: Timestamp,
    pub hop_limit: u16,
    pub visited: Vec<PodAddress>,
    pub attestation: Digest,
    pub egress_policy: EgressPolicy,
}

impl PodLink {
    pub fn validate(
        &self,
        target_artifact: &ptr_types::ArtifactId,
        target_capabilities: &[CapabilityId],
        principal: &PrincipalId,
        now: Timestamp,
    ) -> Result<(), PodLinkError> {
        self.source
            .validate()
            .map_err(|_| PodLinkError::InvalidAddress)?;
        self.target
            .validate()
            .map_err(|_| PodLinkError::InvalidAddress)?;
        if self.trace_id.is_empty() || self.hop_limit == 0 || self.deadline.0 <= now.0 {
            return Err(PodLinkError::DeadlineOrHopLimit);
        }
        if self.source.project != self.target.address.project
            || self.source.namespace != self.target.address.namespace
        {
            return Err(PodLinkError::ScopeMismatch);
        }
        if self.source == self.target.address {
            return Err(PodLinkError::CycleDetected);
        }
        if self.artifact_id != *target_artifact {
            return Err(PodLinkError::ArtifactMismatch);
        }
        if !target_capabilities.contains(&self.capability) {
            return Err(PodLinkError::CapabilityUnavailable);
        }
        if !self.acl.is_empty() && !self.acl.contains(principal) {
            return Err(PodLinkError::AccessDenied);
        }
        if self.attestation == [0; 32] {
            return Err(PodLinkError::MissingAttestation);
        }
        self.egress_policy.validate()?;
        let mut unique = self.visited.clone();
        unique.sort();
        unique.dedup();
        if unique.len() != self.visited.len() || self.visited.contains(&self.target.address) {
            return Err(PodLinkError::CycleDetected);
        }
        Ok(())
    }

    pub fn next_hop(&self, source: PodAddress) -> Result<Self, PodLinkError> {
        if self.hop_limit <= 1 || self.visited.contains(&source) {
            return Err(PodLinkError::CycleDetected);
        }
        let mut next = self.clone();
        next.visited.push(source);
        next.hop_limit -= 1;
        Ok(next)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PodLinkError {
    InvalidAddress,
    ScopeMismatch,
    ArtifactMismatch,
    CapabilityUnavailable,
    AccessDenied,
    MissingAttestation,
    DeadlineOrHopLimit,
    CycleDetected,
    InvalidEgressPolicy,
}
