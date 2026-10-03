use crate::pod_address::MeshEndpointBinding;
use crate::{Digest, FencingToken, Generation, InvitationId, NetworkId, PeerId, Revision};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeshTunnelEventKind {
    InvitationCreated,
    InvitationAccepted,
    MembershipActivated,
    MembershipRevoked,
    RouteChanged,
    TunnelEstablished,
    TunnelRevoked,
    TunnelReleased,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshTunnelLifecycleEvent {
    pub network_id: NetworkId,
    pub peer_id: PeerId,
    pub invitation_id: Option<InvitationId>,
    pub tunnel_id: Option<String>,
    pub generation: Generation,
    pub revision: Revision,
    pub kind: MeshTunnelEventKind,
    pub endpoint: Option<MeshEndpointBinding>,
    pub placement_epoch: u64,
    pub fencing_token: FencingToken,
    pub event_digest: Digest,
}

impl MeshTunnelLifecycleEvent {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.network_id.0.is_empty() || self.peer_id.0.is_empty() {
            return Err("mesh identity");
        }
        if self.generation.0 == 0 || self.revision.0 == 0 || self.placement_epoch == 0 {
            return Err("mesh revision");
        }
        if matches!(
            self.kind,
            MeshTunnelEventKind::TunnelEstablished | MeshTunnelEventKind::RouteChanged
        ) && self.endpoint.is_none()
        {
            return Err("mesh endpoint");
        }
        if let Some(endpoint) = &self.endpoint {
            endpoint.validate().map_err(|_| "mesh endpoint")?;
            if endpoint.network_id != self.network_id || endpoint.peer_id != self.peer_id {
                return Err("mesh endpoint identity");
            }
        }
        if self.event_digest == [0; 32] {
            return Err("event digest");
        }
        Ok(())
    }
}
