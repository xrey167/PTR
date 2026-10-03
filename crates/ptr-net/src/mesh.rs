use ptr_types::{
    Digest, FencingToken, Generation, InvitationId, MeshEndpointBinding, MeshRouteKind,
    MeshTunnelEventKind, MeshTunnelLifecycleEvent, NetworkId, PeerId, Revision,
};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshPeerIdentity {
    pub peer_id: PeerId,
    pub public_key_digest: Digest,
    pub network_id: NetworkId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MeshRoute {
    Direct,
    Relay { relay_id: PeerId },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshEndpoint {
    pub peer: MeshPeerIdentity,
    pub route: MeshRoute,
    pub membership_generation: Generation,
    pub route_revision: Revision,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MembershipState {
    Pending,
    Active,
    Revoked,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshMembership {
    pub peer: MeshPeerIdentity,
    pub generation: Generation,
    pub revision: Revision,
    pub state: MembershipState,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshInvitation {
    pub id: InvitationId,
    pub network_id: NetworkId,
    pub peer: MeshPeerIdentity,
    pub generation: Generation,
    pub one_time: bool,
    pub consumed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MeshError {
    EmptyIdentity,
    NetworkMismatch,
    UnknownPeer,
    AlreadyMember,
    RevokedPeer,
    InvitationConsumed,
    InvitationMismatch,
    StaleGeneration,
    ConflictingRevision,
}

#[derive(Default)]
pub struct MeshRegistry {
    memberships: BTreeMap<(NetworkId, PeerId), MeshMembership>,
    invitations: BTreeMap<InvitationId, MeshInvitation>,
}

impl MeshRegistry {
    pub fn apply_event(&mut self, event: &MeshTunnelLifecycleEvent) -> Result<(), MeshError> {
        event
            .validate()
            .map_err(|_| MeshError::ConflictingRevision)?;
        let key = (event.network_id.clone(), event.peer_id.clone());
        match event.kind {
            MeshTunnelEventKind::InvitationCreated => Ok(()),
            MeshTunnelEventKind::InvitationAccepted | MeshTunnelEventKind::MembershipActivated => {
                let peer = MeshPeerIdentity {
                    peer_id: event.peer_id.clone(),
                    public_key_digest: event.event_digest,
                    network_id: event.network_id.clone(),
                };
                if let Some(existing) = self.memberships.get(&key) {
                    // A revoked membership is terminal for its generation: only a
                    // fresh invitation may bring the peer back, never a replayed or
                    // late activation event.
                    if existing.state == MembershipState::Revoked {
                        return Err(MeshError::RevokedPeer);
                    }
                    if existing.generation != event.generation || existing.revision > event.revision
                    {
                        return Err(MeshError::ConflictingRevision);
                    }
                    // The admitted key material may never change through an
                    // activation event; key rotation has its own explicit path.
                    if existing.peer != peer {
                        return Err(MeshError::ConflictingRevision);
                    }
                    if existing.revision == event.revision {
                        return Ok(());
                    }
                }
                self.memberships.insert(
                    key,
                    MeshMembership {
                        peer,
                        generation: event.generation,
                        revision: event.revision,
                        state: MembershipState::Active,
                    },
                );
                Ok(())
            }
            MeshTunnelEventKind::MembershipRevoked => {
                let membership = self
                    .memberships
                    .get_mut(&key)
                    .ok_or(MeshError::UnknownPeer)?;
                if membership.revision > event.revision {
                    return Err(MeshError::StaleGeneration);
                }
                membership.revision = event.revision;
                membership.state = MembershipState::Revoked;
                Ok(())
            }
            MeshTunnelEventKind::RouteChanged => {
                let membership = self
                    .memberships
                    .get_mut(&key)
                    .ok_or(MeshError::UnknownPeer)?;
                if membership.state != MembershipState::Active
                    || membership.generation != event.generation
                {
                    return Err(MeshError::RevokedPeer);
                }
                // The membership revision is the single monotone counter for the
                // peer, so a replayed route event with an old revision is stale.
                if event.revision <= membership.revision {
                    return Err(MeshError::StaleGeneration);
                }
                membership.revision = event.revision;
                Ok(())
            }
            MeshTunnelEventKind::TunnelEstablished
            | MeshTunnelEventKind::TunnelRevoked
            | MeshTunnelEventKind::TunnelReleased => Ok(()),
        }
    }

    pub fn endpoint_binding(
        &self,
        network_id: &NetworkId,
        peer_id: &PeerId,
        route: MeshRoute,
    ) -> Result<MeshEndpointBinding, MeshError> {
        let endpoint = self.endpoint(network_id, peer_id, route)?;
        Ok(endpoint.into())
    }
    pub fn invite(
        &mut self,
        id: InvitationId,
        network_id: NetworkId,
        peer: MeshPeerIdentity,
        generation: Generation,
    ) -> Result<MeshInvitation, MeshError> {
        validate_peer(&peer, &network_id)?;
        if generation.0 == 0
            || self
                .memberships
                .contains_key(&(network_id.clone(), peer.peer_id.clone()))
        {
            return Err(MeshError::AlreadyMember);
        }
        let invitation = MeshInvitation {
            id: id.clone(),
            network_id,
            peer,
            generation,
            one_time: true,
            consumed: false,
        };
        self.invitations.insert(id, invitation.clone());
        Ok(invitation)
    }

    pub fn accept(&mut self, id: &InvitationId) -> Result<MeshMembership, MeshError> {
        let invitation = self.invitations.get_mut(id).ok_or(MeshError::UnknownPeer)?;
        if invitation.consumed {
            return Err(MeshError::InvitationConsumed);
        }
        let key = (
            invitation.network_id.clone(),
            invitation.peer.peer_id.clone(),
        );
        if let Some(existing) = self.memberships.get(&key) {
            return if existing.peer == invitation.peer {
                Err(MeshError::AlreadyMember)
            } else {
                Err(MeshError::ConflictingRevision)
            };
        }
        invitation.consumed = invitation.one_time;
        let membership = MeshMembership {
            peer: invitation.peer.clone(),
            generation: invitation.generation,
            revision: Revision(1),
            state: MembershipState::Active,
        };
        self.memberships.insert(key, membership.clone());
        Ok(membership)
    }

    pub fn revoke(&mut self, network_id: &NetworkId, peer_id: &PeerId) -> Result<(), MeshError> {
        let membership = self
            .memberships
            .get_mut(&(network_id.clone(), peer_id.clone()))
            .ok_or(MeshError::UnknownPeer)?;
        membership.state = MembershipState::Revoked;
        membership.revision = membership.revision.next();
        Ok(())
    }

    pub fn endpoint(
        &self,
        network_id: &NetworkId,
        peer_id: &PeerId,
        route: MeshRoute,
    ) -> Result<MeshEndpoint, MeshError> {
        let membership = self
            .memberships
            .get(&(network_id.clone(), peer_id.clone()))
            .ok_or(MeshError::UnknownPeer)?;
        if membership.state != MembershipState::Active {
            return Err(MeshError::RevokedPeer);
        }
        Ok(MeshEndpoint {
            peer: membership.peer.clone(),
            route,
            membership_generation: membership.generation,
            route_revision: membership.revision,
        })
    }
}

impl From<MeshEndpoint> for MeshEndpointBinding {
    fn from(endpoint: MeshEndpoint) -> Self {
        let (route, relay_id) = match endpoint.route {
            MeshRoute::Direct => (MeshRouteKind::Direct, None),
            MeshRoute::Relay { relay_id } => (MeshRouteKind::Relay, Some(relay_id)),
        };
        Self {
            network_id: endpoint.peer.network_id,
            peer_id: endpoint.peer.peer_id,
            route,
            relay_id,
            membership_generation: endpoint.membership_generation,
            membership_revision: endpoint.route_revision,
        }
    }
}

impl From<MeshEndpoint> for MeshTunnelLifecycleEvent {
    fn from(endpoint: MeshEndpoint) -> Self {
        let binding: MeshEndpointBinding = endpoint.clone().into();
        Self {
            network_id: endpoint.peer.network_id,
            peer_id: endpoint.peer.peer_id,
            invitation_id: None,
            tunnel_id: None,
            generation: endpoint.membership_generation,
            revision: endpoint.route_revision,
            kind: MeshTunnelEventKind::RouteChanged,
            endpoint: Some(binding),
            placement_epoch: endpoint.route_revision.0,
            fencing_token: FencingToken(endpoint.route_revision.0 as u128),
            event_digest: endpoint.peer.public_key_digest,
        }
    }
}

fn validate_peer(peer: &MeshPeerIdentity, network_id: &NetworkId) -> Result<(), MeshError> {
    if peer.peer_id.0.is_empty()
        || peer.network_id != *network_id
        || peer.public_key_digest == [0; 32]
    {
        return Err(if peer.network_id != *network_id {
            MeshError::NetworkMismatch
        } else {
            MeshError::EmptyIdentity
        });
    }
    Ok(())
}
