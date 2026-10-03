use ptr_net::{MembershipState, MeshError, MeshPeerIdentity, MeshRegistry, MeshRoute};
use ptr_types::{Generation, InvitationId, NetworkId, PeerId};

fn peer(network: &str, name: &str) -> MeshPeerIdentity {
    MeshPeerIdentity {
        peer_id: PeerId::from(name),
        public_key_digest: [7; 32],
        network_id: NetworkId::from(network),
    }
}

#[test]
fn invitation_acceptance_creates_direct_or_relayable_membership() {
    let network = NetworkId::from("mesh");
    let mut registry = MeshRegistry::default();
    registry
        .invite(
            InvitationId::from("invite-1"),
            network.clone(),
            peer("mesh", "alice"),
            Generation(1),
        )
        .unwrap();
    let membership = registry.accept(&InvitationId::from("invite-1")).unwrap();
    assert_eq!(membership.state, MembershipState::Active);
    let endpoint = registry
        .endpoint(
            &network,
            &PeerId::from("alice"),
            MeshRoute::Relay {
                relay_id: PeerId::from("relay-1"),
            },
        )
        .unwrap();
    assert!(matches!(endpoint.route, MeshRoute::Relay { .. }));
}

#[test]
fn revoked_membership_cannot_resolve_an_endpoint_or_reuse_invitation() {
    let network = NetworkId::from("mesh");
    let mut registry = MeshRegistry::default();
    registry
        .invite(
            InvitationId::from("invite-1"),
            network.clone(),
            peer("mesh", "alice"),
            Generation(1),
        )
        .unwrap();
    registry.accept(&InvitationId::from("invite-1")).unwrap();
    registry.revoke(&network, &PeerId::from("alice")).unwrap();
    assert_eq!(
        registry.endpoint(&network, &PeerId::from("alice"), MeshRoute::Direct),
        Err(MeshError::RevokedPeer)
    );
    assert_eq!(
        registry.accept(&InvitationId::from("invite-1")),
        Err(MeshError::InvitationConsumed)
    );
}

#[test]
fn invitation_rejects_cross_network_peer_identity() {
    let mut registry = MeshRegistry::default();
    assert_eq!(
        registry.invite(
            InvitationId::from("invite-1"),
            NetworkId::from("mesh-a"),
            peer("mesh-b", "alice"),
            Generation(1),
        ),
        Err(MeshError::NetworkMismatch)
    );
}
