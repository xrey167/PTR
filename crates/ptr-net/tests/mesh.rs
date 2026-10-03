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

fn lifecycle_event(
    kind: ptr_types::MeshTunnelEventKind,
    name: &str,
    revision: u64,
    digest_byte: u8,
) -> ptr_types::MeshTunnelLifecycleEvent {
    let endpoint = if matches!(kind, ptr_types::MeshTunnelEventKind::RouteChanged) {
        Some(ptr_types::MeshEndpointBinding {
            network_id: NetworkId::from("mesh"),
            peer_id: PeerId::from(name),
            route: ptr_types::MeshRouteKind::Direct,
            relay_id: None,
            membership_generation: Generation(1),
            membership_revision: ptr_types::Revision(revision),
        })
    } else {
        None
    };
    ptr_types::MeshTunnelLifecycleEvent {
        network_id: NetworkId::from("mesh"),
        peer_id: PeerId::from(name),
        invitation_id: None,
        tunnel_id: None,
        generation: Generation(1),
        revision: ptr_types::Revision(revision),
        kind,
        endpoint,
        placement_epoch: revision,
        fencing_token: ptr_types::FencingToken(revision as u128),
        event_digest: [digest_byte; 32],
    }
}

#[test]
fn late_activation_cannot_resurrect_a_revoked_membership() {
    use ptr_types::MeshTunnelEventKind::{MembershipActivated, MembershipRevoked};
    let mut registry = MeshRegistry::default();
    registry
        .apply_event(&lifecycle_event(MembershipActivated, "alice", 1, 7))
        .unwrap();
    registry
        .apply_event(&lifecycle_event(MembershipRevoked, "alice", 2, 7))
        .unwrap();
    assert_eq!(
        registry.apply_event(&lifecycle_event(MembershipActivated, "alice", 3, 7)),
        Err(MeshError::RevokedPeer)
    );
    assert_eq!(
        registry.endpoint(
            &NetworkId::from("mesh"),
            &PeerId::from("alice"),
            MeshRoute::Direct
        ),
        Err(MeshError::RevokedPeer)
    );
}

#[test]
fn a_later_activation_keeps_the_admitted_key_digest_and_is_not_rejected() {
    use ptr_types::MeshTunnelEventKind::MembershipActivated;
    let mut registry = MeshRegistry::default();
    registry
        .apply_event(&lifecycle_event(MembershipActivated, "alice", 1, 7))
        .unwrap();
    // A legitimate activation at a higher revision carries a different
    // event_digest; it must be accepted, but must not replace the admitted key.
    registry
        .apply_event(&lifecycle_event(MembershipActivated, "alice", 2, 9))
        .unwrap();
    let endpoint = registry
        .endpoint(
            &NetworkId::from("mesh"),
            &PeerId::from("alice"),
            MeshRoute::Direct,
        )
        .unwrap();
    assert_eq!(endpoint.peer.public_key_digest, [7; 32]);
    assert_eq!(endpoint.route_revision, ptr_types::Revision(2));
}

#[test]
fn stale_route_changes_are_rejected_after_a_newer_route() {
    use ptr_types::MeshTunnelEventKind::{MembershipActivated, RouteChanged};
    let mut registry = MeshRegistry::default();
    registry
        .apply_event(&lifecycle_event(MembershipActivated, "alice", 1, 7))
        .unwrap();
    registry
        .apply_event(&lifecycle_event(RouteChanged, "alice", 5, 7))
        .unwrap();
    assert_eq!(
        registry.apply_event(&lifecycle_event(RouteChanged, "alice", 3, 7)),
        Err(MeshError::StaleGeneration)
    );
    assert_eq!(
        registry.apply_event(&lifecycle_event(RouteChanged, "alice", 5, 7)),
        Err(MeshError::StaleGeneration)
    );
}

#[test]
fn a_revocation_must_be_newer_than_the_last_event_and_replays_are_idempotent() {
    use ptr_types::MeshTunnelEventKind::{MembershipActivated, MembershipRevoked, RouteChanged};
    let mut registry = MeshRegistry::default();
    registry
        .apply_event(&lifecycle_event(MembershipActivated, "alice", 1, 7))
        .unwrap();
    registry
        .apply_event(&lifecycle_event(RouteChanged, "alice", 5, 7))
        .unwrap();
    // Same revision as the route change: refused, whatever order they arrive in.
    assert_eq!(
        registry.apply_event(&lifecycle_event(MembershipRevoked, "alice", 5, 7)),
        Err(MeshError::StaleGeneration)
    );
    registry
        .apply_event(&lifecycle_event(MembershipRevoked, "alice", 6, 7))
        .unwrap();
    // A replay of the applied revocation is accepted, an older one is not.
    registry
        .apply_event(&lifecycle_event(MembershipRevoked, "alice", 6, 7))
        .unwrap();
    assert_eq!(
        registry.apply_event(&lifecycle_event(MembershipRevoked, "alice", 4, 7)),
        Err(MeshError::StaleGeneration)
    );
}
