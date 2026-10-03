use ptr_net::{
    MeshPeerIdentity, MeshRoute, MeshTunnelExecutor, ReferenceMeshTunnelExecutor, TunnelError,
    TunnelProfile, TunnelState, WireguardDevice, WireguardUserspaceExecutor,
};
use ptr_types::{Generation, NetworkId, PeerId, Revision};

fn profile() -> TunnelProfile {
    TunnelProfile {
        network_id: NetworkId::from("mesh"),
        peer: MeshPeerIdentity {
            peer_id: PeerId::from("peer-a"),
            public_key_digest: [1; 32],
            network_id: NetworkId::from("mesh"),
        },
        route: MeshRoute::Direct,
        generation: Generation(1),
        membership_revision: Revision(1),
        placement_epoch: 1,
        fencing_token: 9,
        interface_name: "ptr-mesh0".into(),
    }
}

#[test]
fn reference_tunnel_is_fenced_and_cleanup_is_idempotent() {
    let mut executor = ReferenceMeshTunnelExecutor::default();
    let lease = executor.admit(&profile()).unwrap();
    assert_eq!(executor.establish(&lease), Ok(TunnelState::Established));
    assert_eq!(executor.revoke(&lease), Ok(()));
    assert_eq!(executor.release(lease.clone()), Ok(()));
    assert_eq!(executor.release(lease), Ok(()));
}

#[test]
fn stale_lease_cannot_revoke_a_new_fenced_lease() {
    let mut executor = ReferenceMeshTunnelExecutor::default();
    let old = executor.admit(&profile()).unwrap();
    let mut replacement_profile = profile();
    replacement_profile.fencing_token = 10;
    let replacement = executor.admit(&replacement_profile).unwrap();
    let stale = ptr_net::TunnelLease {
        lease_id: replacement.lease_id.clone(),
        profile: old.profile.clone(),
        state: old.state,
    };
    assert_eq!(executor.revoke(&stale), Err(TunnelError::StaleFencing));
    assert_eq!(executor.revoke(&replacement), Ok(()));
    assert_ne!(old.profile.fencing_token, replacement.profile.fencing_token);
}

#[test]
fn invalid_profile_is_rejected_before_tunnel_activation() {
    let mut invalid = profile();
    invalid.fencing_token = 0;
    let mut executor = ReferenceMeshTunnelExecutor::default();
    assert_eq!(executor.admit(&invalid), Err(TunnelError::InvalidProfile));
}

#[derive(Default)]
struct FakeWireguardDevice {
    events: Vec<String>,
    fail_configure: bool,
}

impl WireguardDevice for FakeWireguardDevice {
    type Error = String;
    fn create_interface(&mut self, name: &str) -> Result<(), Self::Error> {
        self.events.push(format!("create:{name}"));
        Ok(())
    }
    fn configure_peer(&mut self, name: &str, peer: &MeshPeerIdentity) -> Result<(), Self::Error> {
        if self.fail_configure {
            return Err("configure failed".into());
        }
        self.events.push(format!(
            "peer:{name}:{}:{}",
            peer.peer_id, peer.public_key_digest[0]
        ));
        Ok(())
    }
    fn remove_peer(&mut self, name: &str, peer: &MeshPeerIdentity) -> Result<(), Self::Error> {
        self.events.push(format!(
            "unpeer:{name}:{}:{}",
            peer.peer_id, peer.public_key_digest[0]
        ));
        Ok(())
    }
    fn remove_interface(&mut self, name: &str) -> Result<(), Self::Error> {
        self.events.push(format!("remove:{name}"));
        Ok(())
    }
}

#[test]
fn userspace_executor_binds_device_io_to_the_fenced_lifecycle() {
    let mut executor = WireguardUserspaceExecutor::new(FakeWireguardDevice::default());
    let lease = executor.admit(&profile()).unwrap();
    executor.establish(&lease).unwrap();
    executor.release(lease).unwrap();
    assert_eq!(
        executor.device().events,
        [
            "create:ptr-mesh0",
            "peer:ptr-mesh0:peer-a:1",
            "remove:ptr-mesh0"
        ]
    );
}

#[test]
fn revoke_tears_down_the_interface_and_release_does_not_remove_it_twice() {
    let mut executor = WireguardUserspaceExecutor::new(FakeWireguardDevice::default());
    let lease = executor.admit(&profile()).unwrap();
    executor.establish(&lease).unwrap();
    executor.revoke(&lease).unwrap();
    assert_eq!(executor.device().events.last().unwrap(), "remove:ptr-mesh0");
    executor.release(lease).unwrap();
    let removals = executor
        .device()
        .events
        .iter()
        .filter(|event| event.starts_with("remove:"))
        .count();
    assert_eq!(removals, 1);
}

#[test]
fn rotation_keeps_the_original_lease_valid_and_removes_the_old_peer() {
    let mut executor = WireguardUserspaceExecutor::new(FakeWireguardDevice::default());
    let lease = executor.admit(&profile()).unwrap();
    executor.establish(&lease).unwrap();
    let mut rotated = profile().peer;
    rotated.public_key_digest = [2; 32];
    executor.rotate_peer(&lease, &rotated).unwrap();
    assert_eq!(
        executor.device().events,
        [
            "create:ptr-mesh0",
            "peer:ptr-mesh0:peer-a:1",
            "peer:ptr-mesh0:peer-a:2",
            "unpeer:ptr-mesh0:peer-a:1",
        ]
    );
    executor.revoke(&lease).unwrap();
    executor.release(lease).unwrap();
}

#[test]
fn rotation_rejects_an_all_zero_key() {
    let mut executor = WireguardUserspaceExecutor::new(FakeWireguardDevice::default());
    let lease = executor.admit(&profile()).unwrap();
    executor.establish(&lease).unwrap();
    let mut zeroed = profile().peer;
    zeroed.public_key_digest = [0; 32];
    assert_eq!(
        executor.rotate_peer(&lease, &zeroed),
        Err(TunnelError::RevokedPeer)
    );
}

#[test]
fn failed_peer_configuration_does_not_leak_the_interface_and_allows_retry() {
    let mut executor = WireguardUserspaceExecutor::new(FakeWireguardDevice {
        fail_configure: true,
        ..Default::default()
    });
    let lease = executor.admit(&profile()).unwrap();
    assert_eq!(
        executor.establish(&lease),
        Err(TunnelError::UnsupportedPlatform)
    );
    assert_eq!(
        executor.device().events,
        ["create:ptr-mesh0", "remove:ptr-mesh0"]
    );
    executor.device_mut().fail_configure = false;
    assert_eq!(executor.establish(&lease), Ok(TunnelState::Established));
}

#[cfg(all(feature = "wintun-backend", target_os = "windows"))]
#[test]
fn wintun_dll_and_driver_are_loadable_when_provisioned() {
    let Some(path) = std::env::var_os("WINTUN_DLL") else {
        return;
    };
    let device = ptr_net::WintunDevice::new(path);
    let version = device
        .probe_driver_version()
        .expect("Wintun DLL/driver must load");
    assert!(!version.is_empty());
}
