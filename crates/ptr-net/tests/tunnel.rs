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

#[test]
fn reference_rotation_keeps_the_callers_lease_valid() {
    let mut executor = ReferenceMeshTunnelExecutor::default();
    let lease = executor.admit(&profile()).unwrap();
    executor.establish(&lease).unwrap();
    let mut rotated = profile().peer;
    rotated.public_key_digest = [2; 32];
    executor.rotate_peer(&lease, &rotated).unwrap();
    // The same lease can rotate again and can still be revoked and released.
    executor.rotate_peer(&lease, &rotated).unwrap();
    assert_eq!(executor.active_peer(&lease), Some(&rotated));
    assert_eq!(executor.revoke(&lease), Ok(()));
    assert_eq!(executor.active_peer(&lease), None);
    assert_eq!(executor.release(lease), Ok(()));
}

#[test]
fn reference_active_peer_is_only_reported_for_the_lease_that_was_admitted() {
    let mut executor = ReferenceMeshTunnelExecutor::default();
    let lease = executor.admit(&profile()).unwrap();
    executor.establish(&lease).unwrap();
    assert_eq!(executor.active_peer(&lease), Some(&profile().peer));
    // The lifecycle calls refuse a lease whose profile differs; the read does too.
    let mut forged = lease.clone();
    forged.profile.fencing_token += 1;
    assert_eq!(executor.active_peer(&forged), None);
}

#[test]
fn reference_rotation_rejects_an_empty_peer_id_and_a_zero_key() {
    let mut executor = ReferenceMeshTunnelExecutor::default();
    let lease = executor.admit(&profile()).unwrap();
    executor.establish(&lease).unwrap();
    let mut empty_id = profile().peer;
    empty_id.peer_id = PeerId::from("");
    let mut zero_key = profile().peer;
    zero_key.public_key_digest = [0; 32];
    for peer in [empty_id, zero_key] {
        assert_eq!(
            executor.rotate_peer(&lease, &peer),
            Err(TunnelError::RevokedPeer)
        );
    }
}

#[test]
fn reference_revoke_establish_and_release_are_idempotent_and_fenced() {
    let mut executor = ReferenceMeshTunnelExecutor::default();
    let lease = executor.admit(&profile()).unwrap();
    assert_eq!(executor.establish(&lease), Ok(TunnelState::Established));
    assert_eq!(executor.establish(&lease), Ok(TunnelState::Established));
    assert_eq!(executor.revoke(&lease), Ok(()));
    assert_eq!(executor.revoke(&lease), Ok(()));
    // release is fenced by the whole profile, not only the fencing token.
    let mut other_peer = lease.clone();
    other_peer.profile.peer.public_key_digest = [3; 32];
    assert_eq!(executor.release(other_peer), Err(TunnelError::StaleFencing));
    assert_eq!(executor.release(lease.clone()), Ok(()));
    assert_eq!(executor.release(lease.clone()), Ok(()));
    assert_eq!(executor.revoke(&lease), Err(TunnelError::InvalidState));
}

#[derive(Default)]
struct FakeWireguardDevice {
    events: Vec<String>,
    fail_configure: bool,
    fail_remove_interface: bool,
    fail_remove_peer: bool,
    fail_remove_digest: Option<[u8; 32]>,
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
        if self.fail_remove_peer || self.fail_remove_digest == Some(peer.public_key_digest) {
            return Err("remove_peer failed".into());
        }
        self.events.push(format!(
            "unpeer:{name}:{}:{}",
            peer.peer_id, peer.public_key_digest[0]
        ));
        Ok(())
    }
    fn remove_interface(&mut self, name: &str) -> Result<(), Self::Error> {
        if self.fail_remove_interface {
            return Err("remove failed".into());
        }
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

#[test]
fn failed_cleanup_keeps_ownership_so_a_retry_removes_the_leaked_interface() {
    let mut executor = WireguardUserspaceExecutor::new(FakeWireguardDevice {
        fail_configure: true,
        fail_remove_interface: true,
        ..Default::default()
    });
    let lease = executor.admit(&profile()).unwrap();
    assert_eq!(
        executor.establish(&lease),
        Err(TunnelError::UnsupportedPlatform)
    );
    // The device recovers: the retry first removes what the failed attempt left
    // behind, then creates the interface again.
    executor.device_mut().fail_configure = false;
    executor.device_mut().fail_remove_interface = false;
    assert_eq!(executor.establish(&lease), Ok(TunnelState::Established));
    assert_eq!(
        executor.device().events,
        [
            "create:ptr-mesh0",
            "remove:ptr-mesh0",
            "create:ptr-mesh0",
            "peer:ptr-mesh0:peer-a:1"
        ]
    );
}

#[test]
fn rotation_that_cannot_remove_either_peer_fails_closed() {
    let mut executor = WireguardUserspaceExecutor::new(FakeWireguardDevice::default());
    let lease = executor.admit(&profile()).unwrap();
    executor.establish(&lease).unwrap();
    let mut rotated = profile().peer;
    rotated.public_key_digest = [2; 32];
    executor.device_mut().fail_remove_peer = true;
    assert_eq!(
        executor.rotate_peer(&lease, &rotated),
        Err(TunnelError::UnsupportedPlatform)
    );
    // The device may carry both peers, so the interface is torn down and the
    // lease no longer accepts work, instead of recording a guess.
    assert_eq!(executor.device().events.last().unwrap(), "remove:ptr-mesh0");
    executor.device_mut().fail_remove_peer = false;
    assert_eq!(
        executor.rotate_peer(&lease, &rotated),
        Err(TunnelError::InvalidState)
    );
    assert_eq!(executor.release(lease), Ok(()));
    let removals = executor
        .device()
        .events
        .iter()
        .filter(|event| event.starts_with("remove:"))
        .count();
    assert_eq!(removals, 1, "the interface is not removed twice");
}

#[test]
fn rotation_rollback_failure_with_a_stuck_interface_keeps_ownership_for_release() {
    let mut executor = WireguardUserspaceExecutor::new(FakeWireguardDevice::default());
    let lease = executor.admit(&profile()).unwrap();
    executor.establish(&lease).unwrap();
    let mut rotated = profile().peer;
    rotated.public_key_digest = [2; 32];
    executor.device_mut().fail_remove_peer = true;
    executor.device_mut().fail_remove_interface = true;
    assert!(executor.rotate_peer(&lease, &rotated).is_err());
    executor.device_mut().fail_remove_peer = false;
    executor.device_mut().fail_remove_interface = false;
    // Release still reaches the interface and removes it.
    executor.release(lease).unwrap();
    assert_eq!(executor.device().events.last().unwrap(), "remove:ptr-mesh0");
}

#[test]
fn revoking_a_lease_revoked_with_a_stuck_interface_retries_the_removal() {
    let mut executor = WireguardUserspaceExecutor::new(FakeWireguardDevice::default());
    let lease = executor.admit(&profile()).unwrap();
    executor.establish(&lease).unwrap();
    let mut rotated = profile().peer;
    rotated.public_key_digest = [2; 32];
    // Neither peer nor the interface can be removed: the lease is revoked but
    // still owns a live interface.
    executor.device_mut().fail_remove_peer = true;
    executor.device_mut().fail_remove_interface = true;
    assert!(executor.rotate_peer(&lease, &rotated).is_err());
    // While the device keeps failing, revoke must say so instead of reporting Ok.
    assert_eq!(
        executor.revoke(&lease),
        Err(TunnelError::UnsupportedPlatform)
    );
    // Once the device recovers, the same call removes the interface.
    executor.device_mut().fail_remove_interface = false;
    assert_eq!(executor.revoke(&lease), Ok(()));
    assert_eq!(executor.device().events.last().unwrap(), "remove:ptr-mesh0");
    // After that revoke is a plain idempotent no-op and release does not
    // remove the interface a second time.
    assert_eq!(executor.revoke(&lease), Ok(()));
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
fn failed_cleanup_can_still_be_released() {
    let mut executor = WireguardUserspaceExecutor::new(FakeWireguardDevice {
        fail_configure: true,
        fail_remove_interface: true,
        ..Default::default()
    });
    let lease = executor.admit(&profile()).unwrap();
    assert!(executor.establish(&lease).is_err());
    executor.device_mut().fail_remove_interface = false;
    executor.release(lease).unwrap();
    assert_eq!(
        executor.device().events,
        ["create:ptr-mesh0", "remove:ptr-mesh0"]
    );
}

#[test]
fn failed_rotation_rolls_back_the_new_peer_and_can_retry_with_the_original_lease() {
    let mut executor = WireguardUserspaceExecutor::new(FakeWireguardDevice::default());
    let lease = executor.admit(&profile()).unwrap();
    executor.establish(&lease).unwrap();
    let mut rotated = profile().peer;
    rotated.public_key_digest = [2; 32];
    executor.device_mut().fail_remove_digest = Some([1; 32]);
    assert_eq!(
        executor.rotate_peer(&lease, &rotated),
        Err(TunnelError::UnsupportedPlatform)
    );
    assert_eq!(
        &executor.device().events[2..],
        ["peer:ptr-mesh0:peer-a:2", "unpeer:ptr-mesh0:peer-a:2"]
    );
    // The previous identity remains active after rollback, so rotating to it is a no-op.
    let before = executor.device().events.clone();
    executor.rotate_peer(&lease, &lease.profile.peer).unwrap();
    assert_eq!(executor.device().events, before);
    executor.device_mut().fail_remove_digest = None;
    executor.rotate_peer(&lease, &rotated).unwrap();
    assert_eq!(
        executor.device().events.last().unwrap(),
        "unpeer:ptr-mesh0:peer-a:1"
    );
    executor.release(lease).unwrap();
}

#[test]
fn repeated_rotations_remove_the_current_peer_and_identical_rotation_does_no_io() {
    let mut executor = WireguardUserspaceExecutor::new(FakeWireguardDevice::default());
    let lease = executor.admit(&profile()).unwrap();
    executor.establish(&lease).unwrap();
    for digest in [2, 3] {
        let mut rotated = profile().peer;
        rotated.public_key_digest = [digest; 32];
        executor.rotate_peer(&lease, &rotated).unwrap();
        assert_eq!(
            executor.device().events.last().unwrap(),
            &format!("unpeer:ptr-mesh0:peer-a:{}", digest - 1)
        );
        let before = executor.device().events.clone();
        executor.rotate_peer(&lease, &rotated).unwrap();
        assert_eq!(executor.device().events, before);
    }
    executor.revoke(&lease).unwrap();
    executor.revoke(&lease).unwrap();
    executor.release(lease).unwrap();
    assert_eq!(
        executor
            .device()
            .events
            .iter()
            .filter(|event| event.starts_with("remove:"))
            .count(),
        1
    );
}

#[test]
fn invalid_rotation_identities_are_rejected_before_device_io() {
    let mut executor = WireguardUserspaceExecutor::new(FakeWireguardDevice::default());
    let lease = executor.admit(&profile()).unwrap();
    executor.establish(&lease).unwrap();
    let before = executor.device().events.clone();
    let mut empty = profile().peer;
    empty.peer_id = PeerId::from("");
    let mut foreign = profile().peer;
    foreign.network_id = NetworkId::from("other");
    let mut zero = profile().peer;
    zero.public_key_digest = [0; 32];
    for peer in [empty, foreign, zero] {
        assert_eq!(
            executor.rotate_peer(&lease, &peer),
            Err(TunnelError::RevokedPeer)
        );
        assert_eq!(executor.device().events, before);
    }
    executor.release(lease).unwrap();
}

#[test]
fn releasing_an_unestablished_lease_never_removes_an_interface() {
    let mut executor = WireguardUserspaceExecutor::new(FakeWireguardDevice::default());
    let lease = executor.admit(&profile()).unwrap();
    executor.revoke(&lease).unwrap();
    executor.release(lease.clone()).unwrap();
    executor.release(lease).unwrap();
    assert!(executor.device().events.is_empty());
}

#[cfg(all(feature = "wireguard-uapi-backend", target_os = "linux"))]
mod linux_device {
    use super::*;
    use ptr_net::LinuxWireguardDevice;
    use std::io::{Read, Write};
    use std::os::unix::net::UnixListener;
    use std::sync::{Arc, Mutex};

    /// A stand-in for the WireGuard UAPI socket: records every request and
    /// answers `errno=0`.
    fn mock_socket(dir: &std::path::Path, name: &str) -> Arc<Mutex<Vec<String>>> {
        let listener = UnixListener::bind(dir.join(format!("{name}.sock"))).unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut request = Vec::new();
                let mut byte = [0u8; 1];
                while stream
                    .read(&mut byte)
                    .map(|read| read == 1)
                    .unwrap_or(false)
                {
                    request.push(byte[0]);
                    if request.ends_with(b"\n\n") {
                        break;
                    }
                }
                recorded
                    .lock()
                    .unwrap()
                    .push(String::from_utf8_lossy(&request).into_owned());
                let _ = stream.write_all(b"errno=0\n\n");
            }
        });
        requests
    }

    fn identity(network: &str, digest: u8) -> MeshPeerIdentity {
        MeshPeerIdentity {
            peer_id: PeerId::from("peer-a"),
            public_key_digest: [digest; 32],
            network_id: NetworkId::from(network),
        }
    }

    fn socket_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ptr-wg-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn remove_peer_sends_a_remove_request_for_the_registered_key() {
        let dir = socket_dir("remove");
        let requests = mock_socket(&dir, "ptrwg0");
        let mut device = LinuxWireguardDevice::new(&dir);
        let peer = identity("mesh", 1);
        device.register_public_key(&peer, [0xab; 32]);
        device.configure_peer("ptrwg0", &peer).unwrap();
        device.remove_peer("ptrwg0", &peer).unwrap();
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests[1].contains("remove=true"), "{}", requests[1]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn rotation_that_keeps_the_wireguard_key_does_not_remove_the_replacement() {
        let dir = socket_dir("alias");
        let requests = mock_socket(&dir, "ptrwg0");
        let mut device = LinuxWireguardDevice::new(&dir);
        let (old, new) = (identity("mesh", 1), identity("mesh", 2));
        device.register_public_key(&old, [0xab; 32]);
        device.register_public_key(&new, [0xab; 32]);
        device.configure_peer("ptrwg0", &old).unwrap();
        device.configure_peer("ptrwg0", &new).unwrap();
        device.remove_peer("ptrwg0", &old).unwrap();
        assert!(
            !requests
                .lock()
                .unwrap()
                .iter()
                .any(|r| r.contains("remove=true")),
            "the shared key must stay configured while the replacement uses it"
        );
        device.remove_peer("ptrwg0", &new).unwrap();
        assert!(requests
            .lock()
            .unwrap()
            .iter()
            .any(|r| r.contains("remove=true")));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_same_peer_in_two_networks_keeps_two_keys() {
        let dir = socket_dir("networks");
        let requests = mock_socket(&dir, "ptrwg0");
        let mut device = LinuxWireguardDevice::new(&dir);
        let (a, b) = (identity("mesh-a", 1), identity("mesh-b", 1));
        device.register_public_key(&a, [0x0a; 32]);
        device.register_public_key(&b, [0x0b; 32]);
        device.configure_peer("ptrwg0", &a).unwrap();
        let requests = requests.lock().unwrap();
        assert!(requests[0].contains(&"0a".repeat(32)), "{}", requests[0]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn rotating_the_same_peer_id_removes_only_the_previous_digest_key() {
        let dir = socket_dir("digest-rotation");
        let requests = mock_socket(&dir, "ptrwg0");
        let mut device = LinuxWireguardDevice::new(&dir);
        let (old, new) = (identity("mesh", 1), identity("mesh", 2));
        device.register_public_key(&old, [0x0a; 32]);
        device.register_public_key(&new, [0x0b; 32]);
        device.configure_peer("ptrwg0", &old).unwrap();
        device.configure_peer("ptrwg0", &new).unwrap();
        device.remove_peer("ptrwg0", &old).unwrap();
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert!(requests[0].contains(&"0a".repeat(32)));
        assert!(requests[1].contains(&"0b".repeat(32)));
        assert!(requests[2].contains("remove=true"));
        assert!(requests[2].contains(&"0a".repeat(32)));
        assert!(!requests[2].contains(&"0b".repeat(32)));
        drop(requests);
        std::fs::remove_dir_all(dir).unwrap();
    }
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
