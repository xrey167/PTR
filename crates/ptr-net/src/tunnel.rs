use crate::{MeshPeerIdentity, MeshRoute};
use ptr_types::{Generation, PeerId, Revision};
use std::collections::BTreeMap;
#[cfg(all(feature = "wireguard-uapi-backend", unix))]
#[cfg(windows)]
use std::path::Path;
#[cfg(target_os = "linux")]
use std::path::PathBuf;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TunnelProfile {
    pub network_id: ptr_types::NetworkId,
    pub peer: MeshPeerIdentity,
    pub route: MeshRoute,
    pub generation: Generation,
    pub membership_revision: Revision,
    pub placement_epoch: u64,
    pub fencing_token: u128,
    pub interface_name: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TunnelLease {
    pub lease_id: String,
    pub profile: TunnelProfile,
    pub state: TunnelState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TunnelState {
    Admitted,
    Established,
    Revoked,
    Released,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TunnelError {
    InvalidProfile,
    NotFound,
    InvalidState,
    StaleFencing,
    RevokedPeer,
    UnsupportedPlatform,
    ConflictingLease,
}

pub trait MeshTunnelExecutor: Send + Sync {
    fn admit(&mut self, profile: &TunnelProfile) -> Result<TunnelLease, TunnelError>;
    fn establish(&mut self, lease: &TunnelLease) -> Result<TunnelState, TunnelError>;
    fn rotate_peer(
        &mut self,
        lease: &TunnelLease,
        peer: &MeshPeerIdentity,
    ) -> Result<(), TunnelError>;
    fn revoke(&mut self, lease: &TunnelLease) -> Result<(), TunnelError>;
    fn release(&mut self, lease: TunnelLease) -> Result<(), TunnelError>;
}

/// The narrow device contract used by the real WireGuard adapters.  PTR owns
/// admission, fencing and lifecycle; this trait owns only interface/peer I/O.
/// Implementations must not silently replace a peer or interface after revoke.
pub trait WireguardDevice: Send + Sync {
    type Error: std::fmt::Display;

    fn create_interface(&mut self, name: &str) -> Result<(), Self::Error>;
    fn configure_peer(&mut self, name: &str, peer: &MeshPeerIdentity) -> Result<(), Self::Error>;
    fn remove_interface(&mut self, name: &str) -> Result<(), Self::Error>;
}

/// A userspace WireGuard executor.  The concrete device is injected so the
/// same fenced lifecycle is used by boringtun/userspace, Linux kernel, Wintun,
/// and macOS Network Extension adapters without putting platform APIs in PTR.
pub struct WireguardUserspaceExecutor<D> {
    device: D,
    leases: BTreeMap<String, TunnelLease>,
    next_id: u64,
}

impl<D> WireguardUserspaceExecutor<D> {
    pub fn new(device: D) -> Self {
        Self {
            device,
            leases: BTreeMap::new(),
            next_id: 1,
        }
    }

    pub fn device(&self) -> &D {
        &self.device
    }
    pub fn device_mut(&mut self) -> &mut D {
        &mut self.device
    }
}

impl<D> MeshTunnelExecutor for WireguardUserspaceExecutor<D>
where
    D: WireguardDevice,
{
    fn admit(&mut self, profile: &TunnelProfile) -> Result<TunnelLease, TunnelError> {
        validate_profile(profile)?;
        let lease_id = format!("wg-{}", self.next_id);
        self.next_id += 1;
        let lease = TunnelLease {
            lease_id: lease_id.clone(),
            profile: profile.clone(),
            state: TunnelState::Admitted,
        };
        if self.leases.values().any(|existing| {
            existing.profile.interface_name == profile.interface_name
                && existing.state != TunnelState::Released
        }) {
            return Err(TunnelError::ConflictingLease);
        }
        self.leases.insert(lease_id, lease.clone());
        Ok(lease)
    }

    fn establish(&mut self, lease: &TunnelLease) -> Result<TunnelState, TunnelError> {
        let current = self
            .leases
            .get_mut(&lease.lease_id)
            .ok_or(TunnelError::NotFound)?;
        ensure_same_profile(current, lease)?;
        match current.state {
            TunnelState::Admitted => {
                self.device
                    .create_interface(&current.profile.interface_name)
                    .map_err(|_| TunnelError::UnsupportedPlatform)?;
                self.device
                    .configure_peer(&current.profile.interface_name, &current.profile.peer)
                    .map_err(|_| TunnelError::UnsupportedPlatform)?;
                current.state = TunnelState::Established;
                Ok(current.state)
            }
            TunnelState::Established => Ok(TunnelState::Established),
            _ => Err(TunnelError::InvalidState),
        }
    }

    fn rotate_peer(
        &mut self,
        lease: &TunnelLease,
        peer: &MeshPeerIdentity,
    ) -> Result<(), TunnelError> {
        let current = self
            .leases
            .get_mut(&lease.lease_id)
            .ok_or(TunnelError::NotFound)?;
        ensure_same_profile(current, lease)?;
        if current.state != TunnelState::Established {
            return Err(TunnelError::InvalidState);
        }
        if peer.network_id != current.profile.network_id {
            return Err(TunnelError::RevokedPeer);
        }
        self.device
            .configure_peer(&current.profile.interface_name, peer)
            .map_err(|_| TunnelError::UnsupportedPlatform)?;
        current.profile.peer = peer.clone();
        Ok(())
    }

    fn revoke(&mut self, lease: &TunnelLease) -> Result<(), TunnelError> {
        let current = self
            .leases
            .get_mut(&lease.lease_id)
            .ok_or(TunnelError::NotFound)?;
        ensure_same_profile(current, lease)?;
        match current.state {
            TunnelState::Revoked => Ok(()),
            TunnelState::Released => Err(TunnelError::InvalidState),
            _ => {
                current.state = TunnelState::Revoked;
                Ok(())
            }
        }
    }

    fn release(&mut self, lease: TunnelLease) -> Result<(), TunnelError> {
        let current = self
            .leases
            .get_mut(&lease.lease_id)
            .ok_or(TunnelError::NotFound)?;
        ensure_same_profile(current, &lease)?;
        if current.state == TunnelState::Released {
            return Ok(());
        }
        self.device
            .remove_interface(&current.profile.interface_name)
            .map_err(|_| TunnelError::UnsupportedPlatform)?;
        current.state = TunnelState::Released;
        Ok(())
    }
}

/// Linux WireGuard device adapter using the kernel's userspace UAPI. The
/// public-key material is deliberately supplied out-of-band by admission; a
/// PTR identity digest is never guessed or reinterpreted as a WireGuard key.
#[cfg(all(feature = "wireguard-uapi-backend", target_os = "linux"))]
pub struct LinuxWireguardDevice {
    socket_root: PathBuf,
    public_keys: BTreeMap<String, [u8; 32]>,
}

#[cfg(all(feature = "wireguard-uapi-backend", target_os = "linux"))]
impl LinuxWireguardDevice {
    pub fn new(socket_root: impl Into<PathBuf>) -> Self {
        Self {
            socket_root: socket_root.into(),
            public_keys: BTreeMap::new(),
        }
    }

    pub fn register_public_key(&mut self, peer: &MeshPeerIdentity, key: [u8; 32]) {
        self.public_keys.insert(peer.peer_id.0.clone(), key);
    }

    fn socket_path(&self, name: &str) -> PathBuf {
        self.socket_root.join(format!("{name}.sock"))
    }
}

#[cfg(all(feature = "wireguard-uapi-backend", target_os = "linux"))]
impl WireguardDevice for LinuxWireguardDevice {
    type Error = String;

    fn create_interface(&mut self, name: &str) -> Result<(), Self::Error> {
        validate_interface_name(name)?;
        let status = std::process::Command::new("ip")
            .args(["link", "add", "dev", name, "type", "wireguard"])
            .status()
            .map_err(|error| error.to_string())?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("ip link add failed with {status}"))
        }
    }

    fn configure_peer(&mut self, name: &str, peer: &MeshPeerIdentity) -> Result<(), Self::Error> {
        validate_interface_name(name)?;
        let key = self
            .public_keys
            .get(&peer.peer_id.0)
            .copied()
            .ok_or_else(|| "missing admitted WireGuard public key".to_owned())?;
        let request = wireguard_uapi::xplatform::set::Device {
            peers: vec![wireguard_uapi::xplatform::set::Peer::from_public_key(key)],
            ..Default::default()
        };
        wireguard_uapi::xplatform::Client::create(self.socket_path(name))
            .set(request)
            .map_err(|error| error.to_string())
    }

    fn remove_interface(&mut self, name: &str) -> Result<(), Self::Error> {
        validate_interface_name(name)?;
        let status = std::process::Command::new("ip")
            .args(["link", "del", "dev", name])
            .status()
            .map_err(|error| error.to_string())?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("ip link del failed with {status}"))
        }
    }
}

#[cfg(all(feature = "wireguard-uapi-backend", target_os = "linux"))]
fn validate_interface_name(name: &str) -> Result<(), String> {
    if name.is_empty()
        || name.len() > 15
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
    {
        Err("invalid WireGuard interface name".to_owned())
    } else {
        Ok(())
    }
}

/// Windows Wintun adapter. Wintun supplies the TUN interface; WireGuard
/// cryptographic peer handling remains a separate userspace dataplane (for
/// example BoringTun) and is never confused with the PTR identity digest.
#[cfg(all(feature = "wintun-backend", target_os = "windows"))]
pub struct WintunDevice {
    dll_path: std::path::PathBuf,
    adapter: Option<std::sync::Arc<wintun::Adapter>>,
    session: Option<wintun::Session>,
}

#[cfg(all(feature = "wintun-backend", target_os = "windows"))]
impl WintunDevice {
    pub fn new(dll_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            dll_path: dll_path.into(),
            adapter: None,
            session: None,
        }
    }

    pub fn probe_driver_version(&self) -> Result<String, String> {
        let wintun =
            unsafe { wintun::load_from_path(&self.dll_path) }.map_err(|error| error.to_string())?;
        wintun::get_running_driver_version(&wintun)
            .map(|version| version.to_string())
            .map_err(|error| error.to_string())
    }
}

#[cfg(all(feature = "wintun-backend", target_os = "windows"))]
impl WireguardDevice for WintunDevice {
    type Error = String;

    fn create_interface(&mut self, name: &str) -> Result<(), Self::Error> {
        if name.is_empty() || name.len() > 64 {
            return Err("invalid Wintun adapter name".into());
        }
        let wintun =
            unsafe { wintun::load_from_path(&self.dll_path) }.map_err(|error| error.to_string())?;
        let adapter = wintun::Adapter::create(&wintun, name, "PTR Mesh", None)
            .map_err(|error| error.to_string())?;
        let session = adapter
            .start_session(wintun::MAX_RING_CAPACITY)
            .map_err(|error| error.to_string())?;
        self.adapter = Some(adapter);
        self.session = Some(session);
        Ok(())
    }

    fn configure_peer(&mut self, _name: &str, peer: &MeshPeerIdentity) -> Result<(), Self::Error> {
        if peer.peer_id.0.is_empty() || peer.public_key_digest == [0; 32] {
            return Err("invalid admitted mesh peer".into());
        }
        if self.session.is_none() {
            return Err("Wintun session is not established".into());
        }
        Ok(())
    }

    fn remove_interface(&mut self, _name: &str) -> Result<(), Self::Error> {
        if let Some(session) = self.session.take() {
            session.shutdown().map_err(|error| error.to_string())?;
        }
        if let Some(adapter) = self.adapter.take() {
            match std::sync::Arc::try_unwrap(adapter) {
                Ok(adapter) => adapter.delete().map_err(|error| error.to_string())?,
                Err(_) => return Err("Wintun adapter still has active handles".into()),
            }
        }
        Ok(())
    }
}

#[derive(Default)]
pub struct ReferenceMeshTunnelExecutor {
    next_id: u64,
    leases: std::collections::BTreeMap<String, TunnelLease>,
}

impl MeshTunnelExecutor for ReferenceMeshTunnelExecutor {
    fn admit(&mut self, profile: &TunnelProfile) -> Result<TunnelLease, TunnelError> {
        if profile.network_id.0.is_empty()
            || profile.peer.peer_id.0.is_empty()
            || profile.peer.public_key_digest == [0; 32]
            || profile.peer.network_id != profile.network_id
            || profile.generation.0 == 0
            || profile.membership_revision.0 == 0
            || profile.placement_epoch == 0
            || profile.fencing_token == 0
            || profile.interface_name.is_empty()
        {
            return Err(TunnelError::InvalidProfile);
        }
        let lease_id = format!("tunnel-{}", self.next_id + 1);
        self.next_id += 1;
        let lease = TunnelLease {
            lease_id: lease_id.clone(),
            profile: profile.clone(),
            state: TunnelState::Admitted,
        };
        self.leases.insert(lease_id, lease.clone());
        Ok(lease)
    }

    fn establish(&mut self, lease: &TunnelLease) -> Result<TunnelState, TunnelError> {
        let current = self
            .leases
            .get_mut(&lease.lease_id)
            .ok_or(TunnelError::NotFound)?;
        if current.profile != lease.profile || current.state != TunnelState::Admitted {
            return Err(TunnelError::InvalidState);
        }
        current.state = TunnelState::Established;
        Ok(current.state)
    }

    fn rotate_peer(
        &mut self,
        lease: &TunnelLease,
        peer: &MeshPeerIdentity,
    ) -> Result<(), TunnelError> {
        let current = self
            .leases
            .get_mut(&lease.lease_id)
            .ok_or(TunnelError::NotFound)?;
        if current.profile != lease.profile || current.state != TunnelState::Established {
            return Err(TunnelError::InvalidState);
        }
        if peer.network_id != current.profile.network_id || peer.public_key_digest == [0; 32] {
            return Err(TunnelError::RevokedPeer);
        }
        current.profile.peer = peer.clone();
        current.profile.membership_revision = current.profile.membership_revision.next();
        Ok(())
    }

    fn revoke(&mut self, lease: &TunnelLease) -> Result<(), TunnelError> {
        let current = self
            .leases
            .get_mut(&lease.lease_id)
            .ok_or(TunnelError::NotFound)?;
        if current.profile != lease.profile {
            return Err(TunnelError::StaleFencing);
        }
        if matches!(current.state, TunnelState::Released | TunnelState::Revoked) {
            return Err(TunnelError::InvalidState);
        }
        current.state = TunnelState::Revoked;
        Ok(())
    }

    fn release(&mut self, lease: TunnelLease) -> Result<(), TunnelError> {
        let current = self
            .leases
            .get_mut(&lease.lease_id)
            .ok_or(TunnelError::NotFound)?;
        if current.profile.fencing_token != lease.profile.fencing_token {
            return Err(TunnelError::StaleFencing);
        }
        if current.state == TunnelState::Released {
            return Ok(());
        }
        current.state = TunnelState::Released;
        Ok(())
    }
}

#[allow(dead_code)]
fn _peer_key(peer: &MeshPeerIdentity) -> PeerId {
    peer.peer_id.clone()
}

fn validate_profile(profile: &TunnelProfile) -> Result<(), TunnelError> {
    if profile.network_id.0.is_empty()
        || profile.peer.peer_id.0.is_empty()
        || profile.peer.public_key_digest == [0; 32]
        || profile.peer.network_id != profile.network_id
        || profile.generation.0 == 0
        || profile.membership_revision.0 == 0
        || profile.placement_epoch == 0
        || profile.fencing_token == 0
        || profile.interface_name.is_empty()
    {
        Err(TunnelError::InvalidProfile)
    } else {
        Ok(())
    }
}

fn ensure_same_profile(current: &TunnelLease, presented: &TunnelLease) -> Result<(), TunnelError> {
    if current.profile != presented.profile {
        Err(TunnelError::StaleFencing)
    } else {
        Ok(())
    }
}
