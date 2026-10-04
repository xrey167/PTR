use ptr_types::NodeId;

mod authenticated;
pub use authenticated::AuthenticatedPeer;

mod signing;
pub use signing::{verify, NodeSigner, SigningDomain, SIGNATURE_BYTES};

mod mesh;
pub use mesh::{
    MembershipState, MeshEndpoint, MeshError, MeshInvitation, MeshMembership, MeshPeerIdentity,
    MeshRegistry, MeshRoute,
};

mod tunnel;
#[cfg(all(feature = "wireguard-uapi-backend", target_os = "linux"))]
pub use tunnel::LinuxWireguardDevice;
#[cfg(all(feature = "wintun-backend", target_os = "windows"))]
pub use tunnel::WintunDevice;
pub use tunnel::{
    MeshTunnelExecutor, ReferenceMeshTunnelExecutor, TunnelError, TunnelLease, TunnelProfile,
    TunnelState, WireguardDevice, WireguardUserspaceExecutor,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelayMode {
    Disabled,
    DirectOnly,
    DirectWithRelayFallback,
}

pub const ALPN_RAFT: &[u8] = b"ptr-raft/1";
pub const ALPN_PODWIRE: &[u8] = b"ptr-podwire/1";
pub const ALPN_MODEL: &[u8] = b"ptr-model/1";
pub const ALPN_BLOB: &[u8] = b"ptr-blob/1";
pub const ALPN_EVENTS: &[u8] = b"ptr-events/1";
/// The execution wire: an action requested across a network boundary.
///
/// Its own ALPN rather than `ALPN_PODWIRE`, because the pod wire is Pod access and
/// this one asks for effects. Two protocols sharing one ALPN is how a request meant
/// for one gets parsed by the other, and the parse that succeeds by accident is the
/// dangerous one.
pub const ALPN_EXEC: &[u8] = b"ptr-exec/1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NodeIdentity {
    pub id: NodeId,
    pub public_key: String,
}

pub trait Transport {
    fn send(&self, peer: &NodeIdentity, alpn: &[u8], payload: &[u8]) -> Result<(), String>;
}

#[cfg(feature = "iroh-backend")]
mod iroh_backend {
    use super::{AuthenticatedPeer, NodeIdentity, NodeSigner};
    use iroh::{
        endpoint::{Connection, SendStream},
        Endpoint, EndpointAddr,
    };
    use ptr_types::NodeId;
    use std::net::{Ipv4Addr, SocketAddrV4};
    use std::sync::Arc;
    use tokio::sync::Semaphore;

    pub struct IrohTransport {
        endpoint: Endpoint,
        relay_mode: super::RelayMode,
    }

    pub struct IrohIncoming {
        /// The key the QUIC handshake verified. Never a value from the payload.
        pub peer: AuthenticatedPeer,
        pub payload: Vec<u8>,
        #[allow(dead_code)]
        connection: Connection,
        send: SendStream,
        close_after_respond: bool,
    }

    /// A reusable authenticated QUIC connection. Each request still gets its
    /// own bidirectional stream, while the expensive endpoint handshake is
    /// shared. The semaphore is deliberately non-blocking: callers get an
    /// explicit backpressure error instead of accumulating unbounded work.
    #[derive(Clone)]
    pub struct IrohSession {
        connection: Connection,
        peer: AuthenticatedPeer,
        permits: Arc<Semaphore>,
    }

    pub struct IrohServerSession {
        /// The key the QUIC handshake verified.
        pub peer: AuthenticatedPeer,
        connection: Connection,
    }

    impl IrohTransport {
        pub async fn bind(alpns: &[&[u8]]) -> Result<Self, String> {
            Self::bind_with_relay_mode(alpns, super::RelayMode::Disabled).await
        }

        pub async fn bind_with_relay_mode(
            alpns: &[&[u8]],
            relay_mode: super::RelayMode,
        ) -> Result<Self, String> {
            let iroh_relay_mode = match relay_mode {
                super::RelayMode::Disabled | super::RelayMode::DirectOnly => {
                    iroh::RelayMode::Disabled
                }
                super::RelayMode::DirectWithRelayFallback => iroh::RelayMode::Default,
            };
            let endpoint = Endpoint::builder(iroh::endpoint::presets::Minimal)
                .relay_mode(iroh_relay_mode)
                .clear_ip_transports()
                .bind_addr(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))
                .map_err(|error| error.to_string())?
                .alpns(alpns.iter().map(|alpn| alpn.to_vec()).collect())
                .bind()
                .await
                .map_err(|error| error.to_string())?;

            Ok(Self {
                endpoint,
                relay_mode,
            })
        }

        pub fn relay_mode(&self) -> super::RelayMode {
            self.relay_mode
        }

        pub fn identity(&self) -> NodeIdentity {
            let public_key = self.endpoint.id().to_string();
            NodeIdentity {
                id: NodeId(public_key.clone()),
                public_key,
            }
        }

        /// A signer over this endpoint's own key (D2 in issue #23).
        ///
        /// The key that signs is the key the QUIC handshake proves, so a statement
        /// this signer makes is attributable to exactly the identity a connection to
        /// this endpoint authenticates — and to no other. The secret never leaves
        /// `ptr-net`: the signer signs under a domain and has no accessor for it.
        pub fn signer(&self) -> NodeSigner {
            NodeSigner::from_secret(&self.endpoint.secret_key().to_bytes())
        }

        pub fn direct_addr(&self) -> EndpointAddr {
            self.endpoint.addr()
        }

        pub async fn request(
            &self,
            peer: EndpointAddr,
            alpn: &[u8],
            payload: &[u8],
            max_response: usize,
        ) -> Result<Vec<u8>, String> {
            let expected_peer = peer.id;
            let connection = self
                .endpoint
                .connect(peer, alpn)
                .await
                .map_err(|error| error.to_string())?;

            let authenticated_peer = connection.remote_id();
            if authenticated_peer != expected_peer {
                return Err("authenticated Iroh peer does not match requested endpoint id".into());
            }

            let (mut send, mut recv) = connection
                .open_bi()
                .await
                .map_err(|error| error.to_string())?;
            send.write_all(payload)
                .await
                .map_err(|error| error.to_string())?;
            send.finish().map_err(|error| error.to_string())?;

            let response = recv
                .read_to_end(max_response)
                .await
                .map_err(|error| error.to_string())?;
            connection.close(0u32.into(), b"ptr request complete");
            Ok(response)
        }

        pub async fn connect_session(
            &self,
            peer: EndpointAddr,
            alpn: &[u8],
            max_in_flight: usize,
        ) -> Result<IrohSession, String> {
            if max_in_flight == 0 {
                return Err("max_in_flight must be greater than zero".to_owned());
            }
            let expected_peer = peer.id;
            let connection = self
                .endpoint
                .connect(peer, alpn)
                .await
                .map_err(|error| error.to_string())?;
            let authenticated_peer = connection.remote_id();
            if authenticated_peer != expected_peer {
                connection.close(0u32.into(), b"ptr peer mismatch");
                return Err("authenticated Iroh peer does not match requested endpoint id".into());
            }
            Ok(IrohSession {
                connection,
                peer: AuthenticatedPeer::from_transport(NodeId(authenticated_peer.to_string())),
                permits: Arc::new(Semaphore::new(max_in_flight)),
            })
        }

        pub async fn accept_once(&self, max_request: usize) -> Result<IrohIncoming, String> {
            let incoming =
                self.endpoint.accept().await.ok_or_else(|| {
                    "Iroh endpoint closed before accepting a connection".to_string()
                })?;
            let connection = incoming.await.map_err(|error| error.to_string())?;
            read_request(connection, max_request).await
        }

        pub async fn accept_session(&self) -> Result<IrohServerSession, String> {
            let incoming =
                self.endpoint.accept().await.ok_or_else(|| {
                    "Iroh endpoint closed before accepting a connection".to_owned()
                })?;
            let connection = incoming.await.map_err(|error| error.to_string())?;
            Ok(IrohServerSession {
                peer: AuthenticatedPeer::from_transport(NodeId(connection.remote_id().to_string())),
                connection,
            })
        }

        pub async fn close(&self) {
            self.endpoint.close().await;
        }
    }

    /// Read the one request a connection carries: the per-connection half of
    /// `accept_once`, shared with the accept loop so the two cannot drift.
    async fn read_request(
        connection: Connection,
        max_request: usize,
    ) -> Result<IrohIncoming, String> {
        let peer = AuthenticatedPeer::from_transport(NodeId(connection.remote_id().to_string()));
        let (send, mut recv) = connection
            .accept_bi()
            .await
            .map_err(|error| error.to_string())?;
        let payload = recv
            .read_to_end(max_request)
            .await
            .map_err(|error| error.to_string())?;
        Ok(IrohIncoming {
            peer,
            payload,
            connection,
            send,
            close_after_respond: true,
        })
    }

    impl IrohIncoming {
        pub async fn respond(mut self, payload: &[u8]) -> Result<(), String> {
            self.send
                .write_all(payload)
                .await
                .map_err(|error| error.to_string())?;
            self.send.finish().map_err(|error| error.to_string())?;
            if self.close_after_respond {
                self.connection.closed().await;
            }
            Ok(())
        }
    }

    impl IrohSession {
        pub fn peer(&self) -> &AuthenticatedPeer {
            &self.peer
        }

        pub async fn request(
            &self,
            payload: &[u8],
            max_response: usize,
        ) -> Result<Vec<u8>, String> {
            let _permit = self
                .permits
                .clone()
                .try_acquire_owned()
                .map_err(|_| "pod session backpressure limit reached".to_owned())?;
            let (mut send, mut recv) = self
                .connection
                .open_bi()
                .await
                .map_err(|error| error.to_string())?;
            send.write_all(payload)
                .await
                .map_err(|error| error.to_string())?;
            send.finish().map_err(|error| error.to_string())?;
            recv.read_to_end(max_response)
                .await
                .map_err(|error| error.to_string())
        }

        pub async fn close(&self) {
            self.connection.close(0u32.into(), b"ptr session closed");
            self.connection.closed().await;
        }

        /// Wait until the connection is closed, by either side.
        pub async fn wait_closed(&self) {
            self.connection.closed().await;
        }
    }

    impl IrohServerSession {
        pub async fn accept_request(&self, max_request: usize) -> Result<IrohIncoming, String> {
            let (send, mut recv) = self
                .connection
                .accept_bi()
                .await
                .map_err(|error| error.to_string())?;
            let payload = recv
                .read_to_end(max_request)
                .await
                .map_err(|error| error.to_string())?;
            Ok(IrohIncoming {
                peer: self.peer.clone(),
                payload,
                connection: self.connection.clone(),
                send,
                close_after_respond: false,
            })
        }

        pub async fn close(&self) {
            self.connection
                .close(0u32.into(), b"ptr server session closed");
            self.connection.closed().await;
        }

        pub async fn wait_closed(&self) {
            self.connection.closed().await;
        }
    }

    /// The accept loop (D4 in issue #23), in `src/iroh_backend/serve.rs`. A child
    /// module, so it reads a request exactly as `accept_once` does without widening
    /// anything here.
    mod serve;
    pub use serve::{serve, ServeError, ServePolicy, ServeReport};

    pub use IrohIncoming as Incoming;
    pub use IrohServerSession as ServerSession;
    pub use IrohSession as Session;
    pub use IrohTransport as Transport;
    // Re-exported so a composing crate can name an address without declaring its
    // own iroh dependency: two pins of a transport would be two wire formats.
    pub use iroh::EndpointAddr as Address;
    // The id an address is built from, for a host that reads one from its own
    // configuration (`ptrd`'s `[[wire.peers]]`).
    pub use iroh::EndpointId as Id;
}

#[cfg(feature = "iroh-backend")]
pub use iroh_backend::{
    serve, Address as EndpointAddr, Id as EndpointId, Incoming as IrohIncoming, ServeError,
    ServePolicy, ServeReport, ServerSession as IrohServerSession, Session as IrohSession,
    Transport as IrohTransport,
};

/// Where the deployment says each peer can be reached.
///
/// Behind the transport feature because an address is the transport's type: a book of
/// addresses with no transport to dial them would be a map of nowhere.
#[cfg(feature = "iroh-backend")]
mod peers;

#[cfg(feature = "iroh-backend")]
pub use peers::{BookError, PeerAddress, PeerBook};
