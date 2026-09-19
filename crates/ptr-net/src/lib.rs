use ptr_types::NodeId;

pub const ALPN_RAFT: &[u8] = b"ptr-raft/1";
pub const ALPN_PODWIRE: &[u8] = b"ptr-podwire/1";
pub const ALPN_MODEL: &[u8] = b"ptr-model/1";
pub const ALPN_BLOB: &[u8] = b"ptr-blob/1";
pub const ALPN_EVENTS: &[u8] = b"ptr-events/1";

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
    use super::NodeIdentity;
    use iroh::{
        endpoint::{presets, Connection, SendStream},
        Endpoint, EndpointAddr, RelayMode, TransportAddr,
    };
    use ptr_types::NodeId;
    use std::net::{Ipv4Addr, SocketAddrV4};

    pub struct IrohTransport {
        endpoint: Endpoint,
    }

    pub struct IrohIncoming {
        pub peer: NodeIdentity,
        pub payload: Vec<u8>,
        connection: Connection,
        send: SendStream,
    }

    impl IrohTransport {
        pub async fn bind(alpns: &[&[u8]]) -> Result<Self, String> {
            let endpoint = Endpoint::builder(presets::Minimal)
                .relay_mode(RelayMode::Disabled)
                .bind_addr(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))
                .map_err(|error| error.to_string())?
                .alpns(alpns.iter().map(|alpn| alpn.to_vec()).collect())
                .bind()
                .await
                .map_err(|error| error.to_string())?;

            Ok(Self { endpoint })
        }

        pub fn identity(&self) -> NodeIdentity {
            let public_key = self.endpoint.id().to_string();
            NodeIdentity {
                id: NodeId(public_key.clone()),
                public_key,
            }
        }

        pub fn direct_addr(&self) -> EndpointAddr {
            EndpointAddr::from_parts(
                self.endpoint.id(),
                self.endpoint
                    .bound_sockets()
                    .into_iter()
                    .map(TransportAddr::Ip),
            )
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

        pub async fn accept_once(&self, max_request: usize) -> Result<IrohIncoming, String> {
            let incoming =
                self.endpoint.accept().await.ok_or_else(|| {
                    "Iroh endpoint closed before accepting a connection".to_string()
                })?;
            let connection = incoming.await.map_err(|error| error.to_string())?;
            let peer_id = connection.remote_id();
            let public_key = peer_id.to_string();

            let (send, mut recv) = connection
                .accept_bi()
                .await
                .map_err(|error| error.to_string())?;
            let payload = recv
                .read_to_end(max_request)
                .await
                .map_err(|error| error.to_string())?;

            Ok(IrohIncoming {
                peer: NodeIdentity {
                    id: NodeId(public_key.clone()),
                    public_key,
                },
                payload,
                connection,
                send,
            })
        }

        pub async fn close(&self) {
            self.endpoint.close().await;
        }
    }

    impl IrohIncoming {
        pub async fn respond(mut self, payload: &[u8]) -> Result<(), String> {
            self.send
                .write_all(payload)
                .await
                .map_err(|error| error.to_string())?;
            self.send.finish().map_err(|error| error.to_string())?;
            self.connection.closed().await;
            Ok(())
        }
    }

    pub use IrohIncoming as Incoming;
    pub use IrohTransport as Transport;
}

#[cfg(feature = "iroh-backend")]
pub use iroh_backend::{Incoming as IrohIncoming, Transport as IrohTransport};
