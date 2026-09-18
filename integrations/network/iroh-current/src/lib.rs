use iroh::{
    endpoint::{presets, Connection, SendStream},
    Endpoint, EndpointAddr,
};
use ptr_net::NodeIdentity;
use ptr_types::NodeId;

pub struct CurrentIrohEndpoint {
    endpoint: Endpoint,
}

pub struct CurrentIrohIncoming {
    pub peer: NodeIdentity,
    pub payload: Vec<u8>,
    connection: Connection,
    send: SendStream,
}

impl CurrentIrohEndpoint {
    pub async fn bind(alpns: &[&[u8]]) -> Result<Self, String> {
        let endpoint = Endpoint::builder(presets::Minimal)
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

    pub fn addr(&self) -> EndpointAddr {
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

        if connection.remote_id() != expected_peer {
            return Err("authenticated Iroh endpoint id does not match requested peer".into());
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

    pub async fn accept_once(&self, max_request: usize) -> Result<CurrentIrohIncoming, String> {
        let incoming = self
            .endpoint
            .accept()
            .await
            .ok_or_else(|| "Iroh endpoint closed before incoming connection".to_string())?;
        let connection = incoming.await.map_err(|error| error.to_string())?;
        let remote = connection.remote_id();
        let public_key = remote.to_string();
        let (send, mut recv) = connection
            .accept_bi()
            .await
            .map_err(|error| error.to_string())?;
        let payload = recv
            .read_to_end(max_request)
            .await
            .map_err(|error| error.to_string())?;

        Ok(CurrentIrohIncoming {
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

impl CurrentIrohIncoming {
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
