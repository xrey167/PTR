use ptr_pods::{
    EgressPolicy, NativeProtocolRequest, NetworkEndpoint, PodLink, ProtocolBinding,
    TcpProtocolExecutor, UdpProtocolExecutor,
};
use ptr_protocol::TypedPayload;
use ptr_types::{
    ArtifactId, CapabilityId, Generation, NamespaceId, PodAddress, PodId, PodRevisionAddress,
    ProjectId, RequestId, Timestamp, TypeId,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, UdpSocket};

fn link(protocol: ProtocolBinding, port: u16) -> PodLink {
    PodLink {
        trace_id: "native-test".into(),
        source: PodAddress::new(
            ProjectId::from("project"),
            NamespaceId::from("ns"),
            PodId::from("caller"),
        )
        .unwrap(),
        target: PodRevisionAddress {
            address: PodAddress::new(
                ProjectId::from("project"),
                NamespaceId::from("ns"),
                PodId::from("network"),
            )
            .unwrap(),
            semantic_revision: [1; 32],
            generation: Generation(1),
        },
        artifact_id: ArtifactId::from("network-artifact"),
        protocol,
        capability: CapabilityId::from("exchange"),
        acl: Vec::new(),
        deadline: Timestamp(u64::MAX),
        hop_limit: 4,
        visited: Vec::new(),
        attestation: [2; 32],
        egress_policy: EgressPolicy {
            hosts: vec!["127.0.0.1".into()],
            ports: vec![port],
            topics: Vec::new(),
        },
    }
}

fn request(link: PodLink, port: u16) -> NativeProtocolRequest {
    NativeProtocolRequest {
        link,
        input: TypedPayload {
            type_id: TypeId::from("Request"),
            bytes: b"hello".to_vec(),
        },
        request_id: RequestId::from("native-request"),
        endpoint: NetworkEndpoint {
            host: "127.0.0.1".into(),
            port,
        },
    }
}

async fn read_frame<S: AsyncReadExt + Unpin>(stream: &mut S) -> (String, Vec<u8>) {
    let mut header = [0; 8];
    stream.read_exact(&mut header).await.unwrap();
    let type_len = u16::from_be_bytes([header[0], header[1]]) as usize;
    let body_len = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;
    let mut type_bytes = vec![0; type_len];
    let mut body = vec![0; body_len];
    stream.read_exact(&mut type_bytes).await.unwrap();
    stream.read_exact(&mut body).await.unwrap();
    (String::from_utf8(type_bytes).unwrap(), body)
}

fn response_frame(type_id: &str, body: &[u8]) -> Vec<u8> {
    let mut frame = Vec::new();
    frame.extend_from_slice(&(type_id.len() as u16).to_be_bytes());
    frame.extend_from_slice(&[0, 0]);
    frame.extend_from_slice(&(body.len() as u32).to_be_bytes());
    frame.extend_from_slice(type_id.as_bytes());
    frame.extend_from_slice(body);
    frame
}

#[tokio::test]
async fn tcp_executor_uses_bounded_typed_frames_and_policy() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        assert_eq!(
            read_frame(&mut stream).await,
            ("Request".into(), b"hello".to_vec())
        );
        stream
            .write_all(&response_frame("Response", b"ok"))
            .await
            .unwrap();
    });

    let response = TcpProtocolExecutor::default()
        .execute(request(link(ProtocolBinding::Tcp, port), port))
        .await
        .unwrap();
    assert_eq!(response.protocol, ProtocolBinding::Tcp);
    assert_eq!(response.output.type_id, TypeId::from("Response"));
    assert_eq!(response.output.bytes, b"ok");
    server.await.unwrap();
}

#[tokio::test]
async fn udp_executor_round_trips_a_typed_datagram_and_rejects_unlisted_ports() {
    let server = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let port = server.local_addr().unwrap().port();
    let responder = tokio::spawn(async move {
        let mut buffer = [0; 1024];
        let (length, peer) = server.recv_from(&mut buffer).await.unwrap();
        let type_len = u16::from_be_bytes([buffer[0], buffer[1]]) as usize;
        assert_eq!(&buffer[8 + type_len..length], b"hello");
        server
            .send_to(&response_frame("Response", b"udp-ok"), peer)
            .await
            .unwrap();
    });

    let response = UdpProtocolExecutor::default()
        .execute(request(link(ProtocolBinding::Udp, port), port))
        .await
        .unwrap();
    assert_eq!(response.output.bytes, b"udp-ok");
    responder.await.unwrap();

    let rejected = UdpProtocolExecutor::default()
        .execute(request(
            link(ProtocolBinding::Udp, port),
            port.saturating_add(1),
        ))
        .await;
    assert!(matches!(
        rejected,
        Err(ptr_pods::ProtocolError::InvalidEndpoint)
    ));
}
