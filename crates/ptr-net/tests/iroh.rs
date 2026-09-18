use ptr_net::{IrohTransport, ALPN_PODWIRE};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn direct_local_roundtrip_authenticates_peer_identity() {
    let server = IrohTransport::bind(&[ALPN_PODWIRE]).await.unwrap();
    let client = IrohTransport::bind(&[]).await.unwrap();

    let expected_client = client.identity();
    let server_addr = server.direct_addr();

    let server_task = tokio::spawn(async move {
        let incoming = server.accept_once(1024).await.unwrap();
        assert_eq!(incoming.peer.public_key, expected_client.public_key);
        assert_eq!(incoming.payload, b"ping");
        incoming.respond(b"pong").await.unwrap();
        server.close().await;
    });

    let response = client
        .request(server_addr, ALPN_PODWIRE, b"ping", 1024)
        .await
        .unwrap();
    assert_eq!(response, b"pong");

    client.close().await;
    server_task.await.unwrap();
}
