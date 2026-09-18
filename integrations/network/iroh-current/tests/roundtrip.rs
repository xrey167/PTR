use ptr_iroh_current::CurrentIrohEndpoint;
use ptr_net::ALPN_PODWIRE;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn current_iroh_roundtrip_uses_authenticated_endpoint_identity() {
    let server = CurrentIrohEndpoint::bind(&[ALPN_PODWIRE]).await.unwrap();
    let client = CurrentIrohEndpoint::bind(&[]).await.unwrap();

    let client_identity = client.identity();
    let server_addr = server.addr();

    let server_task = tokio::spawn(async move {
        let incoming = server.accept_once(1024).await.unwrap();
        assert_eq!(incoming.peer.public_key, client_identity.public_key);
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
