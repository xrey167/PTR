use ptr_net::{IrohTransport, ALPN_PODWIRE};
use tokio::sync::oneshot;

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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reusable_session_multiplexes_streams_and_enforces_backpressure() {
    let server = IrohTransport::bind(&[ALPN_PODWIRE]).await.unwrap();
    let client = IrohTransport::bind(&[]).await.unwrap();
    let address = server.direct_addr();

    let server_task = tokio::spawn(async move {
        let session = server.accept_session().await.unwrap();
        let mut responses = Vec::new();
        for _ in 0..2 {
            let incoming = session.accept_request(1024).await.unwrap();
            let response = incoming.payload.clone();
            responses.push(tokio::spawn(async move {
                incoming.respond(&response).await.unwrap();
            }));
        }
        for response in responses {
            response.await.unwrap();
        }
        session.wait_closed().await;
    });

    assert!(client
        .connect_session(address.clone(), ALPN_PODWIRE, 0)
        .await
        .is_err());

    let session = client
        .connect_session(address, ALPN_PODWIRE, 2)
        .await
        .unwrap();
    let first = session.request(b"first", 1024);
    let second = session.request(b"second", 1024);
    let (first, second) = tokio::join!(first, second);
    let first = first.unwrap();
    let second = second.unwrap();
    assert_eq!(first, b"first");
    assert_eq!(second, b"second");
    session.close().await;
    client.close().await;
    server_task.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_rejects_work_above_the_in_flight_limit_without_queueing_it() {
    let server = IrohTransport::bind(&[ALPN_PODWIRE]).await.unwrap();
    let client = IrohTransport::bind(&[]).await.unwrap();
    let address = server.direct_addr();
    let (accepted_tx, accepted_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();

    let server_task = tokio::spawn(async move {
        let session = server.accept_session().await.unwrap();
        let incoming = session.accept_request(1024).await.unwrap();
        accepted_tx.send(()).unwrap();
        release_rx.await.unwrap();
        incoming.respond(b"first-response").await.unwrap();
        session.wait_closed().await;
    });

    let session = client
        .connect_session(address, ALPN_PODWIRE, 1)
        .await
        .unwrap();
    let first = tokio::spawn({
        let session = session.clone();
        async move { session.request(b"first", 1024).await }
    });
    accepted_rx.await.unwrap();

    assert_eq!(
        session.request(b"second", 1024).await,
        Err("pod session backpressure limit reached".to_owned())
    );
    release_tx.send(()).unwrap();
    assert_eq!(first.await.unwrap().unwrap(), b"first-response");
    session.close().await;
    client.close().await;
    server_task.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_closed_session_can_be_replaced_by_a_fresh_session() {
    let server = IrohTransport::bind(&[ALPN_PODWIRE]).await.unwrap();
    let client = IrohTransport::bind(&[]).await.unwrap();
    let address = server.direct_addr();

    let server_task = tokio::spawn(async move {
        let first_session = server.accept_session().await.unwrap();
        let first = first_session.accept_request(1024).await.unwrap();
        first.respond(b"first-response").await.unwrap();
        first_session.wait_closed().await;

        let second_session = server.accept_session().await.unwrap();
        let second = second_session.accept_request(1024).await.unwrap();
        second.respond(b"second-response").await.unwrap();
        second_session.wait_closed().await;
        server.close().await;
    });

    let first = client
        .connect_session(address.clone(), ALPN_PODWIRE, 1)
        .await
        .unwrap();
    assert_eq!(
        first.request(b"first", 1024).await.unwrap(),
        b"first-response"
    );
    first.close().await;

    let second = client
        .connect_session(address, ALPN_PODWIRE, 1)
        .await
        .unwrap();
    assert_eq!(
        second.request(b"second", 1024).await.unwrap(),
        b"second-response"
    );
    second.close().await;
    client.close().await;
    server_task.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn closed_session_rejects_new_requests_and_a_new_session_recovers() {
    let server = IrohTransport::bind(&[ALPN_PODWIRE]).await.unwrap();
    let client = IrohTransport::bind(&[]).await.unwrap();
    let address = server.direct_addr();

    let server_task = tokio::spawn(async move {
        let first = server.accept_session().await.unwrap();
        let incoming = first.accept_request(1024).await.unwrap();
        incoming.respond(b"first").await.unwrap();

        let second = server.accept_session().await.unwrap();
        let incoming = second.accept_request(1024).await.unwrap();
        incoming.respond(b"recovered").await.unwrap();
        second.wait_closed().await;
        server.close().await;
    });

    let first = client
        .connect_session(address.clone(), ALPN_PODWIRE, 1)
        .await
        .unwrap();
    assert_eq!(first.request(b"one", 1024).await.unwrap(), b"first");
    first.close().await;
    assert!(first.request(b"after-close", 1024).await.is_err());

    let second = client
        .connect_session(address, ALPN_PODWIRE, 1)
        .await
        .unwrap();
    assert_eq!(second.request(b"retry", 1024).await.unwrap(), b"recovered");
    second.close().await;
    server_task.await.unwrap();
}
