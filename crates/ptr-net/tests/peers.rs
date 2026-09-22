//! Where a peer may be dialled, and what happens when that is wrong.
//!
//! Three layers, and each test here disables reliance on the others so that none of
//! them is load-bearing by accident:
//!
//! 1. **The key is the address.** An address is filed under the id it carries, so
//!    there is no `(peer, address)` pair to transpose and an attacker's address
//!    recorded by mistake lands under the attacker's own name.
//! 2. **An unrecorded peer is refused before a connection is attempted**, which is
//!    the only refusal that costs nothing and reveals nothing.
//! 3. **An address that lies about the right peer fails on the authenticated key.**
//!    This is the one that matters: a wrong address costs a failed request rather
//!    than a request served by the wrong node, and the test proves the node actually
//!    reached serves nothing.
use ptr_net::{BookError, EndpointAddr, IrohTransport, PeerBook, ALPN_BLOB};
use ptr_types::NodeId;

fn peer_of(address: &EndpointAddr) -> NodeId {
    NodeId(address.id.to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_address_is_recorded_under_the_peer_it_names_and_nowhere_else() {
    let alpha = IrohTransport::bind(&[]).await.unwrap();
    let beta = IrohTransport::bind(&[]).await.unwrap();
    let alpha_addr = alpha.direct_addr();
    let beta_addr = beta.direct_addr();

    let mut book = PeerBook::new();
    assert!(book.is_empty());
    assert!(book.record(alpha_addr.clone()).is_none());
    assert_eq!(book.len(), 1);

    // Recorded under the id the address carries. There is no argument by which it
    // could have been filed under anybody else, which is the point: an operator who
    // pastes the wrong address does not thereby point one peer's name at another
    // peer's socket — the wrong address simply appears under its own owner.
    let located = book
        .locate(&peer_of(&alpha_addr))
        .expect("the peer whose address this is");
    assert_eq!(located.peer(), &peer_of(&alpha_addr));
    assert_eq!(located.into_address(), alpha_addr);

    assert_eq!(
        book.locate(&peer_of(&beta_addr))
            .expect_err("beta's address was never recorded"),
        BookError::Unknown {
            peer: peer_of(&beta_addr)
        }
    );
    assert!(!book.knows(&peer_of(&beta_addr)));

    alpha.close().await;
    beta.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn re_recording_replaces_because_a_node_moves() {
    // Deliberately unlike `AdmissionPolicy::admit` and `PodAccessPolicy::admit`,
    // which refuse a second entry. An address is a fact about where a node is, and a
    // node legitimately moves; refusing an update would leave a deployment unable to
    // follow one. An admission grant is a decision, and a decision that changed
    // silently would be authority nobody took.
    let node = IrohTransport::bind(&[]).await.unwrap();
    let first = node.direct_addr();
    let moved = EndpointAddr::from_parts(first.id, []);

    let mut book = PeerBook::new();
    assert!(book.record(first.clone()).is_none());
    let replaced = book
        .record(moved.clone())
        .expect("recording again reports what it replaced");
    assert_eq!(replaced, first);
    assert_eq!(book.len(), 1, "one peer, one address");
    assert_eq!(book.locate(&peer_of(&first)).unwrap().into_address(), moved);

    node.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn forgetting_a_peer_takes_effect_on_the_next_lookup() {
    let node = IrohTransport::bind(&[]).await.unwrap();
    let address = node.direct_addr();
    let peer = peer_of(&address);

    let mut book = PeerBook::new();
    book.record(address);
    assert!(book.locate(&peer).is_ok());

    assert!(book.forget(&peer));
    assert_eq!(
        book.locate(&peer).expect_err("forgotten"),
        BookError::Unknown { peer: peer.clone() }
    );
    assert!(
        !book.forget(&peer),
        "forgetting twice reports that there was nothing to forget"
    );
    assert!(book.is_empty());

    node.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_address_that_lies_about_a_peer_fails_rather_than_reaching_the_liar() {
    // The gate's own sentence. Suppose somebody hands the deployment an address that
    // carries the *honest* node's id and somebody else's socket — the one thing the
    // book cannot detect, because the id it files under is the id it was given.
    //
    // What stops it is not the book. The key at the far end is authenticated: this
    // backend pins it in the TLS handshake, which the impostor cannot complete without
    // the honest node's private key, and `IrohTransport::request` compares the
    // authenticated id against the one asked for as a second line for a backend that
    // did not pin. So the cost of a wrong address is a failed request, not a request
    // served by the wrong node.
    //
    // The test asserts the property rather than an error string: the impostor serves
    // nothing, and the *same* requester reaching the honest node at its true address
    // does succeed — so the failure was about the lie and not about this fixture, the
    // ALPN, or the requester.
    let requester = IrohTransport::bind(&[]).await.unwrap();
    let honest = IrohTransport::bind(&[ALPN_BLOB]).await.unwrap();
    let impostor = IrohTransport::bind(&[ALPN_BLOB]).await.unwrap();

    let honest_addr = honest.direct_addr();
    let honest_id = honest_addr.id;
    let impostor_addr = impostor.direct_addr();
    let lie = EndpointAddr::from_parts(honest_id, impostor_addr.addrs.iter().cloned());

    let mut book = PeerBook::new();
    book.record(lie);
    // The book resolves it, because from here it is indistinguishable from the truth.
    let located = book
        .locate(&NodeId(honest_id.to_string()))
        .expect("the lie is filed under the id it carries");

    let impostor_served = tokio::spawn(async move {
        // Whatever the impostor would have served, it must not get the chance.
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            impostor.accept_once(1024),
        )
        .await;
        let answered = match outcome {
            Ok(Ok(incoming)) => {
                incoming.respond(b"served by the wrong node").await.ok();
                true
            }
            _ => false,
        };
        impostor.close().await;
        answered
    });

    requester
        .request(located.into_address(), ALPN_BLOB, b"anything", 1024)
        .await
        .expect_err("an address pointing one peer's id at another peer's socket");
    assert!(
        !impostor_served.await.unwrap(),
        "the node at that socket must not have served the request"
    );

    // The control, in the same test and with the same requester: the honest node at
    // its true address answers.
    let mut book = PeerBook::new();
    book.record(honest_addr.clone());
    let located = book.locate(&peer_of(&honest_addr)).unwrap();
    let serving = tokio::spawn(async move {
        let incoming = honest.accept_once(1024).await.unwrap();
        incoming.respond(b"pong").await.unwrap();
        honest.close().await;
    });
    let response = requester
        .request(located.into_address(), ALPN_BLOB, b"ping", 1024)
        .await
        .expect("the honest node at its own address answers");
    assert_eq!(response, b"pong");
    serving.await.unwrap();

    requester.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_control_a_truthful_address_from_the_book_does_reach_its_peer() {
    // Without this, every refusal above would also be what a book that resolved
    // nothing, or a transport that connected to nobody, looks like.
    let server = IrohTransport::bind(&[ALPN_BLOB]).await.unwrap();
    let client = IrohTransport::bind(&[]).await.unwrap();
    let address = server.direct_addr();

    let mut book = PeerBook::new();
    book.record(address.clone());
    let located = book.locate(&peer_of(&address)).unwrap();

    let serving = tokio::spawn(async move {
        let incoming = server.accept_once(1024).await.unwrap();
        incoming.respond(b"pong").await.unwrap();
        server.close().await;
    });

    let response = client
        .request(located.into_address(), ALPN_BLOB, b"ping", 1024)
        .await
        .expect("an address the deployment recorded reaches its peer");
    assert_eq!(response, b"pong");

    client.close().await;
    serving.await.unwrap();
}
