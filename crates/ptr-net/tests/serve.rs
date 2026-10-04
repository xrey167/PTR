//! The accept loop over real connections (D4 in issue #23).
//!
//! Each test drives the loop and its clients in one `tokio::join!`, and every wait is
//! on an observable the test controls — a count of handlers started, a release the
//! test sends, a connection's own close — rather than on a sleep that hopes the other
//! side has caught up.
use ptr_net::{serve, IrohIncoming, IrohTransport, ServeError, ServePolicy, ServeReport};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::watch;

/// A test-local protocol label. The loop is protocol-agnostic, and borrowing a real
/// wire's ALPN for bytes that are not that wire would make a label look like a use.
const ALPN_TEST: &[u8] = b"ptr-net-serve-test/1";

/// A policy with the given limit and deadline, and the sender that stops it.
fn policy_with(max_concurrent: usize, deadline: Duration) -> (ServePolicy, watch::Sender<bool>) {
    let (stop, shutdown) = watch::channel(false);
    (
        ServePolicy {
            max_concurrent: NonZeroUsize::new(max_concurrent).unwrap(),
            per_connection_deadline: deadline,
            shutdown,
        },
        stop,
    )
}

/// Wait until `count` reaches `target`, by yielding rather than by guessing a delay.
async fn until(count: &AtomicUsize, target: usize) {
    while count.load(Ordering::SeqCst) < target {
        tokio::task::yield_now().await;
    }
}

/// Answer with the request's own bytes.
async fn echo(incoming: IrohIncoming) -> Result<(), String> {
    let echo = incoming.payload.clone();
    incoming.respond(&echo).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_connection_over_the_limit_is_refused_and_the_limit_is_served() {
    const LIMIT: usize = 3;
    let server = IrohTransport::bind(&[ALPN_TEST]).await.unwrap();
    let address = server.direct_addr();
    let (policy, stop) = policy_with(LIMIT, Duration::from_secs(20));

    // Each handler holds its slot until the test releases it, so the limit is full
    // exactly when the count of started handlers says it is.
    let started = Arc::new(AtomicUsize::new(0));
    let (release, released) = watch::channel(false);
    let handler = {
        let started = Arc::clone(&started);
        move |incoming: IrohIncoming| {
            let started = Arc::clone(&started);
            let mut released = released.clone();
            async move {
                started.fetch_add(1, Ordering::SeqCst);
                released
                    .wait_for(|go| *go)
                    .await
                    .map_err(|error| error.to_string())?;
                echo(incoming).await
            }
        }
    };

    let driving = async {
        let mut held = Vec::new();
        for index in 0..LIMIT {
            let client = IrohTransport::bind(&[]).await.unwrap();
            let address = address.clone();
            let byte = u8::try_from(index).unwrap();
            // The client goes back with its answer and stays open until the end: a
            // peer that vanishes without closing leaves its connection for the
            // deadline to collect, which is the next test's subject, not this one's.
            held.push(tokio::spawn(async move {
                let answer = client.request(address, ALPN_TEST, &[byte], 64).await;
                (answer, client)
            }));
        }
        until(&started, LIMIT).await;

        // One more, while every slot is taken: refused, not parked.
        let extra = IrohTransport::bind(&[]).await.unwrap();
        let refused = tokio::time::timeout(
            Duration::from_secs(10),
            extra.request(address.clone(), ALPN_TEST, b"one too many", 64),
        )
        .await
        .expect("a refusal is immediate rather than a wait for a free slot");

        release.send(true).unwrap();
        let mut answers = Vec::new();
        let mut clients = Vec::new();
        for handle in held {
            let (answer, client) = handle.await.unwrap();
            answers.push(answer);
            clients.push(client);
        }
        stop.send(true).unwrap();
        (answers, refused, clients)
    };
    let (report, (answers, refused, _clients)) =
        tokio::join!(serve(&server, 1024, policy, handler), driving);

    assert!(refused.is_err(), "the connection over the limit is refused");
    for (index, answer) in answers.into_iter().enumerate() {
        assert_eq!(
            answer.expect("the control: every connection within the limit is served"),
            vec![u8::try_from(index).unwrap()]
        );
    }
    assert_eq!(
        report.unwrap(),
        ServeReport {
            handled: 3,
            refused_over_limit: 1,
            cut_at_deadline: 0,
            failed: 0,
        }
    );
    assert_eq!(
        started.load(Ordering::SeqCst),
        LIMIT,
        "the refused connection never reached a handler"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_hanging_peer_is_cut_at_the_deadline_without_holding_up_another() {
    const DEADLINE: Duration = Duration::from_millis(1500);
    let server = IrohTransport::bind(&[ALPN_TEST]).await.unwrap();
    let address = server.direct_addr();
    let (policy, stop) = policy_with(2, DEADLINE);

    let hanging = IrohTransport::bind(&[]).await.unwrap();
    let honest = IrohTransport::bind(&[]).await.unwrap();
    let driving = async {
        // Completes the handshake and then never opens a stream: the loop is left
        // waiting for a request that is not coming.
        let opened = Instant::now();
        let session = hanging
            .connect_session(address.clone(), ALPN_TEST, 1)
            .await
            .unwrap();

        // Another peer, while the first hangs, is answered — well inside the
        // deadline, so it was not waiting behind the hanging one.
        let asked = Instant::now();
        let answer = honest
            .request(address.clone(), ALPN_TEST, b"honest", 64)
            .await;
        let waited = asked.elapsed();

        // The host closes the hanging connection at its deadline.
        let closed = tokio::time::timeout(DEADLINE * 4, session.wait_closed()).await;
        let held_for = opened.elapsed();
        stop.send(true).unwrap();
        (answer, waited, closed, held_for)
    };
    let (report, (answer, waited, closed, held_for)) =
        tokio::join!(serve(&server, 1024, policy, echo), driving);

    assert_eq!(
        answer.unwrap(),
        b"honest",
        "the control: another peer is served"
    );
    assert!(
        waited < DEADLINE,
        "the honest peer waited {waited:?}, which is not behind a {DEADLINE:?} deadline"
    );
    assert!(
        closed.is_ok(),
        "the hanging connection was closed by the host"
    );
    assert!(
        held_for >= DEADLINE,
        "it was closed at the deadline, not before: {held_for:?}"
    );
    assert_eq!(
        report.unwrap(),
        ServeReport {
            handled: 1,
            refused_over_limit: 0,
            cut_at_deadline: 1,
            failed: 0,
        }
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn after_shutdown_nothing_is_accepted() {
    let server = IrohTransport::bind(&[ALPN_TEST]).await.unwrap();
    let address = server.direct_addr();
    let (policy, stop) = policy_with(4, Duration::from_secs(5));
    let calls = Arc::new(AtomicUsize::new(0));
    let handler = {
        let calls = Arc::clone(&calls);
        move |incoming: IrohIncoming| {
            let calls = Arc::clone(&calls);
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                echo(incoming).await
            }
        }
    };
    let client = IrohTransport::bind(&[]).await.unwrap();

    // The control: before shutdown, a request is served.
    let driving = async {
        let before = client
            .request(address.clone(), ALPN_TEST, b"before", 64)
            .await;
        stop.send(true).unwrap();
        before
    };
    let (report, before) = tokio::join!(serve(&server, 1024, policy, handler), driving);
    assert_eq!(before.unwrap(), b"before");
    assert_eq!(report.unwrap().handled, 1);

    // After it, a request finds nobody accepting: it is not answered, and no handler
    // runs.
    let after = tokio::time::timeout(
        Duration::from_millis(1500),
        client.request(address, ALPN_TEST, b"after", 64),
    )
    .await;
    assert!(
        !matches!(after, Ok(Ok(_))),
        "a request after shutdown is not answered"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "no handler ran after shutdown"
    );

    // A loop started after the instruction was given accepts nothing at all.
    let (mut stopped, _unused) = policy_with(4, Duration::from_secs(5));
    let (_keep, already) = watch::channel(true);
    stopped.shutdown = already;
    let report = serve(&server, 1024, stopped, |_incoming: IrohIncoming| async {
        Err::<(), String>("must not run".to_owned())
    })
    .await
    .unwrap();
    assert_eq!(report, ServeReport::default());

    // And a dropped sender is read as shutdown: a loop nobody can stop any more
    // does not keep accepting.
    let (orphaned, sender) = policy_with(4, Duration::from_secs(5));
    drop(sender);
    let report = serve(&server, 1024, orphaned, |_incoming: IrohIncoming| async {
        Err::<(), String>("must not run".to_owned())
    })
    .await
    .unwrap();
    assert_eq!(report, ServeReport::default());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_zero_deadline_is_refused_before_anything_is_accepted() {
    let server = IrohTransport::bind(&[ALPN_TEST]).await.unwrap();
    let (zero, _stop) = policy_with(1, Duration::ZERO);
    assert_eq!(
        serve(&server, 1024, zero, echo).await,
        Err(ServeError::ZeroDeadline)
    );
    // The control: the same policy with a deadline runs, and stops when told.
    let (some, stop) = policy_with(1, Duration::from_millis(50));
    stop.send(true).unwrap();
    assert_eq!(
        serve(&server, 1024, some, echo).await,
        Ok(ServeReport::default())
    );
}
