//! The network wires `ptrd` serves next to its HTTP API (D4 in issue #23).
//!
//! The accept loop is `ptr-net`'s mechanism; everything here is this host's policy,
//! read from `[wire]`: how many connections each service holds, how long one may
//! take, which services run, and where the peers it knows can be reached. Nothing
//! below has a default — a deployment without `[wire]` serves no wire.
//!
//! What runs, and with what authority:
//!
//! - **`exec`** is an `ExecutionHost` over the *same* runtime the HTTP API serves,
//!   shared rather than reopened: two runtimes over one ledger would be two writers of
//!   one history. It admits exactly the peers that runtime's admission policy admits.
//!   `ptrd` installs none, so in this milestone every request is refused with the one
//!   authority code — the wire is served, and nobody has been granted anything on it.
//! - **`pod`** is a `PodHost` over an empty registry — the HTTP API's is empty too —
//!   whose access policy is read from the *same* runtime's committed Pod scopes, so a
//!   scope withdrawn before a restart stays withdrawn after it. `ptrd` admits none,
//!   so every request is answered `NotAdmitted`. Admitting a peer to a project is a
//!   decision `[wire]` does not make on anybody's behalf.
//!
//! Every answer either wire sends is signed with its endpoint key (D2), and the
//! endpoints bind loopback only, as `ptrd`'s HTTP bind does.
use ptr_config::WireConfig;
use ptr_execwire::ExecutionHost;
use ptr_net::{EndpointAddr, EndpointId, PeerBook, ServePolicy, ServeReport};
use ptr_pods::PodRegistry;
use ptr_podwire::{PodAccessPolicy, PodHost};
use ptr_protocol::TypedPayload;
use ptr_runtime::PtrRuntime;
use ptr_verifier::{VerificationReport, Verifier};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::watch;

/// The loop's policy, from a validated `[wire]` section.
///
/// Validation already refused a zero limit; this still refuses rather than assume,
/// because the type it builds has no zero to fall back on.
pub fn serve_policy(
    wire: &WireConfig,
    shutdown: watch::Receiver<bool>,
) -> Result<ServePolicy, String> {
    let max_concurrent = NonZeroUsize::new(wire.max_concurrent)
        .ok_or_else(|| "wire.max_concurrent must be greater than zero".to_owned())?;
    Ok(ServePolicy {
        max_concurrent,
        per_connection_deadline: Duration::from_millis(wire.per_connection_deadline_ms),
        shutdown,
    })
}

/// The address book, rebuilt from the operator's `[[wire.peers]]` at every start.
///
/// Addresses are deployment facts, not history, so they come from configuration and
/// are never journaled (C8, `34-address-authority.md`). Validation has already
/// refused a malformed id, a malformed address and a duplicate; this refuses what
/// only the transport can tell — 64 hex digits that are not a key.
pub fn peer_book(wire: &WireConfig) -> Result<PeerBook, String> {
    let mut book = PeerBook::new();
    for peer in &wire.peers {
        let id: EndpointId = peer
            .id
            .parse()
            .map_err(|error| format!("wire peer {} is not an endpoint key: {error}", peer.id))?;
        let direct: std::net::SocketAddr = peer
            .direct
            .parse()
            .map_err(|error| format!("wire peer {} address {}: {error}", peer.id, peer.direct))?;
        book.record(EndpointAddr::new(id).with_ip_addr(direct));
    }
    Ok(book)
}

/// The wires running.
pub struct Wires {
    tasks: tokio::task::JoinSet<(&'static str, ServeReport)>,
}

impl Wires {
    /// Wait for every wire to stop. Call after the shutdown has been sent.
    pub async fn stopped(mut self) -> Vec<(&'static str, ServeReport)> {
        let mut reports = Vec::new();
        while let Some(done) = self.tasks.join_next().await {
            if let Ok(report) = done {
                reports.push(report);
            }
        }
        reports
    }
}

/// Bind and start every service `[wire]` names, the execution wire sharing `runtime`
/// with the HTTP API and the pod wire reading its Pod scopes from it. Each runs until `shutdown` reads `true`.
pub async fn start(
    wire: &WireConfig,
    runtime: Arc<Mutex<PtrRuntime>>,
    verifier: Arc<dyn Verifier<TypedPayload> + Send + Sync>,
    shutdown: watch::Receiver<bool>,
) -> Result<Wires, String> {
    let mut tasks = tokio::task::JoinSet::new();
    if wire.serves("exec") {
        let host = Arc::new(
            ExecutionHost::bind_shared(Arc::clone(&runtime))
                .await
                .map_err(|error| format!("execution wire: {error}"))?,
        );
        println!("ptrd execution wire {}", host.identity().public_key);
        let policy = serve_policy(wire, shutdown.clone())?;
        tasks.spawn(async move {
            let report = host.serve(policy).await.unwrap_or_default();
            host.close().await;
            ("exec", report)
        });
    }
    if wire.serves("pod") {
        let host = Arc::new(
            PodHost::bind(
                PodRegistry::default(),
                PodAccessPolicy::from_runtime(Arc::clone(&runtime)),
                SharedVerifier(verifier),
            )
            .await
            .map_err(|error| format!("pod wire: {error}"))?,
        );
        println!("ptrd pod wire {}", host.identity().public_key);
        let policy = serve_policy(wire, shutdown.clone())?;
        tasks.spawn(async move {
            let report = host.serve(policy).await.unwrap_or_default();
            host.close().await;
            ("pod", report)
        });
    }
    Ok(Wires { tasks })
}

/// The HTTP API's verifier, shared with the pod wire so both vouch for a Pod's
/// output by one rule.
struct SharedVerifier(Arc<dyn Verifier<TypedPayload> + Send + Sync>);

impl Verifier<TypedPayload> for SharedVerifier {
    fn verify(&self, output: &TypedPayload) -> VerificationReport {
        self.0.verify(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_config::{PtrConfig, WirePeerConfig};
    use ptr_types::NodeId;

    /// A key the transport would accept, taken from a real endpoint.
    async fn real_key() -> String {
        let transport = ptr_net::IrohTransport::bind(&[]).await.unwrap();
        let key = transport.identity().public_key;
        transport.close().await;
        key
    }

    fn wire(peers: Vec<WirePeerConfig>) -> WireConfig {
        WireConfig {
            max_concurrent: 2,
            per_connection_deadline_ms: 2_000,
            services: vec!["exec".into(), "pod".into()],
            peers,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_peer_book_is_rebuilt_from_the_configured_list() {
        let (one, two) = (real_key().await, real_key().await);
        let configured = wire(vec![
            WirePeerConfig {
                id: one.clone(),
                direct: "127.0.0.1:4433".into(),
            },
            WirePeerConfig {
                id: two.clone(),
                direct: "[::1]:5544".into(),
            },
        ]);
        configured.validate().unwrap();
        let book = peer_book(&configured).unwrap();
        assert_eq!(book.len(), 2);
        let located = book.locate(&NodeId(one.clone())).unwrap().into_address();
        assert_eq!(located.id.to_string(), one);
        assert_eq!(
            located.ip_addrs().copied().collect::<Vec<_>>(),
            vec!["127.0.0.1:4433".parse().unwrap()]
        );
        assert!(book.locate(&NodeId(two)).is_ok());
        // The control: a peer nobody configured is not in the book.
        assert!(book.locate(&NodeId(real_key().await)).is_err());
        // And an empty list is an empty book, not a default one.
        assert!(peer_book(&wire(Vec::new())).unwrap().is_empty());
    }

    #[test]
    fn a_configured_id_that_is_not_a_key_is_refused_at_start() {
        // 64 lowercase hex digits pass validation — the spelling is right — and
        // only the transport can tell that these bytes are not a point on the curve.
        let not_a_key = format!("02{}", "0".repeat(62));
        let configured = wire(vec![WirePeerConfig {
            id: not_a_key.clone(),
            direct: "127.0.0.1:4433".into(),
        }]);
        configured.validate().unwrap();
        let refused = peer_book(&configured).expect_err("not a key");
        assert!(refused.contains(&not_a_key), "{refused}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_configured_wires_run_on_the_shared_runtime_and_stop_together() {
        let runtime = Arc::new(Mutex::new(PtrRuntime::new(PtrConfig::default()).unwrap()));
        let before = runtime.lock().unwrap().committed_events().len();
        let (stop, shutdown) = watch::channel(false);
        let configured = wire(Vec::new());
        let wires = start(
            &configured,
            Arc::clone(&runtime),
            ptr_server::default_verifier(),
            shutdown,
        )
        .await
        .unwrap();

        stop.send(true).unwrap();
        let mut stopped = wires.stopped().await;
        stopped.sort_by_key(|(name, _)| *name);
        assert_eq!(
            stopped,
            vec![
                ("exec", ServeReport::default()),
                ("pod", ServeReport::default())
            ],
            "both wires ran and stopped on the one instruction"
        );
        assert_eq!(
            runtime.lock().unwrap().committed_events().len(),
            before,
            "serving wrote nothing to the shared runtime"
        );

        // The control: a section naming one service starts one.
        let (stop, shutdown) = watch::channel(false);
        let only_pod = WireConfig {
            services: vec!["pod".into()],
            ..wire(Vec::new())
        };
        let wires = start(&only_pod, runtime, ptr_server::default_verifier(), shutdown)
            .await
            .unwrap();
        stop.send(true).unwrap();
        assert_eq!(wires.stopped().await, vec![("pod", ServeReport::default())]);
    }

    #[test]
    fn the_serve_policy_is_the_configured_one() {
        let (_stop, shutdown) = watch::channel(false);
        let policy = serve_policy(&wire(Vec::new()), shutdown).unwrap();
        assert_eq!(policy.max_concurrent.get(), 2);
        assert_eq!(policy.per_connection_deadline, Duration::from_millis(2_000));
        let (_stop, shutdown) = watch::channel(false);
        assert!(serve_policy(
            &WireConfig {
                max_concurrent: 0,
                ..wire(Vec::new())
            },
            shutdown
        )
        .is_err());
    }
}
