//! Execution over a real authenticated connection.
//!
//! What is worth a network test here, rather than a queue: that an action actually
//! traverses `ALPN_EXEC` and is executed under the principal *policy* names, and that
//! neither side believes a claim the connection did not make. Everything about what
//! the runtime permits is tested in `ptr-runtime`, where no socket is involved; a
//! network test of that would be a test of this test's own plumbing.
//!
//! Every test drives one exchange with `tokio::join!` rather than a background loop,
//! so a failure is a failure of the protocol and never of a race with a server task.

use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_execwire::{
    decode_request, encode_receipt, encode_request, request_digest, ExecutionClient, ExecutionHost,
    RefusalCode, WireError, WireOutcome, WireReceipt, WireRequest,
};
use ptr_ledger::LedgerEvent;
use ptr_net::{EndpointAddr, IrohTransport, NodeIdentity, PeerAddress, PeerBook, ALPN_EXEC};
use ptr_runtime::execution::{
    ActionExecutor, ActionScope, AdmissionPolicy, DetachedExecutor, ExecutionGrant,
    RequiredVerification, VerifiedDispatch,
};
use ptr_runtime::PtrRuntime;
use ptr_types::{
    CapabilityId, CapsuleId, Effect, Generation, NodeId, Probability, ProjectId, RequestId, TypeId,
    VerificationLevel,
};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const TTL: Duration = Duration::from_secs(60);

/// What the executor saw, so a test can check the principal an action was audited
/// under rather than the one a request asked for.
#[derive(Clone, Default)]
struct Probe {
    observed: Arc<Mutex<Vec<(String, ProjectId, ActionIr)>>>,
}

impl Probe {
    fn executions(&self) -> usize {
        self.observed.lock().unwrap().len()
    }

    fn principals(&self) -> Vec<String> {
        self.observed
            .lock()
            .unwrap()
            .iter()
            .map(|(principal, _, _)| principal.clone())
            .collect()
    }
}

/// Passes anything whose payload is the agreed one, so a test can fail verification
/// on purpose by changing the payload.
struct PassingVerifier;

impl Verifier<ActionIr> for PassingVerifier {
    fn verify(&self, action: &ActionIr) -> VerificationReport {
        VerificationReport {
            status: if action.payload == b"verified payload" {
                VerificationStatus::Pass
            } else {
                VerificationStatus::Fail
            },
            level: VerificationLevel::Deterministic,
            score: Probability::new(1.0).unwrap(),
            findings: vec![],
        }
    }
}

/// How the adapter answers.
#[derive(Clone, Copy)]
enum Mode {
    /// Finishes and returns bytes.
    Success,
    /// Fails without saying whether the effect applied.
    Uncertain,
}

struct Adapter {
    probe: Probe,
    mode: Mode,
}

impl ActionExecutor for Adapter {
    fn execute(&self, dispatch: VerifiedDispatch<'_>) -> Result<Vec<u8>, String> {
        self.probe.observed.lock().unwrap().push((
            dispatch.principal().to_owned(),
            dispatch.project().clone(),
            dispatch.action().clone(),
        ));
        match self.mode {
            Mode::Success => Ok(b"executed".to_vec()),
            Mode::Uncertain => Err("the adapter cannot say whether it applied".to_owned()),
        }
    }
}

/// An adapter that takes work and answers later, which this wire has no channel for.
struct Detached;

impl DetachedExecutor for Detached {
    fn start(&self, _dispatch: VerifiedDispatch<'_>) -> Result<(), String> {
        panic!("a detached grant must be refused before the adapter is reached");
    }
}

/// The action every test asks for.
fn action(runtime: &mut PtrRuntime) -> ActionIr {
    let revision = runtime
        .ingest_text(RequestId::from("request"), "ground state")
        .unwrap();
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from("capsule:a"),
            generation: Generation(1),
        })
        .unwrap();
    let action = ActionIr {
        operation: "write".into(),
        target: "capsule:a".into(),
        capability: CapabilityId::from("file.write"),
        effect: Effect::Mutation,
        input_type: TypeId::from("Bytes"),
        generation: Generation(1),
        revision,
        payload: b"verified payload".to_vec(),
    };
    runtime
        .permissions_mut()
        .capabilities
        .insert(action.capability.clone());
    runtime.permissions_mut().allow_mutation = true;
    action
}

fn scope(action: &ActionIr) -> ActionScope {
    ActionScope {
        project: ProjectId::from("p"),
        target: action.target.clone(),
        operation: action.operation.clone(),
        capability: action.capability.clone(),
        input_type: action.input_type.clone(),
        effect: action.effect,
    }
}

/// The principal the policy names. No request can ask for it, and nothing else may
/// appear in an audit record.
const POLICY_PRINCIPAL: &str = "principal-from-host-policy";

/// A runtime that admits `peer` with one grant for `action`.
fn runtime_admitting(peer: &NodeIdentity, mode: Mode) -> (PtrRuntime, ActionIr, Probe) {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let action = action(&mut runtime);
    let probe = Probe::default();
    let mut policy = AdmissionPolicy::new();
    let scope = scope(&action);
    let grant_probe = probe.clone();
    policy
        .admit(
            NodeId(peer.public_key.clone()),
            POLICY_PRINCIPAL,
            TTL,
            move || {
                vec![ExecutionGrant::new(
                    scope.clone(),
                    RequiredVerification::Deterministic,
                    PassingVerifier,
                    Adapter {
                        probe: grant_probe.clone(),
                        mode,
                    },
                )]
            },
        )
        .unwrap();
    runtime.install_admission_policy(policy);
    (runtime, action, probe)
}

/// A host serving that runtime, and the request template for it.
async fn host_admitting(peer: &NodeIdentity, mode: Mode) -> (Arc<ExecutionHost>, ActionIr, Probe) {
    let (runtime, action, probe) = runtime_admitting(peer, mode);
    let host = ExecutionHost::bind(runtime).await.unwrap();
    (Arc::new(host), action, probe)
}

/// Where to dial, on the deployment's authority.
///
/// An `ExecutionClient` accepts nothing else, so every test here goes through a book
/// — which is the point: a bare address does not satisfy the signature.
fn located(address: EndpointAddr) -> PeerAddress {
    let mut book = PeerBook::new();
    book.record(address.clone());
    book.locate(&NodeId(address.id.to_string()))
        .expect("an address just recorded is locatable")
}

fn request(host: &ExecutionHost, action: &ActionIr, request_id: u64) -> WireRequest {
    WireRequest {
        addressed_to: host.identity().public_key,
        request_id,
        project: ProjectId::from("p"),
        action: action.clone(),
        once_key: None,
    }
}

/// How many events a runtime has committed, so a test can assert that a refusal
/// wrote nothing.
fn committed(host: &ExecutionHost) -> usize {
    host.runtime().committed_events().len()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_admitted_peer_executes_over_the_wire_under_the_principal_policy_names() {
    let client = ExecutionClient::bind().await.unwrap();
    let (host, action, probe) = host_admitting(&client.identity(), Mode::Success).await;
    let before = committed(&host);

    // A request cannot name a principal, so the closest a peer can come is writing
    // one into the payload it controls. The audit must not read it.
    let mut asked = request(&host, &action, 1);
    asked.action.payload = b"verified payload".to_vec();

    let serving = Arc::clone(&host);
    let (answer, served) = tokio::join!(
        client.request(located(host.address()), &asked),
        serving.serve_once()
    );
    let receipt = answer.unwrap();
    let served = served.unwrap();

    assert_eq!(
        receipt.outcome,
        WireOutcome::Applied {
            response: b"executed".to_vec()
        },
        "the effect applied over a real connection"
    );
    assert_eq!(
        receipt.responder,
        host.identity().public_key,
        "the receipt names its author"
    );
    assert_eq!(served.request_id, Some(1));
    assert_eq!(served.refused, None);
    assert_eq!(
        served.peer.public_key,
        client.identity().public_key,
        "the peer is the connection's, not the payload's"
    );

    assert_eq!(probe.executions(), 1);
    assert_eq!(
        probe.principals(),
        vec![POLICY_PRINCIPAL.to_owned()],
        "the audited principal is the policy's"
    );
    assert!(
        committed(&host) > before,
        "an attempt and its settlement are in the ledger"
    );
    assert!(
        host.runtime().unsettled_effects().is_empty(),
        "a settled effect leaves no fence"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_receipt_naming_another_runtime_is_refused_and_an_honest_one_is_not() {
    // The forged receipt. A receipt is a claim that can outlive the connection that
    // carried it, so the only thing that makes its author mean anything is checking
    // it against the peer the connection authenticated — at the moment it arrives.
    let client = ExecutionClient::bind().await.unwrap();
    let elsewhere = IrohTransport::bind(&[ALPN_EXEC]).await.unwrap();
    let impersonated = elsewhere.identity().public_key;

    // A rogue endpoint that answers every request with a well-formed receipt bound to
    // the right request — only its author is wrong.
    let rogue = IrohTransport::bind(&[ALPN_EXEC]).await.unwrap();
    let rogue_key = rogue.identity().public_key;

    let asked = WireRequest {
        addressed_to: rogue_key.clone(),
        request_id: 11,
        project: ProjectId::from("p"),
        action: ActionIr {
            operation: "write".into(),
            target: "capsule:a".into(),
            capability: CapabilityId::from("file.write"),
            effect: Effect::Mutation,
            input_type: TypeId::from("Bytes"),
            generation: Generation(1),
            revision: ptr_types::Revision(1),
            payload: b"verified payload".to_vec(),
        },
        once_key: None,
    };

    let claimed = impersonated.clone();
    let address = rogue.direct_addr();
    let serving = async move {
        let incoming = rogue.accept_once(1 << 20).await.unwrap();
        let decoded = decode_request(&incoming.payload).unwrap();
        let receipt = WireReceipt {
            responder: claimed,
            request_id: decoded.request_id,
            request_digest: request_digest(&incoming.payload),
            outcome: WireOutcome::Applied {
                response: b"executed".to_vec(),
            },
        };
        incoming
            .respond(&encode_receipt(&receipt).unwrap())
            .await
            .unwrap();
    };

    let (answer, ()) = tokio::join!(client.request(located(address), &asked), serving);
    assert_eq!(
        answer.expect_err("a receipt from somebody else"),
        WireError::ForgedReceipt {
            authenticated: rogue_key.clone(),
            claimed: impersonated,
        },
        "the author is checked against the connection, not against the claim"
    );

    // The control: the same rogue, answering honestly, is accepted. So the refusal
    // above is about the name on the receipt and not about that endpoint.
    let rogue = IrohTransport::bind(&[ALPN_EXEC]).await.unwrap();
    let honest_key = rogue.identity().public_key;
    let address = rogue.direct_addr();
    let asked = WireRequest {
        addressed_to: honest_key.clone(),
        ..asked
    };
    let serving = async move {
        let incoming = rogue.accept_once(1 << 20).await.unwrap();
        let decoded = decode_request(&incoming.payload).unwrap();
        let receipt = WireReceipt {
            responder: honest_key,
            request_id: decoded.request_id,
            request_digest: request_digest(&incoming.payload),
            outcome: WireOutcome::Applied {
                response: b"executed".to_vec(),
            },
        };
        incoming
            .respond(&encode_receipt(&receipt).unwrap())
            .await
            .unwrap();
    };
    let (answer, ()) = tokio::join!(client.request(located(address), &asked), serving);
    assert_eq!(
        answer.unwrap().outcome,
        WireOutcome::Applied {
            response: b"executed".to_vec()
        },
        "an honest receipt from the same endpoint is accepted"
    );
    let _ = elsewhere;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_receipt_that_answers_a_different_request_is_refused() {
    // A receipt is bound to the bytes that arrived. Without that, a host — or anything
    // between — could answer an expensive request with a cheap request's receipt, and
    // both would be well formed.
    let client = ExecutionClient::bind().await.unwrap();

    for (label, wrong_id, wrong_digest, expected) in [
        (
            "a receipt for another request id",
            Some(999),
            false,
            WireError::WrongRequest {
                expected: 21,
                answered: 999,
            },
        ),
        (
            "a receipt bound to other bytes",
            None,
            true,
            WireError::UnboundReceipt,
        ),
    ] {
        let rogue = IrohTransport::bind(&[ALPN_EXEC]).await.unwrap();
        let key = rogue.identity().public_key;
        let address = rogue.direct_addr();
        let asked = WireRequest {
            addressed_to: key.clone(),
            request_id: 21,
            project: ProjectId::from("p"),
            action: ActionIr {
                operation: "write".into(),
                target: "capsule:a".into(),
                capability: CapabilityId::from("file.write"),
                effect: Effect::Mutation,
                input_type: TypeId::from("Bytes"),
                generation: Generation(1),
                revision: ptr_types::Revision(1),
                payload: b"verified payload".to_vec(),
            },
            once_key: None,
        };
        let serving = async move {
            let incoming = rogue.accept_once(1 << 20).await.unwrap();
            let decoded = decode_request(&incoming.payload).unwrap();
            let receipt = WireReceipt {
                responder: key,
                request_id: wrong_id.unwrap_or(decoded.request_id),
                request_digest: if wrong_digest {
                    request_digest(b"some other request")
                } else {
                    request_digest(&incoming.payload)
                },
                outcome: WireOutcome::Applied {
                    response: b"executed".to_vec(),
                },
            };
            incoming
                .respond(&encode_receipt(&receipt).unwrap())
                .await
                .unwrap();
        };
        let (answer, ()) = tokio::join!(client.request(located(address), &asked), serving);
        assert_eq!(answer.expect_err(label), expected, "{label}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_request_addressed_to_another_runtime_is_refused_and_writes_no_record() {
    // The relay case. Both runtimes admit this peer and hold the same grant, so the
    // only thing that refuses the request is the key it names.
    let client = ExecutionClient::bind().await.unwrap();
    let (one, action, probe) = host_admitting(&client.identity(), Mode::Success).await;
    let (two, _, other_probe) = host_admitting(&client.identity(), Mode::Success).await;
    let before = committed(&two);

    let asked = WireRequest {
        addressed_to: one.identity().public_key,
        ..request(&two, &action, 1)
    };
    let serving = Arc::clone(&two);
    let (answer, served) = tokio::join!(
        client.request(located(two.address()), &asked),
        serving.serve_once()
    );
    let receipt = answer.unwrap();
    let served = served.unwrap();

    assert_eq!(
        receipt.outcome,
        WireOutcome::Refused {
            code: RefusalCode::Misaddressed
        }
    );
    assert!(matches!(
        served.refused,
        Some(WireError::Misaddressed { .. })
    ));
    assert_eq!(committed(&two), before, "a refusal writes no record");
    assert_eq!(other_probe.executions(), 0);
    assert_eq!(
        probe.executions(),
        0,
        "and the addressed runtime saw nothing"
    );

    // The control: the same request, addressed to the runtime it is sent to, applies.
    let asked = request(&two, &action, 2);
    let serving = Arc::clone(&two);
    let (answer, _) = tokio::join!(
        client.request(located(two.address()), &asked),
        serving.serve_once()
    );
    assert_eq!(
        answer.unwrap().outcome,
        WireOutcome::Applied {
            response: b"executed".to_vec()
        },
        "so the refusal was about the key it named"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unadmitted_peer_is_refused_exactly_as_an_admitted_one_outside_its_grant() {
    // The wire must not be an oracle for host policy. "Your key is not admitted" and
    // "that action is not in your grant" are the same answer, for the same reason a
    // cross-project Pod request is refused identically to a Pod that does not exist.
    let admitted = ExecutionClient::bind().await.unwrap();
    let stranger = ExecutionClient::bind().await.unwrap();
    let (host, action, probe) = host_admitting(&admitted.identity(), Mode::Success).await;
    let before = committed(&host);

    let serving = Arc::clone(&host);
    let asked = request(&host, &action, 1);
    let (answer, _) = tokio::join!(
        stranger.request(located(host.address()), &asked),
        serving.serve_once()
    );
    let unadmitted = answer.unwrap().outcome;

    let mut outside = request(&host, &action, 2);
    outside.action.operation = "delete".into();
    let serving = Arc::clone(&host);
    let (answer, _) = tokio::join!(
        admitted.request(located(host.address()), &outside),
        serving.serve_once()
    );
    let out_of_scope = answer.unwrap().outcome;

    assert_eq!(
        unadmitted,
        WireOutcome::Refused {
            code: RefusalCode::Runtime
        }
    );
    assert_eq!(
        unadmitted, out_of_scope,
        "a peer cannot tell being unknown from being out of scope"
    );

    // And the same holds on a replay, which is why the window is spent only after the
    // peer is admitted. If an unadmitted peer's request id were recorded, its second
    // identical frame would come back "replayed" where an admitted peer's comes back
    // refused, and the difference between those two answers is an oracle for the
    // policy.
    let serving = Arc::clone(&host);
    let (answer, _) = tokio::join!(
        stranger.request(located(host.address()), &asked),
        serving.serve_once()
    );
    assert_eq!(
        answer.unwrap().outcome,
        unadmitted,
        "a stranger's replay is refused the same way, not as a replay"
    );
    assert_eq!(committed(&host), before, "neither refusal wrote a record");
    assert_eq!(probe.executions(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_same_frame_sent_twice_is_refused_by_the_window_and_applies_once() {
    let client = ExecutionClient::bind().await.unwrap();
    let (host, action, probe) = host_admitting(&client.identity(), Mode::Success).await;
    let asked = request(&host, &action, 5);

    let serving = Arc::clone(&host);
    let (first, _) = tokio::join!(
        client.request(located(host.address()), &asked),
        serving.serve_once()
    );
    let serving = Arc::clone(&host);
    let (second, served) = tokio::join!(
        client.request(located(host.address()), &asked),
        serving.serve_once()
    );

    assert_eq!(
        first.unwrap().outcome,
        WireOutcome::Applied {
            response: b"executed".to_vec()
        }
    );
    assert_eq!(
        second.unwrap().outcome,
        WireOutcome::Refused {
            code: RefusalCode::Replayed
        },
        "the identical frame is refused by the window"
    );
    assert_eq!(
        served.unwrap().refused,
        Some(WireError::Replayed { request_id: 5 })
    );
    assert_eq!(probe.executions(), 1, "the effect applied once");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_retry_is_a_new_request_id_with_the_same_key_and_still_applies_once() {
    // The two mechanisms are not interchangeable. The window refuses a frame sent
    // twice; the at-most-once key is what makes a *retry* — a new frame for the same
    // intent — apply once and return the original answer.
    let client = ExecutionClient::bind().await.unwrap();
    let (host, action, probe) = host_admitting(&client.identity(), Mode::Success).await;

    let first = WireRequest {
        once_key: Some("k".to_owned()),
        ..request(&host, &action, 1)
    };
    let retry = WireRequest {
        request_id: 2,
        ..first.clone()
    };

    let serving = Arc::clone(&host);
    let (answer, _) = tokio::join!(
        client.request(located(host.address()), &first),
        serving.serve_once()
    );
    let applied = answer.unwrap().outcome;
    let serving = Arc::clone(&host);
    let (answer, _) = tokio::join!(
        client.request(located(host.address()), &retry),
        serving.serve_once()
    );
    let retried = answer.unwrap().outcome;

    assert_eq!(
        applied,
        WireOutcome::Applied {
            response: b"executed".to_vec()
        }
    );
    assert_eq!(
        retried, applied,
        "the retry is answered from the record, not executed again"
    );
    assert_eq!(probe.executions(), 1, "the effect applied once");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_damaged_frame_is_refused_before_it_reaches_the_runtime() {
    let client = ExecutionClient::bind().await.unwrap();
    let (host, action, probe) = host_admitting(&client.identity(), Mode::Success).await;
    let before = committed(&host);

    // A real frame with one bit changed, sent on a real connection. A requester whose
    // own client refuses to build this has to go around it, which is exactly what an
    // attacker does.
    let mut frame = encode_request(&request(&host, &action, 3)).unwrap();
    let last = frame.len() - 1;
    frame[last] ^= 1;

    let raw = IrohTransport::bind(&[ALPN_EXEC]).await.unwrap();
    let serving = Arc::clone(&host);
    let address = host.address();
    let (answer, served) = tokio::join!(
        raw.request(address, ALPN_EXEC, &frame, 1 << 20),
        serving.serve_once()
    );
    let response = answer.unwrap();
    let served = served.unwrap();

    let receipt = ptr_execwire::decode_receipt(&response).unwrap();
    assert_eq!(
        receipt.outcome,
        WireOutcome::Refused {
            code: RefusalCode::Malformed
        }
    );
    assert_eq!(
        receipt.request_digest,
        request_digest(&frame),
        "even an unparseable request gets a receipt bound to what arrived"
    );
    assert!(matches!(served.refused, Some(WireError::Frame(_))));
    assert_eq!(served.request_id, None, "there was no request id to echo");
    assert_eq!(committed(&host), before, "a refusal writes no record");
    assert_eq!(probe.executions(), 0);
    let _ = client;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_detached_grant_is_refused_over_the_wire_and_leaves_no_fence() {
    // Detached work is audited by an attempt that stays unsettled until the adapter
    // reports back, and this wire has no channel for that report. So a detached grant
    // is refused — by the runtime, before the attempt is committed, which is why the
    // runtime is left unfenced. An outcome the requester can never be told is not an
    // outcome.
    let client = ExecutionClient::bind().await.unwrap();
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let action = action(&mut runtime);
    let mut policy = AdmissionPolicy::new();
    let scope = scope(&action);
    policy
        .admit(
            NodeId(client.identity().public_key),
            POLICY_PRINCIPAL,
            TTL,
            move || {
                vec![ExecutionGrant::detached(
                    scope.clone(),
                    RequiredVerification::Deterministic,
                    PassingVerifier,
                    Detached,
                )]
            },
        )
        .unwrap();
    runtime.install_admission_policy(policy);
    let host = Arc::new(ExecutionHost::bind(runtime).await.unwrap());
    let before = committed(&host);

    let serving = Arc::clone(&host);
    let asked = request(&host, &action, 1);
    let (answer, served) = tokio::join!(
        client.request(located(host.address()), &asked),
        serving.serve_once()
    );

    assert_eq!(
        answer.unwrap().outcome,
        WireOutcome::Refused {
            code: RefusalCode::Runtime
        }
    );
    assert_eq!(
        served.unwrap().refused,
        Some(WireError::Runtime(
            ptr_runtime::execution::ExecutionError::DispatchMismatch { detached: true }
        )),
        "the local side knows which refusal it was"
    );
    assert_eq!(committed(&host), before, "a refusal writes no record");
    assert!(host.runtime().unsettled_effects().is_empty());
    assert!(host.runtime().outstanding_detached().is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_uncertain_outcome_is_not_a_refusal_and_the_fence_reaches_the_next_request() {
    // "The adapter did not answer" is not "nothing happened". Reporting it as a
    // refusal would invite a retry of an effect that may have applied — which is the
    // one thing the runtime's fence exists to prevent, thrown away at the last step.
    let client = ExecutionClient::bind().await.unwrap();
    let (host, action, probe) = host_admitting(&client.identity(), Mode::Uncertain).await;

    let serving = Arc::clone(&host);
    let asked = request(&host, &action, 1);
    let (answer, _) = tokio::join!(
        client.request(located(host.address()), &asked),
        serving.serve_once()
    );
    assert_eq!(
        answer.unwrap().outcome,
        WireOutcome::Uncertain,
        "the requester is told the outcome is unknown"
    );
    assert_eq!(probe.executions(), 1, "the adapter was reached");
    assert_eq!(
        host.runtime().unsettled_effects().len(),
        1,
        "the attempt is the fence, and it is standing"
    );

    // And the fence is not a per-request opinion: the next request is refused because
    // this runtime cannot take on more while it does not know what happened.
    let serving = Arc::clone(&host);
    let asked = request(&host, &action, 2);
    let (answer, served) = tokio::join!(
        client.request(located(host.address()), &asked),
        serving.serve_once()
    );
    assert_eq!(
        answer.unwrap().outcome,
        WireOutcome::Refused {
            code: RefusalCode::Runtime
        },
        "a fenced runtime refuses rather than reporting another uncertainty"
    );
    assert!(matches!(
        served.unwrap().refused,
        Some(WireError::Runtime(
            ptr_runtime::execution::ExecutionError::AmbiguousOutcome { .. }
        ))
    ));
    assert_eq!(probe.executions(), 1, "and nothing else was dispatched");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn withdrawing_a_peer_takes_effect_between_two_requests_over_the_wire() {
    // Authority is re-derived from the policy on every request, so a withdrawal lands
    // at once rather than when a session's TTL happens to run out.
    let client = ExecutionClient::bind().await.unwrap();
    let (host, action, probe) = host_admitting(&client.identity(), Mode::Success).await;

    let serving = Arc::clone(&host);
    let asked = request(&host, &action, 1);
    let (answer, _) = tokio::join!(
        client.request(located(host.address()), &asked),
        serving.serve_once()
    );
    assert_eq!(
        answer.unwrap().outcome,
        WireOutcome::Applied {
            response: b"executed".to_vec()
        }
    );

    host.runtime()
        .withdraw_peer(&NodeId(client.identity().public_key))
        .unwrap();

    let serving = Arc::clone(&host);
    let asked = request(&host, &action, 2);
    let (answer, _) = tokio::join!(
        client.request(located(host.address()), &asked),
        serving.serve_once()
    );
    assert_eq!(
        answer.unwrap().outcome,
        WireOutcome::Refused {
            code: RefusalCode::Runtime
        },
        "the withdrawn peer is refused on its very next request"
    );
    assert_eq!(probe.executions(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_runtimes_audit_the_same_peer_independently() {
    // The cross-runtime case over a network boundary. An at-most-once key is a fact in
    // one runtime's ledger; it says nothing in another's, and a wire that answered the
    // second request from the first runtime's record would be inventing an agreement
    // between two ledgers that never spoke.
    let client = ExecutionClient::bind().await.unwrap();
    let (one, action, first_probe) = host_admitting(&client.identity(), Mode::Success).await;
    let (two, _, second_probe) = host_admitting(&client.identity(), Mode::Success).await;

    let asked = WireRequest {
        once_key: Some("shared-key".to_owned()),
        ..request(&one, &action, 1)
    };
    let serving = Arc::clone(&one);
    let (answer, _) = tokio::join!(
        client.request(located(one.address()), &asked),
        serving.serve_once()
    );
    assert_eq!(
        answer.unwrap().outcome,
        WireOutcome::Applied {
            response: b"executed".to_vec()
        }
    );

    let asked = WireRequest {
        addressed_to: two.identity().public_key,
        ..asked
    };
    let serving = Arc::clone(&two);
    let (answer, _) = tokio::join!(
        client.request(located(two.address()), &asked),
        serving.serve_once()
    );
    assert_eq!(
        answer.unwrap().outcome,
        WireOutcome::Applied {
            response: b"executed".to_vec()
        },
        "the second runtime never heard of that key, so it executes"
    );

    assert_eq!(first_probe.executions(), 1);
    assert_eq!(second_probe.executions(), 1);
    assert_eq!(
        one.runtime().committed_events().len(),
        two.runtime().committed_events().len(),
        "each runtime recorded its own attempt and settlement"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_runtimes_decide_concurrently_and_one_withdrawal_does_not_reach_the_other() {
    // `21-scoped-execution.md` says a genuinely concurrent race needs two runtimes,
    // because within one process a session holds no authority to mutate anything. Two
    // runtimes exist here, and both decide at the same time. What is asserted is the
    // outcomes and not the interleaving: a test that asserted an order would be
    // asserting something about the scheduler.
    let client = ExecutionClient::bind().await.unwrap();
    let (one, action, first_probe) = host_admitting(&client.identity(), Mode::Success).await;
    let (two, _, second_probe) = host_admitting(&client.identity(), Mode::Success).await;

    one.runtime()
        .withdraw_peer(&NodeId(client.identity().public_key))
        .unwrap();

    let refused_at = request(&one, &action, 1);
    let applied_at = request(&two, &action, 1);
    let serving_one = Arc::clone(&one);
    let serving_two = Arc::clone(&two);
    let (refused, applied, _, _) = tokio::join!(
        client.request(located(one.address()), &refused_at),
        client.request(located(two.address()), &applied_at),
        serving_one.serve_once(),
        serving_two.serve_once()
    );

    assert_eq!(
        refused.unwrap().outcome,
        WireOutcome::Refused {
            code: RefusalCode::Runtime
        },
        "the runtime that withdrew the peer refuses"
    );
    assert_eq!(
        applied.unwrap().outcome,
        WireOutcome::Applied {
            response: b"executed".to_vec()
        },
        "and the other one, deciding at the same moment, does not"
    );
    assert_eq!(first_probe.executions(), 0);
    assert_eq!(second_probe.executions(), 1);
}
