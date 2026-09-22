//! Pod access over a real authenticated connection.
//!
//! What is worth a network test here, rather than a direct call: that a request
//! actually traverses `ALPN_PODWIRE` and is resolved inside the project *policy*
//! names, and that neither side believes a claim the connection did not make.
//! Everything about which Pod answers which request is tested in `tests/access.rs`,
//! where no socket is involved.
//!
//! Every test drives one exchange with `tokio::join!` rather than a background
//! loop, so a failure is a failure of the protocol and never of a race with a
//! server task.
use ptr_net::{EndpointAddr, IrohTransport, PeerAddress, PeerBook, ALPN_PODWIRE};
use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_podwire::{
    decode_answer, decode_request, encode_answer, encode_request, request_digest, PodAccessPolicy,
    PodAnswer, PodClient, PodHost, PodOutcome, PodRequest, PodScope, PodWireError, RefusalCode,
    MAX_BODY_BYTES,
};
use ptr_protocol::TypedPayload;
use ptr_types::{
    CapabilityId, Effect, NodeId, PodId, Probability, ProjectId, TypeId, VerificationLevel,
};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};
use std::sync::{Arc, Mutex};

const SUMMARIZE: &str = "Summarize<Document>";
const DOCUMENT: &str = "Document";
const SUMMARY: &str = "Summary";

/// A Pod that stamps its own project onto whatever it is given, so a test can see
/// **which** Pod answered rather than only that one did.
struct Stamping {
    manifest: PodManifest,
    invocations: Arc<Mutex<usize>>,
}

impl DynPod for Stamping {
    fn manifest(&self) -> &PodManifest {
        &self.manifest
    }

    fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String> {
        *self.invocations.lock().unwrap() += 1;
        let mut bytes = self.manifest.project.0.clone().into_bytes();
        bytes.push(b':');
        bytes.extend_from_slice(&input.bytes);
        Ok(TypedPayload {
            type_id: TypeId::from(SUMMARY),
            bytes,
        })
    }
}

fn pod(project: &str, invocations: &Arc<Mutex<usize>>) -> Arc<Stamping> {
    Arc::new(Stamping {
        manifest: PodManifest {
            project: ProjectId::from(project),
            id: PodId::from("summarizer"),
            capabilities: vec![CapabilityId::from(SUMMARIZE)],
            accepts: vec![TypeId::from(DOCUMENT)],
            produces: vec![TypeId::from(SUMMARY)],
            effects: vec![Effect::Pure],
            protocol_version: 1,
        },
        invocations: Arc::clone(invocations),
    })
}

/// A Pod whose output cannot fit in a frame.
struct Enormous {
    manifest: PodManifest,
}

impl DynPod for Enormous {
    fn manifest(&self) -> &PodManifest {
        &self.manifest
    }

    fn invoke(&self, _: TypedPayload) -> Result<TypedPayload, String> {
        Ok(TypedPayload {
            type_id: TypeId::from(SUMMARY),
            bytes: vec![0; MAX_BODY_BYTES + 1],
        })
    }
}

struct Pass;
impl Verifier<TypedPayload> for Pass {
    fn verify(&self, _: &TypedPayload) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Pass,
            level: VerificationLevel::Deterministic,
            score: Probability::new(1.0).unwrap(),
            findings: vec![],
        }
    }
}

fn scope(project: &str) -> PodScope {
    PodScope::new(ProjectId::from(project))
        .allow(CapabilityId::from(SUMMARIZE), TypeId::from(DOCUMENT))
}

/// Where to dial, on the deployment's authority.
///
/// A `PodClient` accepts nothing else, so every test here goes through a book — which
/// is the point: a bare address does not satisfy the signature.
fn located(address: EndpointAddr) -> PeerAddress {
    let mut book = PeerBook::new();
    book.record(address.clone());
    book.locate(&NodeId(address.id.to_string()))
        .expect("an address just recorded is locatable")
}

/// The request every test sends. Note what it does **not** contain: a project, a
/// session, or anything naming its sender.
fn request(addressed_to: String, request_id: u64) -> PodRequest {
    PodRequest {
        addressed_to,
        request_id,
        capability: CapabilityId::from(SUMMARIZE),
        expect_protocol: 1,
        payload: TypedPayload {
            type_id: TypeId::from(DOCUMENT),
            bytes: b"a document".to_vec(),
        },
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_admitted_peer_reaches_its_project_s_pod_over_a_real_connection() {
    let client = PodClient::bind().await.unwrap();
    let invocations = Arc::new(Mutex::new(0));
    let mut registry = PodRegistry::default();
    registry.register(pod("alpha", &invocations));
    let mut policy = PodAccessPolicy::new();
    policy
        .admit(NodeId(client.identity().public_key), scope("alpha"))
        .unwrap();
    let host = PodHost::bind(registry, policy, Pass).await.unwrap();

    let asked = request(host.identity().public_key, 1);
    let address = host.address();
    let (answered, served) =
        tokio::join!(client.request(located(address), &asked), host.serve_once());
    let answered = answered.unwrap();
    let served = served.unwrap();

    assert_eq!(
        answered.outcome,
        PodOutcome::Answered {
            output: TypedPayload {
                type_id: TypeId::from(SUMMARY),
                bytes: b"alpha:a document".to_vec(),
            }
        },
        "the Pod ran over a real connection and its output came back typed"
    );
    assert_eq!(
        answered.responder,
        host.identity().public_key,
        "the answer names its author"
    );
    assert_eq!(answered.request_id, 1);
    assert_eq!(served.request_id, Some(1));
    assert_eq!(served.refused, None);
    assert_eq!(
        served.peer.public_key,
        client.identity().public_key,
        "the peer is the connection's, and the frame has no field it could have been"
    );
    assert_eq!(*invocations.lock().unwrap(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_peers_send_the_same_bytes_and_reach_two_different_projects_pods() {
    // The protocol's central claim, driven end to end. Both requesters send frames
    // that are **byte-for-byte identical** — same capability, same payload type,
    // same payload, same request id, same host key. Nothing in either frame could
    // select a project, because the format has nowhere to put one. They reach
    // different Pods, and each reaches only its own project's.
    let alpha_client = PodClient::bind().await.unwrap();
    let beta_client = PodClient::bind().await.unwrap();
    let alpha_runs = Arc::new(Mutex::new(0));
    let beta_runs = Arc::new(Mutex::new(0));

    let mut registry = PodRegistry::default();
    registry.register(pod("alpha", &alpha_runs));
    registry.register(pod("beta", &beta_runs));
    let mut policy = PodAccessPolicy::new();
    policy
        .admit(NodeId(alpha_client.identity().public_key), scope("alpha"))
        .unwrap();
    policy
        .admit(NodeId(beta_client.identity().public_key), scope("beta"))
        .unwrap();
    let host = PodHost::bind(registry, policy, Pass).await.unwrap();

    let asked = request(host.identity().public_key, 7);

    let address = host.address();
    let (from_alpha, _) = tokio::join!(
        alpha_client.request(located(address), &asked),
        host.serve_once()
    );
    let address = host.address();
    let (from_beta, _) = tokio::join!(
        beta_client.request(located(address), &asked),
        host.serve_once()
    );

    // One request value, so the frames the two sent are the same bytes. Asserted
    // rather than assumed, because the whole claim rests on it.
    assert_eq!(
        encode_request(&asked).unwrap(),
        encode_request(&asked).unwrap()
    );

    let alpha_output = match from_alpha.unwrap().outcome {
        PodOutcome::Answered { output } => output,
        other => panic!("alpha's peer was refused: {other:?}"),
    };
    let beta_output = match from_beta.unwrap().outcome {
        PodOutcome::Answered { output } => output,
        other => panic!("beta's peer was refused: {other:?}"),
    };
    assert_eq!(alpha_output.bytes, b"alpha:a document");
    assert_eq!(beta_output.bytes, b"beta:a document");
    assert_eq!(
        *alpha_runs.lock().unwrap(),
        1,
        "alpha's Pod ran once, for alpha's peer"
    );
    assert_eq!(
        *beta_runs.lock().unwrap(),
        1,
        "beta's Pod ran once, for beta's peer"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_peer_the_host_never_admitted_is_refused_and_no_pod_runs() {
    let stranger = PodClient::bind().await.unwrap();
    let invocations = Arc::new(Mutex::new(0));
    let mut registry = PodRegistry::default();
    registry.register(pod("alpha", &invocations));
    let host = PodHost::bind(registry, PodAccessPolicy::new(), Pass)
        .await
        .unwrap();

    let asked = request(host.identity().public_key, 2);
    let address = host.address();
    let (answered, served) = tokio::join!(
        stranger.request(located(address), &asked),
        host.serve_once()
    );

    assert_eq!(
        answered.unwrap().outcome,
        PodOutcome::Refused {
            code: RefusalCode::NotAdmitted
        }
    );
    assert_eq!(
        served.unwrap().refused,
        Some(PodWireError::Refused(RefusalCode::NotAdmitted))
    );
    assert_eq!(
        *invocations.lock().unwrap(),
        0,
        "an unadmitted peer must not reach a Pod at all"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn withdrawing_a_peer_takes_effect_on_its_very_next_request_over_the_wire() {
    let client = PodClient::bind().await.unwrap();
    let invocations = Arc::new(Mutex::new(0));
    let mut registry = PodRegistry::default();
    registry.register(pod("alpha", &invocations));
    let mut policy = PodAccessPolicy::new();
    policy
        .admit(NodeId(client.identity().public_key), scope("alpha"))
        .unwrap();
    let mut host = PodHost::bind(registry, policy, Pass).await.unwrap();

    // The control first: the same peer, the same request, admitted.
    let asked = request(host.identity().public_key, 1);
    let address = host.address();
    let (answered, _) = tokio::join!(client.request(located(address), &asked), host.serve_once());
    assert!(matches!(
        answered.unwrap().outcome,
        PodOutcome::Answered { .. }
    ));

    host.policy_mut()
        .withdraw(&NodeId(client.identity().public_key));

    let asked = request(host.identity().public_key, 2);
    let address = host.address();
    let (answered, _) = tokio::join!(client.request(located(address), &asked), host.serve_once());
    assert_eq!(
        answered.unwrap().outcome,
        PodOutcome::Refused {
            code: RefusalCode::NotAdmitted
        },
        "no authority survives the withdrawal, not even for one more request"
    );
    assert_eq!(
        *invocations.lock().unwrap(),
        1,
        "only the admitted request reached the Pod"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_request_addressed_to_another_host_is_refused_and_no_pod_runs() {
    let client = PodClient::bind().await.unwrap();
    let elsewhere = IrohTransport::bind(&[ALPN_PODWIRE]).await.unwrap();
    let invocations = Arc::new(Mutex::new(0));
    let mut registry = PodRegistry::default();
    registry.register(pod("alpha", &invocations));
    let mut policy = PodAccessPolicy::new();
    policy
        .admit(NodeId(client.identity().public_key), scope("alpha"))
        .unwrap();
    let host = PodHost::bind(registry, policy, Pass).await.unwrap();

    // Sent to this host, addressed to another. Without this check a request
    // composed for one host could be relayed to a second and be indistinguishable
    // there from one meant for it.
    let mut asked = request(host.identity().public_key, 3);
    asked.addressed_to = elsewhere.identity().public_key;
    let address = host.address();
    let (answered, served) =
        tokio::join!(client.request(located(address), &asked), host.serve_once());

    assert_eq!(
        answered.unwrap().outcome,
        PodOutcome::Refused {
            code: RefusalCode::Misaddressed
        }
    );
    assert!(matches!(
        served.unwrap().refused,
        Some(PodWireError::Misaddressed { .. })
    ));
    assert_eq!(*invocations.lock().unwrap(), 0);

    // The control: the same peer and the same host, addressed correctly.
    let asked = request(host.identity().public_key, 4);
    let address = host.address();
    let (answered, _) = tokio::join!(client.request(located(address), &asked), host.serve_once());
    assert!(matches!(
        answered.unwrap().outcome,
        PodOutcome::Answered { .. }
    ));
    elsewhere.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_damaged_frame_is_refused_before_any_pod_and_still_answered() {
    // Bytes this build cannot parse, sent by a peer the host *does* admit — so the
    // refusal is about the frame and not about the peer. Sent through a raw
    // transport, because a client would have encoded them correctly.
    let raw = IrohTransport::bind(&[ALPN_PODWIRE]).await.unwrap();
    let invocations = Arc::new(Mutex::new(0));
    let mut registry = PodRegistry::default();
    registry.register(pod("alpha", &invocations));
    let mut policy = PodAccessPolicy::new();
    policy
        .admit(NodeId(raw.identity().public_key), scope("alpha"))
        .unwrap();
    let host = PodHost::bind(registry, policy, Pass).await.unwrap();

    let garbage = b"PTRPWREQ\x01\x00 and then nothing that parses".to_vec();
    let digest = request_digest(&garbage);
    let address = host.address();
    let sending = raw.request(address, ALPN_PODWIRE, &garbage, 1 << 20);
    let (response, served) = tokio::join!(sending, host.serve_once());
    let response = response.unwrap();
    let served = served.unwrap();

    let answered = decode_answer(&response).unwrap();
    assert_eq!(
        answered.outcome,
        PodOutcome::Refused {
            code: RefusalCode::Malformed
        }
    );
    assert_eq!(
        answered.request_digest, digest,
        "a request this build cannot parse still gets an answer its sender can check"
    );
    assert_eq!(
        answered.request_id, 0,
        "there was no request id to answer, and the answer says so rather than inventing one"
    );
    assert!(matches!(served.refused, Some(PodWireError::Frame(_))));
    assert_eq!(*invocations.lock().unwrap(), 0);
    raw.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_answer_naming_another_host_is_refused_and_an_honest_one_is_not() {
    // An answer can be stored and forwarded, so the only thing that makes its
    // author mean anything is checking it against the peer the connection
    // authenticated, at the moment it arrives.
    let client = PodClient::bind().await.unwrap();
    let elsewhere = IrohTransport::bind(&[ALPN_PODWIRE]).await.unwrap();
    let impersonated = elsewhere.identity().public_key;

    // A rogue endpoint whose answer is well formed and correctly bound to the
    // request. Only its author is wrong.
    let rogue = IrohTransport::bind(&[ALPN_PODWIRE]).await.unwrap();
    let rogue_key = rogue.identity().public_key;
    let rogue_address = rogue.direct_addr();
    let asked = request(rogue_key.clone(), 11);

    let claimed = impersonated.clone();
    let serving = async move {
        let incoming = rogue.accept_once(1 << 20).await.unwrap();
        let decoded = decode_request(&incoming.payload).unwrap();
        let answered = PodAnswer {
            responder: claimed,
            request_id: decoded.request_id,
            request_digest: request_digest(&incoming.payload),
            outcome: PodOutcome::Answered {
                output: TypedPayload {
                    type_id: TypeId::from(SUMMARY),
                    bytes: b"a summary nobody here produced".to_vec(),
                },
            },
        };
        incoming
            .respond(&encode_answer(&answered).unwrap())
            .await
            .unwrap();
    };

    let (answered, ()) = tokio::join!(client.request(located(rogue_address), &asked), serving);
    match answered.expect_err("an answer naming another endpoint") {
        PodWireError::ForgedAnswer {
            authenticated,
            claimed,
        } => {
            assert_eq!(authenticated, rogue_key);
            assert_eq!(claimed, impersonated);
        }
        other => panic!("expected a forged answer, got {other:?}"),
    }

    // The control: the same endpoint answering honestly is not refused. Without it
    // the refusal above would be consistent with that endpoint being unreachable.
    let honest = IrohTransport::bind(&[ALPN_PODWIRE]).await.unwrap();
    let honest_key = honest.identity().public_key;
    let honest_address = honest.direct_addr();
    let asked = request(honest_key.clone(), 12);
    let responder = honest_key.clone();
    let serving = async move {
        let incoming = honest.accept_once(1 << 20).await.unwrap();
        let decoded = decode_request(&incoming.payload).unwrap();
        let answered = PodAnswer {
            responder,
            request_id: decoded.request_id,
            request_digest: request_digest(&incoming.payload),
            outcome: PodOutcome::Refused {
                code: RefusalCode::Unavailable,
            },
        };
        incoming
            .respond(&encode_answer(&answered).unwrap())
            .await
            .unwrap();
    };
    let (answered, ()) = tokio::join!(client.request(located(honest_address), &asked), serving);
    assert_eq!(
        answered.expect("an honest author is not refused").responder,
        honest_key
    );
    elsewhere.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_answer_to_another_request_and_one_bound_to_other_bytes_are_each_refused() {
    let client = PodClient::bind().await.unwrap();

    // An answer to a request this requester did not send. The digest is right, so
    // only the id catches it.
    let wrong_id = IrohTransport::bind(&[ALPN_PODWIRE]).await.unwrap();
    let address = wrong_id.direct_addr();
    let key = wrong_id.identity().public_key;
    let asked = request(key.clone(), 21);
    let responder = key.clone();
    let serving = async move {
        let incoming = wrong_id.accept_once(1 << 20).await.unwrap();
        let answered = PodAnswer {
            responder,
            request_id: 22,
            request_digest: request_digest(&incoming.payload),
            outcome: PodOutcome::Refused {
                code: RefusalCode::Unavailable,
            },
        };
        incoming
            .respond(&encode_answer(&answered).unwrap())
            .await
            .unwrap();
    };
    let (answered, ()) = tokio::join!(client.request(located(address), &asked), serving);
    assert_eq!(
        answered.expect_err("an answer to another request"),
        PodWireError::WrongRequest {
            expected: 21,
            answered: 22
        }
    );

    // An answer bound to other bytes. The id is right, so only the digest catches
    // it — a well-formed reply to a question nobody asked here.
    let wrong_digest = IrohTransport::bind(&[ALPN_PODWIRE]).await.unwrap();
    let address = wrong_digest.direct_addr();
    let key = wrong_digest.identity().public_key;
    let asked = request(key.clone(), 31);
    let responder = key.clone();
    let serving = async move {
        let incoming = wrong_digest.accept_once(1 << 20).await.unwrap();
        let answered = PodAnswer {
            responder,
            request_id: 31,
            request_digest: request_digest(b"some other request entirely"),
            outcome: PodOutcome::Refused {
                code: RefusalCode::Unavailable,
            },
        };
        incoming
            .respond(&encode_answer(&answered).unwrap())
            .await
            .unwrap();
    };
    let (answered, ()) = tokio::join!(client.request(located(address), &asked), serving);
    assert_eq!(
        answered.expect_err("an answer bound to other bytes"),
        PodWireError::UnboundAnswer
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_answer_that_will_not_fit_in_a_frame_is_still_an_answer() {
    // The failure this guards against is not a wrong reply, it is *no* reply: an
    // answer this side cannot encode would leave the requester holding an open
    // connection until its deadline, because the host could not phrase what it
    // wanted to say. Nothing applied, so the refusal costs the requester nothing to
    // act on — which is exactly what the execution wire cannot say in this position.
    let client = PodClient::bind().await.unwrap();
    let mut registry = PodRegistry::default();
    registry.register(Arc::new(Enormous {
        manifest: PodManifest {
            project: ProjectId::from("alpha"),
            id: PodId::from("enormous"),
            capabilities: vec![CapabilityId::from(SUMMARIZE)],
            accepts: vec![TypeId::from(DOCUMENT)],
            produces: vec![TypeId::from(SUMMARY)],
            effects: vec![Effect::Pure],
            protocol_version: 1,
        },
    }));
    let mut policy = PodAccessPolicy::new();
    policy
        .admit(NodeId(client.identity().public_key), scope("alpha"))
        .unwrap();
    let host = PodHost::bind(registry, policy, Pass).await.unwrap();

    let asked = request(host.identity().public_key, 41);
    let address = host.address();
    let (answered, served) =
        tokio::join!(client.request(located(address), &asked), host.serve_once());

    let answered = answered.expect("the requester is answered rather than left waiting");
    assert_eq!(
        answered.outcome,
        PodOutcome::Refused {
            code: RefusalCode::AnswerTooLarge
        }
    );
    assert_eq!(
        answered.request_id, 41,
        "and the answer is still bound to the request"
    );

    // The operator learns which bound it was; the requester does not need to.
    let served = served.unwrap();
    assert_eq!(served.reported, answered.outcome);
    assert!(
        matches!(served.refused, Some(PodWireError::Frame(_))),
        "the host records the framing refusal behind it, got {:?}",
        served.refused
    );

    // The control: a Pod whose output does fit is answered, so the refusal above is
    // about the size and not about this host.
    let mut registry = PodRegistry::default();
    let invocations = Arc::new(Mutex::new(0));
    registry.register(pod("alpha", &invocations));
    let mut policy = PodAccessPolicy::new();
    policy
        .admit(NodeId(client.identity().public_key), scope("alpha"))
        .unwrap();
    let host = PodHost::bind(registry, policy, Pass).await.unwrap();
    let asked = request(host.identity().public_key, 42);
    let address = host.address();
    let (answered, _) = tokio::join!(client.request(located(address), &asked), host.serve_once());
    assert!(matches!(
        answered.unwrap().outcome,
        PodOutcome::Answered { .. }
    ));
}
