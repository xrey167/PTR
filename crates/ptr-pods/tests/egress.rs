//! What a Pod's link lets it send out: destinations, topics and rate.
//!
//! Every refusal here is a policy decision, so each must carry its reason and a redacted
//! target and must leave nothing sent. The payload never appears in a refusal.
use ptr_pods::{
    redact_topic, EgressPolicy, FrameRateLimit, FrameRateLimiter, NativeProtocolRequest,
    NetworkEndpoint, PodLink, PodLinkError, ProtocolBinding, ProtocolError, TcpProtocolExecutor,
    UdpProtocolExecutor,
};
use ptr_protocol::TypedPayload;
use ptr_types::{
    ArtifactId, CapabilityId, EgressRefusalReason, Generation, NamespaceId, PodAddress, PodId,
    PodRevisionAddress, ProjectId, RequestId, Timestamp, TypeId,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const SECRET: &[u8] = b"the-payload-must-never-be-recorded";

fn policy(topics: &[&str], hosts: &[&str], ports: &[u16]) -> EgressPolicy {
    EgressPolicy {
        topics: topics.iter().map(|value| (*value).to_owned()).collect(),
        hosts: hosts.iter().map(|value| (*value).to_owned()).collect(),
        ports: ports.to_vec(),
    }
}

fn link(protocol: ProtocolBinding, egress_policy: EgressPolicy) -> PodLink {
    PodLink {
        trace_id: "egress-test".into(),
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
        execution_manifest: [3; 32],
        protocol,
        capability: CapabilityId::from("exchange"),
        acl: Vec::new(),
        deadline: Timestamp(u64::MAX),
        hop_limit: 4,
        visited: Vec::new(),
        attestation: [2; 32],
        egress_policy,
    }
}

fn request(
    protocol: ProtocolBinding,
    egress_policy: EgressPolicy,
    host: &str,
    port: u16,
    topic: Option<&str>,
) -> NativeProtocolRequest {
    NativeProtocolRequest {
        link: link(protocol, egress_policy),
        input: TypedPayload {
            type_id: TypeId::from("Request"),
            bytes: SECRET.to_vec(),
        },
        request_id: RequestId::from("egress-request"),
        endpoint: NetworkEndpoint {
            host: host.into(),
            port,
        },
        topic: topic.map(str::to_owned),
    }
}

fn refusal(error: ProtocolError) -> (String, EgressRefusalReason) {
    match error {
        ProtocolError::EgressRefused { target, reason } => (target, reason),
        other => panic!("expected an egress refusal, got {other:?}"),
    }
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

async fn answering_listener() -> (u16, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut header = [0; 8];
        stream.read_exact(&mut header).await.unwrap();
        let type_len = u16::from_be_bytes([header[0], header[1]]) as usize;
        let body_len = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;
        let mut rest = vec![0; type_len + body_len];
        stream.read_exact(&mut rest).await.unwrap();
        stream
            .write_all(&response_frame("Response", b"ok"))
            .await
            .unwrap();
    });
    (port, server)
}

#[test]
fn a_topic_must_match_a_listed_filter_with_mqtt_wildcards() {
    let policy = policy(&["sensors/+/temp", "alerts/#"], &[], &[]);
    assert_eq!(policy.check_topic("sensors/kitchen/temp"), Ok(()));
    assert_eq!(policy.check_topic("alerts"), Ok(()));
    assert_eq!(policy.check_topic("alerts/fire/floor2"), Ok(()));
    for refused in ["sensors/kitchen/humidity", "sensors/a/b/temp", "other/x"] {
        assert_eq!(
            policy.check_topic(refused),
            Err(EgressRefusalReason::TopicNotAllowed),
            "{refused}"
        );
    }
}

#[test]
fn a_publication_that_names_a_filter_instead_of_a_topic_is_invalid_not_merely_unlisted() {
    let policy = policy(&["#"], &[], &[]);
    for invalid in ["", "sensors/+", "sensors/#", "has space", "line\nbreak"] {
        assert_eq!(
            policy.check_topic(invalid),
            Err(EgressRefusalReason::InvalidTopic),
            "{invalid:?}"
        );
    }
    assert_eq!(
        policy.check_topic("$SYS/broker"),
        Err(EgressRefusalReason::TopicNotAllowed),
        "a leading wildcard never reaches a system topic"
    );
}

#[test]
fn a_policy_with_a_malformed_topic_filter_is_not_a_policy() {
    for filter in ["a/#/b", "a+", "a/b#"] {
        assert_eq!(
            policy(&[filter], &[], &[]).validate(),
            Err(PodLinkError::InvalidEgressPolicy),
            "{filter}"
        );
    }
    assert_eq!(policy(&["a/+/c", "d/#"], &["h"], &[1]).validate(), Ok(()));
}

#[tokio::test]
async fn an_unlisted_host_is_refused_with_its_redacted_target_and_nothing_is_sent() {
    let allowed = policy(&[], &["127.0.0.1"], &[9]);
    let error = TcpProtocolExecutor::default()
        .execute(request(
            ProtocolBinding::Tcp,
            allowed,
            "evil.example",
            9,
            None,
        ))
        .await
        .unwrap_err();
    let (target, reason) = refusal(error);
    assert_eq!(reason, EgressRefusalReason::HostNotAllowed);
    assert_eq!(target, "evil.example:9");
    assert!(
        !target.as_bytes().windows(SECRET.len()).any(|w| w == SECRET),
        "the payload is not part of the recorded target"
    );
}

#[tokio::test]
async fn an_unlisted_port_and_an_unlisted_topic_each_name_their_own_reason() {
    let allowed = policy(&["sensors/#"], &["127.0.0.1"], &[1883]);
    let port_error = UdpProtocolExecutor::default()
        .execute(request(
            ProtocolBinding::Udp,
            allowed.clone(),
            "127.0.0.1",
            1884,
            None,
        ))
        .await
        .unwrap_err();
    assert_eq!(
        refusal(port_error),
        (
            "127.0.0.1:1884".to_owned(),
            EgressRefusalReason::PortNotAllowed
        )
    );

    let topic_error = UdpProtocolExecutor::default()
        .execute(request(
            ProtocolBinding::Udp,
            allowed.clone(),
            "127.0.0.1",
            1883,
            Some("secrets/vault"),
        ))
        .await
        .unwrap_err();
    assert_eq!(
        refusal(topic_error),
        (
            redact_topic("secrets/vault"),
            EgressRefusalReason::TopicNotAllowed
        ),
        "a topic refusal names a digest of the topic, not the topic and not the broker"
    );

    let wildcard_error = UdpProtocolExecutor::default()
        .execute(request(
            ProtocolBinding::Udp,
            allowed,
            "127.0.0.1",
            1883,
            Some("sensors/#"),
        ))
        .await
        .unwrap_err();
    assert_eq!(refusal(wildcard_error).1, EgressRefusalReason::InvalidTopic);
}

#[tokio::test]
async fn a_malformed_endpoint_stays_an_invalid_endpoint_and_is_not_a_policy_refusal() {
    let allowed = policy(&[], &["127.0.0.1"], &[9]);
    let error = TcpProtocolExecutor::default()
        .execute(request(ProtocolBinding::Tcp, allowed, "", 9, None))
        .await
        .unwrap_err();
    assert_eq!(error, ProtocolError::InvalidEndpoint);
}

#[tokio::test]
async fn a_lease_that_used_its_budget_is_refused_until_the_bucket_refills() {
    let (port, server) = answering_listener().await;
    let allowed = policy(&[], &["127.0.0.1"], &[port]);
    let limiter = Arc::new(
        FrameRateLimiter::new(
            FrameRateLimit {
                max_per_window: 1,
                window: Duration::from_secs(3600),
            },
            Instant::now(),
        )
        .unwrap(),
    );
    let executor = TcpProtocolExecutor::default().with_rate_limit(limiter);

    let first = executor
        .execute(request(
            ProtocolBinding::Tcp,
            allowed.clone(),
            "127.0.0.1",
            port,
            None,
        ))
        .await
        .unwrap();
    assert_eq!(first.output.bytes, b"ok");
    server.await.unwrap();

    let second = executor
        .execute(request(
            ProtocolBinding::Tcp,
            allowed,
            "127.0.0.1",
            port,
            None,
        ))
        .await
        .unwrap_err();
    assert_eq!(
        refusal(second),
        (
            format!("127.0.0.1:{port}"),
            EgressRefusalReason::RateLimited
        )
    );
}

#[test]
fn a_flood_is_bounded_by_the_bucket_and_never_queued() {
    let start = Instant::now();
    let limiter = FrameRateLimiter::new(
        FrameRateLimit {
            max_per_window: 8,
            window: Duration::from_secs(1),
        },
        start,
    )
    .unwrap();
    let admitted = (0..10_000)
        .filter(|_| limiter.try_acquire_at(start).is_ok())
        .count();
    assert_eq!(admitted, 8);
}

#[tokio::test]
async fn a_policy_that_lists_topics_refuses_a_request_that_names_none() {
    let allowed = policy(&["sensors/#"], &["127.0.0.1"], &[1883]);
    let error = UdpProtocolExecutor::default()
        .execute(request(
            ProtocolBinding::Udp,
            allowed.clone(),
            "127.0.0.1",
            1883,
            None,
        ))
        .await
        .unwrap_err();
    assert_eq!(
        refusal(error),
        (
            "127.0.0.1:1883".to_owned(),
            EgressRefusalReason::TopicRequired
        ),
        "omitting the topic must not be a way around the topic policy"
    );
    assert_eq!(
        allowed.check_publication(None),
        Err(EgressRefusalReason::TopicRequired)
    );
    assert_eq!(policy(&[], &[], &[]).check_publication(None), Ok(()));
}

#[test]
fn a_refused_topic_is_recorded_as_a_digest_prefix_and_a_length() {
    let target = redact_topic("tokens/s3cr3t-credential");
    assert!(target.starts_with("topic:"), "{target}");
    assert!(!target.contains("s3cr3t"), "{target}");
    assert!(target.ends_with(":24"), "{target}");
    assert_eq!(target, redact_topic("tokens/s3cr3t-credential"));
    assert_ne!(target, redact_topic("tokens/other-credential!!"));
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[tokio::test]
async fn a_deadline_is_milliseconds_since_the_epoch_and_an_expired_one_is_refused() {
    let allowed = policy(&[], &["127.0.0.1"], &[9]);
    for expired in [1, now_millis() - 1] {
        let mut asked = request(ProtocolBinding::Tcp, allowed.clone(), "127.0.0.1", 9, None);
        asked.link.deadline = Timestamp(expired);
        assert_eq!(
            TcpProtocolExecutor::default()
                .execute(asked)
                .await
                .unwrap_err(),
            ProtocolError::Deadline,
            "{expired}"
        );
    }
    // A deadline written in seconds is, read as milliseconds, long past.
    let mut in_seconds = request(ProtocolBinding::Tcp, allowed, "127.0.0.1", 9, None);
    in_seconds.link.deadline = Timestamp(now_millis() / 1000 + 3600);
    assert_eq!(
        TcpProtocolExecutor::default()
            .execute(in_seconds)
            .await
            .unwrap_err(),
        ProtocolError::Deadline
    );
}

#[tokio::test]
async fn a_future_millisecond_deadline_reaches_the_peer() {
    let (port, server) = answering_listener().await;
    let allowed = policy(&[], &["127.0.0.1"], &[port]);
    let mut asked = request(ProtocolBinding::Tcp, allowed, "127.0.0.1", port, None);
    asked.link.deadline = Timestamp(now_millis() + 60_000);
    let response = TcpProtocolExecutor::default().execute(asked).await.unwrap();
    assert_eq!(response.output.bytes, b"ok");
    server.await.unwrap();
}
