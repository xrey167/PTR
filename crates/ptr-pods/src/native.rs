use crate::egress::{redact_target, redact_topic};
use crate::{
    FrameRateLimiter, NativeProtocolRequest, NativeProtocolResponse, ProtocolBinding, ProtocolError,
};
use ptr_types::EgressRefusalReason;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpStream, UdpSocket};
use tokio::time::{timeout, Duration};

pub const MAX_NATIVE_FRAME_BYTES: usize = 1 << 20;

#[derive(Clone, Debug)]
pub struct TcpProtocolExecutor {
    pub timeout: Duration,
    rate_limit: Option<Arc<FrameRateLimiter>>,
}

impl Default for TcpProtocolExecutor {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(10),
            rate_limit: None,
        }
    }
}

impl TcpProtocolExecutor {
    /// Send on behalf of one Pod lease: every request takes a token from the lease's limiter
    /// and is refused with [`EgressRefusalReason::RateLimited`] when none is left.
    pub fn with_rate_limit(mut self, limiter: Arc<FrameRateLimiter>) -> Self {
        self.rate_limit = Some(limiter);
        self
    }

    pub async fn execute(
        &self,
        request: NativeProtocolRequest,
    ) -> Result<NativeProtocolResponse, ProtocolError> {
        validate_request(&request, &ProtocolBinding::Tcp)?;
        take_frame_token(self.rate_limit.as_deref(), &request)?;
        let mut stream = timeout(
            self.timeout,
            TcpStream::connect(request.endpoint.socket_addr()),
        )
        .await
        .map_err(|_| ProtocolError::Deadline)?
        .map_err(|error| ProtocolError::Transport(error.to_string()))?;
        let frame = encode_frame(&request.input)?;
        timeout(self.timeout, stream.write_all(&frame))
            .await
            .map_err(|_| ProtocolError::Deadline)?
            .map_err(|error| ProtocolError::Transport(error.to_string()))?;
        let mut header = [0u8; 8];
        timeout(self.timeout, stream.read_exact(&mut header))
            .await
            .map_err(|_| ProtocolError::Deadline)?
            .map_err(|error| ProtocolError::Transport(error.to_string()))?;
        let type_len = u16::from_be_bytes([header[0], header[1]]) as usize;
        let body_len = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;
        let total = type_len
            .checked_add(body_len)
            .ok_or(ProtocolError::FrameTooLarge)?;
        if total > MAX_NATIVE_FRAME_BYTES || type_len == 0 {
            return Err(ProtocolError::FrameTooLarge);
        }
        let mut type_bytes = vec![0; type_len];
        let mut body = vec![0; body_len];
        timeout(self.timeout, stream.read_exact(&mut type_bytes))
            .await
            .map_err(|_| ProtocolError::Deadline)?
            .map_err(|error| ProtocolError::Transport(error.to_string()))?;
        timeout(self.timeout, stream.read_exact(&mut body))
            .await
            .map_err(|_| ProtocolError::Deadline)?
            .map_err(|error| ProtocolError::Transport(error.to_string()))?;
        let type_id = String::from_utf8(type_bytes)
            .map_err(|_| ProtocolError::Transport("response type is not UTF-8".into()))?;
        Ok(NativeProtocolResponse {
            output: ptr_protocol::TypedPayload {
                type_id: ptr_types::TypeId(type_id),
                bytes: body,
            },
            request_id: request.request_id,
            protocol: ProtocolBinding::Tcp,
        })
    }
}

#[derive(Clone, Debug)]
pub struct UdpProtocolExecutor {
    pub timeout: Duration,
    rate_limit: Option<Arc<FrameRateLimiter>>,
}

impl Default for UdpProtocolExecutor {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(10),
            rate_limit: None,
        }
    }
}

impl UdpProtocolExecutor {
    /// Send on behalf of one Pod lease; see [`TcpProtocolExecutor::with_rate_limit`].
    pub fn with_rate_limit(mut self, limiter: Arc<FrameRateLimiter>) -> Self {
        self.rate_limit = Some(limiter);
        self
    }

    pub async fn execute(
        &self,
        request: NativeProtocolRequest,
    ) -> Result<NativeProtocolResponse, ProtocolError> {
        validate_request(&request, &ProtocolBinding::Udp)?;
        take_frame_token(self.rate_limit.as_deref(), &request)?;
        let socket = UdpSocket::bind("0.0.0.0:0")
            .await
            .map_err(|error| ProtocolError::Transport(error.to_string()))?;
        let frame = encode_frame(&request.input)?;
        timeout(
            self.timeout,
            socket.send_to(&frame, request.endpoint.socket_addr()),
        )
        .await
        .map_err(|_| ProtocolError::Deadline)?
        .map_err(|error| ProtocolError::Transport(error.to_string()))?;
        let mut buffer = vec![0; MAX_NATIVE_FRAME_BYTES];
        let (length, _) = timeout(self.timeout, socket.recv_from(&mut buffer))
            .await
            .map_err(|_| ProtocolError::Deadline)?
            .map_err(|error| ProtocolError::Transport(error.to_string()))?;
        decode_response(&buffer[..length], request.request_id, ProtocolBinding::Udp)
    }
}

impl crate::protocol::NetworkEndpoint {
    fn socket_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

/// The redacted target a refusal names, chosen by the reason: a refused topic is named by its
/// digest prefix and length ([`redact_topic`]), everything else (a host, a port, a missing
/// topic, a spent budget) by `host:port`. Never the payload.
fn refusal_target(request: &NativeProtocolRequest, reason: EgressRefusalReason) -> String {
    match (&request.topic, reason) {
        (Some(topic), EgressRefusalReason::TopicNotAllowed | EgressRefusalReason::InvalidTopic) => {
            redact_topic(topic)
        }
        _ => redact_target(&request.endpoint.socket_addr()),
    }
}

fn egress_refused(request: &NativeProtocolRequest, reason: EgressRefusalReason) -> ProtocolError {
    ProtocolError::EgressRefused {
        target: refusal_target(request, reason),
        reason,
    }
}

/// Take one frame's token from the lease's limiter, when it has one.
fn take_frame_token(
    limiter: Option<&FrameRateLimiter>,
    request: &NativeProtocolRequest,
) -> Result<(), ProtocolError> {
    match limiter {
        Some(limiter) => limiter
            .try_acquire()
            .map_err(|reason| egress_refused(request, reason)),
        None => Ok(()),
    }
}

fn validate_request(
    request: &NativeProtocolRequest,
    expected: &ProtocolBinding,
) -> Result<(), ProtocolError> {
    let mqtt = request.link.protocol == ProtocolBinding::Mqtt;
    if (request.link.protocol != *expected && !mqtt)
        || request.endpoint.host.is_empty()
        || request.endpoint.port == 0
    {
        return Err(ProtocolError::InvalidEndpoint);
    }
    // What the policy lists is a decision about this request, not a malformed one, so a
    // refusal carries its reason and its redacted target for the ledger.
    request
        .link
        .egress_policy
        .check_endpoint(&request.endpoint.host, request.endpoint.port)
        .map_err(|reason| egress_refused(request, reason))?;
    // These executors send `request.input` as it is and never read a topic out of it, so the
    // topic a request declares is metadata they cannot tie to the wire: a Pod could declare an
    // allowed topic and publish another. Under a topic policy, or on an MQTT link, they refuse
    // (fail closed) instead of trusting the declaration. `EgressPolicy::check_publication` is
    // for an executor that takes the topic from the frame it encodes.
    if mqtt || !request.link.egress_policy.topics.is_empty() {
        return Err(egress_refused(
            request,
            EgressRefusalReason::TopicUnverifiable,
        ));
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| ProtocolError::Deadline)?
        .as_millis() as u64;
    if request.link.deadline.0 <= now || request.link.hop_limit == 0 {
        return Err(ProtocolError::Deadline);
    }
    Ok(())
}

fn encode_frame(payload: &ptr_protocol::TypedPayload) -> Result<Vec<u8>, ProtocolError> {
    let type_bytes = payload.type_id.0.as_bytes();
    if type_bytes.is_empty() || type_bytes.len() > u16::MAX as usize {
        return Err(ProtocolError::FrameTooLarge);
    }
    if payload.bytes.len() > MAX_NATIVE_FRAME_BYTES
        || type_bytes.len().saturating_add(payload.bytes.len()) > MAX_NATIVE_FRAME_BYTES
    {
        return Err(ProtocolError::FrameTooLarge);
    }
    let mut frame = Vec::with_capacity(8 + type_bytes.len() + payload.bytes.len());
    frame.extend_from_slice(&(type_bytes.len() as u16).to_be_bytes());
    frame.extend_from_slice(&[0, 0]);
    frame.extend_from_slice(&(payload.bytes.len() as u32).to_be_bytes());
    frame.extend_from_slice(type_bytes);
    frame.extend_from_slice(&payload.bytes);
    Ok(frame)
}

fn decode_response(
    frame: &[u8],
    request_id: ptr_types::RequestId,
    protocol: ProtocolBinding,
) -> Result<NativeProtocolResponse, ProtocolError> {
    if frame.len() < 8 {
        return Err(ProtocolError::Transport("short native frame".into()));
    }
    let type_len = u16::from_be_bytes([frame[0], frame[1]]) as usize;
    let body_len = u32::from_be_bytes([frame[4], frame[5], frame[6], frame[7]]) as usize;
    let expected = 8usize
        .checked_add(type_len)
        .and_then(|value| value.checked_add(body_len))
        .ok_or(ProtocolError::FrameTooLarge)?;
    if expected != frame.len() || expected > MAX_NATIVE_FRAME_BYTES || type_len == 0 {
        return Err(ProtocolError::FrameTooLarge);
    }
    let type_end = 8 + type_len;
    let type_id = String::from_utf8(frame[8..type_end].to_vec())
        .map_err(|_| ProtocolError::Transport("response type is not UTF-8".into()))?;
    Ok(NativeProtocolResponse {
        output: ptr_protocol::TypedPayload {
            type_id: ptr_types::TypeId(type_id),
            bytes: frame[type_end..].to_vec(),
        },
        request_id,
        protocol,
    })
}
