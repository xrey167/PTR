use crate::{NativeProtocolRequest, NativeProtocolResponse, ProtocolBinding, ProtocolError};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpStream, UdpSocket};
use tokio::time::{timeout, Duration};

pub const MAX_NATIVE_FRAME_BYTES: usize = 1 << 20;

#[derive(Clone, Debug)]
pub struct TcpProtocolExecutor {
    pub timeout: Duration,
}

impl Default for TcpProtocolExecutor {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(10),
        }
    }
}

impl TcpProtocolExecutor {
    pub async fn execute(
        &self,
        request: NativeProtocolRequest,
    ) -> Result<NativeProtocolResponse, ProtocolError> {
        validate_request(&request, &ProtocolBinding::Tcp)?;
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
}

impl Default for UdpProtocolExecutor {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(10),
        }
    }
}

impl UdpProtocolExecutor {
    pub async fn execute(
        &self,
        request: NativeProtocolRequest,
    ) -> Result<NativeProtocolResponse, ProtocolError> {
        validate_request(&request, &ProtocolBinding::Udp)?;
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

fn validate_request(
    request: &NativeProtocolRequest,
    expected: &ProtocolBinding,
) -> Result<(), ProtocolError> {
    if request.link.protocol != *expected
        || request.endpoint.host.is_empty()
        || request.endpoint.port == 0
        || !request
            .link
            .egress_policy
            .hosts
            .contains(&request.endpoint.host)
        || !request
            .link
            .egress_policy
            .ports
            .contains(&request.endpoint.port)
    {
        return Err(ProtocolError::InvalidEndpoint);
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| ProtocolError::Deadline)?
        .as_secs();
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
