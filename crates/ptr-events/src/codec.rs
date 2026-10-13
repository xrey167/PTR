//! CRC32 and the on-disk frame format of the file log.
//!
//! Frame: `len: u32 LE | crc32(body): u32 LE | body`, with body
//! `offset u64 | tick u64 | class u8 | topic_len u16 | key_len u16 | topic | key | payload`
//! (all little endian).

use crate::bus::{BusRecord, EventClass, NewRecord, Offset};

pub(crate) const FRAME_HEADER: usize = 8;
const BODY_FIXED: usize = 8 + 8 + 1 + 2 + 2;
pub(crate) const MIN_BODY: usize = BODY_FIXED;
/// Longest topic or key, in bytes.
pub(crate) const MAX_FIELD_BYTES: usize = 256;

const fn crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

static CRC_TABLE: [u32; 256] = crc_table();

/// CRC-32 (IEEE 802.3, reflected, as used by zlib and Ethernet).
pub(crate) fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for byte in data {
        crc = CRC_TABLE[usize::from((crc as u8) ^ byte)] ^ (crc >> 8);
    }
    !crc
}

fn class_byte(class: EventClass) -> u8 {
    match class {
        EventClass::Projection => 1,
        EventClass::Telemetry => 2,
    }
}

fn class_from_byte(byte: u8) -> Option<EventClass> {
    match byte {
        1 => Some(EventClass::Projection),
        2 => Some(EventClass::Telemetry),
        _ => None,
    }
}

/// Largest legal frame body for a payload bound.
pub(crate) fn max_body(max_payload: usize) -> usize {
    BODY_FIXED + 2 * MAX_FIELD_BYTES + max_payload
}

/// Encode a complete frame. The caller has validated field lengths.
pub(crate) fn encode_frame(offset: Offset, tick: u64, record: &NewRecord) -> Vec<u8> {
    let topic = record.topic.as_bytes();
    let key = record.key.as_bytes();
    let body_len = BODY_FIXED + topic.len() + key.len() + record.payload.len();
    let mut body = Vec::with_capacity(body_len);
    body.extend_from_slice(&offset.0.to_le_bytes());
    body.extend_from_slice(&tick.to_le_bytes());
    body.push(class_byte(record.class));
    body.extend_from_slice(&(topic.len() as u16).to_le_bytes());
    body.extend_from_slice(&(key.len() as u16).to_le_bytes());
    body.extend_from_slice(topic);
    body.extend_from_slice(key);
    body.extend_from_slice(&record.payload);
    let mut frame = Vec::with_capacity(FRAME_HEADER + body.len());
    frame.extend_from_slice(&(body.len() as u32).to_le_bytes());
    frame.extend_from_slice(&crc32(&body).to_le_bytes());
    frame.extend_from_slice(&body);
    frame
}

/// Decode a frame body that already passed its checksum.
pub(crate) fn decode_body(body: &[u8]) -> Option<BusRecord> {
    if body.len() < BODY_FIXED {
        return None;
    }
    let offset = u64::from_le_bytes(body[0..8].try_into().ok()?);
    let tick = u64::from_le_bytes(body[8..16].try_into().ok()?);
    let class = class_from_byte(body[16])?;
    let topic_len = usize::from(u16::from_le_bytes(body[17..19].try_into().ok()?));
    let key_len = usize::from(u16::from_le_bytes(body[19..21].try_into().ok()?));
    let rest = &body[BODY_FIXED..];
    if rest.len() < topic_len + key_len {
        return None;
    }
    let topic = String::from_utf8(rest[..topic_len].to_vec()).ok()?;
    let key = String::from_utf8(rest[topic_len..topic_len + key_len].to_vec()).ok()?;
    let payload = rest[topic_len + key_len..].to_vec();
    Some(BusRecord::from_parts(
        Offset(offset),
        tick,
        class,
        topic,
        key,
        payload,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn frames_round_trip_and_detect_a_flipped_bit() {
        let new = NewRecord {
            class: EventClass::Telemetry,
            topic: "t".into(),
            key: "k".into(),
            payload: vec![0, 1, 2, 255],
        };
        let frame = encode_frame(Offset(9), 4, &new);
        let len = u32::from_le_bytes(frame[0..4].try_into().unwrap()) as usize;
        let crc = u32::from_le_bytes(frame[4..8].try_into().unwrap());
        let body = &frame[FRAME_HEADER..];
        assert_eq!(len, body.len());
        assert_eq!(crc, crc32(body));
        let record = decode_body(body).unwrap();
        assert_eq!(record.offset, Offset(9));
        assert_eq!(record.tick, 4);
        assert_eq!(record.class, EventClass::Telemetry);
        assert_eq!(record.payload, new.payload);
        let mut bad = body.to_vec();
        bad[BODY_FIXED] ^= 1;
        assert_ne!(crc32(&bad), crc);
    }

    #[test]
    fn undecodable_bodies_are_refused() {
        assert!(decode_body(&[0; 3]).is_none());
        let new = NewRecord {
            class: EventClass::Projection,
            topic: "t".into(),
            key: "k".into(),
            payload: vec![],
        };
        let frame = encode_frame(Offset(1), 0, &new);
        let mut body = frame[FRAME_HEADER..].to_vec();
        body[16] = 9;
        assert!(decode_body(&body).is_none());
        body[16] = 1;
        body[17] = 200;
        assert!(decode_body(&body).is_none());
    }
}
