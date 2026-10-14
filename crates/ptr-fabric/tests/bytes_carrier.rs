//! The `Bytes` carrier through the public API: a roundtrip and every refusal.

mod common;

use common::handle;
use ptr_fabric::{export_bytes, import_bytes, FabricError};
use ptr_protocol::{content_digest, Carrier, CarrierLocator, MAX_INLINE_BYTES};

fn tensor(len: usize) -> Vec<u8> {
    (0..len).map(|index| (index % 251) as u8).collect()
}

fn bytes_handle(bytes: &[u8]) -> ptr_protocol::TensorHandle {
    let locator = export_bytes(bytes).expect("export");
    handle(locator, bytes.len() as u64, content_digest(bytes))
}

#[test]
fn roundtrip_returns_the_exported_bytes() {
    let bytes = tensor(256);
    let handle = bytes_handle(&bytes);
    assert_eq!(handle.carrier, Carrier::Bytes);
    assert_eq!(import_bytes(&handle).expect("import"), bytes.as_slice());
}

#[test]
fn the_bound_itself_is_accepted_and_one_more_byte_is_refused() {
    let at_bound = tensor(MAX_INLINE_BYTES as usize);
    assert!(export_bytes(&at_bound).is_ok());
    let above = tensor(MAX_INLINE_BYTES as usize + 1);
    assert_eq!(
        export_bytes(&above),
        Err(FabricError::TooLarge {
            len: MAX_INLINE_BYTES + 1,
            max: MAX_INLINE_BYTES,
        })
    );
}

#[test]
fn an_empty_tensor_is_refused() {
    assert_eq!(export_bytes(&[]), Err(FabricError::EmptyTensor));
}

#[test]
fn a_wrong_digest_is_refused() {
    let mut handle = bytes_handle(&tensor(64));
    handle.content_digest[0] ^= 1;
    assert_eq!(import_bytes(&handle), Err(FabricError::DigestMismatch));
}

#[test]
fn a_tampered_inline_payload_is_refused() {
    let mut handle = bytes_handle(&tensor(64));
    let CarrierLocator::Bytes { inline } = &mut handle.locator else {
        panic!("not a Bytes locator");
    };
    inline[10] ^= 0xff;
    assert_eq!(import_bytes(&handle), Err(FabricError::DigestMismatch));
}

#[test]
fn a_wrong_locator_variant_is_refused() {
    let bytes = tensor(64);
    let mut handle = bytes_handle(&bytes);
    handle.locator = CarrierLocator::HostShm {
        fd_slot: 1,
        offset: 0,
    };
    assert_eq!(import_bytes(&handle), Err(FabricError::WrongLocator));
}

#[test]
fn a_handle_of_another_carrier_is_refused() {
    let bytes = tensor(64);
    let mut handle = bytes_handle(&bytes);
    handle.carrier = Carrier::HostShm;
    assert_eq!(import_bytes(&handle), Err(FabricError::WrongCarrier));
}

#[test]
fn a_length_mismatch_is_refused_in_both_directions() {
    let bytes = tensor(64);
    let mut longer = bytes_handle(&bytes);
    longer.byte_len = 128;
    assert_eq!(
        import_bytes(&longer),
        Err(FabricError::LengthMismatch {
            expected: 128,
            found: 64,
        })
    );
    let mut shorter = bytes_handle(&bytes);
    shorter.byte_len = 32;
    assert_eq!(
        import_bytes(&shorter),
        Err(FabricError::LengthMismatch {
            expected: 32,
            found: 64,
        })
    );
}
