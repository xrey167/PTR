//! The `HostShm` carrier through the public API: sealing, the consumer's checks and the
//! mapping. Linux only, with feature `shm-backend` (the Cargo manifest requires the feature).
#![cfg(all(feature = "shm-backend", target_os = "linux"))]

mod common;

use std::fs::File;
use std::io::Write;
use std::os::fd::OwnedFd;
use std::sync::{Mutex, MutexGuard};

use common::handle;
use ptr_fabric::{host_page_size, FabricError, Seal, ShmExpectation, ShmExport, ShmMapping};
use ptr_protocol::{content_digest, Carrier, CarrierLocator, MAX_HANDLE_BYTES};
use rustix::fs::{
    fcntl_add_seals, fcntl_get_seals, ftruncate, memfd_create, MemfdFlags, SealFlags,
};
use rustix::io::Errno;

static SERIAL: Mutex<()> = Mutex::new(());

/// The tests count this process's memfd mappings, so every test that maps runs alone.
fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn tensor(len: usize) -> Vec<u8> {
    (0..len).map(|index| (index % 253) as u8).collect()
}

fn expectation(bytes: &[u8]) -> ShmExpectation {
    ShmExpectation {
        byte_len: bytes.len() as u64,
        offset: 0,
        content_digest: content_digest(bytes),
    }
}

fn dup(export: &ShmExport) -> OwnedFd {
    export.fd().try_clone_to_owned().expect("dup")
}

/// A memfd holding `bytes` that carries exactly the seals in `seals`.
fn hand_built(bytes: &[u8], seals: &[SealFlags]) -> OwnedFd {
    let fd = memfd_create(
        "test-region",
        MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING,
    )
    .expect("memfd");
    let mut file = File::from(fd);
    file.write_all(bytes).expect("write");
    let fd = OwnedFd::from(file);
    for seal in seals {
        fcntl_add_seals(&fd, *seal).expect("seal");
    }
    fd
}

const ALL: [SealFlags; 4] = [
    SealFlags::WRITE,
    SealFlags::SHRINK,
    SealFlags::GROW,
    SealFlags::SEAL,
];

#[test]
fn roundtrip_has_exact_size_and_offset_zero() {
    let _serial = serial();
    let bytes = tensor(5000);
    let export = ShmExport::create(&bytes).expect("export");
    assert_eq!(export.byte_len(), 5000);
    assert_eq!(export.content_digest(), content_digest(&bytes));
    assert_eq!(
        export.locator(7),
        CarrierLocator::HostShm {
            fd_slot: 7,
            offset: 0,
        }
    );
    let handle = handle(export.locator(7), 5000, export.content_digest());
    assert_eq!(handle.carrier, Carrier::HostShm);
    let expected = ShmExpectation::from_handle(&handle).expect("expectation");
    assert_eq!(expected.offset, 0);
    let mapping = ShmMapping::map(export.into_fd(), &expected).expect("map");
    assert_eq!(mapping.as_bytes(), bytes.as_slice());
    assert_eq!(mapping.len(), 5000);
    assert!(!mapping.is_empty());
}

#[test]
fn the_export_carries_all_four_seals_and_the_exact_file_size() {
    let _serial = serial();
    let export = ShmExport::create(&tensor(100)).expect("export");
    let seals = fcntl_get_seals(export.fd()).expect("seals");
    for seal in ALL {
        assert!(seals.contains(seal), "missing {seal:?}");
    }
    let stat = rustix::fs::fstat(export.fd()).expect("fstat");
    assert_eq!(stat.st_size, 100);
}

#[test]
fn writing_growing_or_shrinking_after_export_is_refused() {
    let _serial = serial();
    let export = ShmExport::create(&tensor(100)).expect("export");
    let file = File::from(dup(&export));
    // The duplicate shares the open file description, which was opened read-write by
    // memfd_create, so the refusal below is the seal and not the access mode.
    let mut file = file;
    let written = file.write(b"x");
    assert_eq!(
        written.expect_err("write refused").raw_os_error(),
        Some(Errno::PERM.raw_os_error())
    );
    assert_eq!(ftruncate(export.fd(), 200), Err(Errno::PERM));
    assert_eq!(ftruncate(export.fd(), 10), Err(Errno::PERM));
    assert_eq!(
        fcntl_add_seals(export.fd(), SealFlags::FUTURE_WRITE),
        Err(Errno::PERM)
    );
}

#[test]
fn a_dup_of_the_descriptor_imports_the_same_bytes() {
    let _serial = serial();
    let bytes = tensor(4097);
    let export = ShmExport::create(&bytes).expect("export");
    let received = dup(&export);
    let mapping = ShmMapping::map(received, &expectation(&bytes)).expect("map");
    assert_eq!(mapping.as_bytes(), bytes.as_slice());
    // The producer's own descriptor still maps the same region.
    let again = ShmMapping::map(export.into_fd(), &expectation(&bytes)).expect("map");
    assert_eq!(again.as_bytes(), mapping.as_bytes());
}

#[test]
fn a_region_missing_any_one_seal_is_refused() {
    let _serial = serial();
    let bytes = tensor(64);
    let names = [Seal::Write, Seal::Shrink, Seal::Grow, Seal::Seal];
    for (missing, name) in names.iter().enumerate() {
        let present: Vec<SealFlags> = ALL
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != missing)
            .map(|(_, flag)| *flag)
            .collect();
        let fd = hand_built(&bytes, &present);
        assert_eq!(
            ShmMapping::map(fd, &expectation(&bytes)).unwrap_err(),
            FabricError::MissingSeal(*name),
        );
    }
    let unsealed = hand_built(&bytes, &[]);
    assert_eq!(
        ShmMapping::map(unsealed, &expectation(&bytes)).unwrap_err(),
        FabricError::MissingSeal(Seal::Write)
    );
}

#[test]
fn a_wrong_file_size_is_refused() {
    let _serial = serial();
    let bytes = tensor(64);
    let fd = hand_built(&bytes, &ALL);
    let mut bigger = expectation(&bytes);
    bigger.byte_len = 128;
    assert_eq!(
        ShmMapping::map(fd, &bigger).unwrap_err(),
        FabricError::LengthMismatch {
            expected: 128,
            found: 64,
        }
    );
    let fd = hand_built(&bytes, &ALL);
    let mut smaller = expectation(&bytes);
    smaller.byte_len = 32;
    assert_eq!(
        ShmMapping::map(fd, &smaller).unwrap_err(),
        FabricError::LengthMismatch {
            expected: 32,
            found: 64,
        }
    );
}

#[test]
fn a_nonzero_offset_is_refused() {
    let _serial = serial();
    let bytes = tensor(64);
    let export = ShmExport::create(&bytes).expect("export");
    let mut expected = expectation(&bytes);
    expected.offset = 8;
    assert_eq!(
        ShmMapping::map(export.into_fd(), &expected).unwrap_err(),
        FabricError::NonZeroOffset { offset: 8 }
    );
}

#[test]
fn a_handle_with_a_nonzero_locator_offset_is_refused_through_the_expectation() {
    let _serial = serial();
    let bytes = tensor(64);
    let export = ShmExport::create(&bytes).expect("export");
    let handle = handle(
        CarrierLocator::HostShm {
            fd_slot: 1,
            offset: 64,
        },
        64,
        export.content_digest(),
    );
    let expected = ShmExpectation::from_handle(&handle).expect("expectation");
    assert_eq!(
        ShmMapping::map(export.into_fd(), &expected).unwrap_err(),
        FabricError::NonZeroOffset { offset: 64 }
    );
}

#[test]
fn a_wrong_digest_is_refused_and_leaves_nothing_mapped() {
    let _serial = serial();
    let bytes = tensor(64);
    let export = ShmExport::create(&bytes).expect("export");
    let mut expected = expectation(&bytes);
    expected.content_digest[0] ^= 1;
    let before = mapped_regions();
    assert_eq!(
        ShmMapping::map(export.into_fd(), &expected).unwrap_err(),
        FabricError::DigestMismatch
    );
    assert_eq!(mapped_regions(), before);
}

/// The number of mappings of this process that name a memfd (from `/proc/self/maps`).
fn mapped_regions() -> usize {
    std::fs::read_to_string("/proc/self/maps")
        .expect("maps")
        .lines()
        .filter(|line| line.contains("memfd:"))
        .count()
}

#[test]
fn the_mapping_is_exactly_byte_len_with_no_byte_beyond() {
    let _serial = serial();
    // A tensor of one page and one byte: the mapping is two pages of address space, but the
    // slice handed out is exactly the tensor and the file behind it is exactly the tensor.
    let page = host_page_size() as usize;
    let bytes = tensor(page + 1);
    let export = ShmExport::create(&bytes).expect("export");
    let mapping = ShmMapping::map(export.into_fd(), &expectation(&bytes)).expect("map");
    assert_eq!(mapping.len(), page + 1);
    assert_eq!(mapping.as_bytes().len(), page + 1);
    assert_eq!(mapping.as_bytes().last(), bytes.last());
}

#[test]
fn an_empty_tensor_is_refused_on_both_sides() {
    let _serial = serial();
    assert_eq!(
        ShmExport::create(&[]).unwrap_err(),
        FabricError::EmptyTensor
    );
    let fd = hand_built(&[], &ALL);
    let expected = ShmExpectation {
        byte_len: 0,
        offset: 0,
        content_digest: content_digest(&[]),
    };
    assert_eq!(
        ShmMapping::map(fd, &expected).unwrap_err(),
        FabricError::EmptyTensor
    );
}

#[test]
fn a_handle_of_another_carrier_or_locator_gives_no_expectation() {
    let _serial = serial();
    let bytes = tensor(8);
    let inline = handle(
        CarrierLocator::Bytes {
            inline: bytes.clone(),
        },
        8,
        content_digest(&bytes),
    );
    assert_eq!(
        ShmExpectation::from_handle(&inline).unwrap_err(),
        FabricError::WrongCarrier
    );
    let mut mislabelled = inline;
    mislabelled.carrier = Carrier::HostShm;
    assert_eq!(
        ShmExpectation::from_handle(&mislabelled).unwrap_err(),
        FabricError::WrongLocator
    );
}

#[test]
fn the_footprint_is_rounded_up_to_whole_pages() {
    let _serial = serial();
    let page = host_page_size();
    let footprint = |len: u64| {
        ShmExport::create(&tensor(len as usize))
            .expect("export")
            .footprint_bytes()
    };
    assert_eq!(footprint(1), page);
    assert_eq!(footprint(page), page);
    assert_eq!(footprint(page + 1), 2 * page);
    assert_eq!(footprint(2 * page), 2 * page);
}

#[test]
fn mapping_and_dropping_in_a_loop_does_not_leak() {
    let _serial = serial();
    let bytes = tensor(8192);
    let export = ShmExport::create(&bytes).expect("export");
    let expected = expectation(&bytes);
    let before = mapped_regions();
    for _ in 0..3000 {
        let mapping = ShmMapping::map(dup(&export), &expected).expect("map");
        assert_eq!(mapping.len(), bytes.len());
    }
    assert_eq!(mapped_regions(), before);
}

#[test]
fn a_descriptor_that_is_not_a_memfd_is_refused_as_not_sealable() {
    let _serial = serial();
    let before = mapped_regions();
    let path = std::env::temp_dir().join(format!("ptr-fabric-not-memfd-{}", std::process::id()));
    let mut file = File::create(&path).expect("regular file");
    file.write_all(&tensor(64)).expect("write");
    let fd = OwnedFd::from(file);
    let bytes = tensor(64);
    assert_eq!(
        ShmMapping::map(fd, &expectation(&bytes)).unwrap_err(),
        FabricError::NotSealable
    );
    assert_eq!(mapped_regions(), before);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_length_above_the_handle_bound_is_refused_before_the_descriptor_is_touched() {
    let _serial = serial();
    let before = mapped_regions();
    let path = std::env::temp_dir().join(format!("ptr-fabric-too-large-{}", std::process::id()));
    let fd = OwnedFd::from(File::create(&path).expect("regular file"));
    let too_large = ShmExpectation {
        byte_len: MAX_HANDLE_BYTES + 1,
        offset: 0,
        content_digest: [0; 32],
    };
    // A regular file would be `NotSealable` if the seal query ran first; the size error shows
    // no `fcntl`, `fstat` or `mmap` was reached.
    assert_eq!(
        ShmMapping::map(fd.try_clone().expect("dup"), &too_large).unwrap_err(),
        FabricError::TooLarge {
            len: MAX_HANDLE_BYTES + 1,
            max: MAX_HANDLE_BYTES,
        }
    );
    let huge = ShmExpectation {
        byte_len: u64::MAX,
        ..too_large
    };
    assert!(matches!(
        ShmMapping::map(fd.try_clone().expect("dup"), &huge).unwrap_err(),
        FabricError::TooLarge { .. }
    ));
    // At the bound the descriptor is reached, so the next refusal is the seal query.
    let at_bound = ShmExpectation {
        byte_len: MAX_HANDLE_BYTES,
        ..too_large
    };
    assert_eq!(
        ShmMapping::map(fd, &at_bound).unwrap_err(),
        FabricError::NotSealable
    );
    assert_eq!(mapped_regions(), before);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn an_expectation_from_a_handle_above_the_bound_is_refused() {
    let locator = CarrierLocator::HostShm {
        fd_slot: 0,
        offset: 0,
    };
    let oversize = handle(locator, MAX_HANDLE_BYTES + 4, [0; 32]);
    assert_eq!(
        ShmExpectation::from_handle(&oversize).unwrap_err(),
        FabricError::TooLarge {
            len: MAX_HANDLE_BYTES + 4,
            max: MAX_HANDLE_BYTES,
        }
    );
}
