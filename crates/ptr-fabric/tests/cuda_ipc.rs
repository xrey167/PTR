//! The CUDA foundations through the public API. Linux only, feature `cuda-ipc`.
//!
//! The CPU tests run everywhere (also on a host without a driver, where they check the typed
//! "no driver" path). The GPU tests are `#[ignore]`d: they need an NVIDIA GPU and driver and are
//! run on the owner's PC per docs/GPU_HANDOVER.md. Nothing in them has run in this repository's
//! CI or on the host that wrote them.
#![cfg(all(feature = "cuda-ipc-core", target_os = "linux"))]

use ptr_fabric::{
    audit_export_padding, check_import, footprint_legacy, footprint_memfd, footprint_vmm,
    host_page_size, import_and_copy, CudaDriver, CudaError, CudaProbe, ExportRecord,
    ImportExpectation, PinnedRegistration, ShmExpectation, ShmExport, ShmMapping,
    StagingAllocation, LEGACY_STAGING_UNIT,
};
use ptr_protocol::content_digest;

const GPU_ONLY: &str =
    "needs an NVIDIA GPU and driver; run on the owner's PC per docs/GPU_HANDOVER.md";

// ---- CPU tests -------------------------------------------------------------------------------

#[test]
fn probe_is_absent_without_a_driver_and_never_panics() {
    let probe = CudaDriver::probe();
    assert!(probe.is_consistent(), "{probe:?}");
    match CudaDriver::open() {
        Err(CudaError::NoDriver { reason }) => {
            assert!(!reason.is_empty());
            let CudaProbe::Absent { .. } = probe else {
                panic!("open found no driver but the probe is {probe:?}");
            };
        }
        Ok(_) => {
            // A host with a driver: the report must describe devices, each with a granularity
            // only when it can export a VMM allocation.
            if let CudaProbe::Present { devices, .. } = &probe {
                for device in devices {
                    assert!(!device.name.is_empty());
                    if device.allocation_granularity.is_some() {
                        assert!(device.vmm_supported && device.posix_fd_supported);
                    }
                }
            }
        }
        Err(other) => panic!("open failed with an unexpected error: {other}"),
    }
}

#[test]
fn probing_twice_gives_the_same_answer() {
    assert_eq!(CudaDriver::probe(), CudaDriver::probe());
}

#[test]
fn footprints_charge_the_allocation_not_the_length() {
    assert_eq!(footprint_vmm(1000, 2 << 20), Some(2 << 20));
    assert_eq!(footprint_legacy(1000), Some(LEGACY_STAGING_UNIT));
    assert_eq!(
        footprint_memfd(1000, host_page_size()),
        Some(host_page_size())
    );
    assert_eq!(footprint_vmm(0, 2 << 20), None);
    assert_eq!(footprint_vmm(u64::MAX, 2 << 20), None);
}

fn record() -> ExportRecord {
    ExportRecord {
        producer_session: 1,
        consumer_session: 2,
        operation: 3,
        device: 0,
        size: 2 << 20,
    }
}

fn expectation() -> ImportExpectation {
    ImportExpectation {
        byte_len: 4096,
        content_digest: content_digest(&[7u8; 4096]),
        consumer_session: 2,
        operation: 3,
        source_device: 0,
        consumer_device: 0,
        locator_alloc_size: 2 << 20,
    }
}

#[test]
fn the_record_is_checked_before_any_driver_call() {
    assert_eq!(check_import(&record(), &expectation()), Ok(()));
    let mut forwarded = expectation();
    forwarded.consumer_session = 99;
    assert_eq!(
        check_import(&record(), &forwarded),
        Err(CudaError::RecordMismatch {
            field: "consumer_session"
        })
    );
    let mut too_big = expectation();
    too_big.byte_len = (2 << 20) + 1;
    assert!(matches!(
        check_import(&record(), &too_big),
        Err(CudaError::ExtentMismatch { .. })
    ));
}

#[test]
fn open_without_a_driver_is_a_typed_no_driver_and_matches_the_probe() {
    // Every GPU function takes a `CudaDriver`, which only `open` hands out, so on a host without
    // a driver the whole path ends here before any cudarc call.
    match CudaDriver::open() {
        Err(error) => {
            assert!(matches!(error, CudaError::NoDriver { .. }));
            assert!(!error.to_string().is_empty());
            assert!(!CudaDriver::probe().is_present());
        }
        Ok(_) => assert!(CudaDriver::probe().is_consistent()),
    }
}

// ---- GPU tests -------------------------------------------------------------------------------

fn gpu() -> CudaDriver {
    CudaDriver::open().unwrap_or_else(|error| panic!("{GPU_ONLY}: {error}"))
}

fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|index| (index % 251) as u8 ^ 0x5a).collect()
}

/// A filled allocation of `len` bytes on device 0 and one export of it for session 2, operation 3.
fn exported(
    driver: &CudaDriver,
    len: usize,
) -> (StagingAllocation, ptr_fabric::StagingExport, Vec<u8>) {
    let bytes = pattern(len);
    let mut allocation = StagingAllocation::create(driver, 0, len as u64).expect("create");
    allocation.fill(&bytes).expect("fill");
    let record = allocation.record_for(1, 2, 3);
    let export = allocation.export(record).expect("export");
    (allocation, export, bytes)
}

fn expectation_for(allocation: &StagingAllocation, bytes: &[u8]) -> ImportExpectation {
    ImportExpectation {
        byte_len: bytes.len() as u64,
        content_digest: content_digest(bytes),
        consumer_session: 2,
        operation: 3,
        source_device: allocation.device(),
        consumer_device: allocation.device(),
        locator_alloc_size: allocation.footprint(),
    }
}

#[test]
#[ignore = "needs an NVIDIA GPU and driver; run on the owner's PC per docs/GPU_HANDOVER.md"]
fn a_vmm_export_round_trips_in_one_process() {
    let driver = gpu();
    let (allocation, export, bytes) = exported(&driver, 100_000);
    assert_eq!(export.recorded_size, allocation.footprint());
    assert!(allocation.footprint() >= 100_000);
    let expected = expectation_for(&allocation, &bytes);
    let copy = import_and_copy(&driver, export.fd, &export.record, &expected).expect("import");
    assert_eq!(copy, bytes);
}

#[test]
#[ignore = "needs an NVIDIA GPU and driver; run on the owner's PC per docs/GPU_HANDOVER.md"]
fn a_wrong_digest_or_a_forwarded_record_is_refused() {
    let driver = gpu();
    let (allocation, export, bytes) = exported(&driver, 4096);
    let mut wrong_digest = expectation_for(&allocation, &bytes);
    wrong_digest.content_digest = [0; 32];
    assert_eq!(
        import_and_copy(&driver, export.fd, &export.record, &wrong_digest),
        Err(CudaError::DigestMismatch)
    );

    let (allocation, export, bytes) = exported(&driver, 4096);
    let mut forwarded = expectation_for(&allocation, &bytes);
    forwarded.consumer_session = 77;
    assert_eq!(
        import_and_copy(&driver, export.fd, &export.record, &forwarded),
        Err(CudaError::RecordMismatch {
            field: "consumer_session"
        })
    );
}

#[test]
#[ignore = "needs an NVIDIA GPU and driver; run on the owner's PC per docs/GPU_HANDOVER.md"]
fn the_consumer_sees_only_zeros_beyond_the_tensor() {
    let driver = gpu();
    // Not a multiple of the granularity, so padding exists.
    let (allocation, export, bytes) = exported(&driver, 1000);
    assert!(allocation.footprint() > 1000);
    let expected = expectation_for(&allocation, &bytes);
    let audit = audit_export_padding(&driver, export.fd, &export.record, &expected).expect("audit");
    assert_eq!(audit.padding_len, allocation.footprint() - 1000);
    assert!(audit.all_zero, "the staging padding held non-zero bytes");
}

#[test]
#[ignore = "needs an NVIDIA GPU and driver; run on the owner's PC per docs/GPU_HANDOVER.md"]
fn every_export_is_a_fresh_allocation_and_a_filled_one_cannot_be_refilled() {
    let driver = gpu();
    let (mut allocation, _export, bytes) = exported(&driver, 4096);
    assert!(
        allocation.fill(&bytes).is_err(),
        "a second fill must be refused"
    );
    let second = StagingAllocation::create(&driver, 0, 4096).expect("create");
    assert!(second.content_digest().is_none());
    // An export before the fill is refused.
    assert!(second.export(second.record_for(1, 2, 3)).is_err());
}

#[test]
#[ignore = "needs an NVIDIA GPU and driver; run on the owner's PC per docs/GPU_HANDOVER.md"]
fn the_peer_matrix_matches_the_driver() {
    let driver = gpu();
    let CudaProbe::Present { devices, peers } = CudaDriver::probe() else {
        panic!("{GPU_ONLY}: the probe is absent");
    };
    assert!(!devices.is_empty());
    for link in &peers {
        let mut from = 0;
        let mut to = 0;
        let mut can = 0;
        // SAFETY: `gpu()` proved the driver is loaded; all out pointers are valid locals and
        // the ordinals come from the probe.
        unsafe {
            use cudarc::driver::sys;
            assert_eq!(
                sys::cuDeviceGet(&mut from, link.from as i32),
                sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(
                sys::cuDeviceGet(&mut to, link.to as i32),
                sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(
                sys::cuDeviceCanAccessPeer(&mut can, from, to),
                sys::CUresult::CUDA_SUCCESS
            );
        }
        assert_eq!(link.can_access, can != 0, "{link:?}");
    }
    // `cudaDeviceCanAccessPeer` (runtime API) is the owner's cross-check, see docs/GPU_HANDOVER.md.
    let _ = driver;
}

#[test]
#[ignore = "needs an NVIDIA GPU and driver; run on the owner's PC per docs/GPU_HANDOVER.md"]
fn a_sealed_host_shm_region_registers_and_unregisters() {
    let driver = gpu();
    let bytes = pattern(10_000);
    let export = ShmExport::create(&bytes).expect("export");
    let expectation = ShmExpectation {
        byte_len: export.byte_len(),
        offset: 0,
        content_digest: export.content_digest(),
    };
    let mapping = ShmMapping::map(export.into_fd(), &expectation).expect("map");
    {
        let registration = PinnedRegistration::register(&driver, 0, &mapping).expect("register");
        assert_eq!(registration.len(), bytes.len());
        assert_eq!(
            registration.footprint_bytes(),
            footprint_memfd(10_000, host_page_size())
        );
    }
    // After the drop the range is unregistered, so it registers again.
    let again = PinnedRegistration::register(&driver, 0, &mapping).expect("register again");
    drop(again);
    assert_eq!(mapping.as_bytes(), bytes.as_slice());
}
