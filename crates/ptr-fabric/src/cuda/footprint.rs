//! The allocated footprint of an export, per doc 42 rule 3b: the quota is charged with what the
//! memory really costs, padding included, not with the logical `byte_len`. Pure arithmetic,
//! checked: `None` on overflow, on a zero length and on a zero unit.

/// The legacy `cudaMalloc` staging buffer is rounded up to a multiple of 2 MiB (NVIDIA's IPC
/// guide), so a sub-allocation by the driver cannot put other data into the shared block.
pub const LEGACY_STAGING_UNIT: u64 = 2 * 1024 * 1024;

fn round_up(len: u64, unit: u64) -> Option<u64> {
    if len == 0 || unit == 0 {
        return None;
    }
    len.div_ceil(unit).checked_mul(unit)
}

/// The footprint of a VMM (`cuMemCreate`) staging allocation: `len` rounded up to the allocation
/// `granularity` of the device.
pub fn footprint_vmm(len: u64, granularity: u64) -> Option<u64> {
    round_up(len, granularity)
}

/// The footprint of a legacy `cudaMalloc` staging buffer: `len` rounded up to 2 MiB.
pub fn footprint_legacy(len: u64) -> Option<u64> {
    round_up(len, LEGACY_STAGING_UNIT)
}

/// The footprint of a `memfd` (the pinned buffer of `PinnedHost` is one): `len` rounded up to
/// whole pages of `page` bytes.
pub fn footprint_memfd(len: u64, page: u64) -> Option<u64> {
    round_up(len, page)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vmm_rounds_up_to_the_granularity() {
        assert_eq!(footprint_vmm(1, 2 << 20), Some(2 << 20));
        assert_eq!(footprint_vmm(2 << 20, 2 << 20), Some(2 << 20));
        assert_eq!(footprint_vmm((2 << 20) + 1, 2 << 20), Some(4 << 20));
        assert_eq!(footprint_vmm(100, 64 << 10), Some(64 << 10));
    }

    #[test]
    fn legacy_rounds_up_to_two_mebibytes() {
        assert_eq!(footprint_legacy(1), Some(LEGACY_STAGING_UNIT));
        assert_eq!(
            footprint_legacy(LEGACY_STAGING_UNIT),
            Some(LEGACY_STAGING_UNIT)
        );
        assert_eq!(
            footprint_legacy(LEGACY_STAGING_UNIT + 1),
            Some(2 * LEGACY_STAGING_UNIT)
        );
    }

    #[test]
    fn memfd_rounds_up_to_whole_pages() {
        assert_eq!(footprint_memfd(1, 4096), Some(4096));
        assert_eq!(footprint_memfd(4096, 4096), Some(4096));
        assert_eq!(footprint_memfd(4097, 4096), Some(8192));
        assert_eq!(footprint_memfd(1, 65536), Some(65536));
    }

    #[test]
    fn a_zero_length_or_zero_unit_is_refused() {
        assert_eq!(footprint_vmm(0, 4096), None);
        assert_eq!(footprint_vmm(1, 0), None);
        assert_eq!(footprint_legacy(0), None);
        assert_eq!(footprint_memfd(0, 4096), None);
        assert_eq!(footprint_memfd(1, 0), None);
    }

    #[test]
    fn overflow_is_none_not_a_wrapped_value() {
        assert_eq!(footprint_vmm(u64::MAX, 4096), None);
        assert_eq!(footprint_legacy(u64::MAX), None);
        assert_eq!(footprint_memfd(u64::MAX - 1, 4096), None);
        // The largest length that still fits exactly.
        assert_eq!(
            footprint_memfd(u64::MAX - 4095, 4096),
            Some(u64::MAX - 4095)
        );
    }

    #[test]
    fn the_footprint_is_never_below_the_length() {
        for len in [1u64, 4095, 4096, 4097, 1 << 20, (3 << 20) + 7] {
            for unit in [512u64, 4096, 65536, 2 << 20] {
                let footprint = footprint_vmm(len, unit).expect("fits");
                assert!(
                    footprint >= len && footprint.is_multiple_of(unit) && footprint - len < unit
                );
            }
        }
    }
}
