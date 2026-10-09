//! Resident-set size from `/proc/self/status` (Linux). On other platforms every
//! reader returns `None`, and a benchmark must then record the metric as absent
//! rather than zero.

/// Parse the value of `key` (for example `VmHWM`) in `/proc/<pid>/status` text,
/// in KiB. `None` when the key is missing or its value is not `<number> kB`.
pub fn parse_status_kib(status: &str, key: &str) -> Option<u64> {
    status.lines().find_map(|line| {
        let (name, rest) = line.split_once(':')?;
        if name != key {
            return None;
        }
        let mut parts = rest.split_whitespace();
        let value = parts.next()?.parse::<u64>().ok()?;
        (parts.next()? == "kB").then_some(value)
    })
}

fn read_self(key: &str) -> Option<u64> {
    parse_status_kib(&std::fs::read_to_string("/proc/self/status").ok()?, key)
}

/// Peak resident set size of this process (`VmHWM`) in KiB.
pub fn peak_rss_kib() -> Option<u64> {
    read_self("VmHWM")
}

/// Current resident set size of this process (`VmRSS`) in KiB.
pub fn current_rss_kib() -> Option<u64> {
    read_self("VmRSS")
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str =
        "Name:\tcat\nVmPeak:\t  123456 kB\nVmHWM:\t   2048 kB\nVmRSS:\t   1024 kB\nThreads:\t1\n";

    #[test]
    fn parses_known_keys() {
        assert_eq!(parse_status_kib(SAMPLE, "VmHWM"), Some(2048));
        assert_eq!(parse_status_kib(SAMPLE, "VmRSS"), Some(1024));
        assert_eq!(parse_status_kib(SAMPLE, "VmPeak"), Some(123_456));
    }

    #[test]
    fn rejects_missing_keys_prefix_matches_and_bad_units() {
        assert_eq!(parse_status_kib(SAMPLE, "VmSwap"), None);
        assert_eq!(parse_status_kib(SAMPLE, "Vm"), None);
        assert_eq!(parse_status_kib(SAMPLE, "Threads"), None); // no kB unit
        assert_eq!(parse_status_kib("VmHWM:\tabc kB\n", "VmHWM"), None);
        assert_eq!(parse_status_kib("VmHWM:\t12 MB\n", "VmHWM"), None);
        assert_eq!(parse_status_kib("", "VmHWM"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn this_process_has_a_peak_at_least_its_current_size() {
        let current = current_rss_kib().expect("VmRSS on linux");
        let peak = peak_rss_kib().expect("VmHWM on linux");
        assert!(current > 0);
        assert!(peak >= current, "peak {peak} < current {current}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn peak_rss_grows_after_touching_memory() {
        let before = peak_rss_kib().unwrap();
        let mut block = vec![0u8; 64 * 1024 * 1024];
        for i in (0..block.len()).step_by(4096) {
            block[i] = 1;
        }
        let after = peak_rss_kib().unwrap();
        assert!(block.iter().step_by(4096).all(|&b| b == 1));
        assert!(after >= before + 32 * 1024, "before {before} after {after}");
    }
}
