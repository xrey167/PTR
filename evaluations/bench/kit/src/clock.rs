//! UTC timestamps in the two formats the Python runner writes, without a time
//! library: `utc_stamp()` is `%Y%m%dT%H%M%S.%fZ` (file names, `started_at`) and
//! `isoformat()` is Python's `datetime.isoformat()` with a `+00:00` offset
//! (`finished_at`).

use std::time::{SystemTime, UNIX_EPOCH};

/// A UTC instant with microsecond resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Utc {
    /// Calendar year.
    pub year: i64,
    /// Month 1-12.
    pub month: u32,
    /// Day of month 1-31.
    pub day: u32,
    /// Hour 0-23.
    pub hour: u32,
    /// Minute 0-59.
    pub minute: u32,
    /// Second 0-59.
    pub second: u32,
    /// Microsecond 0-999999.
    pub micro: u32,
}

/// Days since 1970-01-01 to (year, month, day), proleptic Gregorian
/// (H. Hinnant's `civil_from_days`).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

impl Utc {
    /// Convert microseconds since the Unix epoch.
    pub fn from_unix_micros(micros: i64) -> Self {
        let secs = micros.div_euclid(1_000_000);
        let micro = micros.rem_euclid(1_000_000) as u32;
        let days = secs.div_euclid(86_400);
        let rem = secs.rem_euclid(86_400);
        let (year, month, day) = civil_from_days(days);
        Self {
            year,
            month,
            day,
            hour: (rem / 3600) as u32,
            minute: (rem % 3600 / 60) as u32,
            second: (rem % 60) as u32,
            micro,
        }
    }

    /// The current time (a clock before 1970 reads as the epoch).
    pub fn now() -> Self {
        let micros = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| i64::try_from(d.as_micros()).unwrap_or(i64::MAX))
            .unwrap_or(0);
        Self::from_unix_micros(micros)
    }

    /// `20261009T120000.123456Z`, the runner's `utc_stamp()`.
    pub fn utc_stamp(&self) -> String {
        format!(
            "{:04}{:02}{:02}T{:02}{:02}{:02}.{:06}Z",
            self.year, self.month, self.day, self.hour, self.minute, self.second, self.micro
        )
    }

    /// `2026-10-09T12:00:00.123456+00:00`, Python's `isoformat()` (the
    /// fractional part is omitted when it is zero, as Python does).
    pub fn isoformat(&self) -> String {
        let fraction = if self.micro == 0 {
            String::new()
        } else {
            format!(".{:06}", self.micro)
        };
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}{}+00:00",
            self.year, self.month, self.day, self.hour, self.minute, self.second, fraction
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_epoch() {
        let t = Utc::from_unix_micros(0);
        assert_eq!(t.utc_stamp(), "19700101T000000.000000Z");
        assert_eq!(t.isoformat(), "1970-01-01T00:00:00+00:00");
    }

    #[test]
    fn a_known_instant() {
        // `date -u -d @1700000000` = 2023-11-14 22:13:20 UTC.
        let t = Utc::from_unix_micros(1_700_000_000_123_456);
        assert_eq!(t.utc_stamp(), "20231114T221320.123456Z");
        assert_eq!(t.isoformat(), "2023-11-14T22:13:20.123456+00:00");
    }

    #[test]
    fn leap_day_and_year_boundaries() {
        // 2000-02-29 00:00:00 UTC = 951782400.
        assert_eq!(
            Utc::from_unix_micros(951_782_400_000_000).isoformat(),
            "2000-02-29T00:00:00+00:00"
        );
        // 2100 is not a leap year: 2100-03-01 00:00:00 UTC = 4107542400.
        assert_eq!(
            Utc::from_unix_micros(4_107_542_400_000_000).isoformat(),
            "2100-03-01T00:00:00+00:00"
        );
        // 1999-12-31 23:59:59 UTC = 946684799.
        assert_eq!(
            Utc::from_unix_micros(946_684_799_999_999).utc_stamp(),
            "19991231T235959.999999Z"
        );
    }

    #[test]
    fn before_the_epoch_uses_floor_semantics() {
        let t = Utc::from_unix_micros(-1);
        assert_eq!(t.utc_stamp(), "19691231T235959.999999Z");
    }

    #[test]
    fn now_is_after_2026_and_stamps_sort_chronologically() {
        let now = Utc::now();
        assert!(now.year >= 2026, "{now:?}");
        let a = Utc::from_unix_micros(1_700_000_000_000_000).utc_stamp();
        let b = Utc::from_unix_micros(1_700_000_000_000_001).utc_stamp();
        assert!(a < b);
    }
}
