//! The crash-image synthesizer: what a file can look like after a crash.
//!
//! Two failure classes stay distinct. A **process abort** keeps the page cache, so every byte
//! the process wrote survives: its image is the file as it stood (policy [`Policy::All`], and
//! the real files the child processes leave). A **power loss** keeps the bytes the file had
//! synced (`durable`) and, of the bytes written after that, a seeded subset. The model assumes
//! an honest fsync, as the preregistration states: bytes synced before the crash are intact,
//! and nothing else is promised. It is not a real power cut and does not model hardware that
//! lies about a flush, filesystem-specific behaviour or a torn sector below the synced length.
//!
//! The honest policies below never touch a byte under `durable`, and [`check_in_model`] says
//! so: an image that differs from the file there, or that holds a byte that is neither the
//! written one nor a zero hole, is a defect of the harness (exit 2), not of the product.
//! [`Policy::Lie`] is the one policy outside the model, a control: it drops bytes the product
//! acknowledged, as a disk that lied about a flush would.

use super::super::rng::Rng;

/// The size of a hole: the unit in which an unsynced write is lost.
pub const SECTOR: usize = 512;

/// How the bytes written after the last sync are treated.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Policy {
    /// None of them survive: the file is its synced prefix.
    None,
    /// All of them survive (a process abort, or a flush that happened to complete).
    All,
    /// A seeded prefix of them survives.
    Prefix,
    /// The first half of them survives.
    Half,
    /// All of them survive in length but seeded 512-byte sectors read back as zeros.
    Holes,
}

impl Policy {
    pub const HONEST: [Policy; 5] = [
        Policy::None,
        Policy::All,
        Policy::Prefix,
        Policy::Half,
        Policy::Holes,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Policy::None => "none",
            Policy::All => "all",
            Policy::Prefix => "prefix",
            Policy::Half => "half",
            Policy::Holes => "holes",
        }
    }
}

/// One synthesized file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Image {
    pub bytes: Vec<u8>,
    pub policy: Policy,
    /// Some byte after the synced length was lost: the image is shorter than what was written
    /// or holds a hole.
    pub lost_bytes: bool,
    /// Some byte of the image differs from the one written (a hole).
    pub damaged: bool,
}

/// The file after a power loss: `written` is the file as the process left it, `durable` the
/// length it had synced.
pub fn power_loss(written: &[u8], durable: usize, policy: Policy, rng: &mut Rng) -> Image {
    let durable = durable.min(written.len());
    let unsynced = written.len() - durable;
    let keep = match policy {
        Policy::None => 0,
        Policy::All | Policy::Holes => unsynced,
        Policy::Prefix if unsynced == 0 => 0,
        Policy::Prefix => rng.range(1, unsynced as u64) as usize,
        Policy::Half => unsynced / 2,
    };
    let mut bytes = written[..durable + keep].to_vec();
    if policy == Policy::Holes && unsynced > 0 {
        let first = durable / SECTOR;
        let last = (written.len() - 1) / SECTOR;
        let mut zeroed = false;
        for sector in first..=last {
            let take = rng.chance(0.5) || (sector == last && !zeroed);
            if take {
                let from = (sector * SECTOR).max(durable);
                let to = ((sector + 1) * SECTOR).min(bytes.len());
                bytes[from..to].iter_mut().for_each(|byte| *byte = 0);
                zeroed = true;
            }
        }
    }
    let damaged = bytes.iter().zip(written).any(|(a, b)| a != b);
    Image {
        lost_bytes: bytes.len() < written.len() || damaged,
        bytes,
        policy,
        damaged,
    }
}

/// The offset of the first byte of `image` that differs from `written`, if any.
#[cfg(test)]
pub fn first_difference(image: &Image, written: &[u8]) -> Option<usize> {
    image.bytes.iter().zip(written).position(|(a, b)| a != b)
}

/// The shape of the damage an image holds in the part written after the synced length (the
/// "tail", `[durable, end)` of the image).
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Shape {
    /// A: every byte of the tail reads zero. The file was extended and none of the new data
    /// reached the disk (the usual state after a power loss with delayed allocation). No
    /// written byte survives, so trimming the tail loses nothing that exists.
    Zeros,
    /// B: a run of zeros at the start of the tail, then the written bytes intact.
    GapThenIntact,
    /// C: the written bytes intact, then zeros to the end.
    IntactThenZeros,
    /// D: any other mix (a hole inside, or at both ends).
    Mixed,
}

impl Shape {
    /// The suffix of the counters of this shape.
    pub fn tag(self) -> &'static str {
        match self {
            Shape::Zeros => "zeros",
            Shape::GapThenIntact => "gap_then_intact",
            Shape::IntactThenZeros => "intact_then_zeros",
            Shape::Mixed => "mixed",
        }
    }
}

/// The shape of the damage of `image` after `durable`, or `None` for an undamaged image.
pub fn damage_shape(image: &Image, written: &[u8], durable: usize) -> Option<Shape> {
    if !image.damaged {
        return None;
    }
    let durable = durable.min(image.bytes.len());
    let tail = &image.bytes[durable..];
    let was = &written[durable..];
    let first = tail.iter().zip(was).position(|(a, b)| a != b)?;
    let last = tail.iter().zip(was).rposition(|(a, b)| a != b)?;
    let zero = |bytes: &[u8]| bytes.iter().all(|byte| *byte == 0);
    Some(if zero(tail) {
        Shape::Zeros
    } else if first == 0 && last + 1 < tail.len() && zero(&tail[..=last]) {
        Shape::GapThenIntact
    } else if first > 0 && last + 1 == tail.len() && zero(&tail[first..]) {
        Shape::IntactThenZeros
    } else {
        Shape::Mixed
    })
}

/// Whether `image` is a file the honest model allows: nothing under the synced length changed,
/// it is no longer than what was written, and every byte after the synced length is the
/// written one or a zero.
pub fn check_in_model(image: &Image, written: &[u8], durable: usize) -> Result<(), String> {
    let durable = durable.min(written.len());
    if image.bytes.len() < durable || image.bytes.len() > written.len() {
        return Err(format!(
            "the {} image has {} bytes, outside {durable}..={}",
            image.policy.name(),
            image.bytes.len(),
            written.len()
        ));
    }
    if image.bytes[..durable] != written[..durable] {
        return Err(format!(
            "the {} image changed a byte under the synced length {durable}",
            image.policy.name()
        ));
    }
    let strange = image.bytes[durable..]
        .iter()
        .zip(&written[durable..])
        .any(|(got, was)| got != was && *got != 0);
    if strange {
        return Err(format!(
            "the {} image holds a byte that is neither written nor zero",
            image.policy.name()
        ));
    }
    Ok(())
}

/// The control outside the model: the file with its tail cut to `keep` bytes, as if the disk
/// had acknowledged a flush it did not perform. `keep` lies below the acknowledged length.
pub fn fsync_lie(written: &[u8], keep: usize) -> Vec<u8> {
    written[..keep.min(written.len())].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 251) as u8 + 1).collect()
    }

    #[test]
    fn honest_images_keep_the_synced_prefix_and_stay_in_the_model() {
        let written = pattern(3000);
        let mut rng = Rng::new(7);
        for durable in [0, 1, 511, 512, 1500, 2999, 3000] {
            for policy in Policy::HONEST {
                for _ in 0..20 {
                    let image = power_loss(&written, durable, policy, &mut rng);
                    assert_eq!(check_in_model(&image, &written, durable), Ok(()));
                    assert_eq!(&image.bytes[..durable], &written[..durable]);
                }
            }
        }
    }

    #[test]
    fn the_policies_differ_where_they_should() {
        let written = pattern(2048);
        let mut rng = Rng::new(3);
        let none = power_loss(&written, 700, Policy::None, &mut rng);
        let all = power_loss(&written, 700, Policy::All, &mut rng);
        let half = power_loss(&written, 700, Policy::Half, &mut rng);
        let holes = power_loss(&written, 700, Policy::Holes, &mut rng);
        assert_eq!(none.bytes.len(), 700);
        assert!(none.lost_bytes && !none.damaged);
        assert_eq!(all.bytes, written);
        assert!(!all.lost_bytes && !all.damaged);
        assert_eq!(half.bytes.len(), 700 + (2048 - 700) / 2);
        assert_eq!(holes.bytes.len(), 2048);
        assert!(holes.damaged && holes.lost_bytes);
        // Everything is synced: no policy can lose anything.
        for policy in Policy::HONEST {
            let image = power_loss(&written, written.len(), policy, &mut rng);
            assert_eq!(image.bytes, written);
            assert!(!image.lost_bytes);
        }
    }

    #[test]
    fn a_hole_lies_in_the_unsynced_part_and_nowhere_else() {
        let written = pattern(4000);
        let mut rng = Rng::new(21);
        for durable in [0, 300, 1024, 2999] {
            let holes = power_loss(&written, durable, Policy::Holes, &mut rng);
            let first = first_difference(&holes, &written).expect("a hole changes a byte");
            assert!(
                first >= durable,
                "{first} under the synced length {durable}"
            );
        }
        let all = power_loss(&written, 100, Policy::All, &mut rng);
        assert_eq!(first_difference(&all, &written), None);
    }

    #[test]
    fn damage_is_classified_by_the_shape_of_the_zeros() {
        let written = vec![7u8; 1000];
        let shape = |from: usize, to: usize| {
            let mut bytes = written.clone();
            bytes[from..to].iter_mut().for_each(|b| *b = 0);
            let image = Image {
                bytes,
                policy: Policy::Holes,
                lost_bytes: true,
                damaged: true,
            };
            damage_shape(&image, &written, 300)
        };
        assert_eq!(shape(300, 1000), Some(Shape::Zeros));
        assert_eq!(shape(300, 600), Some(Shape::GapThenIntact));
        assert_eq!(shape(500, 1000), Some(Shape::IntactThenZeros));
        assert_eq!(shape(500, 600), Some(Shape::Mixed));
        let clean = power_loss(&written, 300, Policy::All, &mut Rng::new(1));
        assert_eq!(damage_shape(&clean, &written, 300), None);
    }

    #[test]
    fn an_image_outside_the_model_is_refused() {
        let written = pattern(1000);
        let durable = 400;
        let mut rng = Rng::new(1);
        let honest = power_loss(&written, durable, Policy::Prefix, &mut rng);
        let mut eaten = honest.clone();
        eaten.bytes[10] ^= 0xff;
        assert!(check_in_model(&eaten, &written, durable).is_err());
        let mut strange = honest.clone();
        strange.bytes.resize(900, 0x5a);
        assert!(check_in_model(&strange, &written, durable).is_err());
        let mut short = honest;
        short.bytes.truncate(399);
        assert!(check_in_model(&short, &written, durable).is_err());
        let lie = Image {
            bytes: fsync_lie(&written, 100),
            policy: Policy::None,
            lost_bytes: true,
            damaged: false,
        };
        assert!(
            check_in_model(&lie, &written, durable).is_err(),
            "a lie drops synced bytes, which the honest model forbids"
        );
    }

    #[test]
    fn the_same_stream_gives_the_same_images() {
        let written = pattern(2500);
        let make = |seed| {
            let mut rng = Rng::new(seed);
            Policy::HONEST
                .iter()
                .map(|p| power_loss(&written, 600, *p, &mut rng).bytes)
                .collect::<Vec<_>>()
        };
        assert_eq!(make(5), make(5));
        assert_ne!(make(5), make(6));
    }
}
