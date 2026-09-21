//! The generated codebook artifact must still describe this kernel.
//!
//! `datasets/generated/codebook.json` is what everything outside Rust reads. If
//! the tables here change and that file does not, the disagreement is silent:
//! dataset validation keeps accepting the old member set and a checkpoint keeps
//! loading under an assignment that has moved. This test is where that fails.
//!
//! It parses the fields it needs by hand rather than pulling a JSON crate in,
//! because `ptr-types` has no dependencies and this is a test of a flat document.

use ptr_types::{exception_width, CodeFamily, Codebook, CodebookVersion, EXCEPTIONS};

const ARTIFACT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../datasets/generated/codebook.json"
);

/// The string value of a `"key": "value"` pair, which is all this document needs.
/// Lowercase hex without building a string per byte.
fn to_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn string_field(document: &str, key: &str) -> String {
    let needle = format!("\"{key}\":");
    let start = document
        .find(&needle)
        .unwrap_or_else(|| panic!("{key} is absent from the artifact"))
        + needle.len();
    let rest = &document[start..];
    let open = rest.find('"').expect("a quoted value") + 1;
    let end = rest[open..].find('"').expect("a closing quote");
    rest[open..open + end].to_owned()
}

fn number_field(document: &str, key: &str) -> u64 {
    let needle = format!("\"{key}\":");
    let start = document
        .find(&needle)
        .unwrap_or_else(|| panic!("{key} is absent from the artifact"))
        + needle.len();
    document[start..]
        .trim_start()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .expect("a numeric value")
}

#[test]
fn the_generated_artifact_carries_this_kernel_s_canonical_bytes() {
    let document = std::fs::read_to_string(ARTIFACT)
        .expect("datasets/generated/codebook.json is missing; run scripts/generate_codebook.py");
    let book = Codebook::at(CodebookVersion::V1).expect("V1 is assigned");

    let expected = to_hex(&book.canonical_bytes());
    assert_eq!(
        string_field(&document, "canonical_bytes_hex"),
        expected,
        "the artifact describes a different codebook; run scripts/generate_codebook.py"
    );
    assert_eq!(
        number_field(&document, "version"),
        CodebookVersion::V1.0 as u64
    );
}

#[test]
fn the_artifact_lists_every_family_with_its_cardinality() {
    let document = std::fs::read_to_string(ARTIFACT).expect("the artifact exists");
    let book = Codebook::at(CodebookVersion::V1).expect("V1 is assigned");

    for family in CodeFamily::ALL {
        // Each family appears with its own name, and every member name of that
        // family appears somewhere in the document. A family the kernel gained
        // and the artifact lacks fails here rather than in a training run.
        assert!(
            document.contains(&format!("\"family\": \"{}\"", family.name())),
            "{} is missing from the artifact",
            family.name()
        );
        for name in book.member_names(family) {
            assert!(
                document.contains(&format!("\"name\": \"{name}\"")),
                "{name} is missing from the artifact"
            );
        }
    }

    let members: usize = CodeFamily::ALL
        .iter()
        .map(|family| book.cardinality(*family) as usize)
        .sum();
    assert_eq!(
        document.matches("\"code\":").count(),
        members,
        "the artifact holds a different number of members than the kernel assigns"
    );
}

#[test]
fn the_artifact_carries_every_recorded_exception_with_its_width() {
    let document = std::fs::read_to_string(ARTIFACT).expect("the artifact exists");

    // Kernel -> artifact. An exception added in Rust and not regenerated here is
    // a width nothing outside Rust can see, which is the state this section was
    // added to end.
    for exception in EXCEPTIONS {
        assert!(
            document.contains(&format!("\"name\": \"{}\"", exception.name)),
            "{} is missing from the artifact; run scripts/generate_codebook.py",
            exception.name
        );
        assert!(
            document.contains(&format!("\"width\": {}", exception.width)),
            "{} is recorded at another width than the kernel's {}",
            exception.name,
            exception.width
        );
        assert!(
            !exception.reason.trim().is_empty(),
            "{} records no reason, so it reads as an oversight",
            exception.name
        );
    }

    // Artifact -> kernel. Counting is what catches an exception the artifact
    // carries and the kernel has dropped: the loop above would pass on it.
    assert_eq!(
        document.matches("\"width\":").count(),
        EXCEPTIONS.len(),
        "the artifact holds a different number of exceptions than the kernel records"
    );
}

#[test]
fn an_exception_is_outside_the_code_assignment() {
    let book = Codebook::at(CodebookVersion::V1).expect("V1 is assigned");
    let canonical = book.canonical_bytes();

    for exception in EXCEPTIONS {
        // The fingerprint commits to an assignment of codes. An exception assigns
        // none, so folding it in would move the fingerprint of a version whose
        // codes had not moved - invalidating every dataset, checkpoint and run
        // manifest bound to it for a change that renamed nothing.
        assert!(
            !canonical
                .windows(exception.name.len())
                .any(|window| window == exception.name.as_bytes()),
            "{} appears in the canonical bytes",
            exception.name
        );
        // And it is not a family, which is the thing it is an exception to.
        assert!(
            CodeFamily::from_name(exception.name).is_none(),
            "{} is recorded as an exception and is also a family",
            exception.name
        );
    }
}

#[test]
fn a_width_is_looked_up_by_name_and_an_unrecorded_name_has_none() {
    assert_eq!(exception_width("provenance_bucket_count"), Some(64));
    // The control: the lookup answers `None` rather than a default, so a caller
    // that misspells the name gets nothing instead of a plausible number.
    assert_eq!(exception_width("provenance_buckets"), None);
    assert_eq!(exception_width("semantic_role"), None);
}
