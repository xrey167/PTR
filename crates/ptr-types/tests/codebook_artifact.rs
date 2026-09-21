//! The generated codebook artifact must still describe this kernel.
//!
//! `datasets/generated/codebook.json` is what everything outside Rust reads. If
//! the tables here change and that file does not, the disagreement is silent:
//! dataset validation keeps accepting the old member set and a checkpoint keeps
//! loading under an assignment that has moved. This test is where that fails.
//!
//! It parses the fields it needs by hand rather than pulling a JSON crate in,
//! because `ptr-types` has no dependencies and this is a test of a flat document.

use ptr_types::{CodeFamily, Codebook, CodebookVersion};

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
