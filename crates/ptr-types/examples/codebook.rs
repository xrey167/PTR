//! Emit the kernel's cognitive codebook so other languages read it instead of
//! retyping it.
//!
//! `validate_dataset.py` carried its own copies of the operator, epistemic and
//! uncertainty member names. A second copy of a table is exactly what the
//! codebook exists to prevent: adding a variant in Rust left the Python set
//! silently disagreeing, and nothing failed.
//!
//! An example rather than a binary crate, because it is built and linted by the
//! ordinary `--all-targets` run and needs no new workspace member.
//!
//! The canonical bytes are printed as hex and not hashed here, so `ptr-types`
//! keeps its no-dependency rule. `scripts/generate_codebook.py` computes the
//! fingerprint from these bytes, and a test in this crate asserts the committed
//! artifact still carries the bytes the kernel produces.

use ptr_types::{CodeFamily, Codebook, CodebookVersion, EXCEPTIONS};

fn main() {
    let version = CodebookVersion::V1;
    let book = Codebook::at(version).expect("V1 is assigned");

    let families: Vec<String> = CodeFamily::ALL
        .iter()
        .map(|family| {
            let members: Vec<String> = book
                .member_names(*family)
                .iter()
                .enumerate()
                .map(|(code, name)| format!("        {{ \"code\": {code}, \"name\": {} }}", quote(name)))
                .collect();
            format!(
                "    {{\n      \"family\": {},\n      \"cardinality\": {},\n      \"members\": [\n{}\n      ]\n    }}",
                quote(family.name()),
                book.cardinality(*family),
                members.join(",\n")
            )
        })
        .collect();

    // Recorded exceptions travel with the families, because "this width is not a
    // family" is a statement about the taxonomy and belongs beside it. They are
    // deliberately outside `canonical_bytes`: the fingerprint identifies an
    // assignment of codes, and an exception assigns none, so folding it in would
    // invalidate every artifact bound to a version whose codes had not moved.
    let exceptions: Vec<String> = EXCEPTIONS
        .iter()
        .map(|exception| {
            format!(
                "    {{\n      \"name\": {},\n      \"width\": {},\n      \"reason\": {}\n    }}",
                quote(exception.name),
                exception.width,
                quote(exception.reason)
            )
        })
        .collect();

    let hex = to_hex(&book.canonical_bytes());

    println!("{{");
    println!("  \"schema\": \"1\",");
    println!("  \"version\": {},", version.0);
    println!("  \"canonical_bytes_hex\": {},", quote(&hex));
    println!("  \"families\": [");
    println!("{}", families.join(",\n"));
    println!("  ],");
    println!("  \"exceptions\": [");
    println!("{}", exceptions.join(",\n"));
    println!("  ]");
    println!("}}");
}

/// Lowercase hex without building a string per byte.
fn to_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Minimal JSON string escaping. Family and member names are ASCII identifiers
/// today; escaping anyway keeps the output valid if that stops being true.
fn quote(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other if (other as u32) < 0x20 => out.push_str(&format!("\\u{{{:04x}}}", other as u32)),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}
