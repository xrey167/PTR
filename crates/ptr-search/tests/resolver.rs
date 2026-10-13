//! Resolving a query's name to a canonical one, or refusing to.
use ptr_search::{AliasError, AliasTable, ResolveError, MAX_ALIASES, MAX_ALIAS_WORDS};

fn table() -> AliasTable {
    let mut table = AliasTable::new();
    table.add("Müller", "supplier:mueller").unwrap();
    table.add("Müller Werke", "supplier:mueller-werke").unwrap();
    table.add("Acme", "supplier:acme").unwrap();
    table.add("Acme Corp", "supplier:acme").unwrap();
    table
}

#[test]
fn a_known_name_resolves_whatever_its_case_and_punctuation() {
    let resolved = table().resolve("What is the lead time from ACME?").unwrap();
    assert_eq!(resolved.canonical, "supplier:acme");
    assert_eq!(resolved.alias, "acme");
}

#[test]
fn the_longest_match_settles_its_shorter_prefix() {
    let resolved = table()
        .resolve("delivery from Müller Werke please")
        .unwrap();
    assert_eq!(resolved.canonical, "supplier:mueller-werke");
    assert_eq!(resolved.alias, "müller werke");
    let shorter = table().resolve("delivery from Müller please").unwrap();
    assert_eq!(shorter.canonical, "supplier:mueller");
}

#[test]
fn two_aliases_of_one_canonical_name_are_not_an_ambiguity() {
    let resolved = table().resolve("Acme and Acme Corp").unwrap();
    assert_eq!(resolved.canonical, "supplier:acme");
}

#[test]
fn a_query_naming_two_different_things_is_ambiguous_and_lists_them_sorted() {
    assert_eq!(
        table().resolve("Müller or Acme?"),
        Err(ResolveError::Ambiguous {
            candidates: vec!["supplier:acme".into(), "supplier:mueller".into()]
        })
    );
}

#[test]
fn a_name_that_is_not_in_the_table_is_unknown_even_when_it_looks_alike() {
    assert_eq!(
        table().resolve("lead time from Mueller"),
        Err(ResolveError::Unknown)
    );
    assert_eq!(
        table().resolve("Acmes and Acmeless firms"),
        Err(ResolveError::Unknown),
        "a match never starts or ends inside a word"
    );
    assert_eq!(table().resolve(""), Err(ResolveError::Unknown));
    assert_eq!(
        AliasTable::new().resolve("Acme"),
        Err(ResolveError::Unknown)
    );
}

#[test]
fn a_query_is_bounded() {
    let long = "word ".repeat(300);
    assert!(matches!(
        table().resolve(&long),
        Err(ResolveError::QueryTooLong {
            limit: 256,
            actual: 300
        })
    ));
}

#[test]
fn the_table_is_a_function_and_is_bounded() {
    let mut table = table();
    assert_eq!(
        table.add("Acme", "supplier:acme"),
        Ok(()),
        "the same pair again is a no-op"
    );
    assert_eq!(
        table.add("acme", "supplier:other"),
        Err(AliasError::Conflict {
            alias: "acme".into(),
            existing: "supplier:acme".into()
        })
    );
    assert_eq!(table.add("  ", "supplier:x"), Err(AliasError::Empty));
    assert_eq!(table.add("x", "--"), Err(AliasError::Empty));
    let long = ["w"; MAX_ALIAS_WORDS + 1].join(" ");
    assert_eq!(
        table.add(&long, "supplier:x"),
        Err(AliasError::TooLong {
            limit: MAX_ALIAS_WORDS,
            actual: MAX_ALIAS_WORDS + 1
        })
    );
    let mut full = AliasTable::new();
    for n in 0..MAX_ALIASES {
        full.add(&format!("alias{n}"), "c").unwrap();
    }
    assert_eq!(full.len(), MAX_ALIASES);
    assert_eq!(
        full.add("one more", "c"),
        Err(AliasError::TableFull { limit: MAX_ALIASES })
    );
}

#[test]
fn every_resolve_refusal_has_a_stable_code() {
    assert_eq!(ResolveError::Unknown.code(), "PTR_SEARCH_RESOLVE_UNKNOWN");
    assert_eq!(
        ResolveError::Ambiguous { candidates: vec![] }.code(),
        "PTR_SEARCH_RESOLVE_AMBIGUOUS"
    );
    assert_eq!(
        ResolveError::QueryTooLong {
            limit: 1,
            actual: 2
        }
        .code(),
        "PTR_SEARCH_RESOLVE_QUERY_TOO_LONG"
    );
}
