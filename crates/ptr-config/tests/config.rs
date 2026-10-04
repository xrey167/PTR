use ptr_config::{PtrConfig, WireConfigError};

#[test]
fn repository_default_config_parses_and_validates() {
    let text = include_str!("../../../config/default.toml");
    let config = PtrConfig::from_toml_str(text).expect("default config parses");
    config.validate().expect("default config validates");
    assert_eq!(config.runtime.mode, "standalone");
}

#[test]
fn daemon_validation_rejects_non_loopback_and_missing_persistence() {
    let mut config = PtrConfig::default();
    config.server.bind = "0.0.0.0:8080".into();
    config.server.data_dir = "definitely-missing-ptr-data".into();
    assert!(config.validate_daemon().is_err());

    config.server.bind = "127.0.0.1:8080".into();
    config.server.data_dir = std::env::temp_dir().to_string_lossy().into_owned();
    assert!(config.validate_daemon().is_ok());
}

#[test]
fn a_misspelled_key_is_refused_instead_of_silently_ignored() {
    let text = "[server]\nbind = \"127.0.0.1:8080\"\ndata_dirr = \"typo\"\n";
    assert!(PtrConfig::from_toml_str(text).is_err());
    let text = "[runtim]\nmode = \"cluster\"\n";
    assert!(PtrConfig::from_toml_str(text).is_err());
}

#[test]
fn loading_from_a_path_validates_the_result() {
    let directory = std::env::temp_dir().join(format!("ptr-config-load-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let invalid = directory.join("invalid.toml");
    std::fs::write(&invalid, "[runtime]\nmailbox_capacity = 0\n").unwrap();
    assert!(PtrConfig::from_path(&invalid).is_err());
    let valid = directory.join("valid.toml");
    std::fs::write(&valid, "[runtime]\nmailbox_capacity = 8\n").unwrap();
    assert!(PtrConfig::from_path(&valid).is_ok());
    std::fs::remove_dir_all(&directory).unwrap();
}

/// A `[wire]` section with every field decided, and one peer.
const WIRE: &str = r#"
[wire]
max_concurrent = 16
per_connection_deadline_ms = 5000
services = ["exec", "pod"]

[[wire.peers]]
id = "3b6a27bcceb6a42d62a3a8d02a6f0d73653215771de243a63ac048a18b59da29"
direct = "127.0.0.1:4433"
"#;

/// The same section with one line replaced.
fn wire_with(line: &str, replacement: &str) -> String {
    assert!(WIRE.contains(line), "the fixture has {line:?}");
    WIRE.replace(line, replacement)
}

/// The refusal a `[wire]` section gets, by its named error.
fn wire_refusal(text: &str) -> WireConfigError {
    let config = PtrConfig::from_toml_str(text).expect("the section parses");
    let wire = config.wire.clone().expect("the section is present");
    let error = wire.validate().expect_err("the section is refused");
    // The whole configuration is refused for the same reason, with the code in its
    // message, so an operator reading `ptrd`'s refusal sees which rule it was.
    let message = config.validate().expect_err("the configuration is refused");
    assert!(
        message.starts_with(error.code()),
        "{message:?} names {}",
        error.code()
    );
    error
}

#[test]
fn a_wire_section_with_every_field_decided_validates_and_no_section_serves_nothing() {
    let config = PtrConfig::from_toml_str(WIRE).unwrap();
    config.validate().unwrap();
    let wire = config.wire.unwrap();
    assert_eq!(wire.max_concurrent, 16);
    assert_eq!(wire.per_connection_deadline_ms, 5000);
    assert!(wire.serves("exec") && wire.serves("pod"));
    assert_eq!(wire.peers.len(), 1);
    assert_eq!(wire.peers[0].direct, "127.0.0.1:4433");

    // The control: without the section there is no wire at all, not a default one.
    assert_eq!(PtrConfig::default().wire, None);
    let repository = PtrConfig::from_toml_str(include_str!("../../../config/default.toml"));
    assert_eq!(repository.unwrap().wire, None);
}

#[test]
fn a_wire_section_that_leaves_a_field_undecided_is_refused() {
    // No defaults: each field missing is a parse error, not a number filled in.
    for line in [
        "max_concurrent = 16\n",
        "per_connection_deadline_ms = 5000\n",
        "services = [\"exec\", \"pod\"]\n",
    ] {
        let text = wire_with(line, "");
        assert!(
            PtrConfig::from_toml_str(&text).is_err(),
            "a section without {line:?} is refused"
        );
    }
    // Peers too: an empty list is a decision, an absent one is not.
    let without_peers = WIRE.split("[[wire.peers]]").next().unwrap();
    assert!(PtrConfig::from_toml_str(without_peers).is_err());
    let with_none = format!("{without_peers}peers = []\n");
    PtrConfig::from_toml_str(&with_none)
        .unwrap()
        .validate()
        .unwrap();
    // And a key the section does not have is refused rather than ignored.
    let misspelled = wire_with("max_concurrent = 16\n", "max_concurent = 16\n");
    assert!(PtrConfig::from_toml_str(&misspelled).is_err());
}

#[test]
fn a_wire_limit_or_deadline_out_of_range_is_refused_by_name() {
    assert_eq!(
        wire_refusal(&wire_with("max_concurrent = 16", "max_concurrent = 0")),
        WireConfigError::Concurrency { value: 0 }
    );
    assert_eq!(
        wire_refusal(&wire_with("max_concurrent = 16", "max_concurrent = 4097")),
        WireConfigError::Concurrency { value: 4097 }
    );
    assert_eq!(
        wire_refusal(&wire_with(
            "per_connection_deadline_ms = 5000",
            "per_connection_deadline_ms = 0"
        )),
        WireConfigError::Deadline { value_ms: 0 }
    );
    assert_eq!(
        wire_refusal(&wire_with(
            "per_connection_deadline_ms = 5000",
            "per_connection_deadline_ms = 600001"
        )),
        WireConfigError::Deadline { value_ms: 600_001 }
    );
    // The controls: each bound itself is accepted.
    for (line, edge) in [
        ("max_concurrent = 16", "max_concurrent = 1"),
        ("max_concurrent = 16", "max_concurrent = 4096"),
        (
            "per_connection_deadline_ms = 5000",
            "per_connection_deadline_ms = 10",
        ),
        (
            "per_connection_deadline_ms = 5000",
            "per_connection_deadline_ms = 600000",
        ),
    ] {
        PtrConfig::from_toml_str(&wire_with(line, edge))
            .unwrap()
            .validate()
            .unwrap_or_else(|error| panic!("{edge}: {error}"));
    }
}

#[test]
fn a_wire_service_list_that_is_empty_unknown_or_repeated_is_refused_by_name() {
    let services = "services = [\"exec\", \"pod\"]";
    assert_eq!(
        wire_refusal(&wire_with(services, "services = []")),
        WireConfigError::NoServices
    );
    assert_eq!(
        wire_refusal(&wire_with(services, "services = [\"exec\", \"raft\"]")),
        WireConfigError::UnknownService {
            service: "raft".into()
        }
    );
    assert_eq!(
        wire_refusal(&wire_with(services, "services = [\"pod\", \"pod\"]")),
        WireConfigError::DuplicateService {
            service: "pod".into()
        }
    );
    // The control: one service alone is enough.
    PtrConfig::from_toml_str(&wire_with(services, "services = [\"pod\"]"))
        .unwrap()
        .validate()
        .unwrap();
}

#[test]
fn a_malformed_or_duplicate_wire_peer_is_refused_by_name() {
    let id = "3b6a27bcceb6a42d62a3a8d02a6f0d73653215771de243a63ac048a18b59da29";
    // Not the transport's spelling of a key: upper case, short, not hex.
    for bad in [id.to_uppercase(), id[..62].to_owned(), "g".repeat(64)] {
        assert_eq!(
            wire_refusal(&wire_with(id, &bad)),
            WireConfigError::MalformedPeerId { id: bad.clone() }
        );
    }
    for bad in ["localhost:4433", "127.0.0.1", "not an address"] {
        assert_eq!(
            wire_refusal(&wire_with("127.0.0.1:4433", bad)),
            WireConfigError::MalformedPeerAddress {
                id: id.to_owned(),
                direct: bad.to_owned(),
            }
        );
    }
    let twice = format!("{WIRE}\n[[wire.peers]]\nid = \"{id}\"\ndirect = \"127.0.0.1:5544\"\n");
    assert_eq!(
        wire_refusal(&twice),
        WireConfigError::DuplicatePeer { id: id.to_owned() }
    );
    // The control: a second peer with its own id is accepted.
    let other = "4b6a27bcceb6a42d62a3a8d02a6f0d73653215771de243a63ac048a18b59da29";
    let two = format!("{WIRE}\n[[wire.peers]]\nid = \"{other}\"\ndirect = \"[::1]:5544\"\n");
    let config = PtrConfig::from_toml_str(&two).unwrap();
    config.validate().unwrap();
    assert_eq!(config.wire.unwrap().peers.len(), 2);
}

#[test]
fn every_wire_refusal_has_its_own_code() {
    let errors = [
        WireConfigError::Concurrency { value: 0 },
        WireConfigError::Deadline { value_ms: 0 },
        WireConfigError::NoServices,
        WireConfigError::UnknownService {
            service: String::new(),
        },
        WireConfigError::DuplicateService {
            service: String::new(),
        },
        WireConfigError::MalformedPeerId { id: String::new() },
        WireConfigError::MalformedPeerAddress {
            id: String::new(),
            direct: String::new(),
        },
        WireConfigError::DuplicatePeer { id: String::new() },
    ];
    let mut codes: Vec<&str> = errors.iter().map(WireConfigError::code).collect();
    codes.sort_unstable();
    codes.dedup();
    assert_eq!(codes.len(), errors.len(), "two refusals share a code");
}
