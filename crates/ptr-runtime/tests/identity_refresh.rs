use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::{Signer, SigningKey};
use ptr_runtime::identity::{
    AutheliaForwardAuthAdapter, JwksDocument, JwksRefresher, JwksStore, OidcError,
    OidcIdentityAdapter,
};
use ptr_types::{IdentityProvider, Timestamp};
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

fn token(key: &SigningKey, kid: &str) -> String {
    let header = URL_SAFE_NO_PAD.encode(json!({"alg": "EdDSA", "kid": kid}).to_string());
    let payload = URL_SAFE_NO_PAD.encode(
        json!({
            "iss": "https://issuer", "aud": "ptr", "sub": "user", "sid": "session",
            "iat": 1, "exp": 100,
        })
        .to_string(),
    );
    let input = format!("{header}.{payload}");
    format!(
        "{input}.{}",
        URL_SAFE_NO_PAD.encode(key.sign(input.as_bytes()).to_bytes())
    )
}

struct Refresher {
    calls: AtomicUsize,
    issuer: &'static str,
    audience: &'static str,
    store: JwksStore,
    error: Option<OidcError>,
}

impl Refresher {
    fn new(key: &SigningKey) -> Self {
        let mut store = JwksStore::default();
        store.insert("rotated", key.verifying_key());
        Self {
            calls: AtomicUsize::new(0),
            issuer: "https://issuer",
            audience: "ptr",
            store,
            error: None,
        }
    }
}

impl JwksRefresher for Refresher {
    fn refresh(&self) -> Result<JwksDocument, OidcError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        Ok(JwksDocument {
            issuer: self.issuer.into(),
            audience: self.audience.into(),
            store: self.store.clone(),
        })
    }
}

fn adapter() -> OidcIdentityAdapter {
    OidcIdentityAdapter::new("https://issuer", "ptr", JwksStore::default())
        .with_refresh_cooldown(Duration::from_secs(3600))
}

#[test]
fn failed_refreshes_are_throttled_across_different_unknown_keys() {
    let key = SigningKey::from_bytes(&[41; 32]);
    let adapter = adapter();
    let mut refresher = Refresher::new(&key);
    refresher.error = Some(OidcError::RefreshFailed("offline".into()));
    assert_eq!(
        adapter.authenticate_at_with_refresh(
            token(&key, "rotated").as_bytes(),
            Timestamp(50),
            &refresher
        ),
        Err(OidcError::RefreshFailed("offline".into()))
    );
    for kid in ["rotated", "unknown-1", "unknown-2"] {
        assert_eq!(
            adapter.authenticate_at_with_refresh(
                token(&key, kid).as_bytes(),
                Timestamp(50),
                &refresher
            ),
            Err(OidcError::UnknownKey(kid.into()))
        );
    }
    assert_eq!(refresher.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn rejected_jwks_metadata_is_not_cached_and_still_consumes_the_cooldown() {
    let key = SigningKey::from_bytes(&[42; 32]);
    let credential = token(&key, "rotated");
    for (issuer, audience, error) in [
        ("https://wrong", "ptr", OidcError::JwksIssuerMismatch),
        ("https://issuer", "wrong", OidcError::JwksAudienceMismatch),
    ] {
        let adapter = adapter();
        let mut refresher = Refresher::new(&key);
        refresher.issuer = issuer;
        refresher.audience = audience;
        assert_eq!(
            adapter.authenticate_at_with_refresh(credential.as_bytes(), Timestamp(50), &refresher),
            Err(error)
        );
        assert_eq!(
            adapter.authenticate_at_with_refresh(credential.as_bytes(), Timestamp(50), &refresher),
            Err(OidcError::UnknownKey("rotated".into()))
        );
        assert_eq!(refresher.calls.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn zero_cooldown_refetches_and_does_not_authenticate_a_removed_cached_key() {
    let key = SigningKey::from_bytes(&[43; 32]);
    let adapter = adapter().with_refresh_cooldown(Duration::ZERO);
    let mut refresher = Refresher::new(&key);
    let credential = token(&key, "rotated");
    assert!(adapter
        .authenticate_at_with_refresh(credential.as_bytes(), Timestamp(50), &refresher)
        .is_ok());
    refresher.store = JwksStore::default();
    assert_eq!(
        adapter.authenticate_at_with_refresh(credential.as_bytes(), Timestamp(50), &refresher),
        Err(OidcError::UnknownKey("rotated".into()))
    );
    assert_eq!(refresher.calls.load(Ordering::SeqCst), 2);
}

#[test]
fn cached_keys_still_require_a_valid_signature() {
    let key = SigningKey::from_bytes(&[44; 32]);
    let adapter = adapter();
    let refresher = Refresher::new(&key);
    assert!(adapter
        .authenticate_at_with_refresh(token(&key, "rotated").as_bytes(), Timestamp(50), &refresher)
        .is_ok());
    let wrong = SigningKey::from_bytes(&[45; 32]);
    assert_eq!(
        adapter.authenticate_at_with_refresh(
            token(&wrong, "rotated").as_bytes(),
            Timestamp(50),
            &refresher
        ),
        Err(OidcError::InvalidSignature)
    );
    assert_eq!(refresher.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn concurrent_unknown_key_requests_share_one_refresh() {
    let key = SigningKey::from_bytes(&[46; 32]);
    let adapter = adapter();
    let refresher = Refresher::new(&key);
    let barrier = std::sync::Barrier::new(8);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let credential = token(&key, &format!("unknown-{i}"));
                let (adapter, refresher, barrier) = (&adapter, &refresher, &barrier);
                scope.spawn(move || {
                    barrier.wait();
                    assert_eq!(
                        adapter.authenticate_at_with_refresh(
                            credential.as_bytes(),
                            Timestamp(50),
                            refresher
                        ),
                        Err(OidcError::UnknownKey(format!("unknown-{i}")))
                    );
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
    });
    assert_eq!(refresher.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn locally_known_key_bypasses_refresh_even_during_a_failed_fetch_cooldown() {
    let key = SigningKey::from_bytes(&[47; 32]);
    let mut store = JwksStore::default();
    store.insert("local", key.verifying_key());
    let adapter = OidcIdentityAdapter::new("https://issuer", "ptr", store)
        .with_refresh_cooldown(Duration::from_secs(3600));
    let mut refresher = Refresher::new(&key);
    refresher.error = Some(OidcError::RefreshFailed("offline".into()));
    assert!(adapter
        .authenticate_at_with_refresh(token(&key, "unknown").as_bytes(), Timestamp(50), &refresher)
        .is_err());
    assert!(adapter
        .authenticate_at_with_refresh(token(&key, "local").as_bytes(), Timestamp(50), &refresher)
        .is_ok());
    assert_eq!(refresher.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn forward_auth_accepts_credentials_without_a_client_clock_and_ignores_supplied_clocks() {
    let adapter = AutheliaForwardAuthAdapter::new("https://authelia");
    let mut credential = json!({"subject": "user", "session_id": "session", "groups": ["a"],
        "acr": "mfa", "issued_at": 1, "expires_at": u64::MAX});
    let expected = adapter
        .authenticate(credential.to_string().as_bytes())
        .unwrap();
    for now in [json!(u64::MAX), json!("not a timestamp"), json!(null)] {
        credential["now"] = now;
        assert_eq!(
            adapter
                .authenticate(credential.to_string().as_bytes())
                .unwrap(),
            expected
        );
    }
    credential["expires_at"] = json!(2);
    credential["now"] = json!(1);
    assert_eq!(
        adapter.authenticate(credential.to_string().as_bytes()),
        Err(OidcError::Expired)
    );
}
