# PTR Pods – Masterplan und Abnahme

Dieser Plan führt die Pod-Architektur vom Research-Vertrag bis zur sicheren,
typisierten Ausführung. PTR bleibt die autoritative Schicht für Manifest,
Admission, Provenance, Verifikation, Promotion, Ledger und Effekte. Externe
Projekte liefern Muster, aber keine zusätzliche Runtime-Autorität.

## Statusübersicht

Legende: `[x]` abgeschlossen und gezielt verifiziert, `[~]` teilweise
implementiert, `[-]` geplant/offen.

| Bereich | Status | Prüfkriterium |
|---|---:|---|
| Research-/PTR-Abgleich | [~] | Research-Entscheidungen sind als Verträge, Grenzen und Tests dokumentiert |
| Pod-Semantik, Identität und ModelVariant | [~] | typisierte PodKind-, Manifest- und Lineage-Prüfung |
| ExecutionManifest und Artifact-Lifecycle | [~] | immutable Digest, Generation, Provenance und Activation-Gates |
| PodAddress, Directory, Route und Endpoint | [x] | logische Adresse bleibt bei Rebalance stabil; Endpoint wird neu aufgelöst |
| PodLink und ProtocolBinding | [~] | ACL, Capability, Attestation, Hop-/Cycle-Schutz |
| PodWire V1/V2, Header und Sessions | [~] | typed Binding, Generation, Revision, Request-ID und Backpressure |
| Identity und Admission | [~] | lokale OIDC-/JWKS-Verifikation, Session-/Policy-Bindung |
| Runtime, Scopes, Leases und Fencing | [~] | journal-first Lifecycle, Cleanup, Recovery und stale-writer-Schutz |
| Native Executor und Candle CUDA | [~] | typed Tensor-/Device-Vertrag; echte GPU-Abnahme noch offen |
| Output Admission und Evidence | [~] | Verifier vor Candidate, Branch, SemDB oder ExecWire |
| Branches, Hypothesen und Merge | [~] | persistente Evidence, deterministischer und autoritativer Merge |
| Knowledge, Ingestion und Context | [~] | immutable Generationen, Provenance, Retention und Digest |
| KV-State und Placement/Fencing | [~] | opaque Handles, Snapshot/Recompute; echter Modell-KV-Cache offen |
| Ledger und Protected State | [x] | durable Replay, AEAD/Anchor-Prüfung und Protected-Snapshot-Bindung |
| ExecWire und Effect Recovery | [x] | uncertain Effects werden durable gefenced und reconciled |
| Sessions, Turn Events und Recovery | [~] | RequestUncertain führt zu Cleanup, Re-Admission und Recompute |
| Mesh, Relay und Plattformadapter | [~] | Contracts vorhanden; privilegierte WireGuard-/Relay-Abnahme offen |
| Sandbox und Environment | [~] | native Standardausführung; isolierte Adapter nachgelagert |
| Identity-to-Recovery-E2E | [~] | lokaler Iroh-/PodWire-Pfad mit Persistenz und Reopen |

## Bereits umgesetzte Kernverträge

- `ptr-pods` beschreibt semantische Pods, ExecutionManifest, PodLink,
  ProtocolBinding, typed Output/Evidence und native Protokolltypen.
- `ptr-runtime` bindet Identity, Policy, Snapshot, Scope, Placement,
  Fencing, Output Admission, Branch Merge und Promotion.
- `ptr-ledger` bleibt die einzige dauerhafte Eventquelle; Runtime- und
  Protected-State-Ereignisse sind replaybar und integritätsgeprüft.
- `ptr-podwire` bleibt ein effektfreier typed Transport. Netzwerk-, Datei-,
  Tool- und Prozesswirkungen laufen ausschließlich über `ptr-execwire`.
- `ptr-net` enthält Mesh-/Tunnel-Verträge und Direct/Relay-Modellierung;
  privilegierte Kernel-/Wintun-Ausführung ist noch ein Plattformadapter.

## Abgeschlossene Blöcke

### Protected-State- und Snapshot-Authority

`ManifestAuthorityRegistry` materialisiert geschützte Snapshot-Digests und
prüft Live-SemDB-Revision/Digest gegen Manifest und Protected State. Replay
validiert `KvSnapshot`-Events; Snapshot-Mismatch führt zu fail-closed
Ablehnung oder Recompute. Gezielte Runtime- und Protected-State-Tests wurden
bestanden.

### ExecWire-Unsicherheit und Recovery

Ein unklarer Effect-Versuch wird als durable Runtime-Fence behandelt. Replay
rekonstruiert den unsettled Effect; weitere Mutation wird bis zu einer
expliziten Reconciliation abgewiesen. Der gezielte `ptr-execwire`-Wire- und
Recovery-Testlauf wurde bestanden.

## Aktueller Block: produktive Identity-/Policy-Adapter

Implementiert:

- optionales Feature `ptr-runtime/oidc-http` mit `reqwest` 0.13.5,
  `blocking` und `rustls-no-provider`; der Resolver bleibt kompatibel mit der
  bestehenden Iroh/rustls-Abhängigkeitskette;
- `ReqwestJwksRefresher`: nur HTTPS, Status-/Größenprüfung, JSON-Decode,
  Issuer-/Audience-Bindung und erneute lokale JWKS-Validierung;
- `HttpPolicyBundleLoader` für injizierbare, testbare Loader und
  `ReqwestPolicyBundleLoader` für HTTPS-Produktionszugriff;
- Loader dekodieren nur. Ed25519-Signatur, Payload-Digest, Revision,
  Gültigkeitszeitraum und Revocation bleiben bei der Runtime-Admission;
- unbekannter JWKS-`kid` darf nur kontrolliert aktualisieren; scheitert die
  Aktualisierung, bleibt der neue Stateful-Aufruf fail-closed.
- Session-Revocation wird als `LedgerEvent::SessionRevoked` journalisiert,
  beim Replay in die Runtime-Projektion übernommen und vor jeder neuen
  Identity-/Stateful-Admission geprüft; wiederholte Revocation ist idempotent.
- Nach validiertem Ledger-Replay wird eine zuvor aktivierte Policy wieder als
  admissionsfähig materialisiert; eine vollständige Revocation aller aktiven
  Policy-Revisionen sperrt die Admission erneut.
- JWKS-/Policy-HTTP-Responses werden schon beim Lesen auf die jeweilige
  Maximalgröße begrenzt; unbekannte oder übergroße Dokumente bleiben
  fail-closed.

Nachweis:

- `cargo check -p ptr-runtime --features oidc-http --offline -j 1` erfolgreich;
- `cargo test -p ptr-runtime --lib --locked --offline -j 1`: 27/27 erfolgreich.
- `cargo test -p ptr-runtime --features oidc-http --lib --locked --offline -j 1`:
  28/28 erfolgreich;
- `cargo clippy -p ptr-runtime --features oidc-http --all-targets --locked
  --offline -- -D warnings` erfolgreich;
- `cargo test --workspace --locked --no-fail-fast -j 1` vollständig erfolgreich.
- Policy-/Authority-Tests: 4/4 erfolgreich; Ledger-Codec-Tests: 19/19
  erfolgreich.

Die Läufe wurden mit `CARGO_TARGET_DIR=E:\\PTR-cargo-target`,
`RUSTFLAGS=-C debuginfo=0` und `CARGO_INCREMENTAL=0` ausgeführt, damit der
Windows-PDB-Limit-/Plattenplatzfehler nicht erneut auftritt. Das ist ein
Build-Umgebungsparameter und ändert keine Runtime-Semantik.

## Nächste Blöcke

1. HTTP-Refresh-Tests mit lokalem HTTPS-Testserver und kontrolliertem JWKS-
   Unknown-`kid`-/Refresh-/Issuer-/Audience-Verhalten.
2. Durable Policy-/Identity-Replay: Aktivierung, Revocation, Session-Revoke
   und Policy-Revision in einer gemeinsamen Runtime-Projektion.
3. Vollständiger Identity-to-Recovery-E2E über PodWire-Session, Protected KV,
   Ledger-Reopen, neue Admission und neue Fencing-Epoch.
4. Echter modellgebundener Candle-KV-Cache: layerweise K/V-Tensoren,
   Snapshot/Restore, Device-/Stream-Lifetime und Recompute.
5. Remote Session-Reuse, RequestUncertain-Recovery und Iroh Direct/Relay.
6. Privilegierte WireGuard-/Wintun-/Network-Extension-Adapter und echte
   MeshEndpoint-Bindung.
7. Cross-generation Training-/Replay-Trace, Branch-/Hypothesis-Merge und
   native Protocol Pods für TCP, UDP, SSH, VPN, MQTT und WebRTC.

## Verifikation und Grenzen

Vor einem Abschluss sind gezielt und sequenziell auszuführen:

```text
cargo fmt --all -- --check
cargo check --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked --no-fail-fast
git diff --check
```

Ein Testlauf zählt nur, wenn er vollständig beendet wurde und kein Cargo-/Rustc-
Prozess oder Tool-Timeout den Nachweis unterbrochen hat. CUDA-, privilegierte
WireGuard- und echte RTX-3090/4090-Abnahmen werden nur für tatsächlich erkannte
Hardware bzw. vorhandene Berechtigungen als bestanden gemeldet.

Bewusste Grenzen: kein Python-Produktionspfad, keine freie JSON-Map als
Autorität, keine LLM-Summary als einzige Wissensrepräsentation, keine
Gleichsetzung von KV-Entfernung mit Wissenslöschung und keine direkten Effekte
aus PodWire.
