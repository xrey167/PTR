# PTR Runtime – ExecutionManifest- und Lineage-Integration

## Ziel

`ptr-runtime` wird der produktive Resolver für `ExecutionManifest`. Ein
PodWire-V3-Digest wird nur akzeptiert, wenn das vollständige Manifest aus
autoritativen PTR-Quellen rekonstruiert und erneut validiert werden kann.

```text
PodWire Digest
    ↓
PtrRuntime Resolver
    ↓
Ledger / SemDB / Knowledge / Artifact
    ↓
Lineage Validation
    ↓
ValidatedExecutionManifest
    ↓
PodWire Binding
```

## 1. Autoritätsgrenzen

- [ ] `ptr-ledger` als dauerhafte Eventquelle bestätigen.
- [ ] `ptr-semdb` als semantische State-Quelle bestätigen.
- [x] `ptr-memory::KnowledgeStore` als Knowledge-Quelle anbinden.
- [x] `ptr-pods::ArtifactCatalog` als Artifact-Quelle anbinden.
- [ ] `ptr-runtime` als Admission-/Validierungs-/Promotion-Autorität definieren.
- [ ] `ptr-podwire` nur als Transport-/Binding-Schicht verwenden.
- [ ] Keine zweite Knowledge- oder Promotion-Autorität einführen.

## 2. Runtime-Resolver-Verträge

```rust
pub trait RuntimeExecutionManifestResolver: Send + Sync {
    fn resolve(
        &self,
        digest: &Digest,
    ) -> Result<ValidatedExecutionManifest, RuntimeManifestError>;
}
```

- [x] `ValidatedExecutionManifest` definieren.
- [x] Vollständiges `ExecutionManifest` enthalten.
- [ ] Validierungsbericht enthalten.
- [x] Knowledge-Lineage enthalten.
- [x] Artifact-Lineage enthalten.
- [ ] Principal-/Policy-Bindung enthalten.
- [ ] Snapshot-/Revision-Bindung enthalten.
- [x] Digest erneut kanonisch berechnen.
- [x] Immutable Registry für admitted Manifeste definieren.
- [x] Idempotentes Re-Register erlauben.
- [x] Konfligierende Wiederaufnahme ablehnen.
- [x] Revocation-Tombstones dauerhaft speichern.

## 3. Manifest-Aufbau

- [ ] Aktuelle Knowledge-Generationen ermitteln.
- [ ] Dependencies generation-aware auflösen.
- [ ] Invalidierte Knowledge-Records ablehnen.
- [ ] Raw Sources und Provenance prüfen.
- [ ] Artifact-ID und Artifact-Generation auflösen.
- [ ] Artifact-Lifecycle prüfen.
- [ ] Model-/Reader-/Adapter-Generation prüfen.
- [ ] Snapshot-Revision und Snapshot-Digest prüfen.
- [ ] Principal und Policy-Revision binden.
- [ ] Ressourcen- und Sicherheitsanforderungen binden.
- [ ] Kanonischen Manifest-Digest berechnen.
- [ ] Manifest immutable in den Runtime-Kontext übernehmen.

## 4. Lineage-Validierung

### Knowledge

- [x] Alle Knowledge-Generationen existieren.
- [x] Alle Dependencies existieren.
- [ ] Dependencies zeigen auf exakt die erwartete Generation.
- [x] Keine Dependency ist invalidiert.
- [ ] Supersedes-/Vorgängerbeziehungen stimmen.
- [x] Raw-Source-Digests stimmen.
- [ ] Provenance-Digests stimmen.

### Artifact

- [x] Artifact existiert.
- [x] Artifact-Generation stimmt.
- [x] Artifact ist nicht revoked.
- [ ] Artifact ist mindestens `Approved` oder `Active`.
- [ ] Manifest-Hash entspricht dem konkreten PodManifest.
- [ ] Input-/Output-Schemas stimmen.
- [ ] Model-/Adapter-Identität stimmt.

### Runtime

- [ ] Projekt stimmt.
- [ ] Namespace stimmt.
- [ ] Pod-ID stimmt.
- [ ] Capability stimmt.
- [ ] Principal ist zugelassen.
- [ ] Policy-Revision ist aktuell.
- [ ] Session ist gültig.
- [ ] Placement-Epoch ist aktuell.
- [ ] Fencing-Token ist aktuell.

## 5. Fehler- und Sicherheitsmodell

- [ ] `UnknownManifestDigest`
- [ ] `ManifestDigestMismatch`
- [ ] `MissingKnowledgeGeneration`
- [ ] `RevokedKnowledgeGeneration`
- [ ] `MissingArtifactGeneration`
- [ ] `RevokedArtifactGeneration`
- [ ] `InvalidReaderBinding`
- [ ] `InvalidProvenance`
- [ ] `InvalidDependencyLineage`
- [ ] `SnapshotMismatch`
- [ ] `PolicyRevisionMismatch`
- [ ] `ProjectMismatch`
- [ ] `SessionMismatch`
- [ ] `StalePlacement`
- [ ] `StaleFencingToken`
- [ ] Kein Resolver-Fehler erreicht `invoke()`.
- [ ] Kein unbekannter Digest wird als leerer Kontext behandelt.
- [ ] Kein stale Manifest wird automatisch ersetzt.
- [ ] Kein Fallback auf eine andere Artifact-Generation.

## 6. PtrRuntime-Integration

- [x] Resolver als Runtime-Komponente registrieren.
- [ ] Resolver-Zugriff nur über `PtrRuntime` erlauben.
- [ ] Runtime-Revision beim Resolve prüfen.
- [ ] Revoked-Generation-Index verwenden.
- [ ] Scope-/Session-Bindung verwenden.
- [ ] Placement-/Lease-Bindung verwenden.
- [x] Resolver-Ergebnis für PodWire-Binding freigeben.
- [ ] Resolver-Ergebnis für Executor-Admission freigeben.
- [ ] Resolver-Ergebnis für Output-Promotion weiterreichen.

## 7. PodWire-V3-Anbindung

- [x] PodWire erhält nur den Digest aus dem Request.
- [x] Runtime löst den Digest auf.
- [ ] Runtime validiert das vollständige Manifest.
- [x] Runtime erzeugt ein validated Binding.
- [x] Host prüft Request gegen dieses Binding.
- [x] Falscher Digest wird vor `invoke()` abgewiesen.
- [ ] Falsche Generation wird vor `invoke()` abgewiesen.
- [ ] Falsche Policy-Revision wird vor `invoke()` abgewiesen.
- [ ] Response bestätigt denselben Digest.
- [x] Live-Registry-Gate prüft Revocation unmittelbar vor `invoke()`.
- [ ] Client weist Response-Mismatch ab.
- [ ] V1 bleibt unverändert.
- [ ] Unaddressed V2 bleibt kompatibel.

## 8. Ledger und Replay

- [x] `ExecutionManifestAdmitted` definieren.
- [x] `ExecutionManifestRevoked` definieren.
- [x] Manifest-Digest im Event speichern.
- [x] Generationen im Event speichern.
- [x] Policy-Revision im Event speichern.
- [ ] Provenance-Digest im Event speichern.
- [x] Artifact-/Knowledge-Abhängigkeiten speichern.
- [x] Replay deterministisch implementieren.
- [x] Widersprüchliche Events ablehnen.
- [x] Revocation nach Neustart erhalten.
- [x] Durable Reopen testen.

## 9. Output- und Promotion-Bindung

- [ ] Output muss denselben ExecutionManifest-Digest nennen.
- [ ] Output-Generation prüfen.
- [ ] Output-Provenance prüfen.
- [ ] Output-Dependencies prüfen.
- [ ] Verifier im selben Runtime-Kontext ausführen.
- [ ] SemDB-Promotion nur über `PtrRuntime` ausführen.
- [ ] Branch-Merge nur über `PtrRuntime` ausführen.
- [ ] ActionProposal ausschließlich an ExecWire weitergeben.
- [ ] Manifest-fremde Outputs vor Promotion ablehnen.

## 10. Tests

### Unit

- [ ] Gültiges Manifest wird aufgelöst.
- [ ] Unbekannter Digest wird abgewiesen.
- [ ] Digest-Tampering wird abgewiesen.
- [ ] Fehlende Knowledge-Generation wird abgewiesen.
- [ ] Revoked Knowledge-Generation wird abgewiesen.
- [ ] Fehlendes Artifact wird abgewiesen.
- [ ] Revoked Artifact wird abgewiesen.
- [ ] Falscher Reader wird abgewiesen.
- [ ] Falsche Provenance wird abgewiesen.
- [ ] Falsche Policy-Revision wird abgewiesen.

### Integration

- [ ] Runtime baut Manifest aus Knowledge-/Artifact-Records.
- [ ] Runtime registriert Manifest idempotent.
- [ ] Widersprüchliches Re-Register wird abgewiesen.
- [ ] PodWire erhält nur validierte Bindings.
- [ ] Falscher Digest erreicht `invoke()` nicht.
- [ ] Stale Generation erreicht `invoke()` nicht.
- [ ] Revocation wird vor `invoke()` wirksam.
- [ ] Runtime-Reopen rekonstruiert den Resolver-State.

### E2E

```text
Knowledge Generation 1
→ Artifact Generation 1
→ ExecutionManifest bauen
→ Manifest-Admit journalisieren
→ PodWire V3 Request
→ Runtime resolve/validate
→ Pod invoke()
→ Output Admission
→ Knowledge Generation 2
→ Generation 1 revoke
→ alter Manifest-Digest abgewiesen
→ neues Manifest bauen
→ neuer Request erfolgreich
```

- [ ] Kompletter E2E-Test implementieren.
- [ ] Generation-Update prüfen.
- [ ] Revocation prüfen.
- [ ] Runtime-Reopen prüfen.
- [ ] Stale Session prüfen.
- [ ] Stale Fencing prüfen.
- [ ] Keine doppelte Promotion prüfen.

## 11. Research-Fragen

- [ ] Welche Knowledge-Objekte muss das Modell sehen?
- [ ] Welche Lineage muss nur verifizierbar, nicht prompt-sichtbar sein?
- [ ] Welche Artefakte gehören zu einem Modell-Pod?
- [ ] Wie werden LoRA-, Quantisierungs- und Reader-Generationen gebunden?
- [ ] Wann erzeugt eine Änderung ein neues ExecutionManifest?
- [ ] Wie wird Manifest-Revoke in Replay-/Trainingsspuren dargestellt?
- [ ] Welche Runtime-Events sollen in den Lernpfad eingehen?

## 12. Abschluss-Gates

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo check --workspace --locked`
- [ ] `cargo clippy --workspace --all-targets --locked -- -D warnings`
- [ ] `cargo test -p ptr-runtime --locked`
- [ ] `cargo test -p ptr-podwire --features podwire-backend --locked`
- [ ] `cargo test -p ptr-memory --locked`
- [ ] `cargo test -p ptr-ledger --locked`
- [ ] `cargo test --workspace --locked --no-fail-fast`
- [ ] `git diff --check`
- [ ] Kein konkurrierender Cargo-/Rustc-Prozess.

## Aktueller Umsetzungsschnitt

- [x] PodLink bindet den ExecutionManifest-Digest.
- [x] PodWire V3 überträgt den Digest in Request und Response.
- [x] PodWire besitzt einen `ExecutionManifestResolver`-Vertrag.
- [x] Binding validiert das aufgelöste Manifest strukturell.
- [x] `PtrRuntime` stellt einen echten Knowledge-/Artifact-Resolver bereit.
- [x] Runtime-Lineage wird gegen autoritative Stores geprüft.
- [x] Runtime-Manifest-Admit-/Revoke-Events werden journalisiert.
- [ ] Vollständiger Runtime-to-PodWire-E2E-Test.
