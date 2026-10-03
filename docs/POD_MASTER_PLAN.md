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
| Multi-Tier Storage und Residency | [~] | CPU/NVMe/OpenDAL, Pins, Dedupe, Drain und Replay implementiert; echte GPU/S3-Abnahme offen |
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

## Vollständige Arbeitsliste

Diese Matrix ist die ausführbare Langfassung des Plans. `[x]` bedeutet, dass
der Vertrag im Repository vorhanden und gezielt geprüft ist, `[~]` bedeutet
Teilimplementierung mit offenem Integrations- oder Hardwarebeweis und `[-]`
steht für noch nicht umgesetzte Arbeit. Die Owner bezeichnen die autoritative
Komponente; Reviewer und Test-Owner sind die fachlichen Prüfrollen.

### A. Research, Semantik und Identität

| Status | Block | Owner | Reviewer | Abnahme |
|---|---|---|---|---|
| [~] | Research-/PTR-Abgleich, ADRs und Research-Gates | `docs`, `ptr-types` | architect | Research-Mapping ohne verlorene Annahmen |
| [~] | Cognitive Contract und versioniertes Codebook | `ptr-types` | verifier | Codebook-Digest, Validity-Masks, positive/negative Tests |
| [~] | `PodKind`, `ModelVariant`, `PodIdentity`, semantisches Manifest | `ptr-pods` | architect | immutable Manifest-Digest und Lineage |
| [~] | Principal-, Identity- und Session-Bindung | `ptr-types`, `ptr-runtime` | security reviewer | Ablauf, Revoke, Issuer/Audience und Session-Fencing |
| [~] | Namespace-, Projekt- und Capability-Isolation | `ptr-runtime`, `ptr-pods` | code-reviewer | Cross-project/cross-namespace deny tests |

### B. ExecutionManifest und Artifact-Lifecycle

| Status | Block | Owner | Reviewer | Abnahme |
|---|---|---|---|---|
| [~] | `ExecutionManifest` mit Knowledge-/Artifact-/Reader-Lineage | `ptr-runtime` | architect | vollständige Bindung, kanonischer Digest |
| [~] | Artifact Admission und Lifecycle-Gates | `ptr-pods`, `ptr-runtime` | verifier | Candidate→Trained→Evaluated→Approved→Active |
| [~] | Manifest Resolver und immutable Runtime Registry | `ptr-runtime` | code-reviewer | stale/revoked/conflicting generations fail-closed |
| [~] | Snapshot-/Policy-/Principal-Bindung | `ptr-runtime`, `ptr-semdb` | security reviewer | aktuelle Revision und Digest erforderlich |
| [-] | Training-/Replay-Trace ohne Secrets/Locator | `ptr-lineage`, `ptr-runtime` | research reviewer | reproduzierbarer Trace und Redaction-Tests |

### C. Pod-Adresse, Link und Routing

| Status | Block | Owner | Reviewer | Abnahme |
|---|---|---|---|---|
| [x] | `PodAddress`, `PodRevisionAddress`, `PodEndpoint` | `ptr-net` | architect | logische Identität bleibt bei Rebalance stabil |
| [~] | `PodDirectory`, `PodRoute`, Constraints und Health | `ptr-net`, `ptr-runtime` | code-reviewer | Endpoint wird nie vom Caller autorisiert vorgegeben |
| [~] | `PodLink`, ACL, Attestation, Egress-Policy | `ptr-pods`, `ptr-execwire` | security reviewer | Capability, Hop-Limit, Cycle- und Namespace-Schutz |
| [~] | Cache-Key aus logischer Identität und semantischer Revision | `ptr-pods`, `ptr-runtime` | test-engineer | Endpoint-Wechsel erhält Cache, Revision-Wechsel invalidiert |
| [-] | Replica-/Region-/Zone-Placement | `ptr-runtime`, `ptr-net` | architect | deterministische Auswahl und stale-writer-Deny |

### D. PodWire, Sessions und native Protokolle

| Status | Block | Owner | Reviewer | Abnahme |
|---|---|---|---|---|
| [~] | PodWire V1-Kompatibilität und V2/V3 typed Header | `ptr-podwire` | code-reviewer | V1 unverändert, V2/V3 bindet Manifest/Generation |
| [~] | langlebige `PodSession`, Streams, bounded queues | `ptr-podwire`, `ptr-net` | test-engineer | Backpressure, timeout, cancellation, dedupe |
| [~] | RequestUncertain und Runtime-Recovery-Adapter | `ptr-podwire`, `ptr-runtime` | verifier | kein automatischer Stateful-Retry |
| [-] | Duplex-/Turn-Events und Resume-State-Machine | `ptr-pods`, `ptr-podwire` | architect | Partial, interrupt, resume und terminal events |
| [~] | TCP-/UDP-/Iroh-/HTTP-/WebSocket-/gRPC-Bindings | `ptr-net`, `ptr-pods` | dependency-expert | typed framing, deadline, peer binding |
| [-] | SSH-/VPN-/WireGuard-/MQTT-/WebRTC-Adapter | `ptr-net`, `ptr-execwire` | security reviewer | Effect-Admission, typed session state, teardown |
| [~] | MCP-Kompatibilitätsadapter | `ptr-pods` | architect | MCP bleibt Adapter, nicht Kernautorität |

### E. Runtime, Scopes, Leases und Recovery

| Status | Block | Owner | Reviewer | Abnahme |
|---|---|---|---|---|
| [~] | Scope-Hierarchie Session→Turn→PodCall→Effect | `ptr-runtime` | architect | Parent-Bindung und terminal lifecycle |
| [~] | Journal-first Scope-/Lease-Lifecycle | `ptr-runtime`, `ptr-ledger` | verifier | Append-Fehler mutiert keinen Materialized State |
| [~] | Placement-Epoch und Fencing-Token | `ptr-runtime` | security reviewer | alter Writer kann nicht continue/append/release |
| [~] | Cleanup-Coordinator und Recovery-Pending | `ptr-runtime` | test-engineer | Intake→Children→Queues→Pod→Resource→Session |
| [~] | Identity-/Policy-/Snapshot-Authority | `ptr-runtime`, `ptr-semdb` | code-reviewer | Replay, Revocation, fail-closed Admission |
| [~] | OIDC/JWKS und Policy-Bundle-HTTP-Loader | `ptr-runtime` | security reviewer | HTTPS, bounded body, unknown-kid fail-closed |
| [-] | vollständige Identity-to-Recovery-E2E | `ptr-runtime`, `ptr-podwire` | test-engineer | Reopen, RequestUncertain, neue Admission/Epoch |

### F. Output, Evidence und Branches

| Status | Block | Owner | Reviewer | Abnahme |
|---|---|---|---|---|
| [~] | `PodOutput` und kanonischer Output-Digest | `ptr-pods`, `ptr-runtime` | verifier | Manifest/Artifact/Generation/Provenance gebunden |
| [~] | `PodEvidenceBundle` und durable Admission-Event | `ptr-runtime`, `ptr-ledger` | code-reviewer | idempotent, widersprüchliche Wiederholung abgewiesen |
| [~] | Observation/Candidate/ToolResult-Routing | `ptr-runtime`, `ptr-semdb` | test-engineer | nicht autoritativ, kein Überschreiben bestehender Fakten |
| [~] | Hypothesis/Branch-Persistenz | `ptr-branch`, `ptr-runtime` | architect | Evidence vollständig und replaybar |
| [~] | `merge_branch` als einzige Veröffentlichung | `ptr-runtime`, `ptr-branch` | verifier | Verifier, Lineage, Confidence und deterministischer Tie-Break |
| [~] | typed ActionProposal→ActionIr→ExecWire | `ptr-runtime`, `ptr-execwire` | security reviewer | keine direkte Pod-Wirkung |

### G. Knowledge, Retention und Context

| Status | Block | Owner | Reviewer | Abnahme |
|---|---|---|---|---|
| [~] | immutable RawEvent-Log und Content-Digest | `ptr-memory` | verifier | Append-only, identisches Re-Append idempotent |
| [~] | Ingestion: Tool-Paare, Entity, Namespace, State, Relation | `ptr-memory` | architect | deterministisch und source-validierend |
| [~] | Knowledge-Generationen und Dependency-Lineage | `ptr-memory` | code-reviewer | Generation-2-Promotion und Generation-1-Invalidation |
| [~] | Jev-kompatible Retention mit Pins/Fallback | `ptr-memory` | test-engineer | Classifier-Fehler bewahrt Verbatim-Kontext |
| [~] | Context Compiler und vollständiger Context-Digest | `ptr-memory` | verifier | deterministische Ordnung, Budget, keine freie Summary |
| [-] | lokaler Neural-/Hybrid-/Replay-Retention-Classifier | `ptr-memory` | research reviewer | reproduzierbarer Replay und Fehlermetriken |

### H. KV-State und Model Runtime

| Status | Block | Owner | Reviewer | Abnahme |
|---|---|---|---|---|
| [~] | opaque KV-Handles, Invalidation und Recompute | `ptr-memory` | verifier | alte Handles unbrauchbar, unabhängige States gültig |
| [~] | Placement-/Fencing-gebundene KV-Registry | `ptr-runtime`, `ptr-memory` | security reviewer | stale Epoch/Token fail-closed |
| [~] | echter layerweiser KV-Tensor-Cache | `ptr-pods`, `ptr-runtime` | architect | Primitive vorhanden; Modell-Continuation, Paging und Tiering offen |
| [~] | Candle-Safetensors-Dense-Executor | `ptr-pods` | test-engineer | Admission von Weight/Bias/Dtype/Shape |
| [-] | Candle-CUDA Transformer-/KV-Executor | `ptr-pods` | dependency-expert | tatsächlicher RTX-3090/4090-Nachweis |
| [-] | Device-/Stream-/Tensor-Lifetime und OOM-Recovery | `ptr-pods`, `ptr-runtime` | security reviewer | RAII, sync vor Release, kein Raw Pointer nach außen |

### I. Durable Storage und geschützter State

| Status | Block | Owner | Reviewer | Abnahme |
|---|---|---|---|---|
| [x] | Ledger-Replay, AEAD und Generation Anchors | `ptr-ledger`, `ptr-storage` | verifier | Tamper-/Rollback-/Key-Revoke-Tests |
| [~] | Protected Ledger/Knowledge/Artifact/KV-Snapshot-Koordination | `ptr-runtime` | code-reviewer | Live-Digest=Protected-Digest=Manifest-Digest |
| [~] | Ed25519 Policy-/Evidence-Signer und Verifier-Adapter | `ptr-runtime`, `ptr-verifier` | security reviewer | falsche Signatur/Key/Revision fail-closed |
| [x] | generischer Chunk-/Manifest-Vertrag | `ptr-storage` | verifier | kanonische Digests, Verify-on-read, immutable Content-ID |
| [x] | CPU↔NVMe Residency, Pins, Dedupe und Drain | `ptr-storage`, `ptr-runtime` | test-engineer | Source bleibt bis Ziel-Commit gültig; Reopen ist replaybar |
| [~] | OpenDAL ObjectStore | `ptr-storage` | dependency-expert | Memory/Filesystem getestet, S3 kompiliert; echte S3-Abnahme offen |
| [x] | geschützte durable KV-Chunks | `ptr-storage`, `ptr-pods` | security reviewer | AEAD, Generation/Revision/Anchor und Chunk-Digest gebunden |
| [~] | paged CUDA-COW und GPU Residency | `ptr-pods`, `ptr-runtime` | GPU reviewer | F32 Page-Tensors, Stream-Sync, COW, compact-GQA, Paged-Tier-V3 und RTX-4090-Roundtrip implementiert; tensor-native Qwen/fused Attention offen |
| [-] | Mooncake-artiger Remote-Transfer | `ptr-runtime`, `ptr-net` | distributed reviewer | Prefill/Decode, RDMA/QUIC, fail-closed fencing |
| [-] | unabhängige Snapshot-Anker und Compaction-Floor | `ptr-ledger` | architect | Retention verliert keine At-most-once-Schlüssel |

### J. Mesh, Sandbox und Plattform

| Status | Block | Owner | Reviewer | Abnahme |
|---|---|---|---|---|
| [~] | MeshMembership, Invitation, Revocation, RouteRevision | `ptr-net`, `ptr-ledger` | security reviewer | Revoked Peer ist nicht mehr auflösbar |
| [~] | Iroh Direct/Relay-Vertrag | `ptr-net` | dependency-expert | Relay disabled default, Direct→Relay Recovery |
| [~] | WireGuard Userspace-Vertrag | `ptr-net` | code-reviewer | Reference Executor und Fencing |
| [-] | privilegierter Linux-Kernel-WireGuard-Adapter | `ptr-net` | platform reviewer | echte Interface-/Peer-/Teardown-Abnahme |
| [-] | Windows Wintun- und macOS Network-Extension-Adapter | `ptr-net` | platform reviewer | zielplattformabhängige Compile-/Hardwaretests |
| [~] | native Sandbox-/Environment-Executor | `ptr-runtime`, `ptr-pods` | security reviewer | Profile, Policy, Timeout, Cleanup |
| [-] | WinRsBox-/smolvm-Adapter | `ptr-runtime`, `ptr-net` | architect | Contract-Tests ohne Core-Abhängigkeit |

### K. Evidence, Training und Abschluss

| Status | Block | Owner | Reviewer | Abnahme |
|---|---|---|---|---|
| [-] | KV-streams/Retention-Evaluation | `research`, `ptr-memory` | research reviewer | Recall, Recompute, Invalidation getrennt messen |
| [-] | Forking-Sequences/Multi-Horizon-Trace | `ptr-runtime`, `ptr-branch` | research reviewer | Branch/Hypothesis-Verlauf replaybar |
| [-] | Training-/Replay-Dataset aus validierten Outputs | `ptr-lineage`, `ptr-runtime` | data reviewer | nur accepted/merged outcomes, keine Secrets |
| [-] | vollständige Unit-/Integration-/E2E-Abdeckung | `tests`, alle Owner | test-engineer | positive und negative Pfade pro Block |
| [~] | Format, Check, Clippy, Workspace-Test und Diff-Check | `CI` | verifier | vollständige Läufe ohne Timeout/Cargo-Konkurrenz |
| [-] | Commit-/Push-/Release-Nachweis pro Abschlussblock | `git`, alle Owner | code-reviewer | kleiner Commit, reproduzierbarer Testbeleg |

### Abhängigkeitsreihenfolge

```text
Research/Types
  → ExecutionManifest/Lineage
  → Address/PodLink/Protocol Header
  → Runtime Admission/Scopes/Fencing
  → PodWire Sessions/Native Executors
  → Output/Evidence/Branch/ExecWire
  → Knowledge/Context/KV
  → Mesh/Sandbox/Platform
  → Training/Replay/Hardware/E2E
```

Jeder Block wird erst als abgeschlossen markiert, wenn Implementation,
positive und negative Tests, aktualisierte Architektur-Dokumentation und ein
frischer Verifikationsnachweis vorhanden sind. Hardware-, privilegierte
Netzwerk- und externe-Provider-Tests bleiben ausdrücklich als separate Gates
markiert und werden nicht durch reine Compile- oder Mock-Ergebnisse ersetzt.

### KV-Research-Mapping

Der KV-Block baut direkt auf den gelieferten Research-Ergebnissen auf:

| Research-Einsicht | PTR-Umsetzung | aktueller Stand |
|---|---|---|
| KV-streams: Kompaktion kann Modellzustand weiterführen, ohne alten Text neu zu formulieren | `KvStateRuntime`, opaque Handles, Invalidation und Recompute | Metadata-, Tensor- und Qwen2-Decoder-Continuation vorhanden; produktive Multi-Layer-/Compaction-Evaluation offen |
| PagedAttention: KV wird block-/seitenweise statt als eine wachsende Sequenz verwaltet | page/block allocator und Prefix-Sharing im `KvTensorBackend` | F32 CPU-/Candle-Page-Bundles, sealed Pages, Refcounts, principal-/manifestgebundenes Prefix-Sharing und COW implementiert; fused paged Attention offen |
| SGLang KV-Quantisierung/NVFP4: Dtype, Skalierung und fused Attention sind ein Vertrag | typed Dtype-/Scale-Metadaten und quantized Backend-Capabilities | aktuell nur F32; FP8/FP4/NVFP4 und fused-kernel-Gate offen |
| HiCache: GPU-, CPU- und Storage-Tiers mit kontrolliertem Attach/Detach | generischer `TierBackend`, `TierResidencyController`, Pins, Drain und Lookahead-Policy | Candle-GPU-, CPU-, NVMe- und OpenDAL-Tiers vorhanden; Page-V3-Roundtrip und neue Fencing-Lease verifiziert, Heat-Policy offen |
| Mooncake: disaggregiertes Prefill/Decode und verteilter KV-Pool | Placement-/Fencing-gebundene Snapshot-/Transfer-Schnittstelle | lokale Placement-Verträge vorhanden; Pool, Transfer Engine und Prefill/Decode-Trennung offen |
| fast-jev/Retention: behalten, strukturieren, auslagern und invalidieren sind getrennte Entscheidungen | Retention Controller plus Knowledge-/Context-/KV-Lifecycle | Rule-Modell vorhanden; lokaler Neural-/Hybrid-Controller und Evaluation offen |

`CandlePagedKvBackend` verwaltet F32-K/V jetzt als layerübergreifende
Page-Bundles auf einem streamgebundenen CUDA-Device. Der Qwen2/Qwen2.5-
Korrektheitspfad implementiert RMSNorm, half-split RoPE, kompakte Grouped Query
Attention, QKV-Biases, SwiGLU und Residual-Reihenfolge mit den kanonischen
Safetensors-Namen. Prefill/Decode wird gegen vollständigen Recompute geprüft.
Die skalare Modellmathematik materialisiert derzeit jedoch noch eine Host-Sicht;
ein tensor-nativer Multi-Layer-Qwen-Produktions-Executor ist damit nicht belegt. Die
Quantisierungsforschung zeigt außerdem, dass Speicherersparnis ohne passenden
fused Attention-Backend erhebliche Laufzeitverluste erzeugen kann. Deshalb
braucht dieser Pfad ein eigenes Accuracy-/Throughput-Gate und darf nicht nur
über einen erfolgreichen Compile-Check als abgeschlossen gelten.

### Multi-Tier-Speicherhierarchie

Der implementierte vertikale Slice trennt logische Objektidentität strikt von
Residency. `TierObjectManifest` bindet Domain, Generation, Revision, Schema und
kanonische Chunk-Digests. `TierResidencyController` verwaltet verifizierte
Replicas, Reader-Pins, In-flight-Deduplizierung, Transferbudgets sowie
online Attach und drain-basiertes Detach. Backend-, Objekt- und
Replica-Lifecycle werden mit stabilen Ledger-Tags replaybar journalisiert.
Der Produktionspfad koppelt physische Attach-, Register- und Transfer-Vorgänge
über die journal-first Runtime-Fassade an diese Events; die direkten
Controller-Operationen bleiben der deterministische Backend-/Testvertrag.

Praktisch verifiziert sind:

- immutable CPU-Replicas und atomisch veröffentlichte NVMe-Chunks;
- OpenDAL Memory- und Filesystem-Roundtrips sowie S3-Feature-Compilation;
- per-State `ManagedKvRegistry` Snapshot → CPU → NVMe → Restore unter neuer
  State-ID und neuer Write-Lease;
- page-ausgerichtetes `PTRKVP3` mit expliziter Query-/KV-Head-Bindung, einem
  Header-Chunk und einem verifizierten Chunk pro KV-Page; `PTRKVP2` bleibt
  lesbar und wird beim Restore auf den neuen Digest-Vertrag gehoben;
- Candle-GPU → CPU → NVMe → Candle-GPU auf der lokalen RTX 4090, einschließlich
  neuer Fencing-Lease und digestgeprüftem Restore;
- AEAD-geschützte KV-Chunks mit Generation-/Revision-/Anchor-Bindung;
- generationsübergreifende Anchor-Ketten und authentisierte Anchor-Metadaten;
- deterministic LRU, begrenztes Lookahead-Prefetch und Mindest-Replica-Pläne;
- Fail-closed Verhalten bei Corruption, stale Fencing, Pins und
  Transfer-Backpressure sowie cancellation-sichere Transfer-Reservierungen;
- CI-Abdeckung für OpenDAL Memory/Filesystem und Compile-/Clippy-Gates für S3.

Noch keine Abschlussbehauptung besteht für einen vollständigen tensor-nativen
Qwen-LM-Stack mit Embedding, allen Decoder-Layern und LM-Head, einen eigenen
unsafe CUDA-Slab, quantisierte/fused Attention, produktive S3-Dienste oder
Mooncake-artigen Node-zu-Node-Transfer. Die RTX-4090-Abnahme belegt nur den
implementierten F32-Candle-Page-/Tier-Pfad; eine RTX-3090 wurde nicht geprüft.
