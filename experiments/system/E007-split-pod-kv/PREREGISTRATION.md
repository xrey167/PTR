# E007 split-pod-kv (the CPU part): preregistration

Frozen text of the preregistration of E007. It is a `file` key of `config.toml`'s `[preregistration]` table, so its SHA-256 is frozen
with the table. Nothing in it is a result. The decisions are made by `criteria.toml` (frozen the same way) and applied by `aggregate.py`;
this file gives their reasons, the modelling choices and what the harness cannot produce. The harness is
`bins/ptr-bench/src/experiments/e007/` (subcommand `split-pod-kv <seed>`).

Entrypoint (`experiment.toml`'s `entrypoint`, the runner's command):
`cargo run --release --locked -p ptr-bench -- split-pod-kv <seed>`; preflight: the same with `--preflight` instead of `<seed>`.

## 0. Scope, and what needs a GPU

**Scope, stated before any declared seed ran (owner decision of 2026-10-09: E007 is the core, the extensions go to a successor).**
E007 tests, on the CPU reference path and in one process, the mechanism of a pipeline (layer) split across two pods (doc 36, section D and
phase P5a): the stage execution of the tiny seeded stack, the activation-frame schemas and their allowlist, the bounded chunking and the
reassembling receiver, and the downstream stage cache that revalidates the upstream chain at every use. Its claims are the ones in
`experiment.toml`. The code under test is new product code, written for this experiment as a separate change (section 8).

**Character.** E007 is a regression guard and a structural exclusion, not an effect measurement. It tests code written for it in the same
change, its a-priori expected outcome is a pass on every seed, and its hypothesis is an exclusion: that none of the enumerated probe kinds of
this in-process harness (the smuggling, chunk, replay, deadline, identity, byte-cap and invalidation probes of section 4, and nothing else)
gets KV or a stale entry across the split. A pass is the expected result and guards those exclusions; the falsifiability is carried by the
probes and by the planted mutations (section 7), which show that each counter can fire. It says nothing about probes it does not contain, a
real transport, a real checkpoint or an adversary with other capabilities.

**What is CPU-feasible, and is therefore here.** Everything in the hypothesis: bit-identity of a 4-layer, 64-wide seeded F32 model split at
layer 2 against the single-pod run (the layer code is shared, so this checks the plumbing and not the math); the 16 MiB frame as a
transport-only case; the allowlist and the smuggling probes; the chunk protocol with its tombstones, bounds, deadlines and replay protection;
the revalidation of the upstream stage generation, manifest digest and placement epoch under revocation, supersession and re-placement with
the `Revoke` message normal, dropped, delayed and in flight. The transport is the real PodWire request frame codec (`encode_request`,
`decode_request`, `MAX_BODY_BYTES` enforced) in process; the clock is virtual.

**What needs a GPU or a real model (not here; E008 Part C).**
1. A real checkpoint split at an internal layer, on the `candle-cuda` path: whether two device contexts reproduce the single-device logits
   bit for bit, which depends on the kernels (reduction order, batch invariance) and is exactly what a tiny CPU model cannot show.
2. The F16 and BF16 boundary dtypes and the cast: the reference path computes in F32 only (`crates/ptr-pods/src/kv.rs`), so the claim "the
   baseline applies the identical boundary cast" is a no-op here and a real statement only on a GPU model.
3. The byte volumes: activation bytes per token against the KV bytes the downstream pod keeps, for a real model width, and the 2048-wide
   frame as a real activation and not as transport-only bytes.
4. Latency and throughput of the split against one pod: reported, never gated.

**What is CPU-feasible but is not here (E008 Parts A and B).** The extension kinds moved out by the owner decision: input lineage and Raw
ancestry in the frame and in the cache key, production mode and taint, persistence rejection, and the expert, adapter, base-model, semantic,
memory and derived kinds of the invalidation matrix (45 cells); and the transport over loopback iroh (`PodHost`/`PodClient` with
`podwire-backend`), which needs a socket and a Pod adapter around the assembler that this package did not build.

**Limits that stay stated.**
- Bit-identity checks plumbing, not math: the split and the single-pod run call the same per-layer function (`forward_layers`). A defect in a
  layer's arithmetic is invisible to it by construction.
- Frames cross the codec in process: no socket, no iroh, no second host, no real clock (skew is a tolerance, not a simulated drift).
- The chunk envelope is authenticated with an Ed25519 signature registered per authenticated principal; key distribution and the binding of a
  principal to a connection are the caller's and are not modelled.
- Only the upstream stage is exercised as an element of the chain. The base model, adapters and experts are carried, signed and revalidated
  by the product (unit-tested in `split_stage.rs`), but E007 has no case for them and no mutation of their revalidation (E008).
- The downstream cache has no persistence at all, so the restore path of the invalidation matrix does not apply (doc 36 says so).
- **Frame rate.** A completed frame keeps one of a principal's 64 table slots until its signed deadline plus the skew bound, so one principal
  sustains at most `64 / (deadline budget + skew)` frames per second (58 per second at the 100 ms budget of the cases, about 11 per second
  at the maximum lifetime of 5000 ms). One frame per decoded token therefore needs short signed deadlines. This is a property of the design as
  specified, found while building the harness; the cases stay inside it (at most 45 slots in use) and the relation is a test in `params.rs`.
- Synthetic seeded token sequences, one machine profile.

## 1. Decision rule

Five seeds (17, 29, 43, 71, 101), one JSON object each, `criteria.toml` applied mechanically (a "< N" prerequisite is written `>= N`, a
"> N" violation `<= N`). The rules have three lists: coverage (the run was not vacuous), falsifiers (observed violations by the product) and
validity (the record describes the frozen run: the model pin, the corpus selection digest, the procedure pins, the late replay window).

**The four outcomes** are COMPLETED, FALSIFIED, NOT_SHOWN and INVALID. E007 is deterministic and has no sampled margin, so no path ends in
NOT_SHOWN; every combination of the five seeds ends in exactly one of the other three. Precedence, highest first:
1. **INVALID, validity first**, for a seed whose validity rule fails (the model pin, the corpus selection digest, the procedure pins or the late
   replay window differs from the frozen run), **whatever else fired**: the run did not follow the preregistered procedure, so its product
   observations say nothing about the product. Its falsifier failures are listed as ignored (`ignored_falsifiers`) and neither falsify nor make
   the seed vacuous. A validity failure whose cause also makes a product counter fire (a receiver built with another reorder window than the pin
   lets an out-of-window attack through, so `attempts_without_typed_rejection` fires too) is therefore INVALID, not FALSIFIED; the record shows
   both lists. The consequence is accepted: a pin that drifts between `config.toml` and the code turns the seed into INVALID, never into a
   finding about the product, and never into a pass.
2. With every validity rule holding, **FALSIFIED** if any falsifier rule fails in any readable record, **even a record that is INVALID for another
   reason** (a control that did not trip, a harness gap, a case that could not run in the same run): a product defect is never turned into
   INVALID by those. This is the order of the repository's L003 and L004 (the hard failure before the incomplete coverage). It also applies if
   a *valid* record fails a coverage rule (the run is vacuous, which counts as falsified); the coverage failures of an INVALID record do not
   falsify. One difference from L003 and L004: they read any nonzero exit as a hard failure; E007 reads exit 2 as the harness (no result,
   INVALID), per the repository rule that exit 2 is the harness and exit 1 a product defect.
3. **INVALID** if a seed has no readable record, a foreign one (another benchmark, seed or scale), a control that did not trip (and no falsifier
   fired), a harness gap, a counter that is missing or mistyped, or an exit status the rules do not explain; or if `results/mutations.json` is
   absent or does not count every mutation killed.
4. **COMPLETED** iff every seed meets every coverage and validity rule, no falsifier fails, and `results/mutations.json` counts every mutation of
   `tests/mutations.toml` killed.

The harness exit status follows the same order. It exits 2 when a validity counter fired (a pin, the selection, the procedure or the late replay
window differs from the frozen run), whatever else fired. With every validity counter at 0, it exits 1 when a falsifier fired from an **observed
product violation**, also when a case could not run or a control did not trip in the same run (a control that runs the product code with its
protection removed cannot tell that from a product whose protection a defect has already removed, so the violation is the finding and the record
is printed); it exits 2 when the harness is otherwise at fault: a case that cannot run or a control that does not trip while no observed
violation fired, a fixture that does not build. The pins, the selection, the procedure, the late window and the sent-versus-observed comparisons
(they only say what an aborted case never observed) are not product findings and never make exit 1 on their own account. It exits 1 for a
vacuous run and 0 when every rule holds. Tests pin the three corners: validity plus falsifier is INVALID (exit 2); falsifier plus a dead control
with clean validity is FALSIFIED (exit 1); a dead control alone is INVALID (exit 2).
**A product regression that kills a real-code control without firing any falsifier therefore surfaces as INVALID** (exit 2), not as a pass and not
as FALSIFIED: the plan has no falsifier that names it, and this is stated as a limit of the plan. The mutation run shows that the regressions it
plants all fire a falsifier.

Seeds: the corpus is a pure function of `corpus_generator_seed`, so the seed does not change the sequences. It changes (a) the order in which
adjacent chunks of a valid frame arrive (disjoint adjacent pairs swapped at random, inside the reorder window), (b) the content of the 16 MiB
frame, (c) the late-replay delay inside [9000, 10000) ms and (d) the corpus of ordering inside the cases that draw. The five seeds are
therefore five different interleavings and delays over one corpus, and they are a check of the schedule, not of the arithmetic (section 6).

## 2. The cases

All of them run for every seed; all counters are read from events the run observed.

- **C1 bit-identity.** 24 prompts of 8 to 64 tokens (`8 + (7 i mod 57)`), prefill plus 7 decode steps (8 frames), three ways: the whole stack on
  one pod (`Qwen2PagedStack::prefill_in_regime`/`decode_in_regime`, the baseline); the same two stages in process through the boundary cast
  (`bytes_f32(f32_bytes(x))`, a no-op in F32); and the two stages with every frame cut into 2048-byte signed chunks, sent over the PodWire
  codec, reassembled, admitted against the live ledger and placement, and fed to stage B. The output is the logits bit patterns of every prompt
  row and decode step plus the K and V bytes of every layer (layers 0-1 on pod A, 2-3 on pod B); any difference is `bitwise_mismatch_count`.
  A valid frame that does not arrive is `valid_frame_bitwise_mismatches`. For every wire frame all chunks must carry one frame digest equal to
  the digest of the bytes sent, computed in the harness (`consistent_digest_frame_id_set_digest` against `valid_sent_frame_id_set_digest`).
- **C1b downstream reuse.** For three sequences the downstream cache stores stage B's cache snapshot under the prompt frame; a use is begun, the
  entry restored and the decode frames continued; the continuation equals the original run bit for bit and the use finishes `Ok`.
- **C2 large frame.** 2048 x 2048 F32 (16 MiB) seeded bytes. Control: sent as one payload, PodWire's codec must refuse it. Then five chunks of
  at most 3.5 MiB in the preregistered order 1, 0, 3, 2, 4 (inside the window of 4); the reassembled bytes must equal the bytes sent and the
  frame digest, recomputed in the harness from SHA-256 and the documented domain, must equal the receiver's.
- **C3 allowlist and smuggling.** Two pods hold real KV after a 48-token prompt. Probes: a KV page mislabelled as the chunk type and as the
  policy type; a KV page under three type ids that are not allowlisted, and a valid signed chunk under a fourth ("ptr.tensor.blob.v1"); inside a
  signed, digest-consistent frame, a KV page appended after a valid frame (trailing bytes), a KV page as the tensor under a header of another
  shape (wrong shape), and a right-shaped tensor with one position too few and one too many. Each ends in the typed rejection the table pins.
  The **control** is an activation-shaped KV payload (16 rows of a layer's K and V as a 16 x 64 frame): valid, admitted, fed to stage B as
  activations; the rows of the smuggled payload are counted in stage B's cache before and after, and the delta must be 0
  (`activation_control_kv_installs`, `kv_pages_installed_from_wire`), while the cache grows by exactly the rows computed.
- **C4 attacks** (each on a fresh receiver, each one a payload a sender with the key, or one holding a valid chunk, could send): a chunk
  signed by another key and one with a flipped signature bit; chain mismatch; duplicate chunk; a chunk outside the window; corrupted bytes under
  an intact envelope and an envelope re-signed over a wrong digest; a frame announced with another chunk count; a splice of two frames under
  one id; a chunk relabelled into another parent operation (without the key, and with it: it must open a frame of its own and leave the first
  frame's delivery bit-identical); all chunks carrying one consistent wrong frame digest; an expired deadline; the five relabellings (frame id,
  operation, index, count, deadline) of a completed frame's chunk. Bounds: the ninth incomplete frame; the 33rd buffered chunk.
- **C5 load.** Replays of a completed frame under the same and another session with fresh request ids; one tuple assembled concurrently from
  two sessions (exactly one delivery); abandonment of a frame with chunks buffered, explicit and by expiry (released chunk ids against sent
  chunk ids, buffered bytes before and after, the remaining chunks and an immediate and a late (within [9000, 10000) ms) replay refused, the
  next frame reassembles); eight admitted frames abandoned together; saturation (64 frames opened and abandoned at 10 ms intervals, the 65th
  refused with `FrameTableFull`, every tombstone still refusing a replay up to the first retention, then exactly one new frame accepted and the
  next refused); shared slots (60 tombstones plus 4 incomplete frames, the next refused); two frames with one chain interleaved.
- **C6 stale upstream.** For each of revocation and supersession and each of normal, lag and in-flight (6 cells; lag and in-flight run the
  `Revoke` message dropped and delayed or delivered), a fresh ledger, placement and downstream entry; the stage generation is invalidated in the
  ledger (`Revoked`, `CapsuleSuperseded`) and the entry is used. Before the invalidation the entry serves and its continuation is bit-identical.
  A stale hit, a use that starts on the entry, or an output released after the invalidation, is a stale hit of the cell. A frame of the old chain
  re-sent after the invalidation is refused (`NotLive`). After a supersession the new generation's frames are admitted and its entry serves.
  Stale worker: the stage is re-placed (`mark_node_failed`, `assign`) with the generation unchanged; the old worker's frame and the entry derived
  under the old placement are refused (`StalePlacement`), a frame with a forged manifest digest is refused (`DigestMismatch`), the new worker's
  frame is admitted.
- **C7 pins.** Model shape, intermediate size, rope theta, key encoding, the tiny-model config digest, the corpus selection digest and the
  receiver's limits as observed from the assembler, against the pins (`model_pin_mismatches`, `selection_digest_mismatches`,
  `procedure_mismatches`; with `late_window_delays_out_of_range` these are the validity rules: a difference makes the seed INVALID).

## 3. Pins, and how each number was derived (planning model, not a pilot)

| value | pinned | derivation |
|---|---|---|
| model | 4 layers, hidden 64, 4 heads, 2 KV heads, head 16, vocabulary 256, intermediate 128, weight_seed 20240607, split_layer 2 | the R005 tiny model (`layered_kv::pins`), digest `sha256:91be3ab2...` reproduced by a test; the split is internal because `0 < 2 < 4` |
| boundary dtype | F32 | the reference path is F32 only |
| corpus | 24 sequences, 8 to 64 prompt tokens, 8 frames each | 192 frames per seed; 24 sequences cover prompt lengths from one page of K/V to four (page 16) and chunk counts of 1 to 9 per prompt frame |
| corpus selection digest | `sha256:52efb0dee3e036edc77172632c28a2ba072ce353fdce3cee55d380402e530aba` | the digest of the 24 sequences the deterministic generator (seed, index, length) produces, domain `ptr.e007.corpus-selection.v1`; a function of the generator and the pinned lengths, read once from the first run and then pinned (section 10); every run recomputes it and any difference is `selection_digest_mismatches` |
| frames crossed, valid frames sent | at least 192 | 24 sequences x (1 prompt frame + 7 decode frames); the large frame and the probes come on top |
| chunk size of valid frames | 2048 B | below a KV page's size (2048 B = 16 rows x 128 B), so a chunk can carry at most a page, never a page and its neighbour: the scan for a page window is meaningful |
| frame budget, spacing | 100 ms, 25 ms | `(budget + skew) / spacing = 44` slots in use against 64: the stream stays inside the table bound by derivation |
| large frame | 2048 x 2048 F32 = 16 777 216 B in 5 chunks of at most 3 670 016 B (3.5 MiB) | `ceil(16 MiB / 3.5 MiB) = 5`; each chunk plus its header stays under `MAX_BODY_BYTES` (4 MiB); order 1, 0, 3, 2, 4 keeps every index below `position + 4` |
| reorder window, deadline, per-peer incomplete frames, buffered chunks | 4, 5000 ms, 8, 32 | the preregistered values |
| tombstones per principal | 64 | the preregistered value; the saturation case opens 65 |
| clock skew, deadline lifetime, replay window | 1000, 5000, 10 000 ms | `10 000 >= 5000 + 2 x 1000`, checked by `AssemblyPolicy::validate`; the late replay is drawn from [0.9 W, W) = [9000, 10 000) |
| saturation stagger | 10 ms | `63 x 10 + 100 < 10 000`: all 64 abandonments fall inside one replay window, and the first expires 10 ms before the second |
| signed envelope | Ed25519 over frame id, parent operation, chunk index and count, deadline, frame digest, chunk digest, chain | a signature, as the design allows "a signature or a MAC"; chosen because `ed25519-dalek` is already a workspace dependency |
| session policy | chunks of one tuple from any session of the principal join one assembly | the design allows "rejected typed or joins"; joining lets a reconnect continue a frame; exactly one delivery is what is tested |

Thresholds on attempts are 1 per kind by design (each attack kind is performed once or more by construction of the case list); no threshold
was set from a pilot count. One was corrected after the first pilot (section 10).

## 4. How the measurements are taken

- **Wire KV.** Every payload crossing the codec is recorded with its role (valid, activation control, probe) before the receiver answers.
  `wire_kv_page_bytes` is, over admitted valid payloads only, the bytes of any payload whose type is not on the allowlist or whose body equals
  or contains (any offset) a KV page of either pod; `wire_kv_handle_count` counts admitted valid payloads containing the digest or a state
  name of either pod's cache. Every chunk is scanned as sent and every valid frame as the receiver reassembled it (a page of 2048 B can straddle two chunks of 2048 B and is whole only in the frame; found when the mutation `stage-a-ships-a-kv-page` was not killed, section 10). The scan reads the pods' caches through their snapshots and the capture, never the admission code. Probe bytes
  and the control are excluded, as the design says. The capture itself is the `wire_capture` object of the run record (the runner records
  stdout only, so a separate `wire_capture.json` file is not an artifact).
- **Typed rejection.** An attempt is typed if the answer is a `Rejection` (or `ChainFault`); it is judged against the variant a hand-written
  table in the cases pins (`verdict_mismatches`). An attempt the receiver admits is `attempts_without_typed_rejection`. The table is the oracle:
  it is written from the design, not read from the code's answers.
- **Independence.** The 16 MiB frame digest and the per-frame digest consistency are recomputed in the harness from SHA-256 and the documented
  domain; the KV pages are scanned from the pods' snapshots; the identity-set digests are over the ids the harness sent and the ids the receiver
  reports released. The release is reported by the receiver (the one thing the harness cannot see from outside) and compared with what was sent.
- **Tombstone retention.** After the table is full, every abandoned frame's chunk is replayed just before the first retention ends and must be
  refused `FrameAbandoned`; the table's retained count must stay 64 (`tombstone_evictions_before_retention`).

## 5. Baselines and controls

- **B0 the single-pod run**, and the same two stages in process through the cast: both are the reference for C1.
- **Five controls**, naive configurations that the same probe must fool (`control_trips`, a coverage rule per control). Two run the **product
  code with the protection removed**: the real `SplitStageCache` fed a liveness frozen at storing (the ledger revokes the stage, the `Revoke`
  message is dropped, the naive cache still serves while the real liveness refuses the same lookup), and the real `FrameAssembler` that
  forgets a frame the moment it completes (the assembler is rebuilt, as a process that lost its state would be, so the replay of the real
  chunks is delivered again while the unmodified assembler delivers once). **They share the shape of the probes, not their code path**: each
  control builds its own small scenario with the same real objects (the naive cache does not run the cell code of the stale-upstream case, the
  forgetful assembler does not run the cases of C5), so a control that trips shows that the kind of probe can tell the protection from its
  absence, not that the probe of the cases is the one. Two are **harness-only reference models**, because the product has no such mode and
  none was added to keep the API honest: a reassembler keyed by the upstream chain alone (fed the real interleaved chunks, it mixes them) and a
  deny-list admission (it takes a real KV page under another name, which the real allowlist refuses). The fifth is the real PodWire codec
  refusing the unchunked 16 MiB frame. A control that does not trip means the probe it shadows could not tell the protection from its absence.
  If no observed violation fired in that run, the run ends with status 2 and the aggregator calls the seed INVALID; if one did (and no validity
  counter did), the record is printed with exit 1 and the seed is FALSIFIED (section 1). A real-code control conflates "the probe discriminates" with "the product still
  holds the protection": a regression that removes the protection kills the control, and unless some falsifier names it the seed is INVALID.
  **The controls are not the evidence that a counter can fire**: that evidence is the planted mutations of section 7, 59 defects in the product
  and the harness, every one of which must be killed.

## 6. Power (sensitivity of the plan)

The model is deterministic and the corpus fixed, so five seeds are not five samples of one distribution; they are five schedules. Sensitivity
is therefore stated per mechanism:
- **Bit-identity.** A defect that perturbs a fraction `q` of the 192 frames per seed (a stage that mishandles rows beyond a position, a
  boundary that alters a value) is missed with probability `(1 - q)^192` per seed and `(1 - q)^960` over five. For `q = 1%` that is 0.145 per
  seed and `6 x 10^-5` over five; for `q = 0.1%` 0.83 per seed and 0.38 over five. A perturbation of one value of one row of one frame
  (`q ~ 0.5%` of frames if it is the first row of each prompt) is detected with probability 0.62 per seed, 0.99 over five; the planted
  mutation (`second-stage-perturbs-the-residual`, one `f32::EPSILON` on one element of every row of stage B) is the calibration: every frame is
  affected, and the question it answers is that the comparison is exact, not how rare a defect it finds. What the plan cannot see is a defect in
  arithmetic shared by the split and the single-pod run.
- **Adversarial kinds.** Each kind is attempted at least once per seed and the answer is exact, so a rejection that does not happen on a
  kind is seen with probability 1 on the first seed; the 59 mutations show that each protective mechanism, removed alone, is seen by a named
  counter.
- **Stale upstream.** Each of the six cells runs at least once per seed, lag and in-flight with two message variants; a revalidation that is
  missing is seen with probability 1 when the cell reaches the use, which the entry-before-invalidation count (>= 1) and the control that must
  serve a stale entry establish.
- **What no count here can establish:** that the receiver is safe against an input the cases do not generate (a fuzzed byte stream, a hostile
  timing). A seeded random-stream differential test against an independent model was not built (section 8).

## 7. Mutation plan

`tests/mutations.toml`: 59 mutations, **50 in the product** (`ptr-podwire` chunk assembler: 29; `ptr-protocol` activation schemas: 6;
`ptr-runtime` downstream stage cache: 14; `ptr-pods` stage execution: 1) and **9 in the harness** (a stage that
ships a KV page, a stage that ships a cache handle, a control row installed as KV, a consistency check that never holds, a late replay outside its
window, a changed corpus rule, a receiver window and a receiver total cap that differ from the pins, and a model pin that differs). The harness mutations show that the measurements are
not vacuous where the product is correct; they are not evidence about the product. Five of them (late window, corpus rule, receiver
window, receiver total cap, model pin) break the validity rules: the harness prints the record and exits 2 (validity first: the receiver-window
mutation also lets an out-of-window attack through, so a product counter fires too, and the exit is still 2), the aggregator reads INVALID, and
the plan says `expect_exit = 2` for them (`scripts/mutation_check.py` then counts exit 2 with the expected counter as killed, and exit 1 as not killed).
All other mutations are killed by exit 1 with an expected counter. Every hard counter (52) is expected by at least one mutation
and each of the six matrix cells has its own counter. The eleven mutations added in the fix round name the review fixes: the replay boundary
tick, the policy signature, the principal-to-stage binding, the signed chain against the frame's chain, the policy bound to its operation, the
global byte cap, the Exact-only admission, finiteness, the sweep of a full cache, the eviction of a refused entry and the downstream stage's own
revalidation. Listed as not killable, on purpose: removing the `Revoke` message path alone (it is an
optimisation under revalidation, which is the property under test); the expert fence epoch (the runtime keeps none, so a chain with an expert is
refused, covered by a real-runtime test only); the deadline-too-far bound and the byte bound of one frame (no case probes them); the
frame's positions against the cache position and the atomic advance of a failed stage run (unit-tested in `ptr-pods`; the harness's downstream
run does not pass the positions, and no backend fault can be injected into a row without weakening the product). The coverage rules have no
mutation evidence beyond the tests that set each counter to a violating value.

**Limits of the invariants, stated.** The downstream cache is bounded (`SplitStageCache::new(capacity)`, invariant 8) and its saturation is
probed only through the sweep of dead entries on a full cache. Invariant 4 (uncommitted state is never authoritative): liveness is read from
the committed lifecycle state of the runtime and no case drives a proposed-but-uncommitted revocation, so that path is not claimed. The set of
principals is bounded only by key registration, and the global byte cap bounds the memory they can hold together.

**More stated limits (round 3).** (a) `AdmittedFrame::revalidate` checks the frame's own upstream chain and placement only, not the downstream
stage's own chain, and nothing in the repository calls it outside tests: the prefill-without-ticket gap is closed by convention, and the harness
does not exercise it. (b) The assembler's clock never goes back, so one forward time spike (a bad clock step) pins it in the future: every later
chunk reads Expired. The direction is safe (nothing is wrongly admitted); the cost is liveness. (c) The rollback test of a failed stage run
injects a revoked cache, not a failure in the middle of a layer; that a cache with uneven layers truncates correctly is the backend's contract
and is not tested here. (d) The sent-versus-observed mirror counters are falsifier rules, so a printed record with a failed case and an observed
product fault also lists the mirrors an abort leaves unobserved (for example `abandoned_chunks_unreleased`) among the failed rules; the verdict is
unchanged (the product fault falsifies), only the list of failed rules is longer. (e) A key or stage-identity change drops what the old key
minted; the registration call is the trusted deployment API and cannot be reached from the wire.

**More stated limits (round 4).** (f) The chunk assembler's release buffer (the ids of the chunks of abandoned or expired frames, about 32 per
frame) is drained only by `expire()`: a receiver that only calls `offer()` grows it without bound, and no production caller exists (the harness
calls `expire()`); calling `expire()` is a duty of the receiver. (g) A `UseTicket` that is dropped without `abandon_use` or `finish_use` leaks its
slot (there is no `Drop` guard), and ticket ids restart at 1 in every cache, so a ticket of one cache passed to another frees that cache's use
with the same id (the owner's own finish then fails closed with `UnknownTicket`; nothing is released, liveness only). (h) In `qwen_stack` an
overflow of `position_offset + sequence_length` is reported as `CacheSchemaMismatch` (a misleading name) and one add (`position_offset + cached`)
stays unchecked because it is unreachable after the admission check on the same `&mut` cache. (i) The round-3 behaviours (the policy drop on a key
rotation, release-once of chunk ids, `abandon_use`) are covered by unit tests, not by planted mutations, and the E007 harness does not reach them.

## 8. What was built for this experiment (a separate, described product change) and what was not

The experiment could not run on the product as it stood: doc 36 lists `ActivationFrame` among the "proposed types (none exists yet)", the
stack had no stage API, PodWire had no chunking, allowlist or reassembly, and the runtime had no downstream cache. So the product change is
part of the package, in four crates with their own tests, docs-as-code records and the registered limits: `ptr-pods` (`run_stage_in_regime`
and its schema, `forward_token` refactored into `forward_layers` with the whole-stack path bit-unchanged: all existing `ptr-pods` tests pass),
`ptr-protocol` (`activation`), `ptr-podwire` (`chunk`) and `ptr-runtime` (`split_stage`). One allowlist deviation from doc 36 is recorded: the
doc says the allowlist lives "in `ptr-protocol` conversion and in PodWire admission"; it lives in `ptr_protocol::activation::admit` (the
stateless schema half) and in `FrameAssembler::offer` (the admission half), and the prost conversion of `PodCall` is not involved because
chunks travel as ordinary `TypedPayload`s of the existing request frame.
Not built: a random-stream differential test of the receiver against an independent model (the typed-variant table of section 4 is the oracle);
the iroh transport and the Pod adapter (E008 Part B); any element of the chain beyond the stage in a case (E008).

## 9. Registration and freeze

Registered (this commit): `[experiment.E007.required]` in `experiments/preregistration.toml` lists `seeds`, `harness`, every non-prose key of
`config.toml`'s `[pinned_thresholds]` with its type (the inline `invalidation_coverage_matrix` is held flat as `_kinds` and `_cells`) and the two
files `preregistration_text` and `criteria`. `config.toml` holds the `[preregistration]` table with those keys (`harness = "split-pod-kv"`) and
`experiment.toml` the `entrypoint` and the `preflight`. E008 is registered as `planned` with its full text.

**Freeze procedure (the head), in this order. Each step is checked on the tree the previous one produced; the script is the head's
`freeze_e007.sh` and any failing step stops it.**
1. Dry-run of the signing tool (`freeze_sign.py --id E007 --dir <experiment dir> --dry-run`) and the presence of the committed pilot reference
   (`pilots/pilot-1001.json`), before any commit or file change.
2. The work commit, with explicit paths only (`experiments`, `crates`, `bins`, `docs`, `scripts/mutation_check.py`, `scripts/tests`); the tree
   must be clean afterwards.
3. Pre-sign checks on that committed tree: `scripts/check_research_gates.py` and a clean tree; `scripts/check_repo.py`;
   `scripts/update_component_docs.py --check`; `scripts/mutation_check.py E007 --check` (the plan against the code); `scripts/run_experiment.py
   validate`; `python3 -m unittest scripts.tests.test_e007_aggregate scripts.tests.test_mutation_check`; `cargo fmt --all -- --check`;
   `cargo clippy --workspace --all-targets --locked -- -D warnings`; `cargo test -p ptr-pods -p ptr-podwire -p ptr-protocol -p ptr-runtime
   --locked`; `cargo test -p ptr-bench --locked e007`.
4. Sign under the guard (a failure between signing and the tag restores the tracked files): `freeze_sign.py` pins `criteria_sha256` and
   `preregistration_text_sha256` and moves the status to `prepared`; `update_component_docs --write`; `run_experiment validate`;
   `check_research_gates`; the aggregate and research-gate unit tests. Freeze commit and tag `e007-prereg` (only paths of the experiment, the
   generated crate READMEs and `docs/components/`).
5. On that committed tree and with nothing else running: the whole `python3 scripts/mutation_check.py E007` (the checker requires HEAD to hold
   the sources it mutates); it writes `results/mutations.json` stamped with HEAD and must report killed == total == 59. Commit it.
6. Pilot seed 1001 rerun in a release build on the frozen commit, compared **by outcome** with the committed `pilots/pilot-1001.json`: every
   field equal except `elapsed_ms`, `timing_ms` and `late_window_replay_delays_ms` (the timings). A difference stops the run.
7. `python3 scripts/run_experiment.py preflight E007` (exit 0), then `prepare E007`; commit.
8. One `run E007 --seed <s>` per seed **read from `config.toml`** (17, 29, 43, 71, 101), each recorded and committed before the next; a seed is
   run once. A seed exit other than 0 or 1 stops the run (1 is a recorded product finding, 2 a harness-only invalid record, anything else, such as
   a panic or a kill, is not a result). **Only the seed's record is committed between seeds**: the runner requires every seed to run on one tree,
   so no other file may change between the first seed and the aggregation.
9. `python3 experiments/system/E007-split-pod-kv/aggregate.py`.

**No edit to any provenance file is allowed after step 5**: a `*.rs` file, a `Cargo.*` file, a `*.proto` file, a `*.sql` file, `rust-toolchain*`,
`.cargo/config*`, `aggregate.py`, `tests/mutations.toml`, `scripts/mutation_check.py` and `scripts/experiment_records.py`
(`experiment_records.mutation_record_paths`): the mutation evidence is stamped with those files and `aggregate.py` refuses evidence that no
longer matches the checkout. Anything pinned in `config.toml` or `experiment.toml` after step 5 is outside that set, which is why the digests are
pinned in step 4, before the mutation run. The full run of step 5 on the committed tree is the only run of all 59 mutations that counts; any
earlier count in this text (section 10) is a report of a different tree.

## 10. What was run before the freeze (disclosure and chronology)

Only pilot seeds 1001, 1002 and 1003 were run. No declared or confirmatory seed has been run, and nothing is frozen (the head freezes).

**Everything in this section is self-reported.** None of E007 was committed before the freeze commit (the package was untracked in the working
tree), so no commit predates any pilot or any change below, and nobody can check the order from history. The freeze tag is the first point the
record can be verified from.

Chronology, in order:

1. **Debug-build pilot, seed 1001.** It was run before the harness was finished. It failed only on the thresholds: the minimum of crossed frames
   was written as 216, and the pilot observed 192, so the check failed on that number and on nothing else. **This is a downward correction of
   a threshold made after the pilot showed the check failing at 192.** The derivation is 24 sequences x (1 prompt frame + 7 decode frames) =
   192, fixed by the case table of section 3 and by `decode_steps = 8`, not by any measurement. Whether 192 predates the pilot cannot be shown
   from history, because the files were untracked; the claim that the case table already held 192 is the author's and is not verifiable. The
   head and owner accepted it as the correction of a transcription error (216 is not a number the case table can produce); it is a loosening of
   a minimum made after a pilot, and is disclosed as one. No margin was moved. The same pilot showed that the selection digest was not yet
   pinned in the plan, so the harness computed one nobody had written down; the digest is now a pin in `config.toml`. Both changes affect a
   planned design, not a prepared one.
2. **Harness changes after that pilot** (error typing, counters for the falsifiers that had none, the tombstone accounting, the wire scan for KV
   pages and handles) were made before the release pilots below.
3. **Release pilots, seeds 1001, 1002, 1003** (after all harness changes, `release --preflight` passing): exit 0, `coverage_unmet` 0,
   `falsifiers_tripped` 0, `hard_failures` 0, 2 to 4 seconds each. No threshold or margin was tuned on them; margins come from the
   planning model in section 3.
4. **Mutation runs.** The first full run killed 44 of 47 planted mutations. Three survived the intended falsifier:
   `superseded-generation-reads-as-live` (killed, but only by `verdict_mismatches`, a different guard), `stage-a-ships-a-kv-page` (the
   harness crashed instead of counting) and `stage-a-ships-its-cache-handle` (killed only through `bitwise_mismatch_count`). The fixes were:
   the supersession mutation now expects the two `stale_hits_in_upstream_stage_generation_*supersession` counters that really measure it; the
   KV-page mutation runs three sequences so a whole 16-row page exists on the wire and the scan counts it (`wire_kv_page_bytes`); the
   handle mutation is counted by `wire_kv_handle_count`. The second full run killed **47 of 47**, each by its expected counter or by a stricter one
   (`e7-mut-all2.log`), and `mutation_check.py E007 --check` passes.
5. **Clippy fixes** from the final definition-of-done pass: `AllowlistedPayload::Chunk` is boxed, `as_chunks` replaces `chunks_exact(4)`,
   `is_multiple_of` in the stage residual check, one unused pinned-constant removed from the harness, a few test-only lint allowances. None
   changes a measured behavior. Because the harness code changed, `release --preflight` and the release pilots 1001, 1002 and 1003 were run again
   afterwards: exit 0, pinned scale, 68 coverage rules all pass, 45 falsifier rules all pass, 0 hard failures, about 2 s each. The mutation
   anchors were re-checked (`mutation_check.py E007 --check`, 47 anchors) but the 47 mutations were not re-run after these style edits.

6. **Fix round** (two independent reviews of the product code and of the design, before any freeze). Product: the replay boundary tick (a tuple is
   tombstoned or expired at every tick), authentication before decoding, signed and operation-bound policies, principal-to-stage binding, signed
   chain against the frame's chain, a global byte cap, eviction of dead cache entries on use and on a full store, the downstream stage's own chain
   in the key, an Exact-only admission, finiteness, frame positions against the cache position, a typed stage-run struct and an atomic stage run.
   Harness: seven counter pairs (coverage 68 -> 75 rules, falsifiers 45 -> 52), eleven new mutations (47 -> 58), controls rewritten to drive the
   real cache and the real assembler with their protection removed (two) or harness-only reference models (two), the `product_fault` test over
   every falsifier counter, a control that does not trip no longer hides a fired falsifier, aggregate: a missing seed record and a foreign
   record are INVALID. The first full mutation run of this round killed 52 of 58 and six "crashed" (exit 2): the mutated protection was the
   one a real-code control relies on, so the control stopped tripping and the harness refused to print. That was a harness defect, fixed by
   letting a fired falsifier take precedence; the six were then killed (58 of 58). Release preflight and pilots 1001, 1002 and 1003 after the fixes:
   exit 0, 75 coverage rules and 52 falsifier rules pass, all five controls trip, 0 hard failures, about 2 s each. The 52 kills of that full
   run were against the harness as it stood before the precedence change; only the six re-run mutations ran on the harness after it. "58 of 58" is
   therefore a composite of two trees, not a run of the final tree.

7. **Fix round 2** (further review, before any freeze). Product: a rollback of a failed stage run truncates to the cache's sequence length (not its
   next position, which differs for a cache with a position offset) and reports a rollback that fails together with its cause
   (`Qwen2StackError::RollbackFailed`); registering a principal again keeps its tombstones, policies and incomplete frames (a different stage
   identity releases the frames bound to the old one); the number of open uses of the downstream cache is bounded (`UseFault::TooManyUses`);
   the assembler never goes back in time; `max_total_buffered_bytes >= max_frame_bytes` is a policy relation; an admitted frame can be
   revalidated before its output is released. Harness: the pins, the selection, the procedure and the late window moved from the falsifiers to a
   new `validity` list (falsifiers 52 -> 48, validity 4; coverage 75 unchanged), exit 2 and INVALID; the sent-versus-observed comparisons no
   longer count as product findings when a case was aborted; the forgetful-assembler control rebuilds the assembler (re-registration no longer
   forgets); `expect_exit = 2` for four validity mutations; tests for the exit-status order and the two branches of the standing check. The
   58 mutations were re-run after the last edit of any source file of this round: 57 killed in one full run and the 58th alone. That one
   (`receiver-window-differs-from-the-pin`) crashed under the plan of that round, which said `expect_exit = 2`: a larger reorder window also lets
   an out-of-window attack through, so an observed violation fired and, falsifier first, the harness exited 1. It was replaced in the plan by
   `receiver-total-cap-differs-from-the-pin`. **That replacement is reversed in round 3** (item 8): the window mutation is back.

8. **Fix round 3** (further review, before any freeze). Decision: **validity first** (section 1). Under the round-2 order a receiver built with another
   window than the pin was FALSIFIED although the cause was a harness fault; the order is now validity, then falsifier, then the rest, in the
   harness exit status, in `aggregate.py` (`judge_seed` lists the falsifiers of a seed with a failing validity rule as ignored) and in
   `criteria.toml`; tests pin validity plus falsifier (INVALID, exit 2), falsifier plus a dead control with clean validity (FALSIFIED, exit 1)
   and a dead control alone (INVALID, exit 2). The window mutation is re-added next to the total-cap one with `expect_exit = 2` and
   `expect = ["procedure_mismatches"]` (59 mutations, 9 in the harness); the 59 are to be run once, whole, on the final tree. Product: a
   changed key or stage identity of a principal drops its verified policies and releases its incomplete frames (a policy signed under a
   compromised old key stops binding; a new-key chunk cannot complete an old-key frame); the chunk ids released inside `offer` and `abandon` are no
   longer lost and are reported exactly once; the downstream cache has `abandon_use` and `clear()` frees the open-use slots; the position
   arithmetic of a stage admission is checked. `mutation_check.py` refuses an `expect_exit` other than the integer 2 (it ignored it).
   The pilots 1001 to 1003 and the one whole-set mutation run of this round are reported in the hand-back and the report file, not here: this
   text is frozen, and the step-3 run at the freeze is the run that counts.

Product finding disclosed here: the 64-slot tombstone table caps one principal at `64 / (deadline budget + skew)` frames per second
(about 58 per second at a 100 ms budget, about 11 at 5000 ms). It is a bound of the table, written into doc 36, not a defect of a mutation.
Negative or surprising results: none of the pilots tripped a falsifier; that is a statement about the pilots only.
