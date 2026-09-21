# P0.10 — Retained-patch retirement and third-party notices

Baseline: `15e4598bd292b5928455b68ba74ee03e0ded81c4`.
Owner: `scripts/`, `vendor/`, `THIRD-PARTY-NOTICES.md`.

Issue #20's *Downstream dependency maintenance* gate. Two obligations, both
stated in the repository before this contract existed and neither carried by an
artifact: retire each retained patch as upstream absorbs it, and ship the notices
that the licences require to travel with redistributed software.

## What was actually missing

The first finding in #20 was wrong and the correction is part of this contract,
because the shape of the fix follows from it.

`docs/VENDOR_PATCH_POLICY.md` already recorded why the patches exist — *"Eighteen
parents switch their real `paste` dependency to pinned `pastey 0.2.3` through a
Cargo package alias"* — and a retirement rule covering all of them. I had searched
`scripts/` for tooling and read `vendor/SECURITY-PATCHES.md`, and concluded from
their silence that nothing existed.

What was missing is narrower: **the reason existed as a claim about a set, and a
sentence cannot notice when it stops being true.** `check_vendor_integrity.py`
verifies hashes, paths and identities; it says nothing about rationale. A
twenty-second patch added for an entirely different reason would pass every check
while silently making that paragraph false.

The same shape applies to notices. `docs/SECURITY_P0_REVIEW_20260919.md` states
the obligation outright — *"Release packaging must carry third-party
notices/source availability"* — and `deny.toml` narrowly allows one MPL-covered
package with *"redistribution notices remain required"*. Both were policy with no
artifact behind them: `NOTICE` was five lines naming no package.

## Per-patch reasons that a check can hold to account

`vendor/RETIREMENT.json` records, per package, which class of patch it is, which
workspace patches it, and what would retire it.
`scripts/check_vendor_retirement.py` re-derives every field from the tree rather
than trusting it:

| Recorded | Checked against |
|---|---|
| the package set | `vendor/ORIGINS.json`, in both directions |
| `workspace` | the `[patch.crates-io]` section that actually patches that directory |
| the class's `touches` allowance | the changed-file list in `vendor/CURRENT.json` |
| class `paste-alias` | the `package = "pastey"` alias in the vendored manifest |

The third row is the one that earns its place: a `Cargo.toml`-only class cannot
quietly grow a source patch. `check_vendor_integrity.py` records that
`src/lib.rs` changed and is content; this refuses it, because the recorded reason
says the patch touches manifests only.

The structure it checks is verified rather than inferred: the production
workspace patches four packages and the isolated A0 workspace patches seventeen,
which is the twenty-one that `ORIGINS.json` pins.

A declared workspace whose manifest has vanished reports *that*, rather than
degrading into "unknown workspace" for the seventeen records that were right. The
first draft did the latter, and a test caught it.

## Observation, separated from judgement

Whether a newer release exists is answered by
`scripts/refresh_vendor_upstream.py`, which reads the crates.io sparse index. Two
details matter and both have tests:

- **Yanked versions are dropped.** A yanked release is not something a patch can
  move to.
- **The newest version is the highest by precedence, not the last line.** The
  index is in publication order and can end on a backport to an older series, so
  taking the last entry would record a version older than one already published.

Refreshing writes observations and neither passes nor fails. The checker stays
offline and reports candidates from what was recorded, so CI never depends on the
network and a report is reproducible from the repository alone. An observation
that could not be made is left as it was rather than replaced with a guess.

Because all twenty-one now carry a dated observation, CI runs the checker with
`--require-observations`: a new patch cannot land unobserved. Staleness is
reported as the oldest recorded date and **never fails a build** — a check that
goes red with the passage of time fails commits that changed nothing.

Whether a newer release *contains* the required change stays a human reading. A
candidate is a prompt to look, not a verdict.

## Notices that cannot silently shrink

`THIRD-PARTY-NOTICES.md` lists every third-party package reachable from an owned
Cargo workspace with its licence expression and the licence text it ships. Three
properties make it worth committing.

**Coverage cannot lose a workspace.** The package set comes from
`security_workspaces.workspaces`, which already refuses to lose one and whose
guarantee the notices need for the same reason. This is not theoretical:
`colored 3.1.1`, the only MPL-covered package, is reachable **only through the A0
lockfile**. A notices file built from the production workspace alone would have
omitted the single package the MPL obligation is about.

**It is reproducible.** Texts come from the pinned `.crate` archives and the
vendored trees, not from whatever a machine happened to build. This mattered in
practice: of the 899 registry packages in the lockfiles, only 436 were present
locally before `cargo fetch --locked` — the rest are platform- or feature-gated
and had never been built here. A document whose contents depend on which machine
generated it is worthless as a checked artifact, so a missing archive is an error
rather than a silent omission.

**A package it cannot describe is a failure.** No licence expression, no
document. Dropping what cannot be described is the one outcome worse than having
no file.

Identical texts appear once and are referenced. Two MIT texts differing only in a
copyright line are **not** identical and both appear, because that line is the
notice.

`scripts/check_notices.py` recomputes the package set from the lockfiles and names
every difference in both directions, plus a package-set digest so a hand-edited
table is caught. It reads no archives and touches no network, so it runs in the
ordinary invariants job; regenerating is the heavier step that needs the pinned
archives.

## Executed evidence

24 script tests added across `test_check_vendor_retirement.py` and
`test_notices.py`; **77 script tests pass.**

Current state of the retirement report, which is a real result rather than a
placeholder:

| Package | Pinned | Newest published | Reported state |
|---|---|---|---|
| macerator | 0.3.4 | 0.4.0 | **not a candidate** — 0.4.0 still declares `paste ^1`, so it does not satisfy `retire_when` |
| netlink-packet-core | 0.8.2 | 0.9.0 | **blocked** — 0.9.0 is paste-free, and `netdev 0.45.1`, `netlink-packet-route 0.31.0`, `netlink-proto 0.12.2` and `netwatch 0.19.3` require `^0.8.x` |
| the other 19 | — | equal to pinned | no upgrade can retire them |

Both were reported as `candidate` until the checker learned to ask about the
condition rather than the version number, and this table said so with them. **The
number of takeable retirements is zero**, and it is zero for two different
reasons: one release does not satisfy its condition, the other satisfies it and
is refused by its dependents. Retiring the second means moving those four
dependents, not raising the vendored copy.

The nineteen are already at the newest published version: the current upstream
release still carries the `paste` dependency the alias exists to redirect.

Notices: **920 third-party packages, 456 distinct licence texts, every package
carrying a licence expression.** 27 PTR-owned packages are excluded. 117 ship no
licence text file, which is recorded per package rather than hidden.

Negative, one per route back in:

- A vendored package with no retirement record; a record for a package that is
  not vendored; a record naming the wrong workspace; a missing workspace
  manifest; an unknown class; an empty retirement condition.
- A `Cargo.toml`-only class whose package grew a `src/lib.rs` change.
- A `paste-alias` record whose manifest does not carry the alias.
- Publication order deciding "newest"; a yanked release being chosen; a
  pre-release outranking its own release or falling behind an older one; an index
  with nothing unyanked yielding a guess instead of no observation.
- A package added to or removed from the graph; a version bump, which is a
  difference in both directions; a vendored package treated as not third-party; a
  hand-written notices file with no digest; a tampered digest with a matching
  table; a missing document.

## What this does not close

- **This is not legal advice and not a distribution clearance.** The gate
  produces artifacts and checks. Whether a particular distribution satisfies a
  particular licence is not something a generator decides, and the document says
  so in its own header.
- **Retirement is not performed, and neither of the two was ever takeable.** This
  document previously said "the two candidates are reported, not taken", which
  reads as effort not yet spent. It was wrong in a way worth recording, because
  the error was the same one this gate exists to catch — a claim that cannot
  notice when it stops being true — one level up, in the checker rather than in
  the prose. The report said CANDIDATE whenever the observed upstream version
  outranked the pinned one, which asks about a version number and not about the
  condition in `retire_when`:
  - **`macerator 0.4.0` never satisfied it.** It still declares `paste ^1`, so the
    alias it would retire has exactly as much left to redirect as before. It was
    a candidate only by arithmetic on a version string.
  - **`netlink-packet-core 0.9.0` does satisfy it** — the release is paste-free —
    and is refused by its dependents: `netdev 0.45.1` at `^0.8`,
    `netlink-packet-route 0.31.0` at `^0.8.0`, `netlink-proto 0.12.2` and
    `netwatch 0.19.3` at `^0.8.1`, none of which admits anything in `0.9`.
    `0.9.0` is in fact already in this tree for other dependents, which shows
    plainly that presence upstream is not what decides whether a patch can go.

  The checker now reports three states — not a candidate, blocked by named
  dependents, or a genuine candidate — and this tree has no package in the third.
  **The actionable for a blocked package is moving its dependents**, not raising
  the vendored version: both patches use the renamed-key form, which patches one
  specific version, so bumping the copy past what a dependent requires leaves that
  dependent on the *unpatched* upstream while cargo merely notes that the patch
  went unused. Moving a patch still requires reading the upstream change, rerunning
  the affected feature/target/MSRV tests, removing the path patch and the source
  copy and regenerating both provenance inventories — the rule
  `VENDOR_PATCH_POLICY.md` already states.
- **A dependent's requirement is an observation, not a repository fact.** The
  checker stays offline, and that property is kept deliberately rather than traded
  to fix a report. A lockfile records resolved versions and never requirements,
  and none of these dependents is vendored, so `blocked_by` records each
  requirement the way `upstream_observed` is recorded: dated, refreshed by
  `refresh_vendor_upstream.py`, visible rather than absent. What *is* re-derived
  offline is the dependent set, in both directions — a dependent the lockfile
  shows and the record omits fails, and so does a recorded dependent the lockfile
  does not pin at that version.
- **An optional dependent no lockfile can show still constrains a bump.**
  `burn-flex 0.22.0-pre.3` requires `macerator ^0.3.4` optionally and is not
  currently selected, so it appears in no resolved graph. It is recorded with
  `in_lockfile = false` rather than dropped.
- **Whether an upstream release satisfies its condition cannot be checked here.**
  `upstream_retire_when` is an observation like the rest. For the `paste-alias`
  class the signal is mechanical — the release either declares a dependency on
  `paste` or does not — but reading it needs the index. For `raft-protobuf` there
  is no mechanical signal at all: "carries that migration" is something a person
  reads, and the class records `retire_when_signal: null` rather than pretending
  otherwise. A record that misstated it would still not produce a false
  actionable, because the blocker check runs regardless — but it would misstate
  the reason.
- **Observations age.** The checker reports the oldest date and deliberately does
  not fail on it, so a stale observation is visible rather than blocking.
  Refreshing stays a deliberate online step.
- **Licence expressions are the packages' own claims.** The generator records
  what each `Cargo.toml` declares; it does not scan sources to confirm that the
  declaration matches the code.
- **117 packages ship no licence text.** Their expression is recorded and the
  absence is stated per package. Obtaining a text a publisher never shipped is
  not something this repository can do.
- **Generation needs the pinned archives.** `cargo fetch --locked` in each owned
  workspace is a prerequisite, and the generator refuses rather than producing a
  partial document.
