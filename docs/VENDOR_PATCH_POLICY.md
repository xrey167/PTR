# Downstream patch maintenance and review boundary

The security remediation retains 21 published dependency source archives. Most
of the diff is unchanged upstream source, tests and notices, not a new PTR
implementation. `vendor/ORIGINS.json` records archive and original file SHA-256
hashes; `vendor/CURRENT.json` enumerates every current file and lists the exact
changes relative to those originals.

Eighteen parents switch their real `paste` dependency to pinned `pastey 0.2.3`
through a Cargo package alias. Three Raft-related packages additionally migrate
legacy protobuf interfaces and hash maps as described in
`SECURITY_P0_REVIEW_20260919.md` and `vendor/SECURITY-PATCHES.md`.

Those two sentences are a claim about a set, and a sentence cannot notice when it
stops being true. `vendor/RETIREMENT.json` therefore records the same thing per
package — which class of patch it is, which workspace patches it, and what would
let it go — and `scripts/check_vendor_retirement.py` re-derives every field from
the tree: the package set against `ORIGINS.json`, the workspace against the two
`[patch.crates-io]` sections, the changed files against the class's allowance,
and the `pastey` alias against the vendored manifest. A twenty-second patch added
for a different reason fails that check instead of quietly making this paragraph
wrong. Upstream package
names and versions are retained. There are no invented patched versions or
RustSec ignore entries. The root and A0 lockfiles describe the actual selected
package graph, including the optional dependencies audited by cargo-deny.

Normal CI runs `python scripts/check_vendor_integrity.py` in check-only mode.
It rejects unrecorded packages, changed identities, unsafe paths, symlinks,
missing/added/changed files and changed original LICENSE/COPYING/NOTICE/PATENTS
files. A maintainer must review a source change before explicitly regenerating
`CURRENT.json` with `--write`. Hashes are tamper-evidence relative to the reviewed
repository commit, not independent signatures or proof of upstream safety.

The copied upstream lockfiles, examples and test data remain provenance material;
they are not independent PTR products. Security scans discover all owned Cargo
workspaces and audit their entire resolved graphs, including these patched
packages. The four current owned workspaces are root, A0, fuzz and the Rust
template. This is not a claim that optional GPU, Apple, HIP, CUDA or image-codec
execution paths have been hardware-tested.

Retirement rule: move each patch back to a compatible maintained upstream release
when it contains the required change, rerun every affected feature/target/MSRV
test, remove its path patch and source copy, regenerate both provenance inventories
and require clean audit/deny evidence. Never retire a patch by merely suppressing
an advisory. This maintenance obligation remains open after a green merge.

Whether a *newer release exists* is answered by `scripts/refresh_vendor_upstream.py`,
which reads the crates.io sparse index, drops yanked versions and records the
highest remaining one by precedence — not the last line, which is publication
order and can end on a backport. `upstream_observed` in `RETIREMENT.json` carries
that observation with the date it was made.

Refreshing is the online half and deliberately not a checker: it writes
observations and neither passes nor fails. `check_vendor_retirement.py` stays
offline and reports candidates from what was recorded, so CI never depends on the
network and a report is reproducible from the repository alone. An observation
that could not be made is left as it was rather than replaced with a guess, and
the checker reports the oldest recorded date so staleness is visible — it never
fails on age, because a check that goes red with the passage of time fails commits
that changed nothing.

Whether a newer release *contains* the required change is still a human reading.
A candidate is a prompt to look, not a verdict.

## A newer version is not a retirement

Reporting a candidate from `upstream_observed` alone asks about a version number,
and the condition recorded in `retire_when` is not about a version number. That
mistake produced two false actionables, and both are instructive:

- `macerator 0.4.0` outranks the pinned `0.3.4` and **still declares `paste ^1`**.
  The alias it would retire has as much left to redirect as before. There was
  never anything to do.
- `netlink-packet-core 0.9.0` genuinely satisfies the condition — it is paste-free
  — but `netdev 0.45.1` requires `^0.8`, `netlink-packet-route 0.31.0` requires
  `^0.8.0`, and `netlink-proto 0.12.2` and `netwatch 0.19.3` require `^0.8.1`.
  None admits anything in `0.9`, so the resolver would never take it.

So a package with a newer upstream records two further fields and the checker
reports three states: **not a candidate** (upstream does not satisfy the
condition), **blocked** (it does, and named dependents refuse it), or **candidate**
(it does, and nothing refuses it).

**Raising the vendored version is not retiring the patch.** Both patches use the
renamed-key form of `[patch.crates-io]`, which patches one specific version. Bump
`vendor/macerator-0.3.4` to `0.4.0` and `burn-ndarray`, `burn-vision` and
`burn-flex` do not follow it — they resolve to the *unpatched* upstream `0.3.4`,
which still uses `paste`, while cargo prints only that the patch went unused. The
actionable for a blocked package is moving its dependents.

**A dependent's existence is a repository fact; its requirement is not.** A
lockfile records resolved versions, never requirements, and none of these
dependents is vendored. So `blocked_by` names each dependent with the requirement
it declares, and the split is deliberate: the *set* is re-derived from the lockfile
on every run, in both directions, while each *requirement string* is a dated
observation refreshed alongside `upstream_observed`. This keeps the checker
offline, which is a property worth more than a report that reads the network.

**An optional dependent that is not selected still constrains a bump.**
`burn-flex 0.22.0-pre.3` requires `macerator ^0.3.4` optionally, is not currently
selected, and therefore appears in no resolved graph at all. It is recorded with
`in_lockfile = false` rather than dropped, and the checker requires that flag to
match what the lockfile actually shows — in both directions.

The one-shot candidate preparation and source-retention workflows are removed
before integration. Normal builds neither fetch a hosted review delta nor grant
CI permission to push to main.
