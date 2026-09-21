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

Whether an upstream release *has* the required change is not answered anywhere in
this repository. `upstream_observed` in `RETIREMENT.json` is a recorded
observation with the date it was made, and every entry is currently unobserved;
the checker reports that as such rather than as "nothing newer exists". It cannot
look for itself: these packages are patched out of the registry, so cargo never
caches their index entries, and the checker performs no network access. Refreshing
an observation is a deliberate online step by a maintainer, after which
`--require-observations` will pass.

The one-shot candidate preparation and source-retention workflows are removed
before integration. Normal builds neither fetch a hosted review delta nor grant
CI permission to push to main.
