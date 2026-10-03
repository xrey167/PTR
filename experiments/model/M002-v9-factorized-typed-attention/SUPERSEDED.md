# M002-v9 supersession record

M002-v9 is `superseded` after five seed records (Seeds 17, 29, 43, 71 and 101,
committed 2026-10-02) and before any decision. No `m002-v9-decision.json`
exists, and this status is not evidence: it never unlocks M009 or a Learned
Backend.

Why: the study binds the complete tracked `model/burn-a0` Git-tree digest.
`ptr-types`, which A0 depends on by path, gained a `sha2` dependency, so
`model/burn-a0/Cargo.lock` no longer satisfied `cargo ... --locked` and every
build and security job of that workspace failed. Refreshing the lockfile
changes the tree and therefore the pinned digest. Re-pinning in place is not
possible: each of the five records names the preregistration digest it ran
under, and the research gate rejects them against any other configuration.

What stays as it was: the v9 manifest, configuration and all five seed records
are not amended. The records were produced under the earlier tree and can be
read as that, not as evidence for the current one. A further confirmatory
study needs a new, separately preregistered successor (v10) bound to the
refreshed tree.
