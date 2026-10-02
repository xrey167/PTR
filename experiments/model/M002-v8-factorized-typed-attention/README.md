# M002-v8 — superseded before execution

M002-v8 was the first prepared study to pin the complete A0 Git-tree digest,
but post-freeze verification exposed two gate implementation defects: the
pair-table check inspected the wrong Rust source file, and the pilot archive
check treated generated pilot artifacts as if they existed at the code-source
commit. No M002-v8 command, seed, dataset load, metric, checkpoint, or runner
record was created.

The v8 configuration remains unchanged and is superseded rather than repaired
in place. M002-v9 uses the corrected gate implementation and must exercise it
before enrolment. M009 and Learned Backend qualification remain locked.
