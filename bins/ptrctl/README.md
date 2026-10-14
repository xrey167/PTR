# ptrctl

Operator CLI for a PTR checkout and its local artifacts. Arguments are parsed by hand (no clap).

```text
ptrctl [doctor [repo-root]|layout|seal <options>]     existing, unchanged
ptrctl ingest <path> [--format tree|json] [--max-bytes N] [--preview]
ptrctl events <dir> [--consumer NAME] [--from OFFSET] [--limit N]
ptrctl trace <otlp.jsonl>
```

`ingest`, `events` and `trace` are opt-in subcommands; running `ptrctl` without them behaves as
before. Each takes `--help` (`-h`), which prints its usage and the exit codes.

## Exit codes (`ingest`, `events`, `trace`)

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | invalid or corrupt input: unreadable, longer than the bound, damaged, or refused |
| 2 | usage or harness error: unknown option, missing or malformed argument |

An ingest decision such as `NeedsFullCheck` is a result and exits 0. The older subcommands keep
their own codes (`doctor` and `seal`: 0, 1 for a failed check or seal, 2 for usage).

## `ptrctl ingest`

Reads one file (at most `--max-bytes`, default 4 MiB; a longer file is an error, not a
truncation), runs `ptr_ingress::IngressPipeline` with the file extension as a hint and renders
the `IngressRecord` through `ptr_inspect`: artifact kind, MIME, encoding and lossy flag, segments
with their raw byte spans, proposal labels, issues, and the verification decision with reasons.
`--format tree` (default) or `--format json` select the renderer. The tree is passed through
`ptr_inspect::redact_sensitive` first.

Raw content is not printed: the record shows the raw length and an FNV-1a fingerprint (not a
cryptographic digest). `--preview` adds a lossy-UTF-8 preview of each segment's raw bytes, capped
by ptr-inspect at 256 bytes; it is off by default because a segment can hold a credential that
no key-name redaction would catch. The classifier is the rule-based reference classifier.

## `ptrctl events`

Lists a `ptr_events::FileEventLog` directory: end offset, low-water mark, the requested
consumer's committed offset and lag, the slowest registered consumer, then one row per record
(offset, class, authority, topic, payload bytes). `--from OFFSET` lists records after OFFSET
(default: everything retained), `--limit N` at most N rows (default 100, at most 10000).

`FileEventLog::open` creates the directory, deletes `offsets.bin.tmp` and truncates a torn tail,
and has no read-only open. So the command works on a snapshot: it copies the `segment-*.log`
files and `offsets.bin` (regular files only, at most 512 MiB) into a private temp directory,
opens the copy and removes it afterwards. On Unix the temp directory is created 0700 and each
copied file 0600 (fresh, never through an existing path), whatever the umask; on other
platforms the temp directory's inherited permissions apply. The source directory is never created, truncated or
rewritten. Each file is copied through a reader bounded to the size counted for the 512 MiB
check; if a live writer grew a file in between, the command exits 1 ("grew beyond") instead of
copying more. If the snapshot has a torn tail, the command exits 1 with a message instead of
listing a log that a writer would repair. A directory with none of those files is refused. The
snapshot is opened with the default record-size bound, and only the slowest consumer is listed
unless `--consumer` names one (the crate exposes no consumer listing).

## `ptrctl trace`

Summarizes a file written by `ptr_observe::OtlpJsonFileExporter`: one tree per trace with each
span's duration in nanoseconds, status (`OK`, `UNSET`, `ERROR "message"`) and event count. A span
whose parent is not in the file is shown as a root and marked. Every value taken from the file
(trace, span and parent ids, names, status messages) is printed escaped, so a hostile file cannot
send terminal control sequences. ptr-observe has no reader, so
`bins/ptrctl/src/trace.rs` and `json.rs` hold a small bounded reader (16 MiB, 100000 spans,
nesting 32) for exactly the exporter's line layout; it is not a general OTLP reader. Times are
the exporter's ticks plus its epoch offset, so durations are meaningful and absolute times may
not be wall-clock. Indentation is capped at depth 64; deeper spans carry a `[depth N]` marker, so
output is linear in the span count even for a 100000-deep parent chain.

## Tests

`tests/` runs the real binary on files created in each test: a JSON file, a UTF-16 text file and
a PNG header for `ingest`, a log written with `FileEventLog` for `events` (including an
unchanged-directory check and a torn tail), and a file written by the real exporter for `trace`.
