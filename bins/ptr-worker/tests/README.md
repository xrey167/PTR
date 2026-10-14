# Tests for bins/ptr-worker

Unit tests sit beside the code in `src/` (frame transport round trip, truncation, oversize
prefix, I/O errors; call/reply payloads; built-in Pods; the serve loop on scripted byte
streams).

`stdio_session.rs` spawns the real binary (`CARGO_BIN_EXE_ptr-worker`) and drives it over its
stdin/stdout: handshake, calls, a duplicate answered from cache, an unknown Pod, a Cancel, and
Bye with exit code 0; plus exit codes for end of stream without Bye, a truncated frame, an
oversize length prefix and an illegal first frame, and the usage/`pods` output. A watchdog
thread kills the child after 30 s so a hang fails the test instead of blocking it. No network,
no wall-clock sleeps.
