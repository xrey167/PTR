# PTR Binaries

| Binary | Role |
|---|---|
| `ptrd` | main daemon / API / runtime host |
| `ptrctl` | administrative and debugging CLI |
| `ptr-worker` | isolated/remote worker host for Pods and background compute |
| `ptr-bench` | benchmark and experiment runner entry point |

`ptrd` stops cleanly on SIGINT or SIGTERM (in-flight requests finish, exit 0). `make vcw-demo` starts it with `config/vcw-demo.toml` and drives the verifier-closed path (driver: `scripts/vcw_demo.py`; see `docs/architecture/39-vcw-moe.md`, section 3.6).

Binaries should contain composition/bootstrap logic, not domain semantics. Reusable behavior belongs in `crates/`.
