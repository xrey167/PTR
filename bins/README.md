# PTR Binaries

| Binary | Role |
|---|---|
| `ptrd` | main daemon / API / runtime host |
| `ptrctl` | administrative and debugging CLI |
| `ptr-worker` | isolated/remote worker host for Pods and background compute |
| `ptr-bench` | benchmark and experiment runner entry point |

Binaries should contain composition/bootstrap logic, not domain semantics. Reusable behavior belongs in `crates/`.
