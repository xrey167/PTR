# Candle CUDA executor

The `candle-cuda` feature provides the first native Rust GPU executor. It loads
F32 dense weights from Safetensors and places them on a selected CUDA device.
The executor is deliberately feature-gated: normal PTR builds remain free of a
CUDA toolkit requirement.

```powershell
cargo check -p ptr-pods --features candle-cuda
```

The current pin is Candle `=0.11.0`, built with the updated Rust toolchain and
its CUDA backend. CUDA 12.8 is the repository's recommended reproducible
Windows toolchain. Newer toolkits can produce PTX newer than a deployed driver
accepts, so their compatibility must be established by the hardware gate rather
than inferred from a successful Rust build.

## KV tensor backend

`CandlePagedKvBackend` is the native model-bound CUDA KV cache. The compatible
`CandleKvTensorBackend` name delegates to it. A cache owns one stream-specific
Candle device and stores F32 K/V as independently managed page bundles over all
layers. The default page size is 16 tokens and custom sizes must be powers of
two in `1..=256`.

Full pages are sealed and immutable. Online snapshot capture seals only the
writable tail, synchronizes the cache stream and returns a lease that pins the
GPU pages. Host materialization happens later through the snapshot lease, so
the authoritative runtime can revalidate placement epoch and fencing token
immediately before tier publication. Writes after a snapshot use copy-on-write.

Prefix sharing requires identical principal, execution manifest, model,
adapter, generation, schema, dtype, page size and CUDA device. Cross-device
reuse always goes through verified snapshot/restore and obtains a new runtime
lease.

`PagedKvTierAdapter` writes `PTRKVP3`: one deterministic header chunk followed
by one verified chunk per logical page. V3 binds query-head and compact
key/value-head counts independently. `CandleGpuTierBackend` implements the
generic `TierBackend` with CUDA-resident byte tensors, enabling the tested
host-staged path:

```text
Candle pages → GPU tier → CPU tier → NVMe tier → GPU tier → paged restore
```

The legacy `PTRKVTR1` and `PTRKVP2` codecs remain readable. New dense snapshots
use `PTRKVTR2`; new paged snapshots use `PTRKVP3`. Legacy records infer
`key_value_heads == attention_heads` and are digest-verified before conversion
to the current in-memory schema. Quantized FP8/FP4/NVFP4 schema values are
declared but still rejected until storage scales, accuracy gates and matching
attention kernels are implemented.

`ReferenceCausalAttentionExecutor` performs real one-layer multi-head causal
attention. Q/K/V/O matrices can be loaded from Safetensors; prefill writes
projected K/V into pages and single-token decode consumes the logical page
order. It is a correctness reference and may materialize a dense attention
view.

`Qwen2PagedDecoder` adds a Qwen2/Qwen2.5-compatible single-layer correctness
path: pre-norm RMSNorm, canonical half-split RoPE, compact grouped-query K/V,
QKV biases, bias-free SwiGLU and residual ordering. It loads the canonical
`model.layers.{L}.*` Safetensors names and rejects unsupported RoPE scaling or
activation modes. Projection and attention math are currently scalar host
reference operations while K/V residency uses the real CUDA page backend. A
tensor-native multi-layer model, embeddings/LM head and fused paged attention
remain separate performance gates.

On Windows, Candle's CUDA kernel build also requires `nvcc` and the MSVC C++
compiler (`cl.exe`) on `PATH`. Install Visual Studio 2022 Build Tools with the
Desktop C++ workload, then open a Developer PowerShell before running Cargo.
The repository cannot bootstrap that system-wide compiler from Cargo itself.

For CUDA 12.8 + MSVC, use the x64 tool environment and the standard-conforming
MSVC preprocessor required by CUDA CCCL:

```powershell
$vs = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:Path = "$env:CUDA_PATH\bin;$env:Path"
cmd /d /s /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" && set NVCC_PREPEND_FLAGS=-Xcompiler=/Zc:preprocessor && set CUDA_COMPUTE_CAP=89 && cd /d $PWD && cargo xcuda-check"
```

`CUDA_COMPUTE_CAP=89` targets the RTX 4090 (compute capability 8.9). For an RTX
3090 use `86` instead. The `NVCC_PREPEND_FLAGS` setting is intentionally scoped to this command;
it must not be applied globally to non-Windows CUDA builds.

The current local hardware gate covers an RTX 4090 with 24 GiB VRAM: paged
allocation, prefix/COW mechanics, causal prefill/decode, Safetensors loading,
compact Qwen2 GQA prefill/decode, GPU-tier roundtrip through CPU/NVMe and
restore under a new fencing token. The
host reported `nvcc 13.3` for this run; this is hardware execution evidence, not
a clean CUDA-12.8 toolchain rebuild. No RTX-3090 result is claimed without that
physical device.

### Persistent per-machine settings

The repository's `.cargo/config.toml` deliberately holds no `[env]` table:
`scripts/check_research_gates.py` rejects one, because it can name programs,
sources and flags that a research record does not bind. To avoid the Developer
PowerShell step on one machine, put the settings in the user-level Cargo
configuration (`%CARGO_HOME%\config.toml`, usually
`%USERPROFILE%\.cargo\config.toml`):

```toml
[env]
CUDA_PATH = { value = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.3", force = true }
NVCC = { value = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin\\nvcc.exe", force = true }
NVCC_CCBIN = { value = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Tools\\MSVC\\14.44.35207\\bin\\Hostx64\\x64", force = true }
NVCC_PREPEND_FLAGS = { value = "-Xcompiler /Zc:preprocessor", force = true }
CUDA_COMPUTE_CAP = { value = "89", force = true }
```

Candle links against the CUDA 13.3 libraries while `cudaforge` compiles runtime
PTX with CUDA 12.8 for the installed driver. Adjust the paths and the compute
capability to the machine; these values are the ones the repository used before
they moved here.

The model contract is:

- `weight_name`: F32 tensor shaped `[output, input]`
- optional `bias_name`: F32 tensor broadcastable to `[output]`
- input payload: little-endian F32 vector with exactly `input` elements
- output payload: little-endian F32 vector typed with the descriptor's output schema

The descriptor, generation, lease, input type and output type are still admitted
through the regular PTR Pod contract before the tensor operation is usable.
