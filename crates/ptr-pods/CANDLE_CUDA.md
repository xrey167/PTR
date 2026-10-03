# Candle CUDA executor

The `candle-cuda` feature provides the first native Rust GPU executor. It loads
F32 dense weights from Safetensors and places them on a selected CUDA device.
The executor is deliberately feature-gated: normal PTR builds remain free of a
CUDA toolkit requirement.

```powershell
cargo check -p ptr-pods --features candle-cuda
```

The current pin is Candle `=0.11.0`, built with the updated Rust toolchain and
its CUDA backend. The supported local toolkit is CUDA 12.8. CUDA 13.x can
produce PTX newer than the installed Windows driver accepts for this Candle
kernel release.

## KV tensor backend

`CandleKvTensorBackend` provides the first real model-bound CUDA KV cache
contract. It stores layer-wise F32 key/value tensors, validates the device and
schema, supports append/truncate/snapshot/restore, and binds snapshots to a
content digest. The backend is intentionally separate from
`CandleDenseExecutor`; it is a tensor-cache primitive, not yet a complete
Transformer or LLM executor.

The cache contract now also carries a digest-bound `KvPageTable`. The reference
backend reserves and releases logical KV pages during append/truncate, and
snapshot restore verifies the page allocation together with the tensor data.
The current Candle implementation still stores each layer as a contiguous F32
tensor; wiring the page table into a paged CUDA allocation and fused attention
kernel is a separate gate. Quantized FP8/FP4/NVFP4 schema values are therefore
declared but rejected by the current F32 backends until their storage scales,
packing and attention kernels are implemented and benchmarked.

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
