# Engine CUDA transfer fix

The fix lives in [`src/cuda_transfer.rs`](../src/cuda_transfer.rs), called by
[`src/buffers.rs`](../src/buffers.rs). The engine uses unmodified crates.io
`ort = 2.0.0-rc.13`; no vendored fork, Cargo patch, or native ORT modification is needed.

## Scope and performance

Prepared CUDA and TensorRT execution uses direct CUDA Driver API copies,
covering `inference()`, `inference_into()`, and prepared `run()`, including CUDA
graph replay and session recreation. CPU execution keeps its ordinary memory
path and does not load the CUDA driver. Dynamic/custom-provider ordinary
`Session::run()` paths do not use this transfer module.

Host tensors were already allocated with ORT's CUDA pinned allocator. The new
path uses those persistent buffers with `cuMemcpyHtoDAsync_v2` and
`cuMemcpyDtoHAsync_v2`. It queues all inputs or outputs in a batch, then waits
once for that batch. It removes Identity-model copy sessions and their arenas.
The public API remains synchronous; this does not overlap separate inferences.

## Safety and synchronization

`#![deny(unsafe_code)]` applies to the crate, with a narrow allowance for the
private transfer module. Before copying it checks equal shape/type, checked
byte sizes, pinned CPU memory, GPU device ID, and non-null pointers. Packed or
non-byte-addressable types are rejected. Zero-sized tensors require no copy.

The driver reports each device allocation's owning CUDA context. The module
pushes that context, verifies the device, and restores the caller's context.
A context wait before each batch orders transfers against ORT's other streams.
Copies use the default stream and finish before Rust can access or release
any borrowed buffer, including after a partial-batch error. CUDA statuses are
returned as errors; unwind cleanup also synchronizes and restores the context.
No raw pointer, CUDA allocation, or context ownership escapes this module.
The process-wide cache contains library symbols only.

This conservative synchronization supports ORT-owned streams without assuming
their handles. Further overlap would require explicit stream/event integration;
removing these waits without that integration would introduce races.
See NVIDIA's [transfer synchronization rules](https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html).

## Upstream defect retained

The old wrapper `copy_into()` cache borrowed allocator names past the lifetime
of their native `OrtMemoryInfo`, leading to accumulating helper sessions.
The engine bypasses that cache; it does not fix other applications using those
wrapper helpers. Updating only the native ORT shared library is insufficient.
The [issue draft](ort-copy-cache-issue.md), [original investigation](../sam3/memory-investigation/REPORT.md),
[owned-key patch](../sam3/memory-investigation/ort-copy-cache-owned-keys.patch),
and [previous vendored-patch evidence](evidence/ort-copy-cache/summary.json) remain.

## Shipping and verification

Ship the ordinary engine sources and dependency manifests. The CUDA driver is
loaded dynamically (`libcuda.so.1` on Linux, `nvcuda.dll` on Windows), so CPU-only
consumers do not acquire a CUDA link dependency. Linux x86-64 was tested;
Windows and multiple GPUs have not been validated in this change.

```bash
cargo fmt --check
cargo test --offline
cargo clippy --offline --all-targets -- -D warnings
cargo test --release --test copy_cache --test cuda_graph --test mixed_io \
  --test tensorrt -- --ignored --nocapture
```

The opt-in regression validates 600 forwards across six session lifetimes,
all three public inference APIs, graphs on/off, correct outputs, and **zero**
copy-helper arenas. It requires native verbose arena logs and a CUDA host.
Mixed FP16/INT64 and TensorRT compile/load checks exercise the same transfers.

Historical controls remain runnable without changing the global Cargo cache:

```bash
PROBE_VERBOSE=1 bash sam3/memory-investigation/reproduce.sh baseline copy 20
PROBE_VERBOSE=1 bash sam3/memory-investigation/reproduce.sh patched copy 100
bash sam3/memory-investigation/reproduce.sh engine folder
```

`baseline` and `patched` deliberately restore the preserved old buffer source;
`patched` additionally applies the historical owned-key fix in a temporary build.

## Measured result (2026-09-22)

The direct path and the preserved owned-key-patched binary ran the same 68
SAM3 images sequentially on the same host with native ORT 1.30.0 and graphs on.

| Path | Mean inference | Sampled steady process VRAM |
| --- | ---: | ---: |
| Previous wrapper-copy path with owned keys | 1067.4 ms/image | 9580 MiB |
| Direct batched pinned transfers | 1063.9 ms/image | 9210 MiB |

All 68 output PNG hashes match each other and the historical reference.
Memory stayed flat in the measured 25–70 second window. The new path saves
**370 MiB** in this comparison. The observed **0.3%** timing difference is too
small to establish a reliable speedup from these single runs. Inference timing
includes output transfers and the first forward, excluding model loading,
preprocessing, and rendering. NVML samples at roughly 250 ms can miss spikes.

The ordinary tests, formatting, warning-free Clippy, CUDA regression, mixed I/O,
and TensorRT checks pass. See [summary and hashes](evidence/direct-cuda/summary.json),
[GPU tests](evidence/direct-cuda/gpu-tests.log), [CPU tests](evidence/direct-cuda/cpu-tests.log),
and [memory samples](evidence/direct-cuda/sam3-direct.csv). This verifies the
observed workloads, rather than guaranteeing every native runtime is leak-free.

The 600-forward regression also passes with native ORT 1.29.1. The preserved
`patched copy 3` reproducer was rebuilt and executed successfully, with exactly
three CUDA arenas; its historical source intentionally emits unused-code
warnings for the newer transfer module that it bypasses.
