# Upstream issue draft: tensor copy cache retains borrowed allocator names

Status: locally reproduced; not submitted. Target: Rust `ort` wrapper, rather
than the native ONNX Runtime repository.

## Environment and symptom

Rust `ort` 2.0.0-rc.13, Linux x86-64, CUDA GPU with 12 GiB VRAM. Reproduced with
native ONNX Runtime 1.29.1 and 1.30.0. Repeated GPU/CPU tensor copies create more
Identity-model sessions and CUDA arenas. SAM3 eventually reports CUDA OOM at
`cudaGraphLaunch(graph_exec, stream_)`; disabling graphs does not fix the cause.

## Suspected ownership defect

In `src/memory.rs`, `AllocationDevice` holds `&'static str`, while
`MemoryInfo::allocation_device()` obtains a name through `MemoryInfoGetName`.
That name belongs to native memory info. In `src/value/impl_tensor/copy.rs`,
`IdentitySessionKey` stores these allocation devices in a global session cache.
A temporary `TensorRefMut` used for download is dropped, freeing its memory
info while the cache retains the name reference. Reused native memory then
invalidates key contents and subsequent cache lookups can create more sessions.

Expected: dropping a temporary tensor view is safe, and copy helper count is
bounded by source/target device IDs and tensor element types.
Actual: helper sessions and GPU arenas accumulate over repeated copies.

## Reproducer and evidence

From this engine checkout, with a compatible CUDA runtime selected:

```bash
export ORT_DYLIB_PATH=/absolute/path/to/libonnxruntime.so
PROBE_VERBOSE=1 bash sam3/memory-investigation/reproduce.sh baseline copy 20
PROBE_VERBOSE=1 bash sam3/memory-investigation/reproduce.sh patched copy 100
```

The copy case uses a tiny fixture model and five FP32 tensor sizes; SAM3 model
weights are not required. Each round changes input values, copies H2D,
overwrites the CPU target, copies D2H, and checks all elements. The script
builds isolated sources; it never modifies installed runtime or registry files.
It requires cached Cargo dependencies for the offline build, CUDA driver
development linking (`-lcuda`), NVML, Python, Rust/Cargo, and `patch`.

Measured native ORT 1.30.0 controls:

| Wrapper | Checked rounds | CUDA arena creations | Peak process VRAM |
| --- | ---: | ---: | ---: |
| Original | 20 | 65 | 3968 MiB |
| Owned cache names | 100 | 3 | 582 MiB |

The fixed count is one fixture arena plus two copy helpers. The original engine
regression also fails on its first repeated forward with an extra arena.
See [full source trace and experiment record](../sam3/memory-investigation/REPORT.md),
[probe source](../sam3/memory-investigation/probe.rs),
[measurements and hashes](../sam3/memory-investigation/summary.json), and
[original regression failure](evidence/ort-copy-cache/baseline-regression.log).

## Proposed upstream repair and downstream workaround

The [minimal tested patch](../sam3/memory-investigation/ort-copy-cache-owned-keys.patch)
stores owned strings for source and destination device names in the cache key.
The broader public `AllocationDevice` lifetime API also warrants review; this
patch only repairs the cache's ownership.

Our engine now bypasses wrapper tensor-copy helpers through direct pinned CUDA
transfers. It uses the unmodified registry crate. This downstream workaround
is independent of the proposed upstream patch and does not resolve the wrapper
API for other callers. Historical evidence is preserved unchanged.
