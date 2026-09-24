# Per-session CUDA graphs — 2026-09-24

## Design

- Every built-in GPU session retains the CUDA primary context and owns one
  `CU_STREAM_NON_BLOCKING` stream. It is supplied to CUDA/TensorRT as
  `user_compute_stream`; providers inside a mixed session share the same stream
  and device. A secondary CUDA candidate on another device remains available
  as a whole-session fallback.
- Pinned uploads, inference and downloads use that stream. Transfers synchronize
  only that stream. Neither `cuCtxSynchronize` nor ORT's device-wide
  `IoBinding::synchronize_inputs` remains in the transfer path.
- A process-wide read/write gate protects ORT's global capture mode. Session
  setup (including initial allocations), compilation, first-run capture and teardown
  take exclusive access. Subsequent inference/replay takes shared access, so
  independent sessions can run concurrently. Native resources are drained and
  dropped under the gate, before releasing the stream/context reference.
- ORT completes its internal warm-up/capture reruns within the first successful
  inference call. Failed runs do not clear the capture-pending state.
- The old per-GPU lifetime reservation is removed. The same gate handles TensorRT
  plugin registration, replacing the separate initialization mutex.
- CUDA/TensorRT `cuda_graph` options default to true. Effective capture requires
  positive fixed I/O, sequential execution, supported operators, and strict GPU
  selection or an EPContext. Automatic placement stays graph-free to preserve
  CPU partitions and fallback. `OnnxSession::cuda_graph_enabled()` reports the
  effective configuration. Explicit false remains supported.
- Sessions remain `!Send`/`!Sync` through their native pointer fields; no extra
  `Rc` marker is needed. A compile-time assertion preserves this contract.
  Raw custom GPU dispatches remain disallowed
  because they could bypass stream lifetime and capture coordination. External
  CUDA/ORT users are outside this gate and must coordinate capture themselves
  or disable graphs.

The source basis is ORT 1.29.1's
[stream creation](https://github.com/microsoft/onnxruntime/blob/v1.29.1/onnxruntime/core/providers/cuda/cuda_execution_provider.cc),
[global capture](https://github.com/microsoft/onnxruntime/blob/v1.29.1/onnxruntime/core/providers/cuda/cuda_graph.cc),
and [first-run capture loop](https://github.com/microsoft/onnxruntime/blob/v1.29.1/onnxruntime/core/session/inference_session.cc),
plus NVIDIA's [capture restrictions](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#prohibited-and-unhandled-operations).

## Validation

Host: RTX 3060, ONNX Runtime 1.29.1, CUDA driver/runtime API 13.2/13.1,
cuDNN 9.26.0, TensorRT 10.15.1. Model hashes are recorded in the
[0.2.0 validation report](RELEASE_0_2_0.md).

- Independent tiny models: three load/run/drop cycles, two workers, 40 changing
  inputs per worker. CUDA/CUDA and TensorRT/TensorRT each cover graph settings
  on/on, on/off and off/off; CUDA/TensorRT also covers on/on. **1,680 checked
  concurrent calls** across seven configurations, with strict placement.
- YOLO11m: CUDA on the original ONNX, TensorRT on the FP16 EPContext. Each backend
  performs three cycles × two graph-enabled workers × 20 calls: **240 calls**
  total. Inputs alternate between 0.25 and 0.75. All outputs are finite, have
  shape `[1,84,8400]`, and match that backend's serial reference within
  `1e-4 + 1e-4 * abs(expected)`. References differ between input values.
- Lifecycle churn: one CUDA graph worker continues checking changing inputs while
  another performs 12 CUDA/TensorRT load/inference/drop cycles, mixing graphs
  on and off. The final run checked **29,390 replays**; this count depends on scheduling.
- Native profiling: six ordinary calls produce six node executions. With graphs,
  CUDA records three (warm-up/capture) and TensorRT two; subsequent calls bypass
  normal node execution through native replay. This checks execution behavior,
  not just provider registration or an enabled flag.
- Failed native loads, coexistence with compilation, mixed INT64/FP16 I/O, BF16,
  prepared-buffer copy lifetime, automatic CPU partitions, and fallback are also
  covered by the GPU suite.

All **12 GPU tests** passed. The normal suite passed **28 unit/integration tests
and 8 doctests**, including with the Rust 1.88 minimum version. Stable Clippy
(all targets/features, warnings denied), rustdoc (warnings denied), formatting,
and package verification also passed. The local multithread YOLO example passes
Clippy with the new defaults. The package remains engine-only (18 files).

Reproduce using the [GPU suite commands](README.md#gpu-suite). Tests and model
assets remain outside the crate archive. Local raw logs are kept under
`sam3/per-session-graphs-validation/` and are not redistributed.

This validates bounded concurrency, replay and numerical consistency. It is not
an accuracy evaluation, throughput benchmark, multi-GPU test, or soak test.
