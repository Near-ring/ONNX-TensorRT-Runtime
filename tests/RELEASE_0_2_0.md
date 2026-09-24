# 0.2.0 release validation — 2026-09-24

This report describes the consolidated release: independent GPU streams,
coordinated CUDA graph capture, and thread-bound sessions without an extra `Rc`
marker. Graphs default on for eligible strict CUDA/TensorRT sessions; automatic
placement, dynamic I/O, and parallel execution remain graph-free.

Observed environment, queried through the native library APIs:

| Component | Version / device |
|---|---|
| Host | Linux x86-64, kernel 7.0.0-31, glibc 2.43 |
| GPU | NVIDIA GeForce RTX 3060 |
| ONNX Runtime | 1.29.1 |
| CUDA driver API / runtime API | 13.2 / 13.1 |
| cuDNN | 9.26.0 |
| TensorRT | 10.15.1 |

These versions describe the tested host, not universal minimum requirements or
a certification of other model/hardware/library combinations.

## Checks passed

- Stable Rust: 28 unit/integration tests plus 8 doctests. The compile-time check
  confirms `OnnxSession` remains neither `Send` nor `Sync` after removing the marker.
- Rust 1.88 minimum version: the same 36 tests passed on the stream implementation.
- Host GPU: all 12 opt-in integration tests passed, including per-session graph
  replay, concurrent lifecycle changes, TensorRT FP16 compile/reload, mixed
  INT64/FP16 I/O, BF16, prepared-copy lifetime, and automatic CPU partitioning.
- Formatting, all-target/all-feature Clippy with warnings denied, and rustdoc
  with warnings denied. The local multithread YOLO example also passed Clippy.
- Cargo package verification passed. The archive contains 18 engine,
  documentation, license, and metadata files, without tests/examples/model assets.
- The dependency check on 2026-09-24 reported no known OSV advisories for the
  48 registry packages in the unchanged release lockfile. This is a dependency
  database check, not a source security audit.

## Concurrent inference and graph replay

Both CUDA and TensorRT YOLO tests use two worker-local, graph-enabled sessions.
Each backend completed three cycles × two workers × 20 calls: **120 calls per
backend, 240 total**. Calls alternate FP32 `[1,3,640,640]` inputs filled with
0.25 and 0.75. Every output has shape `[1,84,8400]`, contains finite values, and
matches that backend's serial reference within
`abs(actual - expected) <= 1e-4 + 1e-4 * abs(expected)`.
The two reference outputs differ, confirming the input changes are observable.
Strict placement is required and no fallback occurred.

The tiny-model tests check **1,680 calls** across CUDA/CUDA, TensorRT/TensorRT,
and mixed CUDA/TensorRT workers with graphs on/on, on/off, and off/off as applicable.
A lifecycle test checked **29,390 replays** while another worker performed 12
CUDA/TensorRT load/inference/drop cycles; the replay count varies with scheduling.
ORT profiling confirms native replay bypasses repeated normal node execution.

See [CUDA graph design and evidence](CUDA_GRAPHS.md) for the detailed configuration
matrix, capture coordination, native profiling results, and local log locations.
There is no one-session-per-GPU restriction. Each worker still constructs, uses,
and drops its own `!Send`/`!Sync` session.

YOLO files used (kept locally; not redistributed):

```text
yolo11m.onnx
0b61260968bedf1e2c4600529817c8b8aaf649dfe555898170b425850f8521dc

yolo11m.trt-fp16.onnx (artifact used by the threaded test)
15372b90d8870d5fca41ac0278430837b21194a0244e4045bbb93ab3c7728b90
```

This verifies bounded lifecycle/concurrency and numerical consistency. It is not
a detection-accuracy evaluation, throughput benchmark, multi-GPU test, or
long-duration soak test. Native macOS/Windows execution remains unverified.
See [test instructions](README.md) for reproduction commands and prerequisites.
