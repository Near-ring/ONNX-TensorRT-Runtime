# Engine validation

Tests require an installed ONNX Runtime shared library. Use the normal system
loader paths, or set `ORT_DYLIB_PATH` before launching Cargo.

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo publish --dry-run
```

The normal suite covers tensor validation, scalar/unknown/dynamic shapes, CPU
inference, compilation/export, fallback diagnostics, runtime discovery without an
explicit path, and independent CPU workers. `runtime_loading.rs` uses subprocesses
so each loader scenario starts with an uninitialized process-global runtime.

All standalone test functions live under this directory. `internal.rs` compiles
selected private implementation modules directly and runs the tests in `unit/`.
This keeps internal helpers private and avoids references to omitted test files
in the published library. Public API behavior is covered through the built crate
by the other integration test targets. API usage examples are also checked as
Rust doctests.

## GPU suite

On a host with the compatible NVIDIA driver, CUDA, cuDNN, TensorRT, and ORT GPU build:

```bash
export NATIVE_ONNX_YOLO_MODEL=/path/to/yolo11m.trt-fp16.onnx
export NATIVE_ONNX_CUDA_YOLO_MODEL=/path/to/yolo11m.onnx
cargo test --release --all-features --tests -- --ignored --test-threads=1 --nocapture
```

The TensorRT YOLO path accepts a YOLO11m ONNX or embedded TensorRT EPContext with FP32
input `images` of shape `[1, 3, 640, 640]` and output `[1, 84, 8400]`. If the
variable is unset, the test looks for the local example artifact at
`examples/model/yolo11/yolo11m.trt-fp16.onnx`. It fails with installation/model
guidance if prerequisites are unavailable; it does not silently skip validation.
The CUDA YOLO test uses `NATIVE_ONNX_CUDA_YOLO_MODEL` (original ONNX only),
or the local `examples/model/yolo11/yolo11m.onnx` by default.

| Target | Main checks |
|---|---|
| `cuda_graph` | Independent graph sessions, failed-load cleanup, compilation coexistence, profiled replay |
| `tensorrt` | FP16 compilation, embedded-context reload, graph coexistence across providers, strict node placement |
| `threading` | Independent CUDA/TensorRT workers; concurrent native load attempts; graph on/off pairs, lifecycle churn, and 120 YOLO calls per backend matched against serial results |
| `copy_cache` | Repeated transfers and session teardown without unbounded copy-cache growth |
| `mixed_io` | CUDA graph replay with INT64 and FP16 I/O |
| `regressions` | GPU-only BF16 and fresh inputs through CUDA/CPU node partitioning |

The threaded tests create/use/drop each session on its worker, without an
application-side initialization mutex. They cover both graph-enabled and disabled
workers, mixed CUDA/TensorRT sessions, and setup/capture/drop while another worker
replays. Graph profiling compares ordinary node executions with captured replay.
Keep GPU test processes sequential; the internal coordinator does not span processes.

YOLO tests use changing synthetic inputs to check numerical consistency and
lifecycle behavior. They do not measure detection accuracy or prove throughput
improvements, multi-GPU behavior, or long-duration stability.

Tests and fixtures stay in Git and are excluded from the crate archive. Local
examples/assets are an independent ignored package and are not prerequisites for
building, linting, or running the normal tests in a fresh checkout.

See [per-session graph validation](CUDA_GRAPHS.md) for the stream/capture design
and its host GPU evidence.
