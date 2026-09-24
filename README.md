# native-onnx

Rust ONNX inference using **your system's ONNX Runtime, CUDA, and TensorRT**.
Load a model, supply typed tensors, and choose a required backend or ordered fallback.
No Python runtime or automatic native-library downloads.

[Public API guide](https://docs.rs/native-onnx/0.2.0/native_onnx/) ·
[OnnxSession methods](https://docs.rs/native-onnx/0.2.0/native_onnx/struct.OnnxSession.html)

## Why use it?

- **Less repeated work:** reuse input/output buffers; GPU paths reuse pinned host
  and device storage. Eligible fixed-shape models can use CUDA graph replay.
- **Faster startup after compilation:** compile a TensorRT model once, then load
  its embedded engine on later launches.
- **Explicit execution:** require CUDA/TensorRT or allow fallback to CPU, with
  recorded failure reasons and checked tensor names, types, shapes, and lengths.

Compute speed comes from ONNX Runtime and its native providers. This crate reduces
buffer setup and allocation work; it does **not** promise a universal speedup over
`ort`, C++, or Python. Benchmark your model and hardware.

## Install the native libraries first

Requires **Rust 1.88+** and a matching native runtime for your process architecture.
**You must install these system libraries yourself:**

| Execution | Required installation |
|---|---|
| CPU | ONNX Runtime **1.27+** shared library (C API 27); no NVIDIA libraries needed |
| CUDA | ONNX Runtime **1.27+ with CUDA EP**; standard GPU builds require **CUDA 13.x (13.0+)**, **cuDNN 9.x for CUDA 13**, and a compatible driver (**at least R580**; the cuDNN/TensorRT build may require a newer driver) |
| TensorRT | The GPU stack above plus the **TensorRT version required by your ORT build**. The validated baseline is **TensorRT 10.15.1 (ABI 10)**; a newer incompatible major version will not work |
| CoreML | macOS **12+** and ONNX Runtime **1.27+ with CoreML EP** |

GPU dependency versions must match the installed ORT build, not merely exceed a
number. Custom CUDA 12 builds need their matching CUDA/cuDNN/TensorRT libraries.
Release tests passed on Linux/RTX 3060 with observed library versions ORT 1.29.1,
CUDA runtime 13.1, cuDNN 9.26, and TensorRT 10.15.1. These are test results, not
universal dependency minimums. macOS and Windows still need native validation.
Check the [cuDNN support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.26.0/reference/support-matrix.html)
for the exact driver requirements of the package you install.
See [ORT installation](https://onnxruntime.ai/docs/install/),
[CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
and [TensorRT compatibility](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.15.1.html).

No runtime path is required in Rust. ONNX Runtime, CUDA, and TensorRT are loaded
from your system. If their libraries are already discoverable by the platform
loader, no environment configuration is needed. For an installation in custom
directories, set the search paths **before launching** your application:

```bash
# Replace these directories with your actual installation paths.
export LD_LIBRARY_PATH=/opt/onnxruntime/lib:/usr/local/cuda/lib64:/opt/cudnn/lib:/opt/tensorrt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# Optional: select a particular ONNX Runtime library instead of loader discovery.
export ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
```

On Windows, add library folders to `PATH`; on macOS, use `DYLD_LIBRARY_PATH` for
custom directories. The loader looks for `onnxruntime.dll`, `libonnxruntime.dylib`,
or `libonnxruntime.so`. The first ORT initialization selects one library for the
whole process, so there is no per-session `runtime_path` option.
Missing/incompatible libraries return installation guidance;
print errors with `eprintln!("{error:#}")` to include the native loader's cause.

## Basic example

```toml
[dependencies]
native-onnx = "0.2.0"
```

For a model with an FP32 input named `x`, shape `[1, 16]`:

```rust,no_run
use native_onnx::{OnnxOptions, OnnxSession, Result, TensorView};

fn main() -> Result<()> {
    let mut model = OnnxSession::load("model.onnx", OnnxOptions::cpu())?;
    let values = [1.0_f32; 16];
    let input = TensorView::f32("x", &[1, 16], &values);

    for output in model.inference(&[input])? {
        println!("{}: {:?}", output.name, output.shape);
    }
    Ok(())
}
```

Replace the path, name, shape, and values for your model. `model.info()` exposes
its specifications. Input storage is contiguous and row-major; preprocessing and
postprocessing belong in your application. Each `OnnxSession` owns one active
ONNX Runtime session plus its provider-fallback state and reusable I/O buffers.

## Select a GPU and compile once

```rust,no_run
use native_onnx::{
    Backend, BackendSelection, CompileOptions, CompileTarget, OnnxOptions,
    OnnxSession, TensorRtOptions, TensorView,
};

let config = TensorRtOptions {
    fp16: true,
    ..TensorRtOptions::default()
};
let compile = CompileOptions::new(CompileTarget::TensorRt(config.clone()));
let values = [1.0_f32; 16];
let input = TensorView::f32("x", &[1, 16], &values);

// Run once. The destination must not already exist.
OnnxSession::compile("model.onnx", "model.ctx.onnx", compile, &[input])?;

// Later launches start here; the original model is not needed.
let options = OnnxOptions {
    backend: BackendSelection::Require(Backend::TensorRt),
    tensorrt: Some(config),
    ..OnnxOptions::default()
};
let mut model = OnnxSession::load("model.ctx.onnx", options)?;
let outputs = model.inference(&[input])?;
# Ok::<(), native_onnx::Error>(())
```

Compilation requires an explicit target and its settings: `CompileTarget::TensorRt(config)`,
`CompileTarget::Cuda(config)`, or `CompileTarget::Cpu`. `CompileOptions::new(target)`
also exposes threading and dimension overrides. It has no default
target or fallback; the artifact is validated with your inputs before saving.
The returned `CompileReport` identifies the backend and saved format.

For **loading/inference**, use `Require(Backend::Cuda)` for CUDA. Use
`Auto(vec![Backend::TensorRt, Backend::Cuda, Backend::Cpu])` to permit fallback.
`backend()` reports the primary provider; in automatic mode some nodes can run on
CPU. `fallback_events()` reports whole-session failures. `Require` rejects CPU
node fallback.

TensorRT export currently requires one partition. CPU/CUDA compilation produces
an optimized ONNX graph. Compiled contexts require their native provider; they
cannot fall back to a CPU engine. Recompile after changing the model, GPU, or
incompatible runtime/provider settings. Raw TensorRT `.engine` files are not accepted.

## Reuse buffers and configure execution

| API / option | Purpose |
|---|---|
| `inference(inputs)` | Borrow inputs and return owned CPU outputs |
| `inference_into(inputs, outputs)` | Fill caller-owned CPU outputs, matched by name |
| `input_mut(name)`, `run()`, `output(name)` | Reuse fixed-shape buffers; inputs start at zero |
| `info()` | Input/output names, element types, and resolved shapes |
| `is_prepared()` | Whether persistent fixed-shape I/O storage is available |
| `cuda_graph_enabled()` | Whether the active session is configured for graph capture/replay |
| `backend()`, `fallback_events()` | Primary provider and chronological whole-session failures |
| `end_profiling()` | Finish active-session profiling and return its file path |
| `CudaOptions`, `TensorRtOptions` | Device, precision, CUDA graphs, and TensorRT profiles |
| `dimension_overrides` | Resolve named dynamic dimensions |
| `parallel_execution`, `inter_threads` | Enable/configure inter-operator CPU parallelism |

**CUDA graphs default to `true` for CUDA and TensorRT.** Capture requires fixed,
positive I/O shapes, sequential execution, and `BackendSelection::Require` or a
compiled EPContext. `Auto` placement stays graph-free to preserve CPU partitioning
and fallback. Check `model.cuda_graph_enabled()`; set `cuda_graph: false` to opt out.
TensorRT defaults to FP32 with TF32 allowed; unset `NVIDIA_TF32_OVERRIDE`, or set
it to `0` with `tf32: false`.
Dynamic or unknown-rank I/O uses ordinary inference. Dense numeric and boolean
tensors are supported; strings, sequences, and sparse I/O are not.

Public inputs and outputs use contiguous, row-major CPU storage, including on
GPU backends. `Tensor`/`TensorBuffer` own data; `TensorView`/`TensorData` borrow it;
`TensorViewMut`/`TensorDataMut` borrow it for mutation. FP32, FP64, FP16, BF16,
signed/unsigned 8/16/32/64-bit integers, and Boolean tensors are supported.
Use the appropriate `TensorData` variant for non-FP32 input; no implicit conversion occurs.

`TensorSpec.shape` uses `None` for unknown rank, `Some(vec![])` for a scalar, and
`None` inside the dimension vector for a dynamic size. Supply every input exactly
once; names, dtypes, shapes, and element counts are checked. Owned inference outputs
remain valid after the next run. Prepared output borrows end before another mutable
session operation; copy with `to_owned()` to retain them.

```rust,no_run
use native_onnx::{OnnxOptions, OnnxSession};

let mut model = OnnxSession::load("model.onnx", OnnxOptions::cpu())?;
if model.is_prepared() {
    model.input_mut("x")?.as_f32_mut()?.fill(1.0);
    model.run()?;
    println!("{:?}", model.output("y")?.as_f32()?);
}
# Ok::<(), native_onnx::Error>(())
```

Input buffers start at zero and retain values until changed. Borrowing an input
for mutation invalidates previous outputs. `run()` must succeed before `output()`.
`inference_into()` avoids creating caller-visible output allocations on the prepared
path; dynamic execution first creates owned results and copies into your buffers.
Neither path promises zero internal allocations or zero CPU/GPU transfers.

Common defaults: one intra-operator thread, one inter-operator thread, sequential
execution, no dimension overrides, and profiling off. `intra_threads: 0` lets ORT
choose; `inter_threads` only applies with `parallel_execution: true`. Set
`profiling: Some("profiles/run".into())` before loading to collect an ORT trace
(create the parent directory first). Provider-specific `None` options use that
provider's defaults. Options for unused providers are ignored.

TensorRT defaults to device 0, a 4 GiB workspace on 64-bit targets, builder level 3,
automatic auxiliary streams, and sparsity allowed. FP16/TF32 affect internal
arithmetic, not I/O types. Dynamic shape profiles require all three of
`min_shapes`, `opt_shapes`, `max_shapes`, using ORT syntax such as `images:1x3x640x640`.
All public fields and error contracts are documented in the
[API reference](https://docs.rs/native-onnx/0.2.0/native_onnx/).

## Independent inference threads

Calls are synchronous. `OnnxSession` is neither `Send` nor `Sync`: construct,
use, and drop one inside each worker. **Multiple CUDA/TensorRT sessions can enable
CUDA graphs on the same GPU.** Each session owns a nonblocking stream used for
transfers and inference; a mixed TensorRT/CUDA session shares its stream internally.

The engine coordinates native setup, first-run warm-up/capture, and teardown.
Ordinary inference and completed graph replays share access and may run concurrently.
There is no lifetime reservation for a GPU. Initialization includes TensorRT plugin
registration, so callers do not need their own initialization mutex.

This coordination covers this crate's sessions in the current process. External
CUDA/ORT users must avoid incompatible work during capture, or disable graphs.
Concurrent streams do not guarantee a speedup on a saturated GPU. Native provider
crashes cannot be recovered as Rust errors.

```rust,no_run
use native_onnx::{Backend, BackendSelection, OnnxOptions, OnnxSession, Result, TensorView};

let workers: Vec<_> = ["first.onnx", "second.onnx"].into_iter().map(|path| {
    std::thread::spawn(move || -> Result<()> {
        let mut model = OnnxSession::load(path, OnnxOptions {
            backend: BackendSelection::Require(Backend::TensorRt),
            ..OnnxOptions::default()
        })?;
        let outputs = model.inference(&[TensorView::f32("x", &[1, 16], &[1.; 16])])?;
        println!("{} outputs", outputs.len());
        Ok(())
    })
}).collect();
for worker in workers {
    worker.join().expect("worker panicked")?;
}
# Ok::<(), native_onnx::Error>(())
```

Additional providers use `Backend::Custom`; CUDA/TensorRT dispatches must instead
use their built-in variants and typed options. Enable the `openvino`
feature for OpenVINO configuration and install its matching native provider separately.

The pinned `ort` rc.13 dependency has an [exit-time CUDA teardown bug](https://github.com/pykeio/ort/issues/609).
This crate retains one global environment until process exit as a workaround;
model sessions and buffers are still released when dropped.

## Development

With native libraries available through the system loader or `ORT_DYLIB_PATH`:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo publish --dry-run
# On a GPU host with the matching native libraries:
# The YOLO threading tests need the original model and a TensorRT-compatible artifact:
# export NATIVE_ONNX_YOLO_MODEL=/path/to/yolo11m.trt-fp16.onnx
# export NATIVE_ONNX_CUDA_YOLO_MODEL=/path/to/yolo11m.onnx
cargo test --release --all-features --tests -- --ignored --test-threads=1
```

The crate archive includes only engine source, build script, README, license,
and Cargo metadata. Tests stay in Git; local examples, experiments, and model
assets are ignored.
Local development examples use their own ignored `examples/Cargo.toml` manifest;
run SAM3 with `cargo run --release --manifest-path examples/Cargo.toml --bin sam3`.

## Changes in 0.2

`OnnxRuntime` → `OnnxSession`; `Compilation` → `CompileReport`. Old aliases and
per-session `runtime_path` fields are removed. CUDA/TensorRT graphs now default on
for eligible strict sessions, each session owns its stream, and native setup is
serialized inside the engine for independent inference workers.

Licensed under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).
See `LICENSE-APACHE` for the full text.
