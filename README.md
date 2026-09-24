# native-onnx

Rust ONNX inference using **your system's ONNX Runtime, CUDA, and TensorRT**.
Load a model, supply typed tensors, and choose a required backend or ordered fallback.
No Python runtime or automatic native-library downloads.

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
| CUDA | ONNX Runtime **1.27+ with CUDA EP**; standard GPU builds require **CUDA 13.x (13.0+)**, **cuDNN 9.x for CUDA 13**, and an NVIDIA driver supporting CUDA 13 (**R580+**) |
| TensorRT | The GPU stack above plus the **TensorRT version required by your ORT build**. The validated baseline is **TensorRT 10.15.1 (ABI 10)**; a newer incompatible major version will not work |
| CoreML | macOS **12+** and ONNX Runtime **1.27+ with CoreML EP** |

GPU dependency versions must match the installed ORT build, not merely exceed a
number. Custom CUDA 12 builds need their matching CUDA/cuDNN/TensorRT libraries.
Validated on Linux with ORT 1.29.1, CUDA 13, cuDNN 9.20, and TensorRT 10.15.1;
macOS and Windows still need native validation on those platforms.
For that Linux stack, the [cuDNN driver minimum is 580.65.06](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.20.0/reference/support-matrix.html).
See [ORT installation](https://onnxruntime.ai/docs/install/),
[CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
and [TensorRT compatibility](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.15.1.html).

On Linux, set the paths **before launching** your application:

```bash
export ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
# GPU only: replace these with your actual native-library directories.
export LD_LIBRARY_PATH=/opt/onnxruntime/lib:/usr/local/cuda/lib64:/opt/cudnn/lib:/opt/tensorrt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

On Windows, point `ORT_DYLIB_PATH` to `onnxruntime.dll` and add the dependency
folders to `PATH`. On macOS, use `libonnxruntime.dylib`. You can also set
`OnnxOptions::runtime_path`; the first ORT initialization selects the library for
the entire process. Missing/incompatible libraries return installation guidance;
print errors with `eprintln!("{error:#}")` to include the native loader's cause.

## Basic example

```toml
[dependencies]
native-onnx = "0.1.0"
```

For a model with an FP32 input named `x`, shape `[1, 16]`:

```rust,no_run
use native_onnx::{OnnxOptions, OnnxRuntime, Result, TensorView};

fn main() -> Result<()> {
    let mut model = OnnxRuntime::load("model.onnx", OnnxOptions::cpu())?;
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
postprocessing belong in your application.

## Select a GPU and compile once

```rust,no_run
use native_onnx::{
    Backend, BackendSelection, CompileOptions, CompileTarget, OnnxOptions,
    OnnxRuntime, TensorRtOptions, TensorView,
};

let config = TensorRtOptions {
    fp16: true,
    ..TensorRtOptions::default()
};
let compile = CompileOptions::new(CompileTarget::TensorRt(config.clone()));
let values = [1.0_f32; 16];
let input = TensorView::f32("x", &[1, 16], &values);

// Run once. The destination must not already exist.
OnnxRuntime::compile("model.onnx", "model.ctx.onnx", compile, &[input])?;

// Later launches start here; the original model is not needed.
let options = OnnxOptions {
    backend: BackendSelection::Require(Backend::TensorRt),
    tensorrt: Some(config),
    ..OnnxOptions::default()
};
let mut model = OnnxRuntime::load("model.ctx.onnx", options)?;
let outputs = model.inference(&[input])?;
# Ok::<(), native_onnx::Error>(())
```

Compilation requires an explicit target and its settings: `CompileTarget::TensorRt(config)`,
`CompileTarget::Cuda(config)`, or `CompileTarget::Cpu`. `CompileOptions::new(target)`
also exposes threading, runtime path, and dimension overrides. It has no default
target or fallback; the artifact is validated with your inputs before saving.

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
| `inference_into(inputs, outputs)` | Fill caller-owned outputs |
| `input_mut(name)`, `run()`, `output(name)` | Reuse fixed-shape buffers; inputs start at zero |
| `CudaOptions`, `TensorRtOptions` | Device, precision, CUDA graphs, and TensorRT profiles |
| `dimension_overrides` | Resolve named dynamic dimensions |
| `parallel_execution`, `inter_threads` | Enable/configure inter-operator CPU parallelism |

CUDA graphs require eligible fixed-shape I/O and sequential execution. CUDA's
capture option defaults off; TensorRT's defaults on. TensorRT defaults to FP32
with TF32 allowed; unset `NVIDIA_TF32_OVERRIDE`, or set it to `0` with `tf32: false`.
Dynamic or unknown-rank I/O uses ordinary inference. Dense numeric and boolean
tensors are supported; strings, sequences, and sparse I/O are not.

Calls are synchronous. `OnnxRuntime` is neither `Send` nor `Sync`: construct one
inside each inference worker. Native provider crashes cannot be recovered as Rust
errors. Additional providers use `Backend::Custom`; enable the `openvino` feature
for OpenVINO configuration and install its matching native provider separately.

The pinned `ort` rc.13 dependency has an [exit-time CUDA teardown bug](https://github.com/pykeio/ort/issues/609).
This crate retains one global environment until process exit as a workaround;
model sessions and buffers are still released when dropped.

## Development

With `ORT_DYLIB_PATH` configured:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo publish --dry-run
# On a GPU host with the matching native libraries:
cargo test --tests -- --ignored
```

The crate archive includes only engine source, build script, README, license,
and Cargo metadata. Tests stay in Git; local examples, experiments, and model
assets are ignored.

Licensed under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).
See `LICENSE-APACHE` for the full text.
