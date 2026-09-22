# safe-inference

Safe Rust ONNX inference using an existing ONNX Runtime installation. The API has three steps:
`compile`, `load`, and `inference`. Compilation is optional when loading an ordinary ONNX model.

```rust,no_run
use safe_inference::{Backend, BackendSelection, OnnxOptions, OnnxRuntime, TensorView};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image_buffer = vec![0.0_f32; 3 * 640 * 640];
    let input = TensorView::f32("images", &[1, 3, 640, 640], &image_buffer);
    let options = OnnxOptions {
        backend: BackendSelection::Require(Backend::TensorRt),
        ..OnnxOptions::default()
    };

    // Compile once. Produces a single file with the TensorRT engine embedded.
    OnnxRuntime::compile("models/yolo11m.onnx", "models/yolo11m.ctx.onnx",
        options.clone(), &[input])?;

    // Later launches start here; the original ONNX file is no longer needed.
    let mut model = OnnxRuntime::load("models/yolo11m.ctx.onnx", options)?;
    let outputs = model.inference(&[input])?;
    println!("{:?}", outputs[0].shape);
    Ok(())
}
```

## Compile

`OnnxRuntime::compile(source, destination, options, inputs)` honors backend selection and returns
`Compilation { backend, format, fallback_events }`. Compilation is a common API; each provider
chooses the representation it can export:

| Backend | Compilation output |
|---|---|
| TensorRT | Embedded `EPContext` containing a native engine; builder level **3** by default |
| CPU / CUDA | Optimized ONNX graph through ORT's `ModelCompiler` |
| Custom providers, including OpenVINO | ORT's `ModelCompiler`, when supported by the installed provider; otherwise an error |

CPU, CUDA, and TensorRT round trips are tested here. OpenVINO/custom native compilation has not
been tested or installed. Unsupported export returns an error; the API does not promise every
provider produces a native engine. An optimized CPU/CUDA graph still requires its runtime and
compatible execution options at load time; it is not a TensorRT plan.

`Require(backend)` compiles for exactly that provider. `Auto([...])` attempts candidates in order,
including retrying after compilation/export/validation failures. Each candidate must support the
whole graph: compilation does not quietly package a CPU fallback under another provider's name.
The returned `backend` and `format` tell the caller what was actually produced. TensorRT export
currently requires a single partition. Every artifact is reopened and run with the supplied
representative inputs before publication. An existing destination is an error; temporary build
files are cleaned up.

```rust,no_run
use safe_inference::{OnnxOptions, OnnxRuntime, TensorView};
let input = TensorView::f32("x", &[1, 16], &[1.; 16]);
let options = OnnxOptions::cpu();
let result = OnnxRuntime::compile("linear.onnx", "linear.cpu.onnx", options.clone(), &[input])?;
println!("{}: {:?}", result.backend, result.format);
let mut model = OnnxRuntime::load("linear.cpu.onnx", options)?;
let output = model.inference(&[input])?;
# Ok::<(), safe_inference::Error>(())
```

Use the intended backend and compatible settings when loading optimized graphs; `load` does not
record or check which backend originally compiled them. There are no model/engine correspondence
checks, hashes, manifests, deployment fingerprints, or automatic invalidation. The caller decides
when to recompile. Native providers still perform their own deserialization checks.

The two reported formats are `CompiledFormat::OptimizedOnnx` and `CompiledFormat::EpContext`.
Both are ONNX files; `.ctx.onnx` is a convention for embedded contexts, while raw TensorRT `.engine`
files are not accepted. See [ORT compilation and EPContext documentation](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html#compile-api)
and [offline graph optimization](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html#onlineoffline-mode).

## Load and backend selection

`OnnxRuntime::load(path, options)` detects EPContext nodes from the file contents, regardless of
its extension:

| Input | Behavior |
|---|---|
| Ordinary ONNX graph | Configure the requested provider(s), then create a session |
| Compiled EPContext | Load with a compatible configured provider; never rebuild the original graph |

Loading an ordinary ONNX graph with TensorRT builds an in-memory engine. To avoid repeated builds,
compile explicitly and load the resulting `.ctx.onnx` on later launches, or set
`ORT_TENSORRT_CACHE_PATH` before launching to persist ONNX Runtime's TensorRT
engine cache across launches. Cache files depend on the model, runtime, TensorRT
version, and GPU; rebuild them when that environment changes.

```rust,no_run
use safe_inference::{Backend, BackendSelection, OnnxOptions, OnnxRuntime};

// Require CUDA: registration, node placement, or inference failure returns an error.
let strict = OnnxOptions {
    backend: BackendSelection::Require(Backend::Cuda),
    ..OnnxOptions::default()
};
let model = OnnxRuntime::load("model.onnx", strict)?;

// Allow automatic fallback, in the given order.
let automatic = OnnxOptions {
    backend: BackendSelection::Auto(vec![Backend::TensorRt, Backend::Cuda, Backend::Cpu]),
    ..OnnxOptions::default()
};
let model = OnnxRuntime::load("model.onnx", automatic)?;
# Ok::<(), safe_inference::Error>(())
```

`Require` disables ORT's implicit CPU node fallback for non-CPU providers and never switches
providers on an inference error. `Auto` permits ORT node partitioning (including CPU fallback),
and tries the next candidate if whole-session creation or inference fails. Invalid input tensors
return errors without triggering fallback. `backend()` reports the primary provider;
`fallback_events()` records whole-session failures. In automatic mode, the primary provider does
not imply every node executes there. `OnnxOptions::cpu()` requires CPU.

A compiled context needs its native provider. `Auto` can try the configured providers to find one
that can load it, but a context cannot turn into a different backend's engine. Inference errors on
compiled contexts are returned directly. Load the original graph explicitly for a different backend.
A TensorRT context cannot execute through CPU/CUDA alone; the same principle applies to other EPs.

Additional installed ORT providers use `Backend::Custom { name, provider }`. Cargo features
`openvino`, `directml`, and `coreml` expose ORT configuration; they do not install native backends.
For example, with `--features openvino`:

```rust,no_run
use safe_inference::{Backend, BackendSelection, OnnxOptions, ep};
let options = OnnxOptions {
    backend: BackendSelection::Require(Backend::Custom {
        name: "OpenVINO".into(),
        provider: ep::OpenVINO::default().build(),
    }),
    ..OnnxOptions::default()
};
```

## Provider options

`OnnxOptions` contains common settings and two optional provider configurations:

```rust,no_run
use safe_inference::{Backend, BackendSelection, CudaOptions, OnnxOptions, TensorRtOptions};

let options = OnnxOptions {
    backend: BackendSelection::Require(Backend::Cuda),
    cuda: Some(CudaOptions { device_id: 0, tf32: false, cuda_graph: true }),
    tensorrt: None,
    ..OnnxOptions::default()
};

let options = OnnxOptions {
    tensorrt: Some(TensorRtOptions {
        builder_optimization_level: 3,
        fp16: true,
        tf32: true,
        cuda_graph: true,
        ..TensorRtOptions::default()
    }),
    ..OnnxOptions::default()
};
```

Both fields default to `None`: the selected provider uses its default settings. `None` means no
customization, not a disabled backend; `backend` controls selection. CPU/OpenVINO/custom providers
ignore both configurations. Validation occurs only when a provider is used, so unused GPU options
do not block another backend. Each built-in GPU provider has its own `device_id` (default 0).

TensorRT's FP16, TF32, CUDA graphs, builder level, workspace, sparsity, auxiliary streams, and
shape profiles belong to `TensorRtOptions`. Set `ORT_TENSORRT_CACHE_PATH` for its engine cache.
CUDA's TF32, device ID, and opt-in CUDA graph setting belong to `CudaOptions`.
CUDA graphs use persistent bindings when all graph inputs and outputs have fixed, positive
shapes and supported dense tensor types. Other graph I/O runs with CUDA graphs disabled.
Capture still requires all nodes to be eligible for the chosen GPU provider. The CUDA
default is `cuda_graph: false`.
Custom providers carry their own configuration in `Backend::Custom::provider`.

## Inference and buffers

| API | Ownership |
|---|---|
| `inference(inputs)` | Borrow input slices, return owned CPU output tensors |
| `inference_into(inputs, outputs)` | Write into caller-owned output buffers |
| `input_mut(name)`, `run()`, `output(name)` | Reuse persistent input/output buffers |
| `info()` | Input/output names, data types, and dimensions |

Dense, contiguous row-major tensors support FP32, FP64, FP16, BF16, signed and unsigned
8/16/32/64-bit integers, and bool.
Models may have multiple inputs/outputs and dynamic dimensions. For fixed, positive shapes,
prepared storage reuses pinned host buffers, device tensors, and I/O bindings across runs.
Dynamic shapes use ordinary ORT runs. `inference` allocates owned outputs;
use `inference_into` or borrowed `output` to avoid those allocations on the prepared path.

TensorRT defaults to FP16 disabled and TF32 enabled, with FP32 I/O where defined by the model,
4 GiB TensorRT workspace, builder level 3, sparsity, CUDA graphs for eligible static I/O,
ORT graph optimization level 3, and one intra/inter-op thread. `dimensions` resolves symbolic
ONNX dimensions; `TensorRtOptions::{min_shapes, opt_shapes, max_shapes}` configure dynamic profiles.
For TensorRT, unset `NVIDIA_TF32_OVERRIDE` for `tf32 = true`; set it to `0` for `tf32 = false`.
The TensorRT EP has no TF32 provider option; the API validates the process setting so the
requested configuration cannot silently be overridden.
CUDA's `CudaOptions::tf32` is configured separately through its provider API.

All crate code forbids `unsafe`. ORT and its native providers still use native code, which can crash;
safe Rust cannot turn a native process crash into a recoverable `Result`. Methods are synchronous.
`OnnxRuntime` is neither `Send` nor `Sync`; construct a separate instance inside each
concurrent inference worker.

## SAM3 folder example

The [Rust example](examples/sam3_render.rs) reads JPEG/PNG files from one folder and
writes green segmentation overlays to another. Edit its `PROMPT` constant to change
the text prompt. A Rust CLIP tokenizer encodes that text from the bundled BPE data;
no token IDs are hardcoded in the example. Image processing, ONNX inference, and
rendering run in Rust.

The retained [Mixed CUDA model](sam3/README.md) uses a mixed precision image
encoder with FP32 graph I/O, plus FP32 text and decoder graphs. The example uses
its text-only decoder. The box-capable decoder is also retained as part of the
selected strategy.

Set `ORT_DYLIB_PATH` to a compatible ONNX Runtime shared library and put its
native CUDA dependencies on `LD_LIBRARY_PATH`. Then run from this directory:

```bash
cargo run --release --example sam3_render -- samples rendered cuda
# CPU fallback (requires a CPU-capable ONNX Runtime):
cargo run --release --example sam3_render -- samples rendered_cpu cpu
```

The sample `.jpg` names may contain PNG data; the example detects the image
format from file contents. It processes images in sorted order and reuses the
model sessions. Each output is named `<input-name>.overlay.png`. It prints
per-image encoder/decoder inference time and the folder average; timing includes
model output transfer to Rust and the first inference call, but excludes
preprocessing, rendering, and model loading. The example leaves CUDA graphs
disabled by default; set `cuda_graph: true` in its `CudaOptions` to try capture.
Its I64 text encoder now has prepared buffers as well.

## Checks

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Native backends are not downloaded by this crate. The loader uses
`options.runtime_path`, then `ORT_DYLIB_PATH`, then the platform's standard
ONNX Runtime library name. CUDA requires a compatible installed runtime, CUDA
libraries, and driver. CPU inference also requires an installed CPU-capable
ONNX Runtime. `src/` contains the library API; `tests/` contains its tests.
