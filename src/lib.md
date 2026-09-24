# native-onnx

Synchronous ONNX inference using installed ONNX Runtime execution providers.
[`OnnxSession`] owns one model session, validates named tensors, handles configured
provider fallback, and reuses host/device buffers for eligible fixed shapes.
It does not download native libraries, preprocess images, or decode predictions.

## Native runtime discovery

Install **ONNX Runtime 1.27+** (C API 27) for the process architecture. The system
loader searches for `libonnxruntime.so` on Linux, `libonnxruntime.dylib` on macOS,
or `onnxruntime.dll` on Windows. For an installation outside its normal search
paths, add the library directory to `LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`, or
`PATH`, respectively. Alternatively set `ORT_DYLIB_PATH` to the library's full path.
Set these variables before launching the application. There is no per-session
runtime path: the first initialization selects the runtime for the whole process.

CPU requires no NVIDIA libraries. GPU execution additionally needs an ORT GPU
build and the matching CUDA/cuDNN libraries; TensorRT also needs the TensorRT
version that ORT was built against. Standard ORT 1.27+ GPU packages use CUDA 13.x
and cuDNN 9.x. The validated TensorRT baseline is 10.15.1 (ABI 10); incompatible
major versions are not substitutes. See the [installation guide](https://github.com/Near-ring/ONNX-TensorRT-Runtime#install-the-native-libraries-first)
and [ORT provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

Missing libraries return installation guidance, including minimum versions.
Print an error with `eprintln!("{error:#}")` to include its native cause.

## Load and infer

```no_run
use native_onnx::{OnnxOptions, OnnxSession, Result, TensorView};

fn main() -> Result<()> {
    let mut model = OnnxSession::load("model.onnx", OnnxOptions::cpu())?;
    let data = [1.0_f32; 16];
    let input = TensorView::f32("x", &[1, 16], &data);
    let outputs = model.inference(&[input])?;
    for output in outputs {
        println!("{} {:?}: {:?}", output.name, output.shape, output.view().as_f32()?);
    }
    Ok(())
}
```

Replace `x`, `[1, 16]`, and the data for your model. Inspect [`OnnxSession::info`]
before constructing inputs. Supply each input exactly once; names, types, shapes,
and element counts must match. Input order is arbitrary. Returned [`Tensor`]s are
owned CPU values in model output order, valid after another inference or session drop.

| Operation | Contract |
|---|---|
| [`OnnxSession::load`] | Load ONNX or embedded EPContext; detect format from contents |
| [`OnnxSession::compile`] | Compile, reopen, validate with sample inputs, then save without overwriting |
| [`OnnxSession::inference`] | Borrow inputs, return owned CPU outputs; supports dynamic shapes |
| [`OnnxSession::inference_into`] | Match caller-owned outputs by name and fill their existing storage |
| [`OnnxSession::input_mut`] / [`OnnxSession::run`] / [`OnnxSession::output`] | Update and use persistent fixed-shape buffers |
| [`OnnxSession::info`] | Input/output [`TensorSpec`]s and resolved dimensions |
| [`OnnxSession::is_prepared`] | Whether the active session has persistent tensor buffers |
| [`OnnxSession::cuda_graph_enabled`] | Whether the active session is configured for graph capture/replay |
| [`OnnxSession::backend`] | Primary provider, or `None` after exhausted fallback |
| [`OnnxSession::fallback_events`] | Chronological whole-session failures, including native error context |
| [`OnnxSession::end_profiling`] | Finish active-session profiling and return the output path |

## Select providers explicitly

```no_run
use native_onnx::{Backend, BackendSelection, CudaOptions, OnnxOptions, OnnxSession};

let options = OnnxOptions {
    backend: BackendSelection::Require(Backend::Cuda),
    cuda: Some(CudaOptions { device_id: 0, tf32: false, ..CudaOptions::default() }),
    ..OnnxOptions::default()
};
let model = OnnxSession::load("model.onnx", options)?;
assert_eq!(model.backend(), Some("CUDA"));
# Ok::<(), native_onnx::Error>(())
```

[`BackendSelection::Require`] rejects provider-registration errors, CPU node
fallback, and inference failure. [`BackendSelection::Auto`] tries ordered
candidates and permits node partitioning inside ORT. For example,
`Auto(vec![Backend::TensorRt, Backend::Cuda, Backend::Cpu])` allows GPU-to-CPU
fallback. When TensorRT is primary and a later CUDA candidate uses the same GPU,
CUDA can also serve as a secondary provider on that session's stream. A CUDA
candidate on another GPU remains available as a whole-session fallback.

`backend()` reports the primary provider; it is not an execution trace. Successful
partitioning does not add a [`FallbackEvent`]. Whole-session failures do. An
ordinary ONNX model can retry the same inputs on the next candidate. EPContext
artifacts require their compatible provider and cannot migrate to CPU on inference failure.

[`OnnxOptions::default`] uses TensorRT → CUDA → CPU on Linux/Windows, CoreML → CPU
on macOS, and CPU on other targets. Use [`OnnxOptions::cpu`] for deterministic CPU
selection. [`Backend::Custom`] supports additional ORT providers via [`ep`]; raw
CUDA/TensorRT dispatches must use the built-in variants so stream ownership and capture coordination are enforced.
The `openvino` Cargo feature enables OpenVINO configuration, not library installation.

## Configuration and defaults

| Type | Settings |
|---|---|
| [`OnnxOptions`] | `backend`, optional `cuda`/`tensorrt`, threading, dimension overrides, profiling |
| [`CudaOptions`] | GPU `device_id=0`, `tf32=true`, `cuda_graph=true` |
| [`TensorRtOptions`] | GPU `device_id=0`, `fp16=false`, `tf32=true`, `cuda_graph=true`, builder options and shape profiles |
| [`CompileOptions`] | Required [`CompileTarget`], threading, dimension overrides; no automatic target |

`cuda`/`tensorrt` set to `None` use provider defaults when that backend is selected.
Settings for an unused provider are ignored. `OnnxOptions::default()` and
`CompileOptions::new(target)` select one intra-operator thread, one inter-operator
thread, and sequential execution.
`intra_threads=0` lets ORT choose. `inter_threads` takes effect only with
`parallel_execution=true`; that mode disables CUDA graph capture.

TensorRT defaults to a 4 GiB workspace on 64-bit targets, builder optimization
level 3, auxiliary streams `-1` (automatic), and sparsity allowed. FP16/TF32 permit
internal arithmetic; they do not change tensor I/O types. TensorRT TF32 follows
the process-wide `NVIDIA_TF32_OVERRIDE`: unset it for `tf32=true`, or set it to `0`
for `tf32=false`. CUDA TF32 is configured through the provider option.

For dynamic TensorRT profiles, set all of `min_shapes`, `opt_shapes`, and
`max_shapes`, or leave all unset. Their ORT syntax is e.g. `images:1x3x640x640`.
To resolve symbolic dimensions before loading/compilation, insert positive sizes
into `dimension_overrides`, keyed by the ONNX dimension name, e.g. `"batch"`.

## Compile and deploy

```no_run
use native_onnx::{
    Backend, BackendSelection, CompileOptions, CompileTarget, OnnxOptions,
    OnnxSession, TensorRtOptions, TensorView,
};

let config = TensorRtOptions { fp16: true, ..TensorRtOptions::default() };
let data = [1.0_f32; 16];
let input = TensorView::f32("x", &[1, 16], &data);
let report = OnnxSession::compile(
    "model.onnx", "model.ctx.onnx",
    CompileOptions::new(CompileTarget::TensorRt(config.clone())), &[input],
)?;
println!("{}: {:?}", report.backend, report.format);

let mut model = OnnxSession::load("model.ctx.onnx", OnnxOptions {
    backend: BackendSelection::Require(Backend::TensorRt),
    tensorrt: Some(config),
    ..OnnxOptions::default()
})?;
let outputs = model.inference(&[input])?;
# Ok::<(), native_onnx::Error>(())
```

[`CompileReport`] describes the provider and [`CompiledFormat`] written at the
destination. CPU/CUDA export optimized ONNX. TensorRT exports an ONNX EPContext
with an embedded engine; currently one TensorRT partition is required. Raw
TensorRT `.engine` files are not accepted. Compilation validates the saved bytes
with the supplied inputs, preserves an existing destination, and does not retain
a partial output on failure. It needs native libraries and may use temporary disk space.

Recompile after incompatible changes to hardware, model, or native provider
versions. TensorRT builder options affect compilation from ONNX; changing them
when loading an already embedded engine does not rebuild or change its precision.

## Tensor ownership and shapes

[`Tensor`] owns a name, shape, and [`TensorBuffer`]. [`TensorView`] and
[`TensorViewMut`] borrow metadata and contiguous CPU storage. Their constructors
do not validate a model contract; inference does. [`TensorData`] and
[`TensorDataMut`] borrow typed slices without converting their element types.

Supported [`DType`]s are FP32, FP64, FP16 ([`struct@f16`]), BF16 ([`bf16`]), signed/unsigned
8/16/32/64-bit integers, and Boolean. Dense tensors only: strings, sparse tensors,
sequences, and maps are not supported. A provider may support fewer operations
or dtypes than the wrapper's tensor representation.

```no_run
use native_onnx::{OnnxOptions, OnnxSession, TensorData, TensorView};
let mut model = OnnxSession::load("tokens.onnx", OnnxOptions::cpu())?;
let tokens = [1_i64, 2, 3];
let outputs = model.inference(&[TensorView {
    name: "tokens", shape: &[1, 3], data: TensorData::I64(&tokens),
}])?;
# Ok::<(), native_onnx::Error>(())
```

[`TensorSpec::shape`] distinguishes unknown rank (`None`), a scalar
(`Some(vec![])`), and dynamic dimensions (`Some(vec![None, Some(3)])`). Scalars
have one element; any zero dimension makes a tensor empty. [`element_count`]
checks multiplication overflow. Inputs are contiguous, row-major arrays.

## Reuse storage

For fixed, positive I/O dimensions, the engine allocates reusable host tensors
and GPU storage where appropriate. `inference()` still copies outputs to owned
CPU tensors. Use `inference_into()` to fill your own arrays, or the prepared API
to borrow the session's persistent buffers:

```no_run
use native_onnx::{OnnxOptions, OnnxSession};
let mut model = OnnxSession::load("model.onnx", OnnxOptions::cpu())?;
if model.is_prepared() {
    model.input_mut("x")?.as_f32_mut()?.fill(1.0);
    model.run()?;
    let output = model.output("y")?;
    println!("{:?}", output.as_f32()?);
}
# Ok::<(), native_onnx::Error>(())
```

Inputs start at zero and retain values until updated. Borrowing an input for
mutation invalidates old outputs. `output()` requires a successful run for the
current inputs; copy with `to_owned()` to retain its data after another operation.
Unknown-rank or dynamic I/O uses ordinary inference instead of prepared buffers.
GPU transfers finish before calls return. This is storage reuse, not a guarantee
of zero allocations or zero host/device copies.

## Threads and CUDA graphs

[`OnnxSession`] is neither `Send` nor `Sync`. Each worker must load, use, and drop
its own session. Native setup is serialized, including TensorRT plugin registration.
Capture and teardown are also coordinated; completed sessions share access during
ordinary inference and replay. Workers can use the same model file or different
models. Concurrent host calls do not guarantee overlapping GPU kernels or
improved throughput.

```no_run
use native_onnx::{Backend, BackendSelection, OnnxOptions, OnnxSession, Result, TensorView};
let workers: Vec<_> = ["first.onnx", "second.onnx"].into_iter().map(|path| {
    std::thread::spawn(move || -> Result<()> {
        let mut model = OnnxSession::load(path, OnnxOptions {
            backend: BackendSelection::Require(Backend::TensorRt),
            ..OnnxOptions::default() // Eligible strict GPU sessions use graphs by default.
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

CUDA graphs are **on by default for both CUDA and TensorRT**. Set the selected
provider's `cuda_graph` to `false` to opt out. Capture requires fixed positive I/O,
sequential ORT execution, provider support, and strict placement (`Require` or a
compiled EPContext). `Auto` placement keeps graphs disabled so CPU partitions and
fallback continue to work. [`OnnxSession::cuda_graph_enabled`] reports the effective
configuration; the first successful run performs warm-up and capture.

Each native GPU session owns a nonblocking CUDA stream that outlives its bindings,
allocators and native session. Transfers and inference use that stream; CUDA and
TensorRT providers inside the same session use the same GPU/stream. Independent
sessions can capture their own graphs and replay concurrently on the same GPU.

A process-wide coordinator gives native setup, capture and teardown exclusive access;
ordinary inference and completed replays share access. This prevents global capture
from overlapping another session's allocations or synchronization, including when
that other session disables graphs. It also serializes TensorRT plugin registration.
A new session's setup/capture temporarily waits for in-flight calls to finish.

The coordinator covers this crate's sessions, not external CUDA/ORT callers or other
processes. Coordinate external GPU work with initialization/capture, or disable
graphs. See [ORT's capture constraints](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#using-cuda-graphs-preview).

## Errors, profiling, and performance

Public fallible operations return [`Result`] with [`Error`] context. Validate
configuration/model compatibility at startup, log full errors with `{error:#}`,
and inspect fallback events when allowing automatic selection. Native crashes
and process termination cannot be recovered as ordinary errors.

Set `OnnxOptions { profiling: Some("profiles/run".into()), .. }` before loading
to enable an ORT profile, then call `end_profiling()` for its filename. The parent
directory must exist. Leave profiling disabled for timing measurements.

Native providers determine compute performance. This crate reduces repeated
buffer setup and supports compiled TensorRT startup and optional graph replay;
benchmark your own model instead of assuming a universal speedup over `ort`,
C++, or Python. The pinned `ort` rc.13 has an [exit-time CUDA teardown issue](https://github.com/pykeio/ort/issues/609):
one global environment reference is retained until process exit. Model sessions
and buffers are released normally.

## Migrating from 0.1

- `OnnxRuntime` is now [`OnnxSession`]; the intermediate `OnnxSessionRunner` name
  and compatibility aliases are removed.
- `Compilation` is now [`CompileReport`].
- `runtime_path` is removed from inference/compilation options. Use the system
  loader or the optional process-wide `ORT_DYLIB_PATH` environment variable.
- CUDA/TensorRT graphs default to `true` for eligible strict sessions. Multiple
  sessions may use graphs on the same GPU through owned streams and coordinated capture.
