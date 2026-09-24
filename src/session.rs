//! ORT initialization, provider registration, session creation, and tensor conversion.
use crate::buffers::{self, InferenceBuffers};
use crate::{Backend, BackendSelection, CudaOptions, OnnxOptions, TensorRtOptions, diagnostics};
use crate::{DType, ModelInfo, Result, Tensor, TensorBuffer, TensorData, TensorSpec, TensorView};
use crate::{cuda_graph, cuda_transfer::CudaStream};
use anyhow::{Context, bail, ensure};
#[cfg(target_os = "macos")]
use ort::ep::{CoreML, coreml::ModelFormat as CoreMlModelFormat};
use ort::{
    ep::{CPU, CUDA, TensorRT},
    session::{
        Session, SessionInputValue,
        builder::{GraphOptimizationLevel, SessionBuilder},
    },
    value::{DynValue, TensorElementType, TensorRef, ValueType},
};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

// ort rc.13's .fini_array hook can call ReleaseEnv after CUDA's C++ teardown,
// causing use-after-free at process exit (https://github.com/pykeio/ort/issues/609).
// Keep exactly one environment reference until the OS reclaims the process.
// Session/buffer allocations still drop normally; nothing is retained per model.
// Revisit when a crates.io ort release includes the manual-environment API (#610).
static ENVIRONMENT: OnceLock<Arc<ort::environment::Environment>> = OnceLock::new();

fn ort_error(error: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{error}")
}

pub(super) fn initialize() -> Result<()> {
    let _guard = cuda_graph::exclusive()?;
    let path = std::env::var_os("ORT_DYLIB_PATH")
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| {
            format!(
                "{}onnxruntime{}",
                std::env::consts::DLL_PREFIX,
                std::env::consts::DLL_SUFFIX
            )
            .into()
        });
    ort::init_from(&path)
        .map_err(|error| diagnostics::runtime_error(&path, error))?
        .commit();
    if ENVIRONMENT.get().is_none() {
        let environment = ort::environment::Environment::current()?;
        let _ = ENVIRONMENT.set(environment);
    }
    Ok(())
}

fn builder(options: &OnnxOptions, profiling: bool) -> Result<SessionBuilder> {
    let mut session_builder = Session::builder()?
        .with_no_environment_execution_providers()
        .map_err(ort_error)?
        .with_intra_threads(options.intra_threads)
        .map_err(ort_error)?
        .with_inter_threads(options.inter_threads)
        .map_err(ort_error)?
        .with_parallel_execution(options.parallel_execution)
        .map_err(ort_error)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_error)?;
    for (name, size) in &options.dimension_overrides {
        session_builder = session_builder
            .with_dimension_override(name, *size as i64)
            .map_err(ort_error)?;
    }
    if profiling && let Some(path) = &options.profiling {
        session_builder = session_builder.with_profiling(path).map_err(ort_error)?;
    }
    Ok(session_builder)
}

fn tensorrt_provider(options: &OnnxOptions, cuda_graph: bool) -> Result<TensorRT> {
    let defaults = TensorRtOptions::default();
    let settings = options.tensorrt.as_ref().unwrap_or(&defaults);
    settings.validate()?;
    check_tf32(settings.tf32)?;
    let mut provider = TensorRT::default()
        .with_device_id(settings.device_id)
        .with_fp16(settings.fp16)
        .with_int8(false)
        .with_sparsity(settings.sparsity)
        .with_max_workspace_size(settings.workspace_bytes)
        .with_builder_optimization_level(settings.builder_optimization_level)
        .with_auxiliary_streams(settings.auxiliary_streams)
        .with_cuda_graph(cuda_graph);
    if let Some(v) = &settings.min_shapes {
        provider = provider.with_profile_min_shapes(v);
    }
    if let Some(v) = &settings.opt_shapes {
        provider = provider.with_profile_opt_shapes(v);
    }
    if let Some(v) = &settings.max_shapes {
        provider = provider.with_profile_max_shapes(v);
    }
    Ok(provider)
}

fn cuda_provider(settings: CudaOptions, cuda_graph: bool) -> Result<CUDA> {
    settings.validate()?;
    Ok(CUDA::default()
        .with_device_id(settings.device_id)
        .with_tf32(settings.tf32)
        .with_cuda_graph(cuda_graph))
}

fn check_tf32(enabled: bool) -> Result<()> {
    let override_value = std::env::var_os("NVIDIA_TF32_OVERRIDE");
    let valid = if enabled {
        override_value.is_none()
    } else {
        override_value.as_deref().and_then(|value| value.to_str()) == Some("0")
    };
    ensure!(
        valid,
        "TensorRT TF32 follows process NVIDIA_TF32_OVERRIDE: unset it for tf32=true, set to 0 for tf32=false"
    );
    Ok(())
}

fn metadata(session: &Session, declared: &ModelInfo) -> Result<ModelInfo> {
    fn specs(outlets: &[ort::value::Outlet], declared: &[TensorSpec]) -> Result<Vec<TensorSpec>> {
        outlets
            .iter()
            .map(|outlet| {
                let ValueType::Tensor { ty, shape, .. } = outlet.dtype() else {
                    bail!("Only dense tensor I/O is supported: {}", outlet.name());
                };
                let dtype = match ty {
                    TensorElementType::Float32 => DType::F32,
                    TensorElementType::Float64 => DType::F64,
                    TensorElementType::Float16 => DType::F16,
                    TensorElementType::Bfloat16 => DType::BF16,
                    TensorElementType::Int64 => DType::I64,
                    TensorElementType::Int32 => DType::I32,
                    TensorElementType::Int16 => DType::I16,
                    TensorElementType::Int8 => DType::I8,
                    TensorElementType::Uint64 => DType::U64,
                    TensorElementType::Uint32 => DType::U32,
                    TensorElementType::Uint16 => DType::U16,
                    TensorElementType::Uint8 => DType::U8,
                    TensorElementType::Bool => DType::Bool,
                    _ => bail!("Unsupported I/O dtype {ty:?}"),
                };
                Ok(TensorSpec {
                    name: outlet.name().into(),
                    dtype,
                    // ORT exposes unknown rank and scalar rank as the same empty
                    // shape. Keep that distinction from the original ONNX graph.
                    shape: if declared
                        .iter()
                        .any(|s| s.name == outlet.name() && s.shape.is_none())
                    {
                        None
                    } else {
                        Some(shape.iter().map(|&d| usize::try_from(d).ok()).collect())
                    },
                })
            })
            .collect()
    }
    Ok(ModelInfo {
        inputs: specs(session.inputs(), &declared.inputs)?,
        outputs: specs(session.outputs(), &declared.outputs)?,
    })
}

/// A successfully opened provider session and its reusable storage.
pub(crate) struct ActiveSession {
    // Option allows Drop to destroy ALL native resources while holding the gate.
    native: Option<NativeSession>,
    pub backend_index: usize,
    pub info: ModelInfo,
}

struct NativeSession {
    // Field order: bindings/tensors/allocators, then ORT, then the owned stream.
    buffers: Option<InferenceBuffers>,
    session: Session,
    stream: Option<CudaStream>,
    graph_enabled: bool,
    capture_pending: bool,
}

impl Drop for ActiveSession {
    fn drop(&mut self) {
        let _guard = cuda_graph::teardown();
        if let Some(stream) = self
            .native
            .as_ref()
            .and_then(|native| native.stream.as_ref())
        {
            // Also drain partial work after an inference error before freeing tensors.
            let _ = stream.synchronize();
        }
        drop(self.native.take());
    }
}

fn stream_for_backend(options: &OnnxOptions, index: usize) -> Result<Option<CudaStream>> {
    let device = match &options.backend.candidates()[index] {
        Backend::Cuda => {
            let settings = options.cuda.unwrap_or_default();
            settings.validate()?;
            Some(settings.device_id)
        }
        Backend::TensorRt => {
            let defaults = TensorRtOptions::default();
            let settings = options.tensorrt.as_ref().unwrap_or(&defaults);
            settings.validate()?;
            check_tf32(settings.tf32)?;
            Some(settings.device_id)
        }
        _ => None,
    };
    device.map(CudaStream::new).transpose()
}

fn configured_builder(
    options: &OnnxOptions,
    index: usize,
    is_ep_context: bool,
    cuda_graph: bool,
    stream: Option<&CudaStream>,
    build_directory: Option<&Path>,
) -> Result<SessionBuilder> {
    let candidates = options.backend.candidates();
    let backend = &candidates[index];
    let strict = is_ep_context || matches!(options.backend, BackendSelection::Require(_));
    if let Backend::Custom { provider, .. } = backend {
        ensure!(
            provider.downcast_ref::<CUDA>().is_none()
                && provider.downcast_ref::<TensorRT>().is_none(),
            "Custom CUDA/TensorRT providers are not supported: use Backend::Cuda with CudaOptions \
             or Backend::TensorRt with TensorRtOptions (CompileTarget::Cuda/TensorRt for compilation) \
             so native-onnx can own the stream and coordinate CUDA graph capture"
        );
    }
    let mut session_builder = builder(options, true)?;
    let cpu_target = matches!(backend, Backend::Cpu)
        || matches!(backend, Backend::Custom { provider, .. } if provider.downcast_ref::<CPU>().is_some());
    if strict && !cpu_target {
        session_builder = session_builder
            .with_disable_cpu_fallback()
            .map_err(ort_error)?;
    }
    match backend {
        Backend::TensorRt => {
            let mut provider = stream
                .expect("TensorRT stream")
                .tensorrt_provider(tensorrt_provider(options, cuda_graph)?)
                .with_engine_cache(false)
                .with_ep_context_embed_mode(1);
            let configured_cache = std::env::var_os("ORT_TENSORRT_CACHE_PATH")
                .filter(|path| !path.is_empty())
                .map(PathBuf::from);
            if let Some(directory) = build_directory.or(configured_cache.as_deref()) {
                let directory = directory
                    .to_str()
                    .context("TensorRT cache path must be UTF-8")?;
                provider = provider
                    .with_engine_cache(true)
                    .with_engine_cache_path(directory);
            }
            let mut providers = vec![provider.build().error_on_failure()];
            if !strict
                && candidates[index + 1..]
                    .iter()
                    .any(|p| matches!(p, Backend::Cuda))
            {
                // A secondary CUDA provider is optional in Auto mode. Invalid options
                // are handled as an error if CUDA is later tried as the primary provider.
                let settings = options.cuda.unwrap_or_default();
                let trt_device = options.tensorrt.as_ref().map_or(0, |s| s.device_id);
                // Providers inside one session must use the same GPU and stream.
                // A different-device CUDA candidate remains a whole-session fallback.
                if settings.device_id == trt_device
                    && let Ok(provider) = cuda_provider(settings, false)
                {
                    providers.push(
                        stream
                            .expect("TensorRT stream")
                            .cuda_provider(provider)
                            .build(),
                    );
                }
            }
            session_builder = session_builder
                .with_execution_providers(providers)
                .map_err(ort_error)?;
        }
        Backend::Cuda => {
            session_builder = session_builder
                .with_execution_providers([stream
                    .expect("CUDA stream")
                    .cuda_provider(cuda_provider(options.cuda.unwrap_or_default(), cuda_graph)?)
                    .build()
                    .error_on_failure()])
                .map_err(ort_error)?;
        }
        #[cfg(target_os = "macos")]
        Backend::CoreMl => {
            session_builder = session_builder
                .with_execution_providers([CoreML::default()
                    .with_model_format(CoreMlModelFormat::MLProgram)
                    .build()
                    .error_on_failure()])
                .map_err(ort_error)?;
        }
        Backend::Cpu => {
            session_builder = session_builder
                .with_execution_providers([CPU::default().build().error_on_failure()])
                .map_err(ort_error)?;
        }
        Backend::Custom { provider, .. } => {
            session_builder = session_builder
                .with_execution_providers([provider.clone().error_on_failure()])
                .map_err(ort_error)?;
        }
    }
    Ok(session_builder)
}

/// Use ORT's provider-independent compiler; EPs determine which native state they export.
pub(crate) fn compile(source: &Path, options: &OnnxOptions) -> Result<Vec<u8>> {
    let _guard = cuda_graph::exclusive()?;
    let stream = stream_for_backend(options, 0)?;
    let builder = configured_builder(options, 0, false, false, stream.as_ref(), None)
        .map_err(|error| diagnostics::provider_error(&options.backend.candidates()[0], error))?;
    let compiled = ort::compiler::ModelCompiler::new(builder)?
        .with_model_from_file(source)?
        .with_embed_ep_context()?
        .compile_to_buffer()?;
    Ok(compiled.as_slice().to_vec())
}

pub(crate) fn load(
    path: &Path,
    options: &OnnxOptions,
    index: usize,
    is_ep_context: bool,
    declared: &ModelInfo,
    build_directory: Option<&Path>,
) -> Result<ActiveSession> {
    let backend = &options.backend.candidates()[index];
    // Includes CUDA initialization and allocation, TensorRT plugin registration,
    // and failed-load cleanup. No session may capture while these operations run.
    let _guard = cuda_graph::exclusive()?;
    let stream = stream_for_backend(options, index)
        .map_err(|error| diagnostics::provider_error(backend, error))?;
    // Automatic placement can include CPU partitions and data-dependent copies.
    // Keep that path graph-free rather than letting the default break fallback.
    let graph_enabled = buffers::eligible(declared)
        && !options.parallel_execution
        && (is_ep_context || matches!(options.backend, BackendSelection::Require(_)))
        && match backend {
            Backend::Cuda => options.cuda.unwrap_or_default().cuda_graph,
            Backend::TensorRt => {
                options
                    .tensorrt
                    .as_ref()
                    .unwrap_or(&TensorRtOptions::default())
                    .cuda_graph
            }
            _ => false,
        };
    let mut session_builder = configured_builder(
        options,
        index,
        is_ep_context,
        graph_enabled,
        stream.as_ref(),
        build_directory,
    )
    .map_err(|error| diagnostics::provider_error(backend, error))?;
    let session = session_builder
        .commit_from_file(path)
        .map_err(|error| diagnostics::provider_error(backend, error.into()))?;
    let info = metadata(&session, declared)?;
    ensure!(
        !graph_enabled || buffers::eligible(&info),
        "CUDA graph capture requires fixed positive native I/O shapes; set cuda_graph=false for this model"
    );
    let device_id = match backend {
        Backend::TensorRt => Some(
            options
                .tensorrt
                .as_ref()
                .map_or(0, |settings| settings.device_id),
        ),
        Backend::Cuda => Some(options.cuda.unwrap_or_default().device_id),
        #[cfg(target_os = "macos")]
        Backend::CoreMl => None,
        Backend::Cpu | Backend::Custom { .. } => None,
    };
    let buffers = if buffers::eligible(&info) && !matches!(backend, Backend::Custom { .. }) {
        Some(InferenceBuffers::new(
            &session,
            &info,
            device_id,
            !is_ep_context && matches!(options.backend, BackendSelection::Auto(_)),
        )?)
    } else {
        None
    };
    Ok(ActiveSession {
        native: Some(NativeSession {
            buffers,
            session,
            stream,
            graph_enabled,
            capture_pending: graph_enabled,
        }),
        backend_index: index,
        info,
    })
}

impl ActiveSession {
    pub fn buffers(&self) -> Option<&InferenceBuffers> {
        self.native.as_ref().expect("live session").buffers.as_ref()
    }

    pub fn buffers_mut(&mut self) -> Option<&mut InferenceBuffers> {
        self.native.as_mut().expect("live session").buffers.as_mut()
    }

    pub fn graph_enabled(&self) -> bool {
        self.native.as_ref().expect("live session").graph_enabled
    }

    pub fn end_profiling(&mut self) -> Result<String> {
        let _guard = cuda_graph::shared()?;
        Ok(self
            .native
            .as_mut()
            .expect("live session")
            .session
            .end_profiling()?)
    }

    pub fn run(&mut self) -> Result<()> {
        let native = self.native.as_mut().expect("live session");
        // ORT completes warm-up and capture inside its first successful Run
        // (including its internal reruns). A failure must not mark it complete.
        let _capture = if native.capture_pending {
            Some(cuda_graph::exclusive()?)
        } else {
            None
        };
        let _execution = if native.capture_pending {
            None
        } else {
            Some(cuda_graph::shared()?)
        };
        native
            .buffers
            .as_mut()
            .context("Prepared buffers unavailable")?
            .run(&mut native.session, native.stream.as_ref())?;
        native.capture_pending = false;
        Ok(())
    }

    /// Inputs have already been validated against the model by the public API.
    pub fn inference(
        &mut self,
        info: &ModelInfo,
        inputs: &[TensorView<'_>],
    ) -> Result<Vec<Tensor>> {
        if let Some(buffers) = self.buffers_mut() {
            for input in inputs {
                let slot = buffers
                    .input_tensors
                    .iter_mut()
                    .find(|x| x.name == input.name)
                    .expect("validated input");
                slot.view_mut()?.data.copy_from(input.data)?;
            }
            self.run()?;
            self.buffers()
                .expect("prepared buffers")
                .output_tensors
                .iter()
                .map(|s| Ok(s.view()?.to_owned()))
                .collect()
        } else {
            let _guard = cuda_graph::shared()?;
            let values = inputs
                .iter()
                .map(|v| Ok((v.name, input_value(v)?)))
                .collect::<Result<Vec<_>>>()?;
            self.native
                .as_mut()
                .expect("live session")
                .session
                .run(values)
                .map_err(ort_error)
                .and_then(|values| {
                    info.outputs
                        .iter()
                        .map(|s| owned_output(&s.name, &values[s.name.as_str()], s.dtype))
                        .collect()
                })
        }
    }
}

fn input_value<'a>(input: &'a TensorView<'a>) -> Result<SessionInputValue<'a>> {
    macro_rules! convert {
        ($($variant:ident),+) => {
            match input.data {
                $(TensorData::$variant(data) => {
                    Ok(TensorRef::from_array_view((input.shape, data))?.into())
                }),+
            }
        };
    }
    convert!(
        F32, F64, F16, BF16, I64, I32, I16, I8, U64, U32, U16, U8, Bool
    )
}

fn owned_output(name: &str, value: &DynValue, dtype: DType) -> Result<Tensor> {
    macro_rules! convert {
        ($($variant:ident: $element:ty),+) => {
            match dtype {
                $(DType::$variant => {
                    let (shape, data) = value.try_extract_tensor::<$element>()?;
                    let shape = shape.iter()
                        .map(|&dimension| usize::try_from(dimension).context("Negative output dimension"))
                        .collect::<Result<_>>()?;
                    Ok(Tensor {
                        name: name.into(),
                        shape,
                        data: TensorBuffer::$variant(data.to_vec()),
                    })
                }),+
            }
        };
    }
    convert!(F32: f32, F64: f64, F16: half::f16, BF16: half::bf16, I64: i64, I32: i32, I16: i16, I8: i8, U64: u64, U32: u32, U16: u16, U8: u8, Bool: bool)
}
