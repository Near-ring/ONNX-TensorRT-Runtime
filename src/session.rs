//! ORT initialization, provider registration, session creation, and tensor conversion.
use crate::buffers::{self, InferenceBuffers};
use crate::{Backend, BackendSelection, CudaOptions, OnnxOptions, TensorRtOptions};
use crate::{DType, ModelInfo, Result, Tensor, TensorBuffer, TensorData, TensorSpec, TensorView};
use anyhow::{Context, bail, ensure};
use ort::{
    ep::{CPU, CUDA, TensorRT},
    session::{
        Session, SessionInputValue,
        builder::{GraphOptimizationLevel, SessionBuilder},
    },
    value::{DynValue, TensorElementType, TensorRef, ValueType},
};
use std::path::{Path, PathBuf};

fn ort_error(error: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{error}")
}

pub(super) fn initialize(options: &OnnxOptions) -> Result<()> {
    let path = options
        .runtime_path
        .clone()
        .or_else(|| std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from))
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
        .with_context(|| format!("Load existing ORT runtime {}", path.display()))?
        .commit();
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
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_error)?;
    for (name, size) in &options.dimensions {
        session_builder = session_builder
            .with_dimension_override(name, *size as i64)
            .map_err(ort_error)?;
    }
    if profiling && let Some(path) = &options.profiling {
        session_builder = session_builder.with_profiling(path).map_err(ort_error)?;
    }
    Ok(session_builder)
}

fn tensorrt_provider(options: &OnnxOptions, static_io_tensor: bool) -> Result<TensorRT> {
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
        .with_cuda_graph(static_io_tensor && settings.cuda_graph);
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

fn cuda_provider(settings: CudaOptions, static_io_tensor: bool) -> Result<CUDA> {
    settings.validate()?;
    Ok(CUDA::default()
        .with_device_id(settings.device_id)
        .with_tf32(settings.tf32)
        .with_cuda_graph(static_io_tensor && settings.cuda_graph))
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

pub(super) fn metadata(session: &Session) -> Result<ModelInfo> {
    fn specs(outlets: &[ort::value::Outlet]) -> Result<Vec<TensorSpec>> {
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
                    shape: shape.iter().map(|&d| usize::try_from(d).ok()).collect(),
                })
            })
            .collect()
    }
    Ok(ModelInfo {
        inputs: specs(session.inputs())?,
        outputs: specs(session.outputs())?,
    })
}

/// A successfully opened provider session and its reusable storage.
pub(crate) struct ActiveSession {
    // Drop buffers before the session that owns their native allocators.
    pub buffers: Option<InferenceBuffers>,
    pub session: Session,
    pub backend_index: usize,
}

/// Inspect graph I/O without running it, before selecting CUDA-graph/prepared storage.
pub(crate) fn inspect_graph_io(path: &Path, options: &OnnxOptions) -> Result<ModelInfo> {
    let probe = builder(options, false)?
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .map_err(ort_error)?
        .with_execution_providers([CPU::default().build().error_on_failure()])
        .map_err(ort_error)?
        .commit_from_file(path)?;
    metadata(&probe)
}

fn configured_builder(
    options: &OnnxOptions,
    index: usize,
    compiled: bool,
    static_io: bool,
    build_directory: Option<&Path>,
) -> Result<SessionBuilder> {
    let candidates = options.backend.candidates();
    let backend = &candidates[index];
    let strict = compiled || matches!(options.backend, BackendSelection::Require(_));
    let mut session_builder = builder(options, true)?;
    if strict && !matches!(backend, Backend::Cpu) {
        session_builder = session_builder
            .with_disable_cpu_fallback()
            .map_err(ort_error)?;
    }
    match backend {
        Backend::TensorRt => {
            let mut provider = tensorrt_provider(options, static_io)?
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
                if let Ok(provider) = cuda_provider(options.cuda.unwrap_or_default(), false) {
                    providers.push(provider.build());
                }
            }
            session_builder = session_builder
                .with_execution_providers(providers)
                .map_err(ort_error)?;
        }
        Backend::Cuda => {
            session_builder = session_builder
                .with_execution_providers([cuda_provider(
                    options.cuda.unwrap_or_default(),
                    static_io,
                )?
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
    let builder = configured_builder(options, 0, false, false, None)?;
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
    compiled: bool,
    static_io: bool,
    build_directory: Option<&Path>,
) -> Result<ActiveSession> {
    let backend = &options.backend.candidates()[index];
    let mut session_builder =
        configured_builder(options, index, compiled, static_io, build_directory)?;
    let session = session_builder.commit_from_file(path)?;
    let info = metadata(&session)?;
    let device_id = match backend {
        Backend::TensorRt => options
            .tensorrt
            .as_ref()
            .map_or(0, |settings| settings.device_id),
        Backend::Cuda => options.cuda.unwrap_or_default().device_id,
        Backend::Cpu | Backend::Custom { .. } => 0,
    };
    let buffers = if buffers::eligible(&info) && !matches!(backend, Backend::Custom { .. }) {
        Some(InferenceBuffers::new(
            &session,
            &info,
            matches!(backend, Backend::TensorRt | Backend::Cuda),
            device_id,
        )?)
    } else {
        None
    };
    Ok(ActiveSession {
        session,
        buffers,
        backend_index: index,
    })
}

impl ActiveSession {
    /// Inputs have already been validated against the model by the public API.
    pub fn inference(
        &mut self,
        info: &ModelInfo,
        inputs: &[TensorView<'_>],
    ) -> Result<Vec<Tensor>> {
        if let Some(buffers) = &mut self.buffers {
            for input in inputs {
                let slot = buffers
                    .inputs
                    .iter_mut()
                    .find(|x| x.name == input.name)
                    .expect("validated input");
                slot.view_mut()?.data.copy_from(input.data)?;
            }
            buffers.run(&mut self.session).and_then(|()| {
                buffers
                    .outputs
                    .iter()
                    .map(|s| Ok(s.view()?.to_owned()))
                    .collect()
            })
        } else {
            let values = inputs
                .iter()
                .map(|v| Ok((v.name, input_value(v)?)))
                .collect::<Result<Vec<_>>>()?;
            self.session
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
