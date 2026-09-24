//! Public backend selection, compilation, and inference configuration.
use ort::ep::ExecutionProviderDispatch;
use std::{collections::BTreeMap, path::PathBuf};

/// An installed ONNX Runtime execution provider.
#[derive(Clone, Debug)]
pub enum Backend {
    /// NVIDIA TensorRT execution provider.
    TensorRt,
    /// NVIDIA CUDA execution provider.
    Cuda,
    /// Apple CoreML execution provider (macOS only).
    #[cfg(target_os = "macos")]
    CoreMl,
    /// ONNX Runtime's CPU execution provider.
    Cpu,
    /// Configure an additional ORT provider without adding its native dependencies.
    /// CUDA/TensorRT dispatches are rejected: use the built-in variants and typed
    /// options so the engine can own streams and coordinate graph capture. Custom implementations must
    /// not register CUDA/TensorRT indirectly.
    Custom {
        /// Display name used in backend reports and fallback events.
        name: String,
        /// Configured provider to register with ONNX Runtime.
        provider: ExecutionProviderDispatch,
    },
}
impl Backend {
    /// Display name used in backend reports and fallback events.
    pub fn name(&self) -> &str {
        match self {
            Self::TensorRt => "TensorRT",
            Self::Cuda => "CUDA",
            #[cfg(target_os = "macos")]
            Self::CoreMl => "CoreML",
            Self::Cpu => "CPU",
            Self::Custom { name, .. } => name,
        }
    }
}

/// Build and execution options used only by TensorRT.
#[derive(Clone, Debug)]
pub struct TensorRtOptions {
    /// GPU ordinal, default 0.
    pub device_id: i32,
    /// Allow FP16 computations internally; does not change model I/O types. Default false.
    pub fp16: bool,
    /// Allow TF32 for TensorRT FP32 operations (default true).
    /// The process-wide NVIDIA_TF32_OVERRIDE must be unset for true, or 0 for false.
    pub tf32: bool,
    /// Enable CUDA graphs for fixed-shape tensor I/O with sequential execution.
    /// Defaults true. Effective only with strict GPU placement or a compiled
    /// EPContext; automatic placement, dynamic I/O and parallel execution disable it.
    /// Each session owns a stream. Setup, first-run capture and teardown are
    /// coordinated across this crate; subsequent sessions can replay concurrently.
    pub cuda_graph: bool,
    /// Maximum builder workspace in bytes, default 4 GiB on 64-bit targets.
    pub workspace_bytes: usize,
    /// Builder optimization level from 0 through 5, default 3.
    pub builder_optimization_level: u8,
    /// Maximum auxiliary streams; -1 lets TensorRT choose (the default).
    pub auxiliary_streams: i8,
    /// Allow sparse weights where supported, default true.
    pub sparsity: bool,
    /// ORT syntax, e.g. `images:1x3x640x640`. Set all three for dynamic models.
    pub min_shapes: Option<String>,
    /// Preferred dynamic shapes, in the same ORT syntax as [`Self::min_shapes`].
    pub opt_shapes: Option<String>,
    /// Maximum dynamic shapes, in the same ORT syntax as [`Self::min_shapes`].
    pub max_shapes: Option<String>,
}
impl Default for TensorRtOptions {
    fn default() -> Self {
        Self {
            device_id: 0,
            fp16: false,
            tf32: true,
            cuda_graph: true,
            workspace_bytes: usize::try_from(4_u64 * 1024 * 1024 * 1024).unwrap_or(usize::MAX),
            builder_optimization_level: 3,
            auxiliary_streams: -1,
            sparsity: true,
            min_shapes: None,
            opt_shapes: None,
            max_shapes: None,
        }
    }
}

/// Options used only by the built-in CUDA execution provider.
#[derive(Clone, Copy, Debug)]
pub struct CudaOptions {
    /// GPU ordinal, default 0.
    pub device_id: i32,
    /// Allow TF32 for CUDA FP32 operations, default true.
    pub tf32: bool,
    /// Capture and replay fixed-shape models through persistent I/O bindings.
    /// Defaults true. Requires fixed positive I/O, sequential execution and strict
    /// GPU placement (or a compiled EPContext). Automatic placement disables it.
    /// Independent sessions can replay concurrently on their own streams;
    /// see [`TensorRtOptions::cuda_graph`] for capture coordination.
    pub cuda_graph: bool,
}
impl Default for CudaOptions {
    fn default() -> Self {
        Self {
            device_id: 0,
            tf32: true,
            cuda_graph: true,
        }
    }
}
impl CudaOptions {
    pub(crate) fn validate(&self) -> crate::Result<()> {
        anyhow::ensure!(self.device_id >= 0, "Negative CUDA device id");
        Ok(())
    }
}

/// One compilation target together with its provider-specific configuration.
/// Compilation never selects a target automatically or falls back to another provider.
#[derive(Clone, Debug)]
pub enum CompileTarget {
    /// Compile an embedded TensorRT engine with these builder/execution settings.
    TensorRt(TensorRtOptions),
    /// Optimize an ONNX graph for CUDA with these provider settings.
    Cuda(CudaOptions),
    /// Compile for CoreML using MLProgram (macOS only).
    #[cfg(target_os = "macos")]
    CoreMl,
    /// Optimize for CPU; threading is configured in [`CompileOptions`].
    Cpu,
    /// Compile with an explicitly configured installed ORT provider.
    /// CUDA/TensorRT must use their typed variants, as with [`Backend::Custom`].
    Custom {
        /// Display name used in compilation reports and errors.
        name: String,
        /// Provider configured with its compilation settings.
        provider: ExecutionProviderDispatch,
    },
}

/// Compilation configuration with a mandatory target and no automatic fallback.
/// Construct with [`Self::new`], then adjust shared settings as needed.
#[derive(Clone, Debug)]
pub struct CompileOptions {
    /// Required target and its provider-specific compilation settings.
    pub target: CompileTarget,
    /// Threads within an operator during compilation/validation, default 1.
    /// Set to 0 to let ONNX Runtime choose.
    pub intra_threads: usize,
    /// Threads across operators, used when parallel execution is enabled; default 1.
    pub inter_threads: usize,
    /// Execute independent operators in parallel, default false.
    /// CUDA graph capture is disabled in this mode.
    pub parallel_execution: bool,
    /// Resolve symbolic ONNX dimensions to fixed sizes before compilation.
    pub dimension_overrides: BTreeMap<String, usize>,
}

impl CompileOptions {
    /// Select a target explicitly and use defaults for the shared settings.
    /// GPU targets must carry their configuration, including when using defaults.
    pub fn new(target: CompileTarget) -> Self {
        Self {
            target,
            intra_threads: 1,
            inter_threads: 1,
            parallel_execution: false,
            dimension_overrides: BTreeMap::new(),
        }
    }

    pub(crate) fn into_runtime_options(self) -> crate::Result<OnnxOptions> {
        let (backend, tensorrt, cuda) = match self.target {
            CompileTarget::TensorRt(config) => {
                config.validate()?;
                (Backend::TensorRt, Some(config), None)
            }
            CompileTarget::Cuda(config) => {
                config.validate()?;
                (Backend::Cuda, None, Some(config))
            }
            #[cfg(target_os = "macos")]
            CompileTarget::CoreMl => (Backend::CoreMl, None, None),
            CompileTarget::Cpu => (Backend::Cpu, None, None),
            CompileTarget::Custom { name, provider } => {
                (Backend::Custom { name, provider }, None, None)
            }
        };
        Ok(OnnxOptions {
            backend: BackendSelection::Require(backend),
            tensorrt,
            cuda,
            intra_threads: self.intra_threads,
            inter_threads: self.inter_threads,
            parallel_execution: self.parallel_execution,
            dimension_overrides: self.dimension_overrides,
            profiling: None,
        })
    }
}

/// Select one required provider, or try candidates in order with fallback.
#[derive(Clone, Debug)]
pub enum BackendSelection {
    /// Fail if registration, node placement, or inference cannot use this provider.
    Require(Backend),
    /// Allow ORT node partitioning and whole-session fallback, in this order.
    Auto(Vec<Backend>),
}
impl BackendSelection {
    pub(crate) fn candidates(&self) -> &[Backend] {
        match self {
            Self::Require(backend) => std::slice::from_ref(backend),
            Self::Auto(backends) => backends,
        }
    }
}
impl Default for BackendSelection {
    fn default() -> Self {
        #[cfg(target_os = "macos")]
        {
            Self::Auto(vec![Backend::CoreMl, Backend::Cpu])
        }
        #[cfg(cuda_platform)]
        {
            Self::Auto(vec![Backend::TensorRt, Backend::Cuda, Backend::Cpu])
        }
        #[cfg(not(any(target_os = "macos", cuda_platform)))]
        {
            Self::Auto(vec![Backend::Cpu])
        }
    }
}

/// Backend selection, threading, profiling, and model-shape configuration.
///
/// ONNX Runtime is loaded once per process through the system library loader.
/// Set `ORT_DYLIB_PATH` before starting the process to override its location.
#[derive(Clone, Debug)]
pub struct OnnxOptions {
    /// Required provider or ordered fallback candidates; defaults depend on the platform.
    pub backend: BackendSelection,
    /// Optional TensorRT overrides. `None` uses [`TensorRtOptions::default()`] when selected.
    /// Ignored by other providers; does not enable/disable TensorRT.
    pub tensorrt: Option<TensorRtOptions>,
    /// Optional CUDA overrides. `None` uses [`CudaOptions::default()`] when selected.
    /// Ignored by other providers; does not enable/disable CUDA.
    pub cuda: Option<CudaOptions>,
    /// Threads within an operator, default 1; 0 lets ONNX Runtime choose.
    pub intra_threads: usize,
    /// Threads across operators, default 1; used with parallel ORT execution.
    pub inter_threads: usize,
    /// Execute independent operators in parallel, default false.
    /// Enables `inter_threads`; CUDA graph capture is disabled in this mode.
    pub parallel_execution: bool,
    /// Resolve symbolic ONNX dimensions to fixed sizes, enabling the prepared path.
    pub dimension_overrides: BTreeMap<String, usize>,
    /// Optional ORT profiling prefix. Leave unset for timing measurements.
    pub profiling: Option<PathBuf>,
}
impl Default for OnnxOptions {
    fn default() -> Self {
        Self {
            backend: BackendSelection::default(),
            tensorrt: None,
            cuda: None,
            intra_threads: 1,
            inter_threads: 1,
            parallel_execution: false,
            dimension_overrides: BTreeMap::new(),
            profiling: None,
        }
    }
}
impl OnnxOptions {
    /// Require CPU inference without attempting GPU providers.
    pub fn cpu() -> Self {
        Self {
            backend: BackendSelection::Require(Backend::Cpu),
            ..Self::default()
        }
    }
    pub(crate) fn validate(&self) -> crate::Result<()> {
        anyhow::ensure!(
            !self.backend.candidates().is_empty(),
            "At least one backend is required"
        );
        anyhow::ensure!(
            self.intra_threads <= i32::MAX as usize && self.inter_threads <= i32::MAX as usize,
            "Thread counts must fit in a signed 32-bit integer"
        );
        anyhow::ensure!(
            self.dimension_overrides
                .values()
                .all(|&d| d > 0 && i64::try_from(d).is_ok()),
            "Invalid dimension override"
        );
        Ok(())
    }
}

impl TensorRtOptions {
    /// Validate only when TensorRT is selected, so unrelated settings cannot block CPU/CUDA.
    pub(crate) fn validate(&self) -> crate::Result<()> {
        anyhow::ensure!(self.device_id >= 0, "Negative TensorRT device id");
        anyhow::ensure!(
            self.builder_optimization_level <= 5,
            "TensorRT build level must be 0..=5"
        );
        anyhow::ensure!(self.workspace_bytes > 0, "Workspace must be positive");
        anyhow::ensure!(
            self.auxiliary_streams >= -1,
            "TensorRT auxiliary streams must be -1 or nonnegative"
        );
        let profiles = [&self.min_shapes, &self.opt_shapes, &self.max_shapes];
        anyhow::ensure!(
            profiles.iter().all(|x| x.is_some()) || profiles.iter().all(|x| x.is_none()),
            "Set all three TensorRT profiles or none"
        );
        Ok(())
    }
}
