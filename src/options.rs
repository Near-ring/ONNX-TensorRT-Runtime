//! Public backend selection and inference configuration.
use ort::ep::ExecutionProviderDispatch;
use std::{collections::BTreeMap, path::PathBuf};

/// An installed ONNX Runtime execution provider.
#[derive(Clone, Debug)]
pub enum Backend {
    TensorRt,
    Cuda,
    Cpu,
    /// Configure any existing ORT provider without adding its native dependencies.
    Custom {
        name: String,
        provider: ExecutionProviderDispatch,
    },
}
impl Backend {
    pub fn name(&self) -> &str {
        match self {
            Self::TensorRt => "TensorRT",
            Self::Cuda => "CUDA",
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
    pub fp16: bool,
    /// Allow TF32 for TensorRT FP32 operations (default true).
    /// The process-wide NVIDIA_TF32_OVERRIDE must be unset for true, or 0 for false.
    pub tf32: bool,
    /// Enable CUDA graphs for fixed-shape, supported tensor I/O.
    pub cuda_graph: bool,
    pub workspace_bytes: usize,
    pub builder_optimization_level: u8,
    pub auxiliary_streams: i8,
    pub sparsity: bool,
    /// ORT syntax, e.g. `images:1x3x640x640`. Set all three for dynamic models.
    pub min_shapes: Option<String>,
    pub opt_shapes: Option<String>,
    pub max_shapes: Option<String>,
}
impl Default for TensorRtOptions {
    fn default() -> Self {
        Self {
            device_id: 0,
            fp16: false,
            tf32: true,
            cuda_graph: true,
            workspace_bytes: 4 * 1024 * 1024 * 1024,
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
    pub tf32: bool,
    /// Capture and replay fixed-shape models through persistent I/O bindings.
    /// Disabled by default; ignored when graph I/O is not eligible for prepared buffers.
    pub cuda_graph: bool,
}
impl Default for CudaOptions {
    fn default() -> Self {
        Self {
            device_id: 0,
            tf32: true,
            cuda_graph: false,
        }
    }
}
impl CudaOptions {
    pub(crate) fn validate(&self) -> crate::Result<()> {
        anyhow::ensure!(self.device_id >= 0, "Negative CUDA device id");
        Ok(())
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
        Self::Auto(vec![Backend::TensorRt, Backend::Cuda, Backend::Cpu])
    }
}

#[derive(Clone, Debug)]
pub struct OnnxOptions {
    /// Existing ORT shared library; otherwise use ORT_DYLIB_PATH or the platform loader:
    /// onnxruntime.dll on Windows, libonnxruntime.so on Linux, libonnxruntime.dylib on macOS.
    /// The process shares one ORT runtime; select it before the first ORT operation.
    pub runtime_path: Option<PathBuf>,
    pub backend: BackendSelection,
    /// Optional TensorRT overrides. `None` uses [`TensorRtOptions::default()`] when selected.
    /// Ignored by CPU, CUDA, and custom providers; does not enable/disable TensorRT.
    pub tensorrt: Option<TensorRtOptions>,
    /// Optional CUDA overrides. `None` uses [`CudaOptions::default()`] when selected.
    /// Ignored by CPU/custom providers; does not enable/disable CUDA.
    pub cuda: Option<CudaOptions>,
    pub intra_threads: usize,
    pub inter_threads: usize,
    /// Resolve symbolic ONNX dimensions to fixed sizes, enabling the prepared path.
    pub dimensions: BTreeMap<String, usize>,
    /// Optional ORT profiling prefix. Leave unset for timing measurements.
    pub profiling: Option<PathBuf>,
}
impl Default for OnnxOptions {
    fn default() -> Self {
        Self {
            runtime_path: None,
            backend: BackendSelection::default(),
            tensorrt: None,
            cuda: None,
            intra_threads: 1,
            inter_threads: 1,
            dimensions: BTreeMap::new(),
            profiling: None,
        }
    }
}
impl OnnxOptions {
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
            self.dimensions
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
        let profiles = [&self.min_shapes, &self.opt_shapes, &self.max_shapes];
        anyhow::ensure!(
            profiles.iter().all(|x| x.is_some()) || profiles.iter().all(|x| x.is_none()),
            "Set all three TensorRT profiles or none"
        );
        Ok(())
    }
}
