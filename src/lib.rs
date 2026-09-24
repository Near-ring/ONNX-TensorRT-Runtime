#![deny(unsafe_code)]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

mod api;
mod buffers;
// The only engine module allowed to call the dynamically loaded CUDA driver.
#[allow(unsafe_code)]
mod cuda_transfer;
mod diagnostics;
mod engine;
mod options;
mod session;
mod tensor;

pub use anyhow::{Error, Result};
pub use api::{Compilation, CompiledFormat, FallbackEvent, ModelInfo, OnnxRuntime};
pub use options::{
    Backend, BackendSelection, CompileOptions, CompileTarget, CudaOptions, OnnxOptions,
    TensorRtOptions,
};
/// Configure additional installed ONNX Runtime providers through [`Backend::Custom`].
pub use ort::ep;
pub use tensor::{
    DType, Tensor, TensorBuffer, TensorData, TensorDataMut, TensorSpec, TensorView, TensorViewMut,
    bf16, element_count, f16,
};
