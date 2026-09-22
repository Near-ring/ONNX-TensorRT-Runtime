#![forbid(unsafe_code)]
//! ONNX inference with explicit TensorRT compilation and selectable provider fallback.
//! Use [`OnnxRuntime::inference_into`] for caller-owned outputs, or
//! [`OnnxRuntime::input_mut`], [`OnnxRuntime::run`], and [`OnnxRuntime::output`]
//! for persistent input buffers and borrowed outputs.

mod api;
mod buffers;
mod engine;
mod options;
mod session;
mod tensor;

pub use anyhow::{Error, Result};
pub use api::{Compilation, CompiledFormat, FallbackEvent, ModelInfo, OnnxRuntime};
pub use options::{Backend, BackendSelection, CudaOptions, OnnxOptions, TensorRtOptions};
/// Configure additional installed ONNX Runtime providers through [`Backend::Custom`].
pub use ort::ep;
pub use tensor::{
    DType, Tensor, TensorBuffer, TensorData, TensorDataMut, TensorSpec, TensorView, TensorViewMut,
    bf16, element_count, f16,
};
