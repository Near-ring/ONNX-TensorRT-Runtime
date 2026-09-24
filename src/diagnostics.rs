//! Actionable native-library failures without disguising model or option errors.
use crate::Backend;
use std::path::Path;

pub(crate) fn runtime_error(path: &Path, error: ort::LoadDynamicError) -> anyhow::Error {
    anyhow::Error::new(error).context(format!(
        "Cannot load ONNX Runtime at '{}'. Install ONNX Runtime >= 1.27 (C API 27) \
         for this process architecture. Make its shared library (libonnxruntime.so, \
         libonnxruntime.dylib, or onnxruntime.dll) discoverable by the system loader, \
         or set ORT_DYLIB_PATH to its full path before starting the process. \
         Make its dependencies visible via LD_LIBRARY_PATH on Linux, DYLD_LIBRARY_PATH \
         on macOS, or PATH on Windows. https://onnxruntime.ai/docs/install/",
        path.display()
    ))
}

pub(crate) fn provider_error(backend: &Backend, error: anyhow::Error) -> anyhow::Error {
    let message = format!("{error:#}").to_ascii_lowercase();
    if message.contains("no cuda-capable device")
        || message.contains("cuda driver version is insufficient")
    {
        return error.context("CUDA cannot access a compatible GPU. Check GPU access in the host/container and install the NVIDIA driver required by your CUDA runtime (R580 or newer for CUDA 13.x). Use OnnxOptions::cpu() for CPU-only inference.");
    }
    let library_failure = [
        "failed to load",
        "cannot open shared object",
        "loadlibrary",
        "not enabled",
        "not available",
        "not supported in this build",
        "unable to load",
        "could not load",
        "code 126",
        "code 127",
        "error 126",
        "error 127",
    ]
    .iter()
    .any(|needle| message.contains(needle));
    if !library_failure {
        return error.context(format!("{} provider", backend.name()));
    }
    let install = match backend {
        Backend::Cuda | Backend::TensorRt => {
            let tensorrt = if matches!(backend, Backend::TensorRt) {
                " For TensorRT install the exact major/minor version required by that ORT build \
                 (validated baseline: TensorRT 10.15.1, ABI 10; TensorRT 11 is not a substitute)."
            } else { "" };
            format!(
                "Install an ONNX Runtime >= 1.27 GPU build with the {} provider. \
                 Standard ORT 1.27+ GPU packages require CUDA 13.x (>= 13.0), cuDNN 9.x \
                 and a compatible NVIDIA driver (at least R580 for CUDA 13; your \
                 cuDNN/TensorRT build may require a newer driver).{tensorrt} \
                 Custom ORT builds need the CUDA/cuDNN/TensorRT versions they were built against. \
                 Add their library directories to LD_LIBRARY_PATH (Linux) or PATH (Windows) \
                 before starting the process. For CPU-only inference use OnnxOptions::cpu(). \
                 https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html",
                backend.name()
            )
        }
        #[cfg(target_os = "macos")]
        Backend::CoreMl => "Install an ONNX Runtime >= 1.27 build with CoreML support and use macOS >= 12 for MLProgram. For CPU-only inference use OnnxOptions::cpu().".into(),
        Backend::Cpu => "Install ONNX Runtime >= 1.27 with its CPU provider and set ORT_DYLIB_PATH to the shared library.".into(),
        Backend::Custom { .. } => "Install an ONNX Runtime >= 1.27 build containing this provider and the native dependency versions required by that build.".into(),
    };
    error.context(format!(
        "{} native libraries are unavailable. {install}",
        backend.name()
    ))
}
