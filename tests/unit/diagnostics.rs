use crate::{Backend, diagnostics::*};
use std::path::Path;

#[test]
fn installation_advice_is_limited_to_library_errors() {
    let error = provider_error(
        &Backend::Cuda,
        anyhow::anyhow!("Failed to load libcudnn.so.9"),
    );
    let text = format!("{error:#}");
    assert!(text.contains("Install") && text.contains("13.0") && text.contains("cuDNN 9"));
    assert!(text.contains("libcudnn.so.9"));
    let error = provider_error(
        &Backend::TensorRt,
        anyhow::anyhow!("LoadLibrary failed with error 126"),
    );
    assert!(error.to_string().contains("10.15.1"));
    let error = provider_error(&Backend::Cuda, anyhow::anyhow!("Invalid model input shape"));
    assert!(!error.to_string().contains("Install"));
}

#[test]
fn unsupported_runtime_version_has_installation_instructions() {
    let path = Path::new("/old/libonnxruntime.so");
    let error = runtime_error(
        path,
        ort::LoadDynamicError::BadVersion {
            version_str: "1.20.0".into(),
            path: path.into(),
        },
    );
    let text = format!("{error:#}");
    assert!(text.contains("Install ONNX Runtime >= 1.27"));
    assert!(text.contains("1.20.0") && text.contains("ORT_DYLIB_PATH"));
}
