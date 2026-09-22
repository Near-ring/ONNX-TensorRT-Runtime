#![forbid(unsafe_code)]
use safe_inference::{
    Backend, BackendSelection, CompiledFormat, CudaOptions, OnnxOptions, OnnxRuntime, Result,
    TensorRtOptions, TensorView,
};
mod common;
use common::fixture;

#[test]
fn cpu_compile_roundtrip_and_auto_fallback() -> Result<()> {
    let root =
        std::env::temp_dir().join(format!("safe-inference-cpu-compile-{}", std::process::id()));
    std::fs::create_dir_all(&root)?;
    let source = root.join("source.onnx");
    std::fs::copy(fixture("linear.onnx"), &source)?;
    let destination = root.join("cpu.onnx");
    let inputs = [TensorView::f32("x", &[1, 16], &[1.; 16])];
    let result = OnnxRuntime::compile(&source, &destination, OnnxOptions::cpu(), &inputs)?;
    assert_eq!(result.backend, "CPU");
    assert_eq!(result.format, CompiledFormat::OptimizedOnnx);
    assert!(result.fallback_events.is_empty());
    std::fs::remove_file(&source)?;
    let mut loaded = OnnxRuntime::load(&destination, OnnxOptions::cpu())?;
    assert_eq!(loaded.inference(&inputs)?[0].view().as_f32()?, &[16.; 16]);
    drop(loaded);
    let before = std::fs::read(&destination)?;
    assert!(
        OnnxRuntime::compile(
            fixture("linear.onnx"),
            &destination,
            OnnxOptions::cpu(),
            &inputs
        )
        .is_err()
    );
    assert_eq!(before, std::fs::read(&destination)?);
    let invalid_trt = TensorRtOptions {
        workspace_bytes: 0,
        ..TensorRtOptions::default()
    };
    let invalid_cuda = CudaOptions {
        device_id: -1,
        ..CudaOptions::default()
    };
    let automatic = OnnxOptions {
        tensorrt: Some(invalid_trt.clone()),
        cuda: Some(invalid_cuda),
        ..OnnxOptions::default()
    };
    let result = OnnxRuntime::compile(
        fixture("linear.onnx"),
        root.join("auto.onnx"),
        automatic,
        &inputs,
    )?;
    assert_eq!(result.backend, "CPU");
    assert_eq!(result.fallback_events.len(), 2);
    let required = OnnxOptions {
        backend: BackendSelection::Require(Backend::TensorRt),
        tensorrt: Some(invalid_trt),
        ..OnnxOptions::default()
    };
    let failed = root.join("failed.onnx");
    assert!(OnnxRuntime::compile(fixture("linear.onnx"), &failed, required, &inputs).is_err());
    assert!(!failed.exists());
    let bad_input = [TensorView::f32("wrong", &[1], &[0.])];
    assert!(
        OnnxRuntime::compile(
            fixture("linear.onnx"),
            &failed,
            OnnxOptions::cpu(),
            &bad_input
        )
        .is_err()
    );
    assert!(!failed.exists());
    std::fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn compiled_cpu_model_does_not_depend_on_external_source_weights() -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "safe-inference-external-compile-{}",
        std::process::id()
    ));
    let source_directory = root.join("source");
    std::fs::create_dir_all(&source_directory)?;
    let source = source_directory.join("model.onnx");
    std::fs::copy(fixture("external.onnx"), &source)?;
    std::fs::copy(fixture("weights.bin"), source_directory.join("weights.bin"))?;
    let mut reference = OnnxRuntime::load(&source, OnnxOptions::cpu())?;
    let info = reference.info().clone();
    let shape = info.inputs[0].fixed_shape().unwrap();
    let values = vec![1.; safe_inference::element_count(&shape)?];
    let inputs = [TensorView::f32(&info.inputs[0].name, &shape, &values)];
    let expected = reference.inference(&inputs)?;
    let destination = root.join("compiled.onnx");
    OnnxRuntime::compile(&source, &destination, OnnxOptions::cpu(), &inputs)?;
    drop(reference);
    std::fs::remove_dir_all(source_directory)?;
    let mut model = OnnxRuntime::load(&destination, OnnxOptions::cpu())?;
    assert_eq!(
        expected[0].view().as_f32()?,
        model.inference(&inputs)?[0].view().as_f32()?
    );
    drop(model);
    std::fs::remove_dir_all(root)?;
    Ok(())
}
