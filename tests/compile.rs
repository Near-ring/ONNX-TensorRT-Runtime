#![forbid(unsafe_code)]
use native_onnx::{
    CompileOptions, CompileReport, CompileTarget, CompiledFormat, CudaOptions, OnnxOptions,
    OnnxSession, Result, TensorData, TensorRtOptions, TensorView, ep,
};
mod common;
use common::fixture;

// A compilation request cannot obtain an implicit target through Default.
static_assertions::assert_not_impl_any!(CompileOptions: Default);
static_assertions::assert_not_impl_any!(CompileTarget: Default);

#[test]
fn cpu_compile_roundtrip_and_output_validation() -> Result<()> {
    let root = std::env::temp_dir().join(format!("native-onnx-cpu-compile-{}", std::process::id()));
    std::fs::create_dir_all(&root)?;
    let source = root.join("source.onnx");
    std::fs::copy(fixture("linear.onnx"), &source)?;
    let destination = root.join("cpu.onnx");
    let inputs = [TensorView::f32("x", &[1, 16], &[1.; 16])];
    let result: CompileReport = OnnxSession::compile(
        &source,
        &destination,
        CompileOptions::new(CompileTarget::Cpu),
        &inputs,
    )?;
    assert_eq!(result.backend, "CPU");
    assert_eq!(result.format, CompiledFormat::OptimizedOnnx);
    std::fs::remove_file(&source)?;
    let mut loaded = OnnxSession::load(&destination, OnnxOptions::cpu())?;
    assert_eq!(loaded.inference(&inputs)?[0].view().as_f32()?, &[16.; 16]);
    drop(loaded);
    let before = std::fs::read(&destination)?;
    assert!(
        OnnxSession::compile(
            fixture("linear.onnx"),
            &destination,
            CompileOptions::new(CompileTarget::Cpu),
            &inputs
        )
        .is_err()
    );
    assert_eq!(before, std::fs::read(&destination)?);
    let failed = root.join("failed.onnx");
    let bad_input = [TensorView::f32("wrong", &[1], &[0.])];
    assert!(
        OnnxSession::compile(
            fixture("linear.onnx"),
            &failed,
            CompileOptions::new(CompileTarget::Cpu),
            &bad_input
        )
        .is_err()
    );
    assert!(!failed.exists());
    std::fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn compile_rejects_invalid_target_settings_without_fallback() -> Result<()> {
    let root = tempfile::tempdir()?;
    let inputs = [TensorView::f32("x", &[1, 16], &[1.; 16])];
    for (target, expected) in [
        (
            CompileTarget::TensorRt(TensorRtOptions {
                workspace_bytes: 0,
                ..TensorRtOptions::default()
            }),
            "Workspace must be positive",
        ),
        (
            CompileTarget::Cuda(CudaOptions {
                device_id: -1,
                ..CudaOptions::default()
            }),
            "Negative CUDA device id",
        ),
    ] {
        let destination = root.path().join("failed.onnx");
        let error = OnnxSession::compile(
            fixture("linear.onnx"),
            &destination,
            CompileOptions::new(target),
            &inputs,
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains(expected), "{error:#}");
        assert!(!destination.exists());
    }
    Ok(())
}

#[test]
fn compile_accepts_a_configured_custom_provider() -> Result<()> {
    let root = tempfile::tempdir()?;
    let destination = root.path().join("custom.onnx");
    let inputs = [TensorView::f32("x", &[1, 16], &[1.; 16])];
    let result = OnnxSession::compile(
        fixture("linear.onnx"),
        &destination,
        CompileOptions::new(CompileTarget::Custom {
            name: "Configured CPU".into(),
            provider: ep::CPU::default().with_arena_allocator(false).build(),
        }),
        &inputs,
    )?;
    assert_eq!(result.backend, "Configured CPU");
    let mut model = OnnxSession::load(destination, OnnxOptions::cpu())?;
    assert_eq!(model.inference(&inputs)?[0].view().as_f32()?, &[16.; 16]);
    Ok(())
}

#[test]
fn compile_applies_dimension_overrides() -> Result<()> {
    let root = tempfile::tempdir()?;
    let destination = root.path().join("fixed.onnx");
    let values = [1_i64, 2, 3, 4, 5, 6];
    let inputs = [TensorView {
        name: "tokens",
        shape: &[2, 3],
        data: TensorData::I64(&values),
    }];
    let mut options = CompileOptions::new(CompileTarget::Cpu);
    options.intra_threads = 2;
    options.inter_threads = 2;
    options.parallel_execution = true;
    options.dimension_overrides.insert("batch".into(), 2);
    OnnxSession::compile(fixture("dynamic.onnx"), &destination, options, &inputs)?;
    let mut model = OnnxSession::load(destination, OnnxOptions::cpu())?;
    assert_eq!(model.info().inputs[0].fixed_shape(), Some(vec![2, 3]));
    let outputs = model.inference(&inputs)?;
    let TensorData::I64(output) = outputs[0].view().data else {
        panic!("Expected an i64 output");
    };
    assert_eq!(output, &values);
    Ok(())
}

#[test]
fn compiled_cpu_model_does_not_depend_on_external_source_weights() -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "native-onnx-external-compile-{}",
        std::process::id()
    ));
    let source_directory = root.join("source");
    std::fs::create_dir_all(&source_directory)?;
    let source = source_directory.join("model.onnx");
    std::fs::copy(fixture("external.onnx"), &source)?;
    std::fs::copy(fixture("weights.bin"), source_directory.join("weights.bin"))?;
    let mut reference = OnnxSession::load(&source, OnnxOptions::cpu())?;
    let info = reference.info().clone();
    let shape = info.inputs[0].fixed_shape().unwrap();
    let values = vec![1.; native_onnx::element_count(&shape)?];
    let inputs = [TensorView::f32(&info.inputs[0].name, &shape, &values)];
    let expected = reference.inference(&inputs)?;
    let destination = root.join("compiled.onnx");
    OnnxSession::compile(
        &source,
        &destination,
        CompileOptions::new(CompileTarget::Cpu),
        &inputs,
    )?;
    drop(reference);
    std::fs::remove_dir_all(source_directory)?;
    let mut model = OnnxSession::load(&destination, OnnxOptions::cpu())?;
    assert_eq!(
        expected[0].view().as_f32()?,
        model.inference(&inputs)?[0].view().as_f32()?
    );
    drop(model);
    std::fs::remove_dir_all(root)?;
    Ok(())
}
