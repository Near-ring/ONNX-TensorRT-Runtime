//! Opt-in GPU integration tests; native TensorRT/CUDA libraries must already exist.
#![forbid(unsafe_code)]
use safe_inference::{
    Backend, BackendSelection, CompiledFormat, CudaOptions, OnnxOptions, OnnxRuntime, Result,
    TensorRtOptions, TensorView,
};
mod common;
use common::fixture;

#[test]
#[ignore = "requires host NVIDIA GPU and existing TensorRT libraries; builds only tiny models"]
fn compile_load_inference_and_strict_backend() -> Result<()> {
    let root = std::env::temp_dir().join(format!("safe-inference-api-test-{}", std::process::id()));
    std::fs::create_dir(&root)?;
    let source = root.join("source.onnx");
    let engine = root.join("compiled.ctx.onnx");
    std::fs::copy(fixture("linear.onnx"), &source)?;
    let options = OnnxOptions::default();
    let x = [1.; 16];
    let input = TensorView::f32("x", &[1, 16], &x);
    let compiled = OnnxRuntime::compile(&source, &engine, options.clone(), &[input])?;
    assert_eq!(compiled.backend, "TensorRT");
    assert_eq!(compiled.format, CompiledFormat::EpContext);
    assert!(OnnxRuntime::compile(&source, &engine, options.clone(), &[input]).is_err());
    std::fs::remove_file(&source)?;
    // The only deployment artifact is the embedded engine; no manifest or source is needed.
    assert_eq!(std::fs::read_dir(&root)?.count(), 1);
    let before = std::fs::read(&engine)?;
    let mtime = std::fs::metadata(&engine)?.modified()?;
    let mut model = OnnxRuntime::load(&engine, options.clone())?;
    assert_eq!(model.backend(), Some("TensorRT"));
    assert_eq!(model.inference(&[input])?[0].view().as_f32()?, &[16.; 16]);
    assert_eq!(
        model.inference(&[TensorView::f32("x", &[1, 16], &[0.; 16])])?[0]
            .view()
            .as_f32()?,
        &[0.; 16]
    );
    assert_eq!(model.inference(&[input])?[0].view().as_f32()?, &[16.; 16]);
    drop(model);
    // CUDA uses the same compile API and produces an optimized graph, not a TensorRT plan.
    let cuda_options = OnnxOptions {
        backend: BackendSelection::Require(Backend::Cuda),
        ..OnnxOptions::default()
    };
    let cuda_model = root.join("cuda.onnx");
    let compiled = OnnxRuntime::compile(
        fixture("linear.onnx"),
        &cuda_model,
        cuda_options.clone(),
        &[input],
    )?;
    assert_eq!(compiled.backend, "CUDA");
    assert_eq!(compiled.format, CompiledFormat::OptimizedOnnx);
    let mut cuda = OnnxRuntime::load(&cuda_model, cuda_options)?;
    assert_eq!(cuda.inference(&[input])?[0].view().as_f32()?, &[16.; 16]);
    drop(cuda);
    assert_eq!(before, std::fs::read(&engine)?);
    assert_eq!(mtime, std::fs::metadata(&engine)?.modified()?);
    // User owns engine/build-setting correspondence; loading ignores a different builder level.
    let mut changed = options.clone();
    changed.tensorrt = Some(TensorRtOptions {
        builder_optimization_level: 2,
        cuda_graph: false,
        ..TensorRtOptions::default()
    });
    let mut model = OnnxRuntime::load(&engine, changed)?;
    assert_eq!(model.inference(&[input])?[0].view().as_f32()?, &[16.; 16]);
    drop(model);
    assert!(OnnxRuntime::load(&engine, OnnxOptions::cpu()).is_err());
    assert!(
        OnnxRuntime::compile(
            &engine,
            root.join("invalid.ctx.onnx"),
            options.clone(),
            &[input]
        )
        .is_err()
    );
    // Contents decide the format, even if the extension changes.
    let renamed = root.join("compiled.bin");
    std::fs::rename(&engine, &renamed)?;
    let mut model = OnnxRuntime::load(&renamed, options.clone())?;
    assert_eq!(model.inference(&[input])?[0].view().as_f32()?, &[16.; 16]);
    drop(model);
    // Strict GPU selection succeeds when all nodes can run there.
    for backend in [Backend::TensorRt, Backend::Cuda] {
        let mut strict = options.clone();
        let expected = backend.name().to_owned();
        if matches!(backend, Backend::Cuda) {
            strict.cuda = Some(CudaOptions {
                tf32: false,
                ..CudaOptions::default()
            });
            strict.tensorrt = Some(TensorRtOptions {
                workspace_bytes: 0,
                ..TensorRtOptions::default()
            });
        } else {
            strict.cuda = Some(CudaOptions {
                device_id: -1,
                ..CudaOptions::default()
            });
        }
        strict.backend = BackendSelection::Require(backend);
        let mut model = OnnxRuntime::load(fixture("linear.onnx"), strict)?;
        assert_eq!(model.backend(), Some(expected.as_str()));
        assert!(model.fallback_events().is_empty());
        assert_eq!(model.inference(&[input])?[0].view().as_f32()?, &[16.; 16]);
    }
    // A CPU-only operator must fail strict CUDA placement, rather than silently use CPU.
    let cpu_graph = root.join("unique.onnx");
    write_cpu_only_graph(&cpu_graph)?;
    let strict = OnnxOptions {
        backend: BackendSelection::Require(Backend::Cuda),
        ..OnnxOptions::default()
    };
    assert!(OnnxRuntime::load(&cpu_graph, strict).is_err());
    let auto = OnnxOptions {
        backend: BackendSelection::Auto(vec![Backend::Cuda, Backend::Cpu]),
        ..OnnxOptions::default()
    };
    let mut model = OnnxRuntime::load(&cpu_graph, auto)?;
    let values = [2., 1., 2., 1.];
    assert_eq!(
        model.inference(&[TensorView::f32("x", &[4], &values)])?[0]
            .view()
            .as_f32()?,
        &[1., 2.]
    );
    drop(model);
    // A broken wrapper returns an error rather than rebuilding or selecting another provider.
    std::fs::write(&renamed, b"invalid model")?;
    assert!(OnnxRuntime::load(&renamed, options).is_err());
    std::fs::remove_dir_all(root)?;
    Ok(())
}

fn write_cpu_only_graph(path: &std::path::Path) -> Result<()> {
    use onnx_rs::ast::*;
    fn value<'a>(name: &'a str, size: Dimension<'a>) -> ValueInfo<'a> {
        ValueInfo {
            name,
            r#type: Some(TypeProto {
                value: Some(TypeValue::Tensor(TensorTypeProto {
                    elem_type: DataType::Float,
                    shape: Some(TensorShape {
                        dim: vec![TensorShapeDimension {
                            value: size,
                            ..Default::default()
                        }],
                    }),
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    }
    let model = Model {
        ir_version: 8,
        opset_import: vec![OperatorSetId {
            domain: "",
            version: 17,
        }],
        graph: Some(Graph {
            name: "cpu_unique",
            input: vec![value("x", Dimension::Value(4))],
            output: vec![value("y", Dimension::Param("count"))],
            node: vec![Node {
                op_type: OpType::Unique,
                input: vec!["x"],
                output: vec!["y"],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    std::fs::write(path, onnx_rs::encode(&model))?;
    Ok(())
}
