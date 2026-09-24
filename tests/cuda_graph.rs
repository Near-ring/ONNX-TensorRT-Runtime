#![forbid(unsafe_code)]
use native_onnx::{
    Backend, BackendSelection, CompileOptions, CompileTarget, CudaOptions, OnnxOptions,
    OnnxSession, Result, TensorView,
};
mod common;
use common::fixture;

#[test]
#[ignore = "requires a host CUDA GPU and compatible ONNX Runtime libraries"]
fn independent_graph_sessions_replay_compile_and_recover_failed_loads() -> Result<()> {
    let options = OnnxOptions {
        backend: BackendSelection::Require(Backend::Cuda),
        cuda: Some(CudaOptions {
            cuda_graph: true,
            ..CudaOptions::default()
        }),
        ..OnnxOptions::default()
    };
    let bytes = std::fs::read(fixture("linear.onnx"))?;
    let mut invalid = onnx_rs::parse(&bytes)?;
    invalid.graph.as_mut().unwrap().node[0].op_type =
        onnx_rs::ast::OpType::Custom("MissingCudaGraphTestOperator");
    let file = tempfile::Builder::new().suffix(".onnx").tempfile()?;
    std::fs::write(file.path(), onnx_rs::encode(&invalid))?;
    let error = OnnxSession::load(file.path(), options.clone())
        .err()
        .expect("invalid operator must fail");
    assert!(format!("{error:#}").contains("MissingCudaGraphTestOperator"));

    let mut first = OnnxSession::load(fixture("linear.onnx"), options.clone())?;
    let mut second = OnnxSession::load(fixture("linear.onnx"), options.clone())?;
    assert!(first.cuda_graph_enabled() && second.cuda_graph_enabled());
    for value in [1., 3., 0., 7.] {
        for model in [&mut first, &mut second] {
            assert_eq!(
                model.inference(&[TensorView::f32("x", &[1, 16], &[value; 16])])?[0]
                    .view()
                    .as_f32()?,
                &[16. * value; 16]
            );
        }
    }
    // Native compilation and validation coexist with already captured sessions.
    let dir = tempfile::tempdir()?;
    OnnxSession::compile(
        fixture("linear.onnx"),
        dir.path().join("compiled.onnx"),
        CompileOptions::new(CompileTarget::Cuda(options.cuda.unwrap())),
        &[TensorView::f32("x", &[1, 16], &[1.; 16])],
    )?;
    drop(second);
    assert_eq!(
        first.inference(&[TensorView::f32("x", &[1, 16], &[2.; 16])])?[0]
            .view()
            .as_f32()?,
        &[32.; 16]
    );

    let dynamic = OnnxSession::load(fixture("dynamic.onnx"), options.clone())?;
    assert!(!dynamic.cuda_graph_enabled());
    let mut parallel = options.clone();
    parallel.parallel_execution = true;
    let mut parallel = OnnxSession::load(fixture("linear.onnx"), parallel)?;
    assert!(!parallel.cuda_graph_enabled());
    assert_eq!(
        parallel.inference(&[TensorView::f32("x", &[1, 16], &[1.; 16])])?[0]
            .view()
            .as_f32()?,
        &[16.; 16]
    );
    let mut automatic = options;
    automatic.backend = BackendSelection::Auto(vec![Backend::Cuda, Backend::Cpu]);
    assert!(!OnnxSession::load(fixture("linear.onnx"), automatic)?.cuda_graph_enabled());
    Ok(())
}

#[test]
fn custom_gpu_dispatches_cannot_bypass_stream_management() -> Result<()> {
    for provider in [
        native_onnx::ep::CUDA::default()
            .with_cuda_graph(true)
            .build(),
        native_onnx::ep::TensorRT::default()
            .with_cuda_graph(true)
            .build(),
    ] {
        let options = OnnxOptions {
            backend: BackendSelection::Require(Backend::Custom {
                name: "opaque provider".into(),
                provider: provider.clone(),
            }),
            ..OnnxOptions::default()
        };
        let error = OnnxSession::load(fixture("linear.onnx"), options)
            .err()
            .expect("custom GPU dispatch must be rejected before provider registration");
        assert!(
            format!("{error:#}").contains("use Backend::Cuda"),
            "{error:#}"
        );
        let directory = tempfile::tempdir()?;
        let error = OnnxSession::compile(
            fixture("linear.onnx"),
            directory.path().join("compiled.onnx"),
            CompileOptions::new(CompileTarget::Custom {
                name: "opaque provider".into(),
                provider,
            }),
            &[TensorView::f32("x", &[1, 16], &[1.; 16])],
        )
        .expect_err("custom compilation must not bypass stream management");
        assert!(
            format!("{error:#}").contains("use Backend::Cuda"),
            "{error:#}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires host CUDA and TensorRT; verifies native graph replay through ORT profiling"]
fn captured_sessions_skip_normal_node_execution_on_replay() -> Result<()> {
    for backend in [Backend::Cuda, Backend::TensorRt] {
        let mut counts = Vec::new();
        for enabled in [false, true] {
            let directory = tempfile::tempdir()?;
            let mut model = OnnxSession::load(
                fixture("linear.onnx"),
                OnnxOptions {
                    backend: BackendSelection::Require(backend.clone()),
                    cuda: Some(CudaOptions {
                        cuda_graph: enabled,
                        ..CudaOptions::default()
                    }),
                    tensorrt: Some(native_onnx::TensorRtOptions {
                        cuda_graph: enabled,
                        ..native_onnx::TensorRtOptions::default()
                    }),
                    profiling: Some(directory.path().join("capture-check")),
                    ..OnnxOptions::default()
                },
            )?;
            assert_eq!(model.cuda_graph_enabled(), enabled);
            for n in 1..=6 {
                let x = [n as f32; 16];
                assert_eq!(
                    model.inference(&[TensorView::f32("x", &[1, 16], &x)])?[0]
                        .view()
                        .as_f32()?,
                    &[16. * n as f32; 16]
                );
            }
            let profile = model.end_profiling()?;
            drop(model);
            let events: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(profile)?)?;
            let count = events
                .iter()
                .filter(|e| e["cat"] == "Node" && e["args"]["provider"].is_string())
                .count();
            counts.push(count);
        }
        println!(
            "{}: node executions for six calls, graphs off/on: {counts:?}",
            backend.name()
        );
        assert!(
            counts[0] >= 6,
            "ordinary inference must record each execution"
        );
        assert!(
            counts[1] > 0 && counts[1] < counts[0],
            "graph replay must bypass repeated normal node execution"
        );
    }
    Ok(())
}
