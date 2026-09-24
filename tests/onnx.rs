#![forbid(unsafe_code)]
use native_onnx::{
    Backend, BackendSelection, CudaOptions, OnnxOptions, OnnxSession, Result, TensorData,
    TensorDataMut, TensorRtOptions, TensorView, TensorViewMut, ep,
};
mod common;
use common::fixture;

// CUDA graph sessions cannot safely move between or be shared by threads.
static_assertions::assert_not_impl_any!(OnnxSession: Send, Sync);

#[test]
fn defaults_use_level_three_and_opt_in_fp16() {
    let config = OnnxOptions::default();
    assert!(config.tensorrt.is_none() && config.cuda.is_none());
    let tensorrt = TensorRtOptions::default();
    assert_eq!(tensorrt.builder_optimization_level, 3);
    assert!(!tensorrt.fp16 && tensorrt.tf32 && tensorrt.cuda_graph && tensorrt.sparsity);
    let cuda = CudaOptions::default();
    assert!(cuda.tf32 && cuda.cuda_graph);
    assert_eq!(tensorrt.workspace_bytes, 4 * 1024 * 1024 * 1024);
    let BackendSelection::Auto(backends) = config.backend else {
        panic!("Expected auto fallback")
    };
    #[cfg(target_os = "macos")]
    assert_eq!(
        backends.iter().map(Backend::name).collect::<Vec<_>>(),
        ["CoreML", "CPU"]
    );
    #[cfg(cuda_platform)]
    assert_eq!(
        backends.iter().map(Backend::name).collect::<Vec<_>>(),
        ["TensorRT", "CUDA", "CPU"]
    );
    #[cfg(not(any(target_os = "macos", cuda_platform)))]
    assert_eq!(
        backends.iter().map(Backend::name).collect::<Vec<_>>(),
        ["CPU"]
    );
}

#[cfg(target_os = "macos")]
#[test]
fn macos_default_runs_with_coreml_or_cpu_fallback() -> Result<()> {
    let mut model = OnnxSession::load(fixture("linear.onnx"), OnnxOptions::default())?;
    assert!(matches!(model.backend(), Some("CoreML" | "CPU")));
    let input = TensorView::f32("x", &[1, 16], &[1.; 16]);
    assert_eq!(model.inference(&[input])?[0].view().as_f32()?, &[16.; 16]);
    Ok(())
}

#[test]
fn cpu_multi_input_buffers_and_validation() -> Result<()> {
    let mut model = OnnxSession::load(fixture("multi.onnx"), OnnxOptions::cpu())?;
    assert_eq!(model.backend(), Some("CPU"));
    assert!(model.is_prepared());
    assert!(model.output("sum").is_err());
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y = [2.0; 6];
    let inputs = [
        TensorView::f32("y", &[2, 3], &y),
        TensorView::f32("x", &[2, 3], &x),
    ];
    let values = model.inference(&inputs)?;
    assert_eq!(values[0].view().as_f32()?, &[3., 4., 5., 6., 7., 8.]);
    let mut product = [0.; 6];
    let mut sum = [0.; 6];
    model.inference_into(
        &inputs,
        &mut [
            TensorViewMut::f32("product", &[2, 3], &mut product),
            TensorViewMut::f32("sum", &[2, 3], &mut sum),
        ],
    )?;
    assert_eq!(product, [2., 4., 6., 8., 10., 12.]);
    assert_eq!(sum, [3., 4., 5., 6., 7., 8.]);
    let mut wrong = [99.; 5];
    assert!(
        model
            .inference_into(
                &inputs,
                &mut [
                    TensorViewMut::f32("sum", &[2, 3], &mut wrong),
                    TensorViewMut::f32("product", &[2, 3], &mut product)
                ]
            )
            .is_err()
    );
    assert_eq!(wrong, [99.; 5]);
    assert!(model.inference(&[inputs[0], inputs[0]]).is_err());
    assert!(model.fallback_events().is_empty());
    model.input_mut("x")?.as_f32_mut()?.fill(0.0);
    assert!(model.output("sum").is_err());
    model.run()?;
    assert_eq!(model.output("sum")?.as_f32()?, &[2.; 6]);
    model.input_mut("x")?.as_f32_mut()?.copy_from_slice(&x);
    model.run()?;
    assert_eq!(model.output("sum")?.as_f32()?, &sum);
    Ok(())
}

#[test]
fn dynamic_int64_and_reused_output() -> Result<()> {
    let mut model = OnnxSession::load(fixture("dynamic.onnx"), OnnxOptions::cpu())?;
    assert!(!model.is_prepared());
    for batch in [1, 4, 2] {
        let shape = [batch, 3];
        let input: Vec<i64> = (0..batch * 3).map(|x| x as i64).collect();
        let mut output = vec![0_i64; input.len()];
        model.inference_into(
            &[TensorView {
                name: "tokens",
                shape: &shape,
                data: TensorData::I64(&input),
            }],
            &mut [TensorViewMut {
                name: "result",
                shape: &shape,
                data: TensorDataMut::I64(&mut output),
            }],
        )?;
        assert_eq!(input, output);
    }
    Ok(())
}

#[test]
fn unavailable_configured_provider_falls_back_with_reason() -> Result<()> {
    use ep::ArbitrarilyConfigurableExecutionProvider;
    let options = OnnxOptions {
        backend: BackendSelection::Auto(vec![
            Backend::Custom {
                name: "unavailable-v1".into(),
                provider: ep::CUDA::default()
                    .with_arbitrary_config("native_onnx_invalid_test_option", "1")
                    .build(),
            },
            Backend::Cpu,
        ]),
        ..OnnxOptions::default()
    };
    let model = OnnxSession::load(fixture("multi.onnx"), options)?;
    assert_eq!(model.backend(), Some("CPU"));
    assert_eq!(model.fallback_events().len(), 1);
    assert!(!model.fallback_events()[0].error.is_empty());
    Ok(())
}

#[test]
fn no_configured_backend_is_an_error() {
    let options = OnnxOptions {
        backend: BackendSelection::Auto(vec![]),
        ..OnnxOptions::default()
    };
    assert!(OnnxSession::load(fixture("multi.onnx"), options).is_err());
}

#[test]
fn required_unavailable_provider_returns_error() {
    use ep::ArbitrarilyConfigurableExecutionProvider;
    let options = OnnxOptions {
        backend: BackendSelection::Require(Backend::Custom {
            name: "unavailable".into(),
            provider: ep::CUDA::default()
                .with_arbitrary_config("native_onnx_invalid_test_option", "1")
                .build(),
        }),
        ..OnnxOptions::default()
    };
    assert!(OnnxSession::load(fixture("multi.onnx"), options).is_err());
}

#[test]
fn ordinary_onnx_detection_does_not_depend_on_extension() -> Result<()> {
    let path =
        std::env::temp_dir().join(format!("native-onnx-graph-{}.ctx.onnx", std::process::id()));
    std::fs::copy(fixture("multi.onnx"), &path)?;
    let model = OnnxSession::load(&path, OnnxOptions::cpu())?;
    assert_eq!(model.backend(), Some("CPU"));
    drop(model);
    std::fs::remove_file(path)?;
    Ok(())
}

#[test]
fn cpu_ignores_unrelated_gpu_settings() -> Result<()> {
    let options = OnnxOptions {
        tensorrt: Some(TensorRtOptions {
            builder_optimization_level: 99,
            ..TensorRtOptions::default()
        }),
        cuda: Some(CudaOptions {
            device_id: -1,
            tf32: false,
            cuda_graph: true,
        }),
        ..OnnxOptions::cpu()
    };
    let mut model = OnnxSession::load(fixture("linear.onnx"), options)?;
    let input = TensorView::f32("x", &[1, 16], &[1.; 16]);
    assert_eq!(model.inference(&[input])?[0].view().as_f32()?, &[16.; 16]);
    Ok(())
}

#[test]
fn invalid_provider_options_obey_backend_selection() -> Result<()> {
    let invalid_trt = TensorRtOptions {
        workspace_bytes: 0,
        ..TensorRtOptions::default()
    };
    let invalid_cuda = CudaOptions {
        device_id: -1,
        ..CudaOptions::default()
    };
    for backend in [Backend::TensorRt, Backend::Cuda] {
        let options = OnnxOptions {
            backend: BackendSelection::Require(backend),
            tensorrt: Some(invalid_trt.clone()),
            cuda: Some(invalid_cuda),
            ..OnnxOptions::default()
        };
        assert!(OnnxSession::load(fixture("linear.onnx"), options).is_err());
    }
    let options = OnnxOptions {
        backend: BackendSelection::Auto(vec![Backend::TensorRt, Backend::Cuda, Backend::Cpu]),
        tensorrt: Some(invalid_trt),
        cuda: Some(invalid_cuda),
        ..OnnxOptions::default()
    };
    let model = OnnxSession::load(fixture("linear.onnx"), options)?;
    assert_eq!(model.backend(), Some("CPU"));
    assert_eq!(model.fallback_events().len(), 2);
    Ok(())
}
