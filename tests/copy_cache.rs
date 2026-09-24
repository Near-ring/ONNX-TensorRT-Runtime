//! Regression for copy helpers retaining dangling allocator-name cache keys.
#![forbid(unsafe_code)]

use native_onnx::{
    Backend, BackendSelection, CudaOptions, OnnxOptions, OnnxRuntime, Result, TensorView,
    TensorViewMut,
};
use ort::{environment::Environment, logging::LogLevel};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

mod common;
use common::fixture;

#[test]
#[ignore = "requires a host CUDA GPU and ORT's verbose CUDA arena creation logs"]
fn gpu_copy_cache_stays_bounded_across_runs_and_session_drops() -> Result<()> {
    let arena_count = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&arena_count);
    let builder = match std::env::var_os("ORT_DYLIB_PATH") {
        Some(path) => ort::init_from(path)?,
        None => ort::init(),
    };
    builder
        .with_logger(Arc::new(move |_, _, _, _, message| {
            if message.starts_with("Creating BFCArena for Cuda with") {
                count.fetch_add(1, Ordering::Relaxed);
            }
        }))
        .commit();
    Environment::current()?.set_log_level(LogLevel::Verbose);

    for cycle in 0..6 {
        let options = OnnxOptions {
            backend: BackendSelection::Require(Backend::Cuda),
            cuda: Some(CudaOptions {
                cuda_graph: cycle % 2 == 0,
                ..CudaOptions::default()
            }),
            ..OnnxOptions::default()
        };
        let mut model = OnnxRuntime::load(fixture("linear.onnx"), options)?;
        assert!(model.is_prepared());
        let output_name = model.info().outputs[0].name.clone();
        let after_load = arena_count.load(Ordering::Relaxed);
        assert!(after_load > 0, "ORT did not emit CUDA arena logs");
        let initial = model.inference(&[TensorView::f32("x", &[1, 16], &[1.0; 16])])?;
        assert_eq!(initial[0].view().as_f32()?, &[16.0; 16]);
        drop(initial);
        let warmed = arena_count.load(Ordering::Relaxed);
        assert_eq!(
            warmed, after_load,
            "direct transfers must not create copy-helper arenas"
        );

        for step in 1..=100 {
            let value = step as f32;
            let input = [value; 16];
            let expected = [16.0 * value; 16];
            match step % 3 {
                0 => {
                    let output = model.inference(&[TensorView::f32("x", &[1, 16], &input)])?;
                    assert_eq!(output[0].view().as_f32()?, &expected);
                }
                1 => {
                    let mut output = [0.0; 16];
                    model.inference_into(
                        &[TensorView::f32("x", &[1, 16], &input)],
                        &mut [TensorViewMut::f32(&output_name, &[1, 16], &mut output)],
                    )?;
                    assert_eq!(output, expected);
                }
                _ => {
                    model.input_mut("x")?.as_f32_mut()?.copy_from_slice(&input);
                    model.run()?;
                    assert_eq!(model.output(&output_name)?.as_f32()?, &expected);
                }
            }
            assert_eq!(
                arena_count.load(Ordering::Relaxed),
                warmed,
                "extra CUDA arena in cycle {cycle}, step {step}"
            );
        }
        drop(model);
    }
    println!("600 checked forwards, 6 sessions, zero copy helpers; CUDA graphs on/off");
    Ok(())
}
