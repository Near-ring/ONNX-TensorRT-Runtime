//! Independent sessions load, run, and drop on their own worker threads.
#![forbid(unsafe_code)]
use native_onnx::{
    Backend, BackendSelection, CudaOptions, OnnxOptions, OnnxSession, Result, Tensor,
    TensorRtOptions, TensorView,
};
use std::{path::Path, sync::Barrier, thread};
mod common;

fn options(backend: Backend) -> OnnxOptions {
    OnnxOptions {
        backend: BackendSelection::Require(backend),
        tensorrt: Some(TensorRtOptions {
            fp16: true,
            cuda_graph: true,
            ..TensorRtOptions::default()
        }),
        ..OnnxOptions::default()
    }
}

fn independent_models(backends: [Backend; 2], graphs: [bool; 2]) -> Result<()> {
    for _ in 0..3 {
        let barrier = Barrier::new(2);
        thread::scope(|scope| -> Result<()> {
            let workers: Vec<_> = (0..2)
                .map(|worker| {
                    let barrier = &barrier;
                    let mut options = options(backends[worker].clone());
                    options.cuda = Some(CudaOptions {
                        cuda_graph: graphs[worker],
                        ..CudaOptions::default()
                    });
                    options.tensorrt.as_mut().unwrap().cuda_graph = graphs[worker];
                    scope.spawn(move || -> Result<()> {
                        let name = if worker == 0 {
                            "linear.onnx"
                        } else {
                            "multi.onnx"
                        };
                        barrier.wait();
                        let model = OnnxSession::load(common::fixture(name), options);
                        barrier.wait(); // All workers reach this even if loading failed.
                        let mut model = model?;
                        assert_eq!(model.cuda_graph_enabled(), graphs[worker]);
                        for round in 1..=40 {
                            let value = round as f32;
                            if worker == 0 {
                                let out = model.inference(&[TensorView::f32(
                                    "x",
                                    &[1, 16],
                                    &[value; 16],
                                )])?;
                                assert_eq!(out[0].view().as_f32()?, &[16. * value; 16]);
                            } else {
                                let out = model.inference(&[
                                    TensorView::f32("x", &[2, 3], &[value; 6]),
                                    TensorView::f32("y", &[2, 3], &[2.; 6]),
                                ])?;
                                for tensor in out {
                                    let expected = match tensor.name.as_str() {
                                        "sum" => value + 2.,
                                        "product" => value * 2.,
                                        name => panic!("unexpected output {name}"),
                                    };
                                    assert_eq!(tensor.view().as_f32()?, &[expected; 6]);
                                }
                            }
                        }
                        assert!(model.fallback_events().is_empty());
                        Ok(())
                    })
                })
                .collect();
            // Join every worker even if one failed.
            let results: Vec<_> = workers.into_iter().map(|worker| worker.join()).collect();
            for result in results {
                result.map_err(|_| anyhow::anyhow!("inference worker panicked"))??;
            }
            Ok(())
        })?;
    }
    Ok(())
}

#[test]
fn cpu_models_on_independent_threads() -> Result<()> {
    independent_models([Backend::Cpu, Backend::Cpu], [false; 2])
}

#[test]
#[ignore = "requires host CUDA and an ONNX Runtime GPU build"]
fn cuda_models_on_independent_threads() -> Result<()> {
    for graphs in [[true, true], [true, false], [false, false]] {
        independent_models([Backend::Cuda, Backend::Cuda], graphs)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires host TensorRT and an ONNX Runtime GPU build"]
fn tensorrt_models_on_independent_threads() -> Result<()> {
    for graphs in [[true, true], [true, false], [false, false]] {
        independent_models([Backend::TensorRt, Backend::TensorRt], graphs)?;
    }
    independent_models([Backend::Cuda, Backend::TensorRt], [true; 2])
}

fn yolo_inference(model: &mut OnnxSession, pixels: &[f32], backend: &str) -> Result<Vec<Tensor>> {
    let output = model.inference(&[TensorView::f32("images", &[1, 3, 640, 640], pixels)])?;
    assert_eq!(model.backend(), Some(backend));
    assert!(model.cuda_graph_enabled());
    assert!(model.fallback_events().is_empty());
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].shape, [1, 84, 8400]);
    assert!(output[0].view().as_f32()?.iter().all(|x| x.is_finite()));
    Ok(output)
}

#[test]
#[ignore = "requires host TensorRT and YOLO11m; set NATIVE_ONNX_YOLO_MODEL to its ONNX/EPContext file"]
fn tensorrt_yolo_matches_serial_reference_from_two_workers() -> Result<()> {
    let path = std::env::var_os("NATIVE_ONNX_YOLO_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("examples/model/yolo11/yolo11m.trt-fp16.onnx")
        });
    anyhow::ensure!(
        path.is_file(),
        "Set NATIVE_ONNX_YOLO_MODEL to a YOLO11m ONNX/EPContext file with input images [1,3,640,640]"
    );
    yolo_workers(Backend::TensorRt, &path)
}

#[test]
#[ignore = "requires host CUDA and YOLO11m; set NATIVE_ONNX_CUDA_YOLO_MODEL to its original ONNX file"]
fn cuda_yolo_matches_serial_reference_from_two_workers() -> Result<()> {
    let path = std::env::var_os("NATIVE_ONNX_CUDA_YOLO_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/model/yolo11/yolo11m.onnx")
        });
    anyhow::ensure!(
        path.is_file(),
        "Set NATIVE_ONNX_CUDA_YOLO_MODEL to the original YOLO11m ONNX file"
    );
    yolo_workers(Backend::Cuda, &path)
}

fn yolo_workers(backend: Backend, path: &Path) -> Result<()> {
    let backend_name = backend.name();
    let options = options(backend.clone());
    let inputs = [vec![0.25; 3 * 640 * 640], vec![0.75; 3 * 640 * 640]];
    let mut reference = OnnxSession::load(path, options.clone())?;
    let expected = [
        yolo_inference(&mut reference, &inputs[0], backend_name)?,
        yolo_inference(&mut reference, &inputs[1], backend_name)?,
    ];
    assert_ne!(
        expected[0][0].view().as_f32()?,
        expected[1][0].view().as_f32()?
    );
    drop(reference);

    let started = std::time::Instant::now();
    for cycle in 0..3 {
        let barrier = Barrier::new(2);
        thread::scope(|scope| -> Result<()> {
            let workers: Vec<_> = (0..2).map(|worker| {
                scope.spawn({
                    let options = options.clone();
                    let inputs = &inputs;
                    let expected = &expected;
                    let barrier = &barrier;
                    move || -> Result<()> {
                        barrier.wait();
                        // No caller-side load mutex: the engine handles native initialization.
                        let model = OnnxSession::load(path, options);
                        barrier.wait();
                        let mut model = model?;
                        for round in 0..20 {
                            let index = (worker + round) % inputs.len();
                            let actual = yolo_inference(&mut model, &inputs[index], backend_name)?;
                            let values = actual[0].view().as_f32()?;
                            let expected = expected[index][0].view().as_f32()?;
                            for (actual, expected) in values.iter().zip(expected) {
                                assert!((actual - expected).abs() <= 1e-4 + 1e-4 * expected.abs(), "prediction differs from serial reference: {actual} vs {expected}");
                            }
                        }
                        println!("cycle {cycle}, worker {worker}: 20 checked {backend_name} YOLO inferences");
                        Ok(())
                    }
                })
            }).collect();
            let results: Vec<_> = workers.into_iter().map(|worker| worker.join()).collect();
            for result in results {
                result.map_err(|_| anyhow::anyhow!("YOLO worker panicked"))??;
            }
            Ok(())
        })?;
    }
    println!(
        "{backend_name}: 120 checked concurrent YOLO calls completed in {:?}",
        started.elapsed()
    );
    Ok(())
}

#[test]
#[ignore = "requires host CUDA and TensorRT; exercises capture while peers load, replay and drop"]
fn session_churn_during_graph_replay() -> Result<()> {
    use std::sync::atomic::{AtomicBool, Ordering};
    struct Completion<'a>(&'a AtomicBool);
    impl Drop for Completion<'_> {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }
    let finished = AtomicBool::new(false);
    let start = Barrier::new(2);
    thread::scope(|scope| -> Result<()> {
        let churn = scope.spawn(|| -> Result<()> {
            let _completion = Completion(&finished);
            start.wait();
            let result = (|| {
                for cycle in 0..12 {
                    let backend = if cycle % 2 == 0 {
                        Backend::Cuda
                    } else {
                        Backend::TensorRt
                    };
                    let mut opts = options(backend);
                    let enabled = cycle % 3 != 0;
                    opts.cuda = Some(CudaOptions {
                        cuda_graph: enabled,
                        ..CudaOptions::default()
                    });
                    opts.tensorrt.as_mut().unwrap().cuda_graph = enabled;
                    let mut model = OnnxSession::load(common::fixture("linear.onnx"), opts)?;
                    assert_eq!(model.cuda_graph_enabled(), enabled);
                    for value in [1., 5.] {
                        assert_eq!(
                            model.inference(&[TensorView::f32("x", &[1, 16], &[value; 16])])?[0]
                                .view()
                                .as_f32()?,
                            &[16. * value; 16]
                        );
                    }
                }
                Ok(())
            })();
            finished.store(true, Ordering::Release);
            result
        });
        let replay = scope.spawn(|| -> Result<()> {
            // Always release the peer even if this load fails.
            let model = OnnxSession::load(
                common::fixture("linear.onnx"),
                OnnxOptions {
                    backend: BackendSelection::Require(Backend::Cuda),
                    cuda: Some(CudaOptions {
                        cuda_graph: true,
                        ..CudaOptions::default()
                    }),
                    ..OnnxOptions::default()
                },
            );
            start.wait();
            let mut model = model?;
            let mut round = 0;
            while round < 100 || !finished.load(Ordering::Acquire) {
                let value = (round % 9) as f32;
                assert_eq!(
                    model.inference(&[TensorView::f32("x", &[1, 16], &[value; 16])])?[0]
                        .view()
                        .as_f32()?,
                    &[16. * value; 16]
                );
                round += 1;
            }
            println!("{round} checked replays during 12 CUDA/TensorRT session lifecycles");
            Ok(())
        });
        let churn = churn.join();
        let replay = replay.join();
        churn.map_err(|_| anyhow::anyhow!("churn worker panicked"))??;
        replay.map_err(|_| anyhow::anyhow!("replay worker panicked"))??;
        Ok(())
    })
}
