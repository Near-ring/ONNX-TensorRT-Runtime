//! Compile for a selected backend, then load the saved model for inference.
#![forbid(unsafe_code)]
use safe_inference::{Backend, BackendSelection, OnnxOptions, OnnxRuntime, TensorView};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[allow(clippy::identity_op)] // Keep the NCHW dimensions visible.
    let image_buffer = vec![0.0_f32; 1 * 3 * 640 * 640];
    let input = TensorView::f32("images", &[1, 3, 640, 640], &image_buffer);
    let options = OnnxOptions {
        // Change this to Backend::Cuda or Backend::Cpu to compile for that provider.
        backend: BackendSelection::Require(Backend::TensorRt),
        ..OnnxOptions::default() // TensorRT builder optimization level 3.
    };

    // 1. Compile once. The destination must not already exist.
    let compiled = OnnxRuntime::compile(
        "models/yolo11m.onnx",
        "models/yolo11m.compiled.onnx",
        options.clone(),
        &[input],
    )?;

    println!("Compiled with {}: {:?}", compiled.backend, compiled.format);

    // 2. On later launches, start here with the same backend options.
    let mut model = OnnxRuntime::load("models/yolo11m.compiled.onnx", options)?;

    // 3. Run inference with FP32 input and output.
    let outputs = model.inference(&[input])?;
    println!("Output shape: {:?}", outputs[0].shape);
    Ok(())
}
