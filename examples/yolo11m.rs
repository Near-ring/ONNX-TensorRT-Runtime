//! Run from the workspace root. Edit the paths and options below for your application.
#![forbid(unsafe_code)]

use safe_inference::{OnnxOptions, OnnxRuntime, TensorView};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load the model. Defaults: TensorRT level 3, then CUDA/CPU fallback.
    // Use OnnxOptions::cpu() here for CPU-only inference.
    let options = OnnxOptions::default();
    let mut model = OnnxRuntime::load("models/yolo11m.onnx", options)?;

    // 2. Create a zero-filled FP32 image tensor: RGB, NCHW [1, 3, 640, 640].
    // For a real image, fill this buffer with preprocessed pixels in [0, 1].
    #[allow(clippy::identity_op)] // Keep all four NCHW dimensions visible.
    let image_buffer: Vec<f32> = vec![0.0_f32; 1 * 3 * 640 * 640];

    // 3. Run inference. The API borrows the input slice and returns owned tensors.
    let outputs =
        model.inference(&[TensorView::f32("images", &[1, 3, 640, 640], &image_buffer)])?;

    // 4. Read the raw predictions. Detection decoding and NMS follow this step.
    let predictions: TensorView = outputs[0].view();
    let values: &[f32] = predictions.as_f32()?;
    println!("Backend: {:?}", model.backend());
    println!("Output shape: {:?}", predictions.shape); // [1, 84, 8400]
    println!("Prediction values: {}", values.len());

    Ok(())
}
