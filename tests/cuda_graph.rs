#![forbid(unsafe_code)]
use native_onnx::{
    Backend, BackendSelection, CudaOptions, OnnxOptions, OnnxRuntime, Result, TensorView,
};
mod common;
use common::fixture;

#[test]
#[ignore = "requires a host CUDA GPU and compatible ONNX Runtime libraries"]
fn cuda_graph_replays_with_stable_bound_buffers() -> Result<()> {
    let options = OnnxOptions {
        backend: BackendSelection::Require(Backend::Cuda),
        cuda: Some(CudaOptions {
            cuda_graph: true,
            ..CudaOptions::default()
        }),
        ..OnnxOptions::default()
    };
    let mut model = OnnxRuntime::load(fixture("linear.onnx"), options)?;
    assert!(model.is_prepared());
    for value in [1.0_f32, 2.0, 3.0] {
        let input = [value; 16];
        let output = model.inference(&[TensorView::f32("x", &[1, 16], &input)])?;
        assert_eq!(output[0].view().as_f32()?, &[16.0 * value; 16]);
    }
    Ok(())
}
