#![forbid(unsafe_code)]
use native_onnx::{DType, Result, TensorSpec, TensorView, element_count};

#[test]
fn checked_dimensions_and_scalar() -> Result<()> {
    assert_eq!(element_count(&[])?, 1);
    assert_eq!(element_count(&[usize::MAX, 0])?, 0);
    assert!(element_count(&[usize::MAX, 2]).is_err());
    let spec = TensorSpec {
        name: "x".into(),
        dtype: DType::F32,
        shape: Some(vec![None, Some(2)]),
    };
    assert!(
        spec.validate(&TensorView::f32("x", &[3, 2], &[0.0; 6]))
            .is_ok()
    );
    assert!(
        spec.validate(&TensorView::f32("x", &[2, 3], &[0.0; 6]))
            .is_err()
    );
    assert!(
        spec.validate(&TensorView::f32("x", &[3, 2], &[0.0; 5]))
            .is_err()
    );
    Ok(())
}
