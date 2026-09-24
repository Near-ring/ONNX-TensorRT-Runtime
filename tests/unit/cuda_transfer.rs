use crate::{Result, cuda_transfer::checked_bytes};
use ort::value::TensorElementType;

#[test]
fn copy_sizes_reject_overflow_and_non_byte_types() -> Result<()> {
    assert_eq!(checked_bytes(&[2, 3], &TensorElementType::Float32)?, 24);
    assert_eq!(checked_bytes(&[], &TensorElementType::Int64)?, 8);
    assert_eq!(
        checked_bytes(&[i64::MAX, 0], &TensorElementType::Float64)?,
        0
    );
    assert!(checked_bytes(&[-1], &TensorElementType::Float32).is_err());
    assert!(checked_bytes(&[i64::MAX, 2], &TensorElementType::Float64).is_err());
    assert!(checked_bytes(&[4], &TensorElementType::String).is_err());
    assert!(checked_bytes(&[4], &TensorElementType::Int4).is_err());
    Ok(())
}
