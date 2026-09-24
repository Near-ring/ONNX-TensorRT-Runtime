use crate::Result;
use anyhow::{bail, ensure};
pub use half::{bf16, f16};

/// Dense, contiguous, row-major tensor metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    /// Input or output name from the ONNX graph.
    pub name: String,
    /// Required element type.
    pub dtype: DType,
    /// `None` means unknown rank; `Some(vec![])` is a scalar.
    /// Within a known rank, `None` is a dynamic dimension and zero is an empty dimension.
    pub shape: Option<Vec<Option<usize>>>,
}

impl TensorSpec {
    /// Concrete dimensions, or `None` if any dimension is dynamic.
    pub fn fixed_shape(&self) -> Option<Vec<usize>> {
        self.shape.as_ref()?.iter().copied().collect()
    }
    /// Check a tensor's name, element type, dimensions, and storage length.
    pub fn validate(&self, tensor: &TensorView<'_>) -> Result<()> {
        ensure!(
            tensor.name == self.name,
            "Tensor name mismatch: {}",
            tensor.name
        );
        ensure!(
            tensor.data.dtype() == self.dtype,
            "Tensor {} expects {:?}, got {:?}",
            tensor.name,
            self.dtype,
            tensor.data.dtype()
        );
        ensure!(
            self.shape.as_ref().is_none_or(|shape| {
                tensor.shape.len() == shape.len()
                    && shape
                        .iter()
                        .zip(tensor.shape)
                        .all(|(expected, actual)| expected.is_none_or(|v| v == *actual))
            }),
            "Tensor {} expects shape {:?}, got {:?}",
            tensor.name,
            self.shape,
            tensor.shape
        );
        ensure!(
            element_count(tensor.shape)? == tensor.data.len(),
            "Tensor length mismatch: {}",
            tensor.name
        );
        Ok(())
    }
}

/// Checked product of a tensor's dimensions.
/// Scalars (`[]`) have one element; any zero dimension makes the tensor empty.
/// Returns an error when a nonempty tensor's element count overflows `usize`.
pub fn element_count(shape: &[usize]) -> Result<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d)
            .ok_or_else(|| anyhow::anyhow!("Tensor element count overflow"))
    })
}

macro_rules! tensor_types {
    ($($variant:ident: $ty:ty),+ $(,)?) => {
        /// Supported dense tensor element types.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum DType {
            $(#[doc = concat!("Elements of type `", stringify!($ty), "`.")]
            $variant),+
        }

        /// Borrowed, contiguous tensor elements.
        #[derive(Clone, Copy, Debug)]
        pub enum TensorData<'a> {
            $(#[doc = concat!("Borrowed `", stringify!($ty), "` elements.")]
            $variant(&'a [$ty])),+
        }

        /// Exclusively borrowed, contiguous tensor elements.
        #[derive(Debug)]
        pub enum TensorDataMut<'a> {
            $(#[doc = concat!("Mutable `", stringify!($ty), "` elements.")]
            $variant(&'a mut [$ty])),+
        }

        /// Owned, contiguous tensor elements stored on the CPU.
        #[derive(Clone, Debug, PartialEq)]
        pub enum TensorBuffer {
            $(#[doc = concat!("Owned `", stringify!($ty), "` elements.")]
            $variant(Vec<$ty>)),+
        }

        impl TensorData<'_> {
            /// Element type of this slice.
            pub fn dtype(&self) -> DType {
                match self { $(Self::$variant(_) => DType::$variant),+ }
            }
            /// Number of elements, not bytes.
            pub fn len(&self) -> usize {
                match self { $(Self::$variant(x) => x.len()),+ }
            }
            /// Whether the slice contains no elements.
            pub fn is_empty(&self) -> bool {
                self.len() == 0
            }
            /// Copy the elements into an owned CPU buffer.
            pub fn to_owned(self) -> TensorBuffer {
                match self { $(Self::$variant(x) => TensorBuffer::$variant(x.to_vec())),+ }
            }
        }

        impl TensorDataMut<'_> {
            /// Element type of this slice.
            pub fn dtype(&self) -> DType {
                match self { $(Self::$variant(_) => DType::$variant),+ }
            }
            /// Number of elements, not bytes.
            pub fn len(&self) -> usize {
                match self { $(Self::$variant(x) => x.len()),+ }
            }
            /// Whether the slice contains no elements.
            pub fn is_empty(&self) -> bool {
                self.len() == 0
            }
            /// Copy elements after checking that types and lengths match.
            pub fn copy_from(&mut self, source: TensorData<'_>) -> Result<()> {
                ensure!(self.len() == source.len(), "Output buffer length mismatch");
                match (self, source) {
                    $((Self::$variant(dst), TensorData::$variant(src)) => dst.copy_from_slice(src)),+,
                    _ => bail!("Output buffer dtype mismatch"),
                }
                Ok(())
            }
        }

        impl TensorBuffer {
            /// Element type of this buffer.
            pub fn dtype(&self) -> DType {
                match self { $(Self::$variant(_) => DType::$variant),+ }
            }
            /// Number of elements, not bytes.
            pub fn len(&self) -> usize {
                match self { $(Self::$variant(x) => x.len()),+ }
            }
            /// Whether the buffer contains no elements.
            pub fn is_empty(&self) -> bool {
                self.len() == 0
            }
            /// Borrow the elements without copying.
            pub fn view(&self) -> TensorData<'_> {
                match self { $(Self::$variant(x) => TensorData::$variant(x)),+ }
            }
            /// Borrow the elements for mutation without copying.
            pub fn view_mut(&mut self) -> TensorDataMut<'_> {
                match self { $(Self::$variant(x) => TensorDataMut::$variant(x)),+ }
            }
        }
    };
}
tensor_types!(F32: f32, F64: f64, F16: f16, BF16: bf16, I64: i64, I32: i32, I16: i16, I8: i8, U64: u64, U32: u32, U16: u16, U8: u8, Bool: bool);

/// Named tensor borrowing contiguous, row-major CPU storage.
/// Creating a view does not validate it; inference validates it against the model.
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a> {
    /// Input or output name from the ONNX graph.
    pub name: &'a str,
    /// Concrete dimensions; use `[]` for a scalar.
    pub shape: &'a [usize],
    /// Borrowed elements in row-major order.
    pub data: TensorData<'a>,
}
impl<'a> TensorView<'a> {
    /// Construct a view over FP32 elements without copying or validating them.
    pub fn f32(name: &'a str, shape: &'a [usize], data: &'a [f32]) -> Self {
        Self {
            name,
            shape,
            data: TensorData::F32(data),
        }
    }
    /// Copy the name, shape, and elements into an owned tensor.
    pub fn to_owned(self) -> Tensor {
        Tensor {
            name: self.name.into(),
            shape: self.shape.to_vec(),
            data: self.data.to_owned(),
        }
    }
    /// Borrow the FP32 elements, or return an error for another element type.
    pub fn as_f32(&self) -> Result<&'a [f32]> {
        match self.data {
            TensorData::F32(x) => Ok(x),
            _ => bail!("Not an FP32 tensor"),
        }
    }
}

/// Named tensor exclusively borrowing contiguous, row-major CPU storage.
#[derive(Debug)]
pub struct TensorViewMut<'a> {
    /// Input or output name from the ONNX graph.
    pub name: &'a str,
    /// Concrete dimensions; use `[]` for a scalar.
    pub shape: &'a [usize],
    /// Exclusively borrowed elements in row-major order.
    pub data: TensorDataMut<'a>,
}
impl<'a> TensorViewMut<'a> {
    /// Construct a mutable FP32 view without copying or validating the elements.
    pub fn f32(name: &'a str, shape: &'a [usize], data: &'a mut [f32]) -> Self {
        Self {
            name,
            shape,
            data: TensorDataMut::F32(data),
        }
    }
    /// Mutably borrow FP32 elements, or return an error for another element type.
    pub fn as_f32_mut(&mut self) -> Result<&mut [f32]> {
        match &mut self.data {
            TensorDataMut::F32(x) => Ok(x),
            _ => bail!("Not an FP32 tensor"),
        }
    }
}

/// Owned, named tensor with contiguous, row-major CPU storage.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    /// Input or output name from the ONNX graph.
    pub name: String,
    /// Concrete dimensions; an empty shape denotes a scalar.
    pub shape: Vec<usize>,
    /// Owned elements in row-major order.
    pub data: TensorBuffer,
}
impl Tensor {
    /// Borrow the tensor without copying its metadata or elements.
    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            name: &self.name,
            shape: &self.shape,
            data: self.data.view(),
        }
    }
    /// Borrow the tensor for element mutation without copying.
    pub fn view_mut(&mut self) -> TensorViewMut<'_> {
        TensorViewMut {
            name: &self.name,
            shape: &self.shape,
            data: self.data.view_mut(),
        }
    }
}

pub(crate) fn copy_outputs(source: &[Tensor], targets: &mut [TensorViewMut<'_>]) -> Result<()> {
    anyhow::ensure!(source.len() == targets.len(), "Output count mismatch");
    for (i, target) in targets.iter().enumerate() {
        anyhow::ensure!(
            !targets[..i].iter().any(|x| x.name == target.name),
            "Duplicate output {}",
            target.name
        );
        let value = source
            .iter()
            .find(|x| x.name == target.name)
            .ok_or_else(|| anyhow::anyhow!("Unknown output {}", target.name))?;
        anyhow::ensure!(
            target.shape == value.shape
                && target.data.dtype() == value.data.dtype()
                && target.data.len() == value.data.len(),
            "Output shape/type/length mismatch: {}",
            target.name
        );
    }
    for target in targets {
        let source = source
            .iter()
            .find(|x| x.name == target.name)
            .expect("validated output");
        target.data.copy_from(source.data.view())?;
    }
    Ok(())
}
