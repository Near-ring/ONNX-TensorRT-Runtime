use crate::Result;
use anyhow::{bail, ensure};

/// Dense, contiguous, row-major tensor. Scalars have shape `[]` and one element.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub dtype: DType,
    /// `None` is dynamic; zero is a real empty dimension.
    pub shape: Vec<Option<usize>>,
}

impl TensorSpec {
    pub fn fixed_shape(&self) -> Option<Vec<usize>> {
        self.shape.iter().copied().collect()
    }
    pub fn validate(&self, tensor: &TensorView<'_>) -> Result<()> {
        ensure!(
            tensor.name == self.name,
            "Tensor name mismatch: {}",
            tensor.name
        );
        ensure!(
            tensor.data.dtype() == self.dtype,
            "Tensor dtype mismatch: {}",
            tensor.name
        );
        ensure!(
            tensor.shape.len() == self.shape.len()
                && self
                    .shape
                    .iter()
                    .zip(tensor.shape)
                    .all(|(expected, actual)| expected.is_none_or(|v| v == *actual)),
            "Tensor shape mismatch: {}",
            tensor.name
        );
        ensure!(
            element_count(tensor.shape)? == tensor.data.len(),
            "Tensor length mismatch: {}",
            tensor.name
        );
        Ok(())
    }
}

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
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum DType { $($variant),+ }
        #[derive(Clone, Copy, Debug)]
        pub enum TensorData<'a> { $($variant(&'a [$ty])),+ }
        #[derive(Debug)]
        pub enum TensorDataMut<'a> { $($variant(&'a mut [$ty])),+ }
        #[derive(Clone, Debug, PartialEq)]
        pub enum TensorBuffer { $($variant(Vec<$ty>)),+ }
        impl TensorData<'_> {
            pub fn dtype(&self) -> DType { match self { $(Self::$variant(_) => DType::$variant),+ } }
            pub fn len(&self) -> usize { match self { $(Self::$variant(x) => x.len()),+ } }
            pub fn is_empty(&self) -> bool { self.len() == 0 }
            pub fn to_owned(self) -> TensorBuffer { match self { $(Self::$variant(x) => TensorBuffer::$variant(x.to_vec())),+ } }
        }
        impl TensorDataMut<'_> {
            pub fn dtype(&self) -> DType { match self { $(Self::$variant(_) => DType::$variant),+ } }
            pub fn len(&self) -> usize { match self { $(Self::$variant(x) => x.len()),+ } }
            pub fn is_empty(&self) -> bool { self.len() == 0 }
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
            pub fn dtype(&self) -> DType { match self { $(Self::$variant(_) => DType::$variant),+ } }
            pub fn len(&self) -> usize { match self { $(Self::$variant(x) => x.len()),+ } }
            pub fn is_empty(&self) -> bool { self.len() == 0 }
            pub fn view(&self) -> TensorData<'_> { match self { $(Self::$variant(x) => TensorData::$variant(x)),+ } }
            pub fn view_mut(&mut self) -> TensorDataMut<'_> { match self { $(Self::$variant(x) => TensorDataMut::$variant(x)),+ } }
        }
    };
}
tensor_types!(F32: f32, F64: f64, I64: i64, I32: i32, U8: u8, I8: i8, Bool: bool);

#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a> {
    pub name: &'a str,
    pub shape: &'a [usize],
    pub data: TensorData<'a>,
}
impl<'a> TensorView<'a> {
    pub fn f32(name: &'a str, shape: &'a [usize], data: &'a [f32]) -> Self {
        Self {
            name,
            shape,
            data: TensorData::F32(data),
        }
    }
    pub fn to_owned(self) -> Tensor {
        Tensor {
            name: self.name.into(),
            shape: self.shape.to_vec(),
            data: self.data.to_owned(),
        }
    }
    pub fn as_f32(&self) -> Result<&'a [f32]> {
        match self.data {
            TensorData::F32(x) => Ok(x),
            _ => bail!("Not an FP32 tensor"),
        }
    }
}

#[derive(Debug)]
pub struct TensorViewMut<'a> {
    pub name: &'a str,
    pub shape: &'a [usize],
    pub data: TensorDataMut<'a>,
}
impl<'a> TensorViewMut<'a> {
    pub fn f32(name: &'a str, shape: &'a [usize], data: &'a mut [f32]) -> Self {
        Self {
            name,
            shape,
            data: TensorDataMut::F32(data),
        }
    }
    pub fn as_f32_mut(&mut self) -> Result<&mut [f32]> {
        match &mut self.data {
            TensorDataMut::F32(x) => Ok(x),
            _ => bail!("Not an FP32 tensor"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: TensorBuffer,
}
impl Tensor {
    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            name: &self.name,
            shape: &self.shape,
            data: self.data.view(),
        }
    }
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
