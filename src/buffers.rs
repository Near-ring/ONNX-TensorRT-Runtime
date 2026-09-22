//! Reusable host/device tensors and I/O bindings for static, dense tensor I/O.
use crate::{DType, ModelInfo, Result, TensorData, TensorDataMut, TensorView, TensorViewMut};
use anyhow::{Context, ensure};
use ort::{
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::{IoBinding, Session},
    value::{DynTensor, DynValue, TensorElementType, TensorRefMut, TensorValueType},
};

pub(crate) struct TensorSlot {
    pub name: String,
    pub shape: Vec<usize>,
    dtype: DType,
    host: DynTensor,
    device: Option<DynTensor>,
}
pub(crate) struct InferenceBuffers {
    pub inputs: Vec<TensorSlot>,
    pub outputs: Vec<TensorSlot>,
    binding: IoBinding,
    gpu: bool,
    pub valid_output: bool,
    // ORT allocated tensors do not retain the Allocator wrapper. Keep both alive
    // until after all tensors and I/O bindings have been dropped (field order).
    _host_allocator: Allocator,
    _device_allocator: Option<Allocator>,
}

pub(crate) fn eligible(info: &ModelInfo) -> bool {
    !info.inputs.is_empty()
        && !info.outputs.is_empty()
        && info
            .inputs
            .iter()
            .chain(&info.outputs)
            .all(|s| s.shape.iter().all(|d| d.is_some_and(|n| n > 0)))
}

fn ort_dtype(dtype: DType) -> TensorElementType {
    match dtype {
        DType::F32 => TensorElementType::Float32,
        DType::F64 => TensorElementType::Float64,
        DType::F16 => TensorElementType::Float16,
        DType::BF16 => TensorElementType::Bfloat16,
        DType::I64 => TensorElementType::Int64,
        DType::I32 => TensorElementType::Int32,
        DType::I16 => TensorElementType::Int16,
        DType::I8 => TensorElementType::Int8,
        DType::U64 => TensorElementType::Uint64,
        DType::U32 => TensorElementType::Uint32,
        DType::U16 => TensorElementType::Uint16,
        DType::U8 => TensorElementType::Uint8,
        DType::Bool => TensorElementType::Bool,
    }
}

macro_rules! dispatch_dtype {
    ($dtype:expr, $callback:ident, $($arg:expr),*) => {
        match $dtype {
            DType::F32 => $callback!($($arg,)* F32, f32),
            DType::F64 => $callback!($($arg,)* F64, f64),
            DType::F16 => $callback!($($arg,)* F16, half::f16),
            DType::BF16 => $callback!($($arg,)* BF16, half::bf16),
            DType::I64 => $callback!($($arg,)* I64, i64),
            DType::I32 => $callback!($($arg,)* I32, i32),
            DType::I16 => $callback!($($arg,)* I16, i16),
            DType::I8 => $callback!($($arg,)* I8, i8),
            DType::U64 => $callback!($($arg,)* U64, u64),
            DType::U32 => $callback!($($arg,)* U32, u32),
            DType::U16 => $callback!($($arg,)* U16, u16),
            DType::U8 => $callback!($($arg,)* U8, u8),
            DType::Bool => $callback!($($arg,)* Bool, bool),
        }
    };
}
macro_rules! view_data {
    ($slot:expr, $variant:ident, $ty:ty) => {
        Ok::<TensorData<'_>, anyhow::Error>(TensorData::$variant(
            $slot.host.try_extract_tensor::<$ty>()?.1,
        ))
    };
}
macro_rules! view_data_mut {
    ($slot:expr, $variant:ident, $ty:ty) => {
        Ok::<TensorDataMut<'_>, anyhow::Error>(TensorDataMut::$variant(
            $slot.host.try_extract_tensor_mut::<$ty>()?.1,
        ))
    };
}
macro_rules! download_typed {
    ($slot:expr, $value:expr, $variant:ident, $ty:ty) => {{
        // Storage stays CUDA-pinned; this borrowed CPU view avoids the ORT 1.29
        // direct-CudaPinned D2H error. Its lifetime is bounded by the slot.
        let mut cpu = TensorRefMut::from_array_view_mut((
            $slot.shape.as_slice(),
            $slot.host.try_extract_tensor_mut::<$ty>()?.1,
        ))?;
        $value
            .downcast_ref::<TensorValueType<$ty>>()?
            .copy_into(&mut cpu)
            .context("Safe D2H transfer")?;
        Ok(())
    }};
}

impl TensorSlot {
    pub fn view(&self) -> Result<TensorView<'_>> {
        let data = dispatch_dtype!(self.dtype, view_data, self)?;
        Ok(TensorView {
            name: &self.name,
            shape: &self.shape,
            data,
        })
    }
    pub fn view_mut(&mut self) -> Result<TensorViewMut<'_>> {
        let data = dispatch_dtype!(self.dtype, view_data_mut, self)?;
        Ok(TensorViewMut {
            name: &self.name,
            shape: &self.shape,
            data,
        })
    }
    fn download(&mut self, value: &DynValue) -> Result<()> {
        dispatch_dtype!(self.dtype, download_typed, self, value)
    }
}

impl InferenceBuffers {
    pub fn new(session: &Session, info: &ModelInfo, gpu: bool, device_id: i32) -> Result<Self> {
        ensure!(
            eligible(info),
            "Prepared buffers require fixed, positive tensor shapes"
        );
        let host = if gpu {
            Allocator::new(
                session,
                MemoryInfo::new(
                    AllocationDevice::CUDA_PINNED,
                    device_id,
                    AllocatorType::Device,
                    MemoryType::CPUOutput,
                )?,
            )?
        } else {
            Allocator::default()
        };
        let device = if gpu {
            Some(Allocator::new(
                session,
                MemoryInfo::new(
                    AllocationDevice::CUDA,
                    device_id,
                    AllocatorType::Device,
                    MemoryType::Default,
                )?,
            )?)
        } else {
            None
        };
        let mut binding = session.create_binding()?;
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for spec in &info.inputs {
            let shape = spec.fixed_shape().expect("validated fixed shape");
            let tensor = DynTensor::new(&host, ort_dtype(spec.dtype), shape.as_slice())?;
            let device = device
                .as_ref()
                .map(|a| DynTensor::new(a, ort_dtype(spec.dtype), shape.as_slice()))
                .transpose()?;
            binding.bind_input(&spec.name, device.as_ref().unwrap_or(&tensor))?;
            inputs.push(TensorSlot {
                name: spec.name.clone(),
                shape,
                dtype: spec.dtype,
                host: tensor,
                device,
            });
        }
        for spec in &info.outputs {
            let shape = spec.fixed_shape().expect("validated fixed shape");
            let tensor = DynTensor::new(&host, ort_dtype(spec.dtype), shape.as_slice())?;
            if let Some(a) = &device {
                binding.bind_output(
                    &spec.name,
                    DynTensor::new(a, ort_dtype(spec.dtype), shape.as_slice())?,
                )?;
            } else {
                // Share ownership through a safe view upgrade; cloning copies data.
                binding.bind_output(
                    &spec.name,
                    tensor
                        .view()
                        .try_upgrade()
                        .map_err(|_| anyhow::anyhow!("Cannot share owned output"))?,
                )?;
            }
            outputs.push(TensorSlot {
                name: spec.name.clone(),
                shape,
                dtype: spec.dtype,
                host: tensor,
                device: None,
            });
        }
        Ok(Self {
            inputs,
            outputs,
            binding,
            gpu,
            valid_output: false,
            _host_allocator: host,
            _device_allocator: device,
        })
    }
    pub fn run(&mut self, session: &mut Session) -> Result<()> {
        self.valid_output = false;
        if self.gpu {
            for input in &mut self.inputs {
                input
                    .host
                    .copy_into(input.device.as_mut().expect("GPU input"))
                    .context("Safe H2D transfer")?;
            }
        }
        let values = session.run_binding(&self.binding).context("Inference")?;
        if self.gpu {
            for output in &mut self.outputs {
                output.download(&values[output.name.as_str()])?;
            }
        }
        self.valid_output = true;
        Ok(())
    }
}
