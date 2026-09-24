//! Reusable host/device tensors and I/O bindings for static, dense tensor I/O.
use crate::cuda_transfer;
use crate::{DType, ModelInfo, Result, TensorData, TensorDataMut, TensorView, TensorViewMut};
use anyhow::{Context, ensure};
use ort::{
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::{IoBinding, Session},
    value::{DynTensor, DynTensorValueType, TensorElementType},
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
    cuda: bool,
    device_id: i32,
    rebind_inputs: bool,
    inputs_bound: bool,
    pub outputs_valid: bool,
    // ORT allocated tensors do not retain the Allocator wrapper. Keep both alive
    // until after all tensors and I/O bindings have been dropped (field order).
    _host_allocator: Allocator,
    _device_allocator: Option<Allocator>,
}

pub(crate) fn eligible(info: &ModelInfo) -> bool {
    !info.inputs.is_empty()
        && !info.outputs.is_empty()
        && info.inputs.iter().chain(&info.outputs).all(|s| {
            s.shape
                .as_ref()
                .is_some_and(|shape| shape.iter().all(|d| d.is_some_and(|n| n > 0)))
        })
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
}

impl InferenceBuffers {
    pub fn new(
        session: &Session,
        info: &ModelInfo,
        cuda: bool,
        device_id: i32,
        rebind_inputs: bool,
    ) -> Result<Self> {
        ensure!(
            eligible(info),
            "Prepared buffers require fixed, positive tensor shapes"
        );
        let host = if cuda {
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
        let device = if cuda {
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
            // ort rc.13 zeroes CPU-accessible allocations in DynTensor::new.
            let tensor = DynTensor::new(&host, ort_dtype(spec.dtype), shape.as_slice())?;
            let device = device
                .as_ref()
                .map(|a| DynTensor::new(a, ort_dtype(spec.dtype), shape.as_slice()))
                .transpose()?;
            // Bind only after uploading initialized data in run(). Binding may
            // copy immediately when ORT places this input on another provider.
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
            cuda,
            device_id,
            rebind_inputs,
            inputs_bound: false,
            outputs_valid: false,
            _host_allocator: host,
            _device_allocator: device,
        })
    }
    pub fn run(&mut self, session: &mut Session) -> Result<()> {
        self.outputs_valid = false;
        if self.cuda {
            cuda_transfer::upload(
                self.inputs
                    .iter_mut()
                    .map(|input| (&input.host, input.device.as_mut().expect("CUDA input"))),
                self.device_id,
            )
            .context("Pinned H2D transfers")?;
        }
        // ORT caches cross-device copies at bind time. Auto mode must rebind after
        // updates so CPU partitions see fresh inputs. Strict GPU sessions keep
        // their original device addresses for CUDA graph replay.
        if !self.inputs_bound || self.rebind_inputs {
            for input in &self.inputs {
                self.binding
                    .bind_input(&input.name, input.device.as_ref().unwrap_or(&input.host))?;
            }
            self.binding.synchronize_inputs()?;
            self.inputs_bound = true;
        }
        let values = session.run_binding(&self.binding).context("Inference")?;
        if self.cuda {
            let device_outputs = self
                .outputs
                .iter()
                .map(|output| values[output.name.as_str()].downcast_ref::<DynTensorValueType>())
                .collect::<ort::Result<Vec<_>>>()?;
            cuda_transfer::download(
                device_outputs
                    .iter()
                    .zip(&mut self.outputs)
                    .map(|(value, output)| (&**value, &mut output.host)),
                self.device_id,
            )
            .context("Pinned D2H transfers")?;
        }
        self.outputs_valid = true;
        Ok(())
    }
}
