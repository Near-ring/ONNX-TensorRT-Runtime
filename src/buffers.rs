//! Reusable host/device tensors and I/O bindings for static, dense tensor I/O.
use crate::cuda_transfer::CudaStream;
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
    host_tensor: DynTensor,
    device_tensor: Option<DynTensor>,
}
pub(crate) struct InferenceBuffers {
    // Persistent caller-visible host tensors for fixed-shape model I/O.
    pub input_tensors: Vec<TensorSlot>,
    pub output_tensors: Vec<TensorSlot>,
    // The ORT I/O binding owns the GPU output tensors when CUDA memory is used.
    io_binding: IoBinding,
    // Auto-provider sessions may cache cross-device copies and must rebind every run.
    rebind_inputs_each_run: bool,
    // Whether bind_input has completed at least once for this I/O binding.
    input_bindings_initialized: bool,
    // Whether host output tensors contain the most recent successful run.
    pub outputs_valid: bool,
    // ORT allocated tensors do not retain the Allocator wrapper. Keep both alive
    // until after all tensors and I/O bindings have been dropped (field order).
    _host_allocator_guard: Allocator,
    _device_allocator_guard: Option<Allocator>,
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
            $slot.host_tensor.try_extract_tensor::<$ty>()?.1,
        ))
    };
}
macro_rules! view_data_mut {
    ($slot:expr, $variant:ident, $ty:ty) => {
        Ok::<TensorDataMut<'_>, anyhow::Error>(TensorDataMut::$variant(
            $slot.host_tensor.try_extract_tensor_mut::<$ty>()?.1,
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
        device_id: Option<i32>,
        rebind_inputs_each_run: bool,
    ) -> Result<Self> {
        ensure!(
            eligible(info),
            "Prepared buffers require fixed, positive tensor shapes"
        );
        let cuda = device_id.is_some();
        let allocator_device_id = device_id.unwrap_or_default();
        let host_allocator = if cuda {
            Allocator::new(
                session,
                MemoryInfo::new(
                    AllocationDevice::CUDA_PINNED,
                    allocator_device_id,
                    AllocatorType::Device,
                    MemoryType::CPUOutput,
                )?,
            )?
        } else {
            Allocator::default()
        };
        let device_allocator = if cuda {
            Some(Allocator::new(
                session,
                MemoryInfo::new(
                    AllocationDevice::CUDA,
                    allocator_device_id,
                    AllocatorType::Device,
                    MemoryType::Default,
                )?,
            )?)
        } else {
            None
        };
        let mut io_binding = session.create_binding()?;
        let mut input_tensors = Vec::new();
        let mut output_tensors = Vec::new();
        for spec in &info.inputs {
            let shape = spec.fixed_shape().expect("validated fixed shape");
            // ort rc.13 zeroes CPU-accessible allocations in DynTensor::new.
            let host_tensor =
                DynTensor::new(&host_allocator, ort_dtype(spec.dtype), shape.as_slice())?;
            let device_tensor = device_allocator
                .as_ref()
                .map(|allocator| DynTensor::new(allocator, ort_dtype(spec.dtype), shape.as_slice()))
                .transpose()?;
            // Bind only after uploading initialized data in run(). Binding may
            // copy immediately when ORT places this input on another provider.
            input_tensors.push(TensorSlot {
                name: spec.name.clone(),
                shape,
                dtype: spec.dtype,
                host_tensor,
                device_tensor,
            });
        }
        for spec in &info.outputs {
            let shape = spec.fixed_shape().expect("validated fixed shape");
            let host_tensor =
                DynTensor::new(&host_allocator, ort_dtype(spec.dtype), shape.as_slice())?;
            if let Some(allocator) = &device_allocator {
                io_binding.bind_output(
                    &spec.name,
                    DynTensor::new(allocator, ort_dtype(spec.dtype), shape.as_slice())?,
                )?;
            } else {
                // Share ownership through a safe view upgrade; cloning copies data.
                io_binding.bind_output(
                    &spec.name,
                    host_tensor
                        .view()
                        .try_upgrade()
                        .map_err(|_| anyhow::anyhow!("Cannot share owned output"))?,
                )?;
            }
            output_tensors.push(TensorSlot {
                name: spec.name.clone(),
                shape,
                dtype: spec.dtype,
                host_tensor,
                device_tensor: None,
            });
        }
        Ok(Self {
            input_tensors,
            output_tensors,
            io_binding,
            rebind_inputs_each_run,
            input_bindings_initialized: false,
            outputs_valid: false,
            _host_allocator_guard: host_allocator,
            _device_allocator_guard: device_allocator,
        })
    }
    pub fn run(&mut self, session: &mut Session, stream: Option<&CudaStream>) -> Result<()> {
        self.outputs_valid = false;
        if let Some(stream) = stream {
            stream
                .upload(self.input_tensors.iter_mut().map(|input| {
                    (
                        &input.host_tensor,
                        input.device_tensor.as_mut().expect("CUDA input"),
                    )
                }))
                .context("Pinned H2D transfers")?;
        }
        // ORT caches cross-device copies at bind time. Auto mode must rebind after
        // updates so CPU partitions see fresh inputs. Strict GPU sessions keep
        // their original device addresses for CUDA graph replay.
        if !self.input_bindings_initialized || self.rebind_inputs_each_run {
            for input in &self.input_tensors {
                self.io_binding.bind_input(
                    &input.name,
                    input.device_tensor.as_ref().unwrap_or(&input.host_tensor),
                )?;
            }
            // bind_input may have queued cross-provider copies. Wait only for
            // our stream; ORT SynchronizeInputs calls cudaDeviceSynchronize.
            if let Some(stream) = stream {
                stream.synchronize()?;
            }
            self.input_bindings_initialized = true;
        }
        let values = session.run_binding(&self.io_binding).context("Inference")?;
        if let Some(stream) = stream {
            let device_outputs = self
                .output_tensors
                .iter()
                .map(|output| values[output.name.as_str()].downcast_ref::<DynTensorValueType>())
                .collect::<ort::Result<Vec<_>>>()?;
            stream
                .download(
                    device_outputs
                        .iter()
                        .zip(&mut self.output_tensors)
                        .map(|(value, output)| (&**value, &mut output.host_tensor)),
                )
                .context("Pinned D2H transfers")?;
        }
        self.outputs_valid = true;
        Ok(())
    }
}
