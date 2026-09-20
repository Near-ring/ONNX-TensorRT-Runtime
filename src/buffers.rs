//! Reusable host/device tensors and I/O bindings for fixed-shape FP32 inference.
use crate::{DType, ModelInfo, Result, TensorData, TensorDataMut, TensorView, TensorViewMut};
use anyhow::{Context, ensure};
use ort::{
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::{IoBinding, Session},
    value::{Tensor, TensorRefMut, TensorValueType},
};

pub(crate) struct TensorSlot {
    pub name: String,
    pub shape: Vec<usize>,
    pub host: Tensor<f32>,
    device: Option<Tensor<f32>>,
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
            .all(|s| s.dtype == DType::F32 && s.shape.iter().all(|d| d.is_some_and(|n| n > 0)))
}
impl TensorSlot {
    pub fn view(&self) -> Result<TensorView<'_>> {
        Ok(TensorView {
            name: &self.name,
            shape: &self.shape,
            data: TensorData::F32(self.host.try_extract_tensor::<f32>()?.1),
        })
    }
    pub fn view_mut(&mut self) -> Result<TensorViewMut<'_>> {
        Ok(TensorViewMut {
            name: &self.name,
            shape: &self.shape,
            data: TensorDataMut::F32(self.host.try_extract_tensor_mut::<f32>()?.1),
        })
    }
}
impl InferenceBuffers {
    pub fn new(session: &Session, info: &ModelInfo, gpu: bool, device_id: i32) -> Result<Self> {
        ensure!(
            eligible(info),
            "InferenceBuffers buffers require fixed, positive FP32 shapes"
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
            let mut tensor = Tensor::<f32>::new(&host, shape.as_slice())?;
            tensor.try_extract_tensor_mut::<f32>()?.1.fill(0.0);
            let device = device
                .as_ref()
                .map(|a| Tensor::<f32>::new(a, shape.as_slice()))
                .transpose()?;
            binding.bind_input(&spec.name, device.as_ref().unwrap_or(&tensor))?;
            inputs.push(TensorSlot {
                name: spec.name.clone(),
                shape,
                host: tensor,
                device,
            });
        }
        for spec in &info.outputs {
            let shape = spec.fixed_shape().expect("validated fixed shape");
            let mut tensor = Tensor::<f32>::new(&host, shape.as_slice())?;
            tensor.try_extract_tensor_mut::<f32>()?.1.fill(0.0);
            if let Some(a) = &device {
                binding.bind_output(&spec.name, Tensor::<f32>::new(a, shape.as_slice())?)?;
            } else {
                // Share ownership through a safe view upgrade; Tensor::clone copies data.
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
                // Storage stays CUDA-pinned; the borrowed CPU view avoids the ORT 1.29
                // direct-CudaPinned D2H error. The view cannot outlive its allocation.
                let mut cpu = TensorRefMut::from_array_view_mut((
                    output.shape.as_slice(),
                    output.host.try_extract_tensor_mut::<f32>()?.1,
                ))?;
                values[output.name.as_str()]
                    .downcast_ref::<TensorValueType<f32>>()?
                    .copy_into(&mut cpu)
                    .context("Safe D2H transfer")?;
            }
        }
        self.valid_output = true;
        Ok(())
    }
}
