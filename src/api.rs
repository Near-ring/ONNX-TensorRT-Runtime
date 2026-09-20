//! Public compile, load, and inference APIs. Provider mechanics live in session.rs.
use crate::engine::ModelFormat;
use crate::session::ActiveSession;
use crate::{
    Backend, BackendSelection, OnnxOptions, Result, Tensor, TensorSpec, TensorView, TensorViewMut,
};
use crate::{buffers, engine, session};
use anyhow::{Context, bail, ensure};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// ONNX input and output names, data types, and shapes.
#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
}

#[derive(Clone, Debug)]
pub struct FallbackEvent {
    pub backend: String,
    pub error: String,
}

/// Format produced by a successful compilation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompiledFormat {
    /// ONNX graph optimized for the selected provider; no embedded native context.
    OptimizedOnnx,
    /// ONNX wrapper with embedded, provider-specific compiled state.
    EpContext,
}

/// Actual compilation result, including any candidates skipped in Auto mode.
#[derive(Clone, Debug)]
pub struct Compilation {
    pub backend: String,
    pub format: CompiledFormat,
    pub fallback_events: Vec<FallbackEvent>,
}

/// An ONNX model session with ordered provider fallback and reusable inference buffers.
/// Methods are synchronous. Use one instance per concurrent inference worker.
pub struct OnnxRuntime {
    path: PathBuf,
    options: OnnxOptions,
    info: ModelInfo,
    active: Option<ActiveSession>,
    fallback_events: Vec<FallbackEvent>,
    compiled: bool,
}

impl OnnxRuntime {
    /// Load an ONNX graph or an EPContext engine file (detected from its contents).
    /// EPContext files need a compatible configured provider; their original graph is not rebuilt.
    pub fn load(path: impl AsRef<Path>, options: OnnxOptions) -> Result<Self> {
        Self::load_internal(path.as_ref(), options, None)
    }

    /// Compile for the selected backend and validate the saved model with representative inputs.
    /// Require fails on that provider's error; Auto tries each candidate in order.
    /// The result reports an optimized ONNX graph or an embedded provider context.
    /// Existing output files are never overwritten; correspondence checks are the caller's job.
    pub fn compile(
        source: impl AsRef<Path>,
        destination: impl AsRef<Path>,
        options: OnnxOptions,
        inputs: &[TensorView<'_>],
    ) -> Result<Compilation> {
        let destination = destination.as_ref();
        ensure!(!destination.exists(), "Compiled output already exists");
        options.validate()?;
        session::initialize(&options)?;
        let source = source.as_ref().canonicalize().context("ONNX source path")?;
        ensure!(
            matches!(engine::inspect(&source)?, ModelFormat::Onnx),
            "Compile requires an ONNX graph, not an existing EPContext"
        );
        let mut fallback_events = Vec::new();
        for backend in options.backend.candidates() {
            let mut selected = options.clone();
            selected.backend = BackendSelection::Require(backend.clone());
            match Self::compile_candidate(&source, selected, inputs) {
                Ok((bytes, format)) => {
                    use std::io::Write;
                    let mut file = fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(destination)?;
                    file.write_all(&bytes)?;
                    file.sync_all()?;
                    return Ok(Compilation {
                        backend: backend.name().into(),
                        format,
                        fallback_events,
                    });
                }
                Err(error) => {
                    if matches!(options.backend, BackendSelection::Require(_)) {
                        return Err(error.context(format!("Compile for {}", backend.name())));
                    }
                    fallback_events.push(FallbackEvent {
                        backend: backend.name().into(),
                        error: format!("{error:#}"),
                    });
                }
            }
        }
        bail!("All configured compilation backends failed: {fallback_events:?}")
    }

    /// Primary registered provider, not a claim that every node runs on that provider.
    pub fn backend(&self) -> Option<&str> {
        self.active
            .as_ref()
            .map(|a| self.options.backend.candidates()[a.backend_index].name())
    }
    pub fn fallback_events(&self) -> &[FallbackEvent] {
        &self.fallback_events
    }
    pub fn is_prepared(&self) -> bool {
        self.active.as_ref().is_some_and(|a| a.buffers.is_some())
    }
    pub fn end_profiling(&mut self) -> Result<String> {
        Ok(self
            .active
            .as_mut()
            .context("No active backend")?
            .session
            .end_profiling()?)
    }
}

impl OnnxRuntime {
    /// Return the ONNX model input/output metadata.
    pub fn info(&self) -> &ModelInfo {
        &self.info
    }
    /// Run inference from borrowed inputs and return owned, CPU-ready output tensors.
    pub fn inference(&mut self, inputs: &[TensorView<'_>]) -> Result<Vec<Tensor>> {
        self.validate_inputs(inputs)?;
        loop {
            let active = self.active.as_mut().context("No active backend")?;
            let result = active.inference(&self.info, inputs);
            match result {
                Ok(values) => {
                    return Ok(values);
                }
                Err(e) => self.fallback(e)?,
            }
        }
    }
    /// Run inference into caller-owned output buffers. Fixed FP32 models reuse prepared storage.
    pub fn inference_into(
        &mut self,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorViewMut<'_>],
    ) -> Result<()> {
        self.validate_inputs(inputs)?;
        if !self.is_prepared() {
            return crate::tensor::copy_outputs(&self.inference(inputs)?, outputs);
        }
        ensure!(
            outputs.len() == self.info.outputs.len(),
            "Output count mismatch"
        );
        for (i, output) in outputs.iter().enumerate() {
            ensure!(
                !outputs[..i].iter().any(|x| x.name == output.name),
                "Duplicate output {}",
                output.name
            );
            let spec = self
                .info
                .outputs
                .iter()
                .find(|s| s.name == output.name)
                .context("Unknown output")?;
            ensure!(
                output.data.dtype() == spec.dtype
                    && spec.fixed_shape().as_deref() == Some(output.shape)
                    && output.data.len() == crate::element_count(output.shape)?,
                "Output shape/type/length mismatch"
            );
        }
        loop {
            if !self.is_prepared() {
                return crate::tensor::copy_outputs(&self.inference(inputs)?, outputs);
            }
            for input in inputs {
                self.input_mut(input.name)?.data.copy_from(input.data)?;
            }
            let active = self.active.as_mut().context("No active backend")?;
            match active
                .buffers
                .as_mut()
                .expect("prepared backend")
                .run(&mut active.session)
            {
                Ok(()) => {
                    break;
                }
                Err(error) => self.fallback(error)?,
            }
        }
        for output in outputs {
            output.data.copy_from(self.output(output.name)?.data)?;
        }
        Ok(())
    }
}

impl OnnxRuntime {
    /// Borrow a persistent input buffer for a fixed-shape FP32 model.
    pub fn input_mut(&mut self, name: &str) -> Result<TensorViewMut<'_>> {
        let buffers = self
            .active
            .as_mut()
            .and_then(|a| a.buffers.as_mut())
            .context("Prepared buffers unavailable; use inference for dynamic or non-FP32 I/O")?;
        buffers.valid_output = false;
        buffers
            .inputs
            .iter_mut()
            .find(|s| s.name == name)
            .context("Unknown input")?
            .view_mut()
    }
    /// Borrow the last successful prepared output. The borrow expires before the next run.
    pub fn output(&self, name: &str) -> Result<TensorView<'_>> {
        let buffers = self
            .active
            .as_ref()
            .and_then(|a| a.buffers.as_ref())
            .context("Prepared buffers unavailable")?;
        ensure!(
            buffers.valid_output,
            "No successful output for current input"
        );
        buffers
            .outputs
            .iter()
            .find(|s| s.name == name)
            .context("Unknown output")?
            .view()
    }
    /// Run the fixed-shape FP32 model using its persistent input and output buffers.
    pub fn run(&mut self) -> Result<()> {
        let active = self.active.as_mut().context("No active backend")?;
        let buffers = active
            .buffers
            .as_mut()
            .context("Prepared buffers unavailable")?;
        match buffers.run(&mut active.session) {
            Ok(()) => Ok(()),
            Err(error) => {
                // Copy input only on failure, before dropping the failed pinned buffers.
                let inputs = buffers
                    .inputs
                    .iter()
                    .map(|s| Ok(s.view()?.to_owned()))
                    .collect::<Result<Vec<_>>>()?;
                self.fallback(error)?;
                let views = inputs.iter().map(Tensor::view).collect::<Vec<_>>();
                self.inference(&views)?;
                ensure!(
                    self.is_prepared(),
                    "Fallback ran successfully but has no prepared output; use inference for custom providers"
                );
                Ok(())
            }
        }
    }
}

impl OnnxRuntime {
    fn compile_candidate(
        source: &Path,
        mut options: OnnxOptions,
        inputs: &[TensorView<'_>],
    ) -> Result<(Vec<u8>, CompiledFormat)> {
        let directory = engine::BuildDirectory::new()?;
        let bytes = if matches!(
            options.backend,
            BackendSelection::Require(Backend::TensorRt)
        ) {
            options.profiling = Some(directory.path().join("compile-profile"));
            let mut model = Self::load_internal(source, options.clone(), Some(directory.path()))?;
            model.inference(inputs)?;
            let profile = model.end_profiling()?;
            engine::export_from_profile(directory.path(), &model.info, Path::new(&profile))?
        } else {
            session::compile(source, &options)?
        };
        let staged = directory.path().join("compiled.onnx");
        fs::write(&staged, &bytes)?;
        let format = match engine::inspect(&staged)? {
            ModelFormat::Onnx => CompiledFormat::OptimizedOnnx,
            ModelFormat::EpContext { .. } => CompiledFormat::EpContext,
        };
        // Reopen and execute the actual exported bytes before publishing the destination.
        options.profiling = None;
        let mut model = Self::load(&staged, options)?;
        model.inference(inputs)?;
        Ok((bytes, format))
    }

    fn load_internal(
        path: &Path,
        options: OnnxOptions,
        build_directory: Option<&Path>,
    ) -> Result<Self> {
        options.validate()?;
        session::initialize(&options)?;
        let path = path.canonicalize().context("Model path")?;
        let format = engine::inspect(&path)?;
        let mut model = Self {
            path,
            options,
            info: ModelInfo {
                inputs: Vec::new(),
                outputs: Vec::new(),
            },
            active: None,
            fallback_events: Vec::new(),
            compiled: matches!(format, ModelFormat::EpContext { .. }),
        };
        if let ModelFormat::EpContext { fixed_fp32 } = format {
            ensure!(
                build_directory.is_none(),
                "Compile requires the original ONNX graph"
            );
            let mut failures = Vec::new();
            for index in 0..model.options.backend.candidates().len() {
                match session::load(&model.path, &model.options, index, true, fixed_fp32, None) {
                    Ok(active) => {
                        model.active = Some(active);
                        break;
                    }
                    Err(error) => failures.push(FallbackEvent {
                        backend: model.options.backend.candidates()[index].name().into(),
                        error: format!("{error:#}"),
                    }),
                }
            }
            ensure!(
                model.active.is_some(),
                "No configured backend could load the EPContext: {failures:?}"
            );
            model.fallback_events = failures;
        } else {
            model.info = session::inspect_graph_io(&model.path, &model.options)?;
            model.activate(0, build_directory)?;
        }
        model.info =
            session::metadata(&model.active.as_ref().context("No active backend")?.session)?;
        Ok(model)
    }

    fn activate(&mut self, start: usize, build_directory: Option<&Path>) -> Result<()> {
        self.active = None; // Release failed GPU session before creating the fallback.
        for index in start..self.options.backend.candidates().len() {
            match session::load(
                &self.path,
                &self.options,
                index,
                false,
                buffers::eligible(&self.info),
                build_directory,
            ) {
                Ok(active) => {
                    self.active = Some(active);
                    return Ok(());
                }
                Err(error) => self.fallback_events.push(FallbackEvent {
                    backend: self.options.backend.candidates()[index].name().into(),
                    error: format!("{error:#}"),
                }),
            }
        }
        bail!(
            "All configured ONNX backends failed: {:?}",
            self.fallback_events
        )
    }

    fn validate_inputs(&self, inputs: &[TensorView<'_>]) -> Result<()> {
        ensure!(
            inputs.len() == self.info.inputs.len(),
            "Input count mismatch"
        );
        for (i, input) in inputs.iter().enumerate() {
            ensure!(
                !inputs[..i].iter().any(|x| x.name == input.name),
                "Duplicate input {}",
                input.name
            );
            self.info
                .inputs
                .iter()
                .find(|s| s.name == input.name)
                .with_context(|| format!("Unknown input {}", input.name))?
                .validate(input)?;
        }
        Ok(())
    }

    fn fallback(&mut self, error: anyhow::Error) -> Result<()> {
        let index = self
            .active
            .as_ref()
            .context("No active backend")?
            .backend_index;
        if self.compiled || matches!(self.options.backend, BackendSelection::Require(_)) {
            return Err(error);
        }
        self.fallback_events.push(FallbackEvent {
            backend: self.options.backend.candidates()[index].name().into(),
            error: format!("{error:#}"),
        });
        self.activate(index + 1, None)
    }
}
