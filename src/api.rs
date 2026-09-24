//! Public compile, load, and inference APIs. Provider mechanics live in session.rs.
use crate::engine::ModelFormat;
use crate::session::ActiveSession;
use crate::{
    Backend, BackendSelection, CompileOptions, OnnxOptions, Result, Tensor, TensorSpec, TensorView,
    TensorViewMut,
};
use crate::{engine, session};
use anyhow::{Context, bail, ensure};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// ONNX input and output names, data types, and shapes.
#[derive(Clone, Debug)]
pub struct ModelInfo {
    /// Required inputs, in model order.
    pub inputs: Vec<TensorSpec>,
    /// Produced outputs, in model order.
    pub outputs: Vec<TensorSpec>,
}

/// A provider failure recorded before trying the next configured backend.
#[derive(Clone, Debug)]
pub struct FallbackEvent {
    /// Name of the provider that failed.
    pub backend: String,
    /// Error message including its context chain.
    pub error: String,
}

fn failure_report(events: &[FallbackEvent]) -> String {
    events
        .iter()
        .map(|event| format!("  {}: {}", event.backend, event.error))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Format produced by a successful compilation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompiledFormat {
    /// ONNX graph optimized for the selected provider; no embedded native context.
    OptimizedOnnx,
    /// ONNX wrapper with embedded, provider-specific compiled state.
    EpContext,
}

/// Summary of an artifact compiled and validated for the explicitly requested target.
#[derive(Clone, Debug)]
pub struct CompileReport {
    /// Provider that successfully compiled and validated the artifact.
    pub backend: String,
    /// Representation stored at the requested destination.
    pub format: CompiledFormat,
}

/// A loaded ONNX model with one active ONNX Runtime session.
///
/// Adds provider fallback, model metadata, and reusable I/O buffers around the native
/// session. Methods are synchronous; load one session inside each inference worker.
///
/// This type is neither `Send` nor `Sync`. Create, use, and drop it on the same
/// thread. Independent workers can own graph-enabled sessions on the same GPU.
/// Native setup, first-run capture and teardown are coordinated; subsequent
/// inference/replay can run concurrently on separate session-owned streams.
/// CUDA graph state in ONNX Runtime is per host thread, so moving a session after
/// capture would require renewed capture coordination, even without concurrent calls.
///
/// Start with [`Self::inference`] for ordinary or dynamic inputs. For fixed tensor
/// shapes, [`Self::input_mut`], [`Self::run`], and [`Self::output`] reuse storage.
/// Public inputs and outputs always use CPU memory, including on GPU backends.
pub struct OnnxSession {
    path: PathBuf,
    options: OnnxOptions,
    model_info: ModelInfo,
    active_session: Option<ActiveSession>,
    fallback_events: Vec<FallbackEvent>,
    is_ep_context: bool,
}

impl OnnxSession {
    /// Load an ONNX graph or an EPContext engine file (detected from its contents).
    /// EPContext files need a compatible configured provider; their original graph is not rebuilt.
    /// The file extension does not determine the format. External ONNX tensor files
    /// must be available relative to the model file when the model references them.
    /// Native libraries must already be installed; see [runtime discovery](crate#native-runtime-discovery).
    ///
    /// # Errors
    /// Returns an error for missing/incompatible libraries, invalid model/configuration,
    /// unsupported dense tensor metadata, or when every selected provider fails.
    /// [`BackendSelection::Auto`] may continue to a later candidate; inspect
    /// [`Self::fallback_events`] and [`Self::backend`] after loading.
    ///
    /// # Example
    /// ```no_run
    /// use native_onnx::{OnnxOptions, OnnxSession};
    /// let model = OnnxSession::load("model.onnx", OnnxOptions::cpu())?;
    /// for input in &model.info().inputs {
    ///     println!("{}: {:?} {:?}", input.name, input.dtype, input.shape);
    /// }
    /// # Ok::<(), native_onnx::Error>(())
    /// ```
    pub fn load(path: impl AsRef<Path>, options: OnnxOptions) -> Result<Self> {
        Self::load_internal(path.as_ref(), options, None)
    }

    /// Compile for an explicit target and validate the saved model with representative inputs.
    /// [`CompileOptions`] requires a target with its provider configuration. Any target
    /// failure is returned directly; compilation never falls back to another provider.
    /// The result reports an optimized ONNX graph or an embedded provider context.
    /// Existing output files are never overwritten; correspondence checks are the caller's job.
    ///
    /// TensorRT compilation currently requires one exportable partition and produces
    /// an ONNX EPContext containing its engine. CPU/CUDA produce optimized ONNX.
    /// The exported artifact is reopened and executed using `inputs` before the
    /// destination is created. TensorRT artifacts depend on compatible hardware and
    /// native libraries; loading with different builder settings does not rebuild them.
    ///
    /// # Errors
    /// Returns an error for an existing destination, a source that is already an
    /// EPContext, invalid inputs/options, compilation or validation failure,
    /// CUDA stream/capture errors, or file I/O failures. No partial destination is
    /// retained on failure. Native compilation may use temporary disk space.
    ///
    /// Inference options with automatic backend selection cannot be used to compile:
    /// ```compile_fail
    /// use native_onnx::{OnnxOptions, OnnxSession};
    /// OnnxSession::compile("model.onnx", "compiled.onnx", OnnxOptions::default(), &[]);
    /// ```
    pub fn compile(
        source: impl AsRef<Path>,
        destination: impl AsRef<Path>,
        options: CompileOptions,
        inputs: &[TensorView<'_>],
    ) -> Result<CompileReport> {
        let destination = destination.as_ref();
        ensure!(!destination.exists(), "Compiled output already exists");
        let options = options.into_runtime_options()?;
        options.validate()?;
        session::initialize()?;
        let source = source.as_ref().canonicalize().context("ONNX source path")?;
        ensure!(
            matches!(engine::inspect(&source, &options)?.0, ModelFormat::Onnx),
            "Compile requires an ONNX graph, not an existing EPContext"
        );
        let backend = options.backend.candidates()[0].name().to_owned();
        let (bytes, format) = Self::compile_and_validate(&source, options, inputs)
            .with_context(|| format!("Compile for {backend}"))?;
        use std::io::Write;
        let parent = destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let mut file = tempfile::Builder::new()
            .prefix(".native-onnx-")
            .tempfile_in(parent)?;
        file.write_all(&bytes)?;
        file.as_file().sync_all()?;
        file.persist_noclobber(destination).map_err(|error| {
            // Dropping PersistError's temporary file removes incomplete output.
            anyhow::anyhow!(
                "Publish compiled model {}: {}",
                destination.display(),
                error.error
            )
        })?;
        Ok(CompileReport { backend, format })
    }

    /// Primary registered provider, not a claim that every node runs on that provider.
    /// `Auto` permits node partitioning across providers. Returns `None` if an
    /// inference failure exhausted all fallback candidates and left no active session.
    pub fn backend(&self) -> Option<&str> {
        self.active_session
            .as_ref()
            .map(|a| self.options.backend.candidates()[a.backend_index].name())
    }
    /// Whole-session provider failures recorded during loading or inference.
    /// Entries are chronological and retain formatted native error context.
    /// Successful node partitioning inside ORT does not add an event here.
    pub fn fallback_events(&self) -> &[FallbackEvent] {
        &self.fallback_events
    }
    /// Whether the active session has reusable buffers for fixed-shape tensor I/O.
    /// All inputs and outputs must have known ranks and positive, fixed dimensions.
    /// Custom providers use ordinary inference. This does not indicate whether
    /// CUDA graphs are enabled, or whether every model operation runs on a GPU.
    pub fn is_prepared(&self) -> bool {
        self.active_session
            .as_ref()
            .is_some_and(|a| a.buffers().is_some())
    }
    /// Whether the selected session is configured for CUDA graph capture/replay.
    /// Requires fixed, positive I/O, sequential execution and strict GPU placement
    /// (or a compiled EPContext). Capture completes during the first successful run.
    pub fn cuda_graph_enabled(&self) -> bool {
        self.active_session
            .as_ref()
            .is_some_and(ActiveSession::graph_enabled)
    }

    /// Finish ONNX Runtime profiling and return the profile's file path.
    /// Enable profiling with [`OnnxOptions::profiling`] before loading the model.
    /// This ends profiling for the active native session. If fallback has occurred,
    /// earlier sessions may have separate profile files under the configured prefix.
    /// Native profiling or filesystem failures are returned with their context.
    pub fn end_profiling(&mut self) -> Result<String> {
        self.active_session
            .as_mut()
            .context("No active backend")?
            .end_profiling()
    }
}

impl OnnxSession {
    /// Return the ONNX model input/output metadata.
    /// Metadata reflects the active provider's resolved shapes. A failed fallback
    /// leaves the last known metadata available even if [`Self::backend`] is `None`.
    pub fn info(&self) -> &ModelInfo {
        &self.model_info
    }
    /// Run inference from borrowed inputs and return owned, CPU-ready output tensors.
    ///
    /// Supply every input exactly once, matched by name; input order is arbitrary.
    /// Outputs are in [`ModelInfo::outputs`] order and remain valid after the next
    /// inference or after dropping the session. Fixed-shape models reuse internal
    /// buffers but copy results into the returned tensors. Dynamic models use
    /// ordinary ORT execution. No image preprocessing or output decoding is applied.
    ///
    /// # Errors
    /// Names, element types, ranks, concrete dimensions, and buffer lengths must
    /// match the model. Provider errors are returned directly in required mode;
    /// automatic mode can retry the same inputs on a later backend. A native crash
    /// cannot be converted to a Rust error.
    pub fn inference(&mut self, inputs: &[TensorView<'_>]) -> Result<Vec<Tensor>> {
        self.validate_inputs(inputs)?;
        loop {
            let active = self.active_session.as_mut().context("No active backend")?;
            let result = active.inference(&self.model_info, inputs);
            match result {
                Ok(values) => {
                    return Ok(values);
                }
                Err(e) => self.fallback(e)?,
            }
        }
    }
    /// Run inference into caller-owned output buffers. Fixed tensor I/O reuses prepared storage.
    ///
    /// Inputs and outputs are matched by name and must each cover their full model
    /// interface without duplicates. Output shapes, types, and lengths must match
    /// the actual results. The dynamic path first creates owned results and then
    /// copies them; this method is not a guarantee of allocation-free execution.
    ///
    /// # Errors
    /// Returns the input/provider errors of [`Self::inference`], or an output
    /// name/shape/type/length mismatch. All output buffers are validated before
    /// result data is copied into them.
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
            outputs.len() == self.model_info.outputs.len(),
            "Output count mismatch"
        );
        for (i, output) in outputs.iter().enumerate() {
            ensure!(
                !outputs[..i].iter().any(|x| x.name == output.name),
                "Duplicate output {}",
                output.name
            );
            let spec = self
                .model_info
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
            let active = self.active_session.as_mut().context("No active backend")?;
            match active.run() {
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

impl OnnxSession {
    /// Borrow a persistent input buffer for fixed-shape tensor I/O.
    /// Inputs are initialized to zero when the session is loaded.
    /// Borrowing an input invalidates the previous prepared outputs, even if no
    /// elements are changed. The borrow must end before another mutable operation
    /// on the session. Other input buffers retain their values until updated.
    ///
    /// # Errors
    /// Returns an error for an unknown name, unavailable prepared buffers, or a
    /// native tensor access failure. Use [`Self::inference`] for dynamic I/O.
    pub fn input_mut(&mut self, name: &str) -> Result<TensorViewMut<'_>> {
        let buffers = self
            .active_session
            .as_mut()
            .and_then(ActiveSession::buffers_mut)
            .context("Prepared buffers unavailable; use inference for dynamic I/O")?;
        buffers.outputs_valid = false;
        buffers
            .input_tensors
            .iter_mut()
            .find(|s| s.name == name)
            .context("Unknown input")?
            .view_mut()
    }
    /// Borrow the last successful prepared output. The borrow expires before the next run.
    /// The data is CPU-accessible even when inference ran on a GPU. To retain it,
    /// call [`TensorView::to_owned`] before changing inputs or running again.
    ///
    /// # Errors
    /// Returns an error before a successful run, after an input was borrowed for
    /// mutation, after a failed run, for an unknown name, or without prepared buffers.
    pub fn output(&self, name: &str) -> Result<TensorView<'_>> {
        let buffers = self
            .active_session
            .as_ref()
            .and_then(ActiveSession::buffers)
            .context("Prepared buffers unavailable")?;
        ensure!(
            buffers.outputs_valid,
            "No successful output for current input"
        );
        buffers
            .output_tensors
            .iter()
            .find(|s| s.name == name)
            .context("Unknown output")?
            .view()
    }
    /// Run the fixed-shape model using its persistent input and output buffers.
    /// Unchanged inputs retain their previous values. Successful results become
    /// available through [`Self::output`]. This call completes GPU-to-host transfers
    /// before returning; it does not launch background inference.
    ///
    /// # Errors
    /// Returns an error without prepared buffers or on an unrecoverable provider
    /// failure. In automatic mode, input data is preserved across fallback. If the
    /// fallback provider has no prepared output API, use [`Self::inference`] instead.
    pub fn run(&mut self) -> Result<()> {
        let active = self.active_session.as_mut().context("No active backend")?;
        match active.run() {
            Ok(()) => Ok(()),
            Err(error) => {
                // Copy input only on failure, before dropping the failed pinned buffers.
                let inputs = active
                    .buffers()
                    .context("Prepared buffers unavailable")?
                    .input_tensors
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

impl OnnxSession {
    fn compile_and_validate(
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
            engine::export_from_profile(directory.path(), &model.model_info, Path::new(&profile))?
        } else {
            session::compile(source, &options)?
        };
        let staged = directory.path().join("compiled.onnx");
        fs::write(&staged, &bytes)?;
        let format = match engine::inspect(&staged, &options)?.0 {
            ModelFormat::Onnx => CompiledFormat::OptimizedOnnx,
            ModelFormat::EpContext => CompiledFormat::EpContext,
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
        session::initialize()?;
        let path = path.canonicalize().context("Model path")?;
        let (format, info) = engine::inspect(&path, &options)?;
        let mut model = Self {
            path,
            options,
            model_info: info,
            active_session: None,
            fallback_events: Vec::new(),
            is_ep_context: matches!(format, ModelFormat::EpContext),
        };
        if let ModelFormat::EpContext = format {
            ensure!(
                build_directory.is_none(),
                "Compile requires the original ONNX graph"
            );
            let mut failures = Vec::new();
            for index in 0..model.options.backend.candidates().len() {
                match session::load(
                    &model.path,
                    &model.options,
                    index,
                    true,
                    &model.model_info,
                    None,
                ) {
                    Ok(active) => {
                        model.active_session = Some(active);
                        break;
                    }
                    Err(error) => failures.push(FallbackEvent {
                        backend: model.options.backend.candidates()[index].name().into(),
                        error: format!("{error:#}"),
                    }),
                }
            }
            ensure!(
                model.active_session.is_some(),
                "No configured backend could load the EPContext:\n{}",
                failure_report(&failures)
            );
            model.fallback_events = failures;
        } else {
            model.activate(0, build_directory)?;
        }
        model.model_info = model
            .active_session
            .as_ref()
            .context("No active backend")?
            .info
            .clone();
        Ok(model)
    }

    fn activate(&mut self, start: usize, build_directory: Option<&Path>) -> Result<()> {
        self.active_session = None; // Release failed GPU session before creating the fallback.
        for index in start..self.options.backend.candidates().len() {
            match session::load(
                &self.path,
                &self.options,
                index,
                false,
                &self.model_info,
                build_directory,
            ) {
                Ok(active) => {
                    self.model_info = active.info.clone();
                    self.active_session = Some(active);
                    return Ok(());
                }
                Err(error) => self.fallback_events.push(FallbackEvent {
                    backend: self.options.backend.candidates()[index].name().into(),
                    error: format!("{error:#}"),
                }),
            }
        }
        bail!(
            "All configured ONNX backends failed:\n{}",
            failure_report(&self.fallback_events)
        )
    }

    fn validate_inputs(&self, inputs: &[TensorView<'_>]) -> Result<()> {
        ensure!(
            inputs.len() == self.model_info.inputs.len(),
            "Input count mismatch"
        );
        for (i, input) in inputs.iter().enumerate() {
            ensure!(
                !inputs[..i].iter().any(|x| x.name == input.name),
                "Duplicate input {}",
                input.name
            );
            self.model_info
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
            .active_session
            .as_ref()
            .context("No active backend")?
            .backend_index;
        if self.is_ep_context || matches!(self.options.backend, BackendSelection::Require(_)) {
            return Err(error);
        }
        self.fallback_events.push(FallbackEvent {
            backend: self.options.backend.candidates()[index].name().into(),
            error: format!("{error:#}"),
        });
        self.activate(index + 1, None)
    }
}
