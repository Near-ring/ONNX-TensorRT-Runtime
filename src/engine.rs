//! Detect provider context models and export TensorRT engines in the EPContext format.
use crate::{DType, ModelInfo, OnnxOptions, Result, TensorSpec};
use anyhow::{Context, bail, ensure};
use onnx_rs::ast::{
    Attribute, AttributeType, DataType, Dimension, Graph, Model, Node, OpType, OperatorSetId,
    TensorShape, TensorShapeDimension, TensorTypeProto, TypeProto, TypeValue, ValueInfo,
};
use std::{collections::BTreeSet, fs, path::Path};

#[derive(Clone, Copy)]
pub(crate) enum ModelFormat {
    Onnx,
    EpContext,
}

/// Detect compiled wrappers from their contents, regardless of the file extension.
pub(crate) fn inspect(path: &Path, options: &OnnxOptions) -> Result<(ModelFormat, ModelInfo)> {
    let bytes = fs::read(path).with_context(|| format!("Read model {}", path.display()))?;
    let model =
        onnx_rs::parse(&bytes).with_context(|| format!("Parse ONNX model {}", path.display()))?;
    let graph = model.graph.context("Missing ONNX graph")?;
    let is_ep_context = graph
        .node
        .iter()
        .any(|node| node.domain == "com.microsoft" && node.op_type == OpType::Custom("EPContext"));
    // Reading protobuf metadata must not instantiate a CPU session: some valid
    // models (for example BF16 Add) have kernels only in the selected GPU provider.
    let initialized = graph
        .initializer
        .iter()
        .map(|t| t.name())
        .chain(
            graph
                .sparse_initializer
                .iter()
                .filter_map(|t| t.values.as_ref().map(|v| v.name())),
        )
        .collect::<BTreeSet<_>>();
    let info = ModelInfo {
        inputs: graph
            .input
            .iter()
            .filter(|v| !initialized.contains(v.name))
            .map(|v| tensor_spec(v, options))
            .collect::<Result<_>>()?,
        outputs: graph
            .output
            .iter()
            .map(|v| tensor_spec(v, options))
            .collect::<Result<_>>()?,
    };
    let format = if is_ep_context {
        ModelFormat::EpContext
    } else {
        ModelFormat::Onnx
    };
    Ok((format, info))
}

fn tensor_spec(value: &ValueInfo<'_>, options: &OnnxOptions) -> Result<TensorSpec> {
    let Some(TypeValue::Tensor(tensor)) = value.r#type.as_ref().and_then(|t| t.value.as_ref())
    else {
        bail!("Only dense tensor I/O is supported: {}", value.name);
    };
    let dtype = match tensor.elem_type {
        DataType::Float => DType::F32,
        DataType::Double => DType::F64,
        DataType::Float16 => DType::F16,
        DataType::Bfloat16 => DType::BF16,
        DataType::Int64 => DType::I64,
        DataType::Int32 => DType::I32,
        DataType::Int16 => DType::I16,
        DataType::Int8 => DType::I8,
        DataType::Uint64 => DType::U64,
        DataType::Uint32 => DType::U32,
        DataType::Uint16 => DType::U16,
        DataType::Uint8 => DType::U8,
        DataType::Bool => DType::Bool,
        dtype => bail!("Unsupported I/O dtype {dtype:?}: {}", value.name),
    };
    let shape = tensor.shape.as_ref().map(|shape| {
        shape
            .dim
            .iter()
            .map(|d| match d.value {
                Dimension::Value(n) => usize::try_from(n).ok(),
                Dimension::Param(name) => options.dimension_overrides.get(name).copied(),
            })
            .collect()
    });
    Ok(TensorSpec {
        name: value.name.into(),
        dtype,
        shape,
    })
}

fn value<'a>(spec: &'a TensorSpec, symbols: &'a [String]) -> Result<ValueInfo<'a>> {
    let elem_type = match spec.dtype {
        DType::F32 => DataType::Float,
        DType::F16 => DataType::Float16,
        DType::BF16 => DataType::Bfloat16,
        DType::U8 => DataType::Uint8,
        DType::U16 => DataType::Uint16,
        DType::U32 => DataType::Uint32,
        DType::U64 => DataType::Uint64,
        DType::I8 => DataType::Int8,
        DType::I16 => DataType::Int16,
        DType::I32 => DataType::Int32,
        DType::I64 => DataType::Int64,
        DType::Bool => DataType::Bool,
        DType::F64 => DataType::Double,
    };
    let dim = spec
        .shape
        .as_deref()
        .unwrap_or_default()
        .iter()
        .zip(symbols)
        .map(|(n, symbol)| {
            Ok(TensorShapeDimension {
                value: match n {
                    Some(n) => Dimension::Value(i64::try_from(*n)?),
                    None => Dimension::Param(symbol),
                },
                ..Default::default()
            })
        })
        .collect::<Result<_>>()?;
    Ok(ValueInfo {
        name: &spec.name,
        r#type: Some(TypeProto {
            value: Some(TypeValue::Tensor(TensorTypeProto {
                elem_type,
                shape: spec.shape.as_ref().map(|_| TensorShape { dim }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    })
}

pub(crate) fn export_from_profile(
    directory: &Path,
    info: &ModelInfo,
    profile: &Path,
) -> Result<Vec<u8>> {
    let events: Vec<serde_json::Value> = serde_json::from_slice(&fs::read(profile)?)?;
    let mut nodes = BTreeSet::new();
    for event in events
        .iter()
        .filter(|e| e["cat"] == "Node" && e["args"]["provider"].is_string())
    {
        ensure!(
            event["args"]["provider"] == "TensorrtExecutionProvider",
            "Full-model engine export requires all compute in TensorRT"
        );
        nodes.insert(
            event["name"]
                .as_str()
                .context("Missing profile node name")?,
        );
    }
    ensure!(
        nodes.len() == 1,
        "Full-model engine export requires exactly one TensorRT partition"
    );
    let engines = fs::read_dir(directory)?
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .filter(|e| e.path().extension().is_some_and(|s| s == "engine"))
        .collect::<Vec<_>>();
    ensure!(
        engines.len() == 1,
        "Full-model engine export requires one engine file"
    );
    let engine = fs::read(engines[0].path())?;
    let attributes = vec![
        Attribute {
            name: "embed_mode",
            i: 1,
            r#type: AttributeType::Int,
            ..Default::default()
        },
        Attribute {
            name: "ep_cache_context",
            s: &engine,
            r#type: AttributeType::String,
            ..Default::default()
        },
        Attribute {
            name: "source",
            s: b"TensorrtExecutionProvider",
            r#type: AttributeType::String,
            ..Default::default()
        },
    ];
    let node = Node {
        name: "compiled_model",
        op_type: OpType::Custom("EPContext"),
        domain: "com.microsoft",
        input: info.inputs.iter().map(|s| s.name.as_str()).collect(),
        output: info.outputs.iter().map(|s| s.name.as_str()).collect(),
        attribute: attributes,
        ..Default::default()
    };
    let symbols = info
        .inputs
        .iter()
        .chain(&info.outputs)
        .map(|s| {
            (0..s.shape.as_ref().map_or(0, Vec::len))
                .map(|i| format!("{}_d{i}", s.name))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut values = info
        .inputs
        .iter()
        .chain(&info.outputs)
        .zip(&symbols)
        .map(|(spec, symbols)| value(spec, symbols));
    let input = values
        .by_ref()
        .take(info.inputs.len())
        .collect::<Result<_>>()?;
    let output = values.collect::<Result<_>>()?;
    let model = Model {
        ir_version: 8,
        producer_name: "native-onnx",
        opset_import: vec![
            OperatorSetId {
                domain: "",
                version: 17,
            },
            OperatorSetId {
                domain: "com.microsoft",
                version: 1,
            },
        ],
        graph: Some(Graph {
            name: "compiled_model",
            node: vec![node],
            input,
            output,
            ..Default::default()
        }),
        ..Default::default()
    };
    Ok(onnx_rs::encode(&model))
}

/// Isolated scratch space for an explicit compilation; no persistent cache or identity record.
pub(crate) struct BuildDirectory(tempfile::TempDir);
impl BuildDirectory {
    pub fn new() -> Result<Self> {
        Ok(Self(
            tempfile::Builder::new()
                .prefix("native-onnx-compile-")
                .tempdir()?,
        ))
    }
    pub fn path(&self) -> &Path {
        self.0.path()
    }
}
