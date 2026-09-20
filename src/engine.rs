//! Detect provider context models and export TensorRT engines in the EPContext format.
use crate::{DType, ModelInfo, Result, TensorSpec};
use anyhow::{Context, ensure};
use onnx_rs::ast::{
    Attribute, AttributeType, DataType, Dimension, Graph, Model, Node, OpType, OperatorSetId,
    TensorShape, TensorShapeDimension, TensorTypeProto, TypeProto, TypeValue, ValueInfo,
};
use std::{collections::BTreeSet, fs, path::Path};

#[derive(Clone, Copy)]
pub(crate) enum ModelFormat {
    Onnx,
    EpContext { fixed_fp32: bool },
}

/// Detect compiled wrappers from their contents, regardless of the file extension.
pub(crate) fn inspect(path: &Path) -> Result<ModelFormat> {
    let bytes = fs::read(path)?;
    let model = onnx_rs::parse(&bytes)?;
    let graph = model.graph.context("Missing ONNX graph")?;
    let compiled = graph
        .node
        .iter()
        .any(|node| node.domain == "com.microsoft" && node.op_type == OpType::Custom("EPContext"));
    let fixed = !graph.input.is_empty()
        && !graph.output.is_empty()
        && graph.input.iter().chain(&graph.output).all(|value| {
            matches!(value.r#type.as_ref().and_then(|t| t.value.as_ref()),
                Some(TypeValue::Tensor(t)) if t.elem_type == DataType::Float
                    && t.shape.as_ref().is_some_and(|s| s.dim.iter().all(|d|
                        matches!(d.value, Dimension::Value(n) if n > 0))))
        });
    Ok(if compiled {
        ModelFormat::EpContext { fixed_fp32: fixed }
    } else {
        ModelFormat::Onnx
    })
}

fn value<'a>(spec: &'a TensorSpec, symbols: &'a [String]) -> Result<ValueInfo<'a>> {
    let elem_type = match spec.dtype {
        DType::F32 => DataType::Float,
        DType::U8 => DataType::Uint8,
        DType::I8 => DataType::Int8,
        DType::I32 => DataType::Int32,
        DType::I64 => DataType::Int64,
        DType::Bool => DataType::Bool,
        DType::F64 => DataType::Double,
    };
    let dim = spec
        .shape
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
                shape: Some(TensorShape { dim }),
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
            (0..s.shape.len())
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
        producer_name: "safe-inference",
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
pub(crate) struct BuildDirectory(std::path::PathBuf);
impl BuildDirectory {
    pub fn new() -> Result<Self> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "safe-inference-compile-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
    }
    pub fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for BuildDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
