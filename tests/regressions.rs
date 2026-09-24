#![forbid(unsafe_code)]

use native_onnx::{
    Backend, BackendSelection, OnnxOptions, OnnxSession, Result, TensorData, TensorView,
    TensorViewMut, bf16,
};
use onnx_rs::ast::{
    Attribute, AttributeType, DataType, Dimension, Graph, Model, Node, OpType, OperatorSetId,
    TensorShape, TensorShapeDimension, TensorTypeProto, TypeProto, TypeValue, ValueInfo,
};

fn value(name: &'static str, dtype: DataType, dims: Option<&[i64]>) -> ValueInfo<'static> {
    ValueInfo {
        name,
        r#type: Some(TypeProto {
            value: Some(TypeValue::Tensor(TensorTypeProto {
                elem_type: dtype,
                shape: dims.map(|dims| TensorShape {
                    dim: dims
                        .iter()
                        .map(|&n| TensorShapeDimension {
                            value: Dimension::Value(n),
                            ..Default::default()
                        })
                        .collect(),
                }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn graph(dtype: DataType, dims: Option<&[i64]>, node: Node<'static>) -> Model<'static> {
    Model {
        ir_version: 8,
        opset_import: vec![
            OperatorSetId {
                domain: "",
                version: 17,
            },
            OperatorSetId {
                domain: "ai.onnx.ml",
                version: 3,
            },
        ],
        graph: Some(Graph {
            name: "regression",
            input: vec![value("x", dtype, dims)],
            output: vec![value("y", dtype, dims)],
            node: vec![node],
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn save(model: &Model<'_>) -> Result<tempfile::NamedTempFile> {
    let file = tempfile::Builder::new().suffix(".onnx").tempfile()?;
    std::fs::write(file.path(), onnx_rs::encode(model))?;
    Ok(file)
}

#[test]
fn unknown_rank_is_not_a_scalar() -> Result<()> {
    let file = save(&graph(
        DataType::Float,
        None,
        Node {
            op_type: OpType::Identity,
            input: vec!["x"],
            output: vec!["y"],
            ..Default::default()
        },
    ))?;
    let mut model = OnnxSession::load(file.path(), OnnxOptions::cpu())?;
    assert!(
        !model.is_prepared(),
        "unknown-rank I/O must not use scalar buffers"
    );
    for dims in [&[][..], &[2][..], &[1, 2][..]] {
        let values = vec![3.0; native_onnx::element_count(dims)?];
        let outputs = model.inference(&[TensorView::f32("x", dims, &values)])?;
        assert_eq!(outputs[0].shape, dims);
        assert_eq!(outputs[0].view().as_f32()?, values);
    }
    Ok(())
}

#[test]
fn scalar_and_fresh_prepared_inputs_are_zero_initialized() -> Result<()> {
    let file = save(&graph(
        DataType::Bool,
        Some(&[]),
        Node {
            op_type: OpType::Identity,
            input: vec!["x"],
            output: vec!["y"],
            ..Default::default()
        },
    ))?;
    let mut model = OnnxSession::load(file.path(), OnnxOptions::cpu())?;
    assert!(model.is_prepared());
    model.run()?;
    assert!(matches!(
        model.output("y")?.data,
        TensorData::Bool(&[false])
    ));
    assert_eq!(model.info().inputs[0].fixed_shape(), Some(vec![]));
    Ok(())
}

#[test]
#[ignore = "requires host CUDA and an ONNX Runtime GPU build"]
fn auto_cuda_cpu_partition_uses_fresh_inputs_for_every_api() -> Result<()> {
    // Scaler is CPU-only. CUDA remains the primary provider, forcing the input
    // binding to copy a device tensor to the CPU rather than borrow its storage.
    let file = save(&graph(
        DataType::Float,
        Some(&[4]),
        Node {
            domain: "ai.onnx.ml",
            op_type: OpType::Custom("Scaler"),
            input: vec!["x"],
            output: vec!["y"],
            attribute: vec![
                Attribute {
                    name: "scale",
                    r#type: AttributeType::Floats,
                    floats: vec![2.0],
                    ..Default::default()
                },
                Attribute {
                    name: "offset",
                    r#type: AttributeType::Floats,
                    floats: vec![0.0],
                    ..Default::default()
                },
            ],
            ..Default::default()
        },
    ))?;
    let mut model = OnnxSession::load(
        file.path(),
        OnnxOptions {
            backend: BackendSelection::Auto(vec![Backend::Cuda, Backend::Cpu]),
            ..OnnxOptions::default()
        },
    )?;
    assert_eq!(model.backend(), Some("CUDA"));
    assert!(model.is_prepared());
    for n in [1.0, 3.0, 5.0] {
        let values = [n; 4];
        let input = TensorView::f32("x", &[4], &values);
        assert_eq!(
            model.inference(&[input])?[0].view().as_f32()?,
            &[2.0 * n; 4]
        );
        let mut output = [0.0; 4];
        model.inference_into(&[input], &mut [TensorViewMut::f32("y", &[4], &mut output)])?;
        assert_eq!(output, [2.0 * n; 4]);
        model.input_mut("x")?.as_f32_mut()?.fill(n + 1.0);
        model.run()?;
        assert_eq!(model.output("y")?.as_f32()?, &[2.0 * (n + 1.0); 4]);
    }
    Ok(())
}

#[test]
#[ignore = "requires host CUDA with BF16 support and an ONNX Runtime GPU build"]
fn cuda_only_bf16_graph_does_not_require_cpu_kernels() -> Result<()> {
    let file = save(&graph(
        DataType::Bfloat16,
        Some(&[4]),
        Node {
            op_type: OpType::Add,
            input: vec!["x", "x"],
            output: vec!["y"],
            ..Default::default()
        },
    ))?;
    let mut model = OnnxSession::load(
        file.path(),
        OnnxOptions {
            backend: BackendSelection::Require(Backend::Cuda),
            ..OnnxOptions::default()
        },
    )?;
    for n in [1.0, 3.0] {
        let values = [bf16::from_f32(n); 4];
        let outputs = model.inference(&[TensorView {
            name: "x",
            shape: &[4],
            data: TensorData::BF16(&values),
        }])?;
        assert!(
            matches!(outputs[0].view().data, TensorData::BF16(values) if values == [bf16::from_f32(n * 2.0); 4])
        );
    }
    Ok(())
}
