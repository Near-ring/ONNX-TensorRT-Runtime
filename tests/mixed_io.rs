#![forbid(unsafe_code)]
use native_onnx::{
    Backend, BackendSelection, CudaOptions, OnnxOptions, OnnxRuntime, Result, TensorData,
    TensorDataMut, TensorView, f16,
};
use onnx_rs::ast::{
    DataType, Dimension, Graph, Model, Node, OpType, OperatorSetId, TensorShape,
    TensorShapeDimension, TensorTypeProto, TypeProto, TypeValue, ValueInfo,
};
use std::path::PathBuf;

fn value(name: &'static str, dtype: DataType) -> ValueInfo<'static> {
    ValueInfo {
        name,
        r#type: Some(TypeProto {
            value: Some(TypeValue::Tensor(TensorTypeProto {
                elem_type: dtype,
                shape: Some(TensorShape {
                    dim: vec![TensorShapeDimension {
                        value: Dimension::Value(4),
                        ..Default::default()
                    }],
                }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn mixed_model() -> Result<PathBuf> {
    let model = Model {
        ir_version: 8,
        opset_import: vec![OperatorSetId {
            domain: "",
            version: 17,
        }],
        graph: Some(Graph {
            name: "mixed_static",
            input: vec![
                value("ids", DataType::Int64),
                value("half", DataType::Float16),
            ],
            output: vec![
                value("ids2", DataType::Int64),
                value("half2", DataType::Float16),
            ],
            node: vec![
                Node {
                    op_type: OpType::Add,
                    input: vec!["ids", "ids"],
                    output: vec!["ids2"],
                    ..Default::default()
                },
                Node {
                    op_type: OpType::Add,
                    input: vec!["half", "half"],
                    output: vec!["half2"],
                    ..Default::default()
                },
            ],
            ..Default::default()
        }),
        ..Default::default()
    };
    let path = std::env::temp_dir().join(format!(
        "native-onnx-mixed-{}-{:?}.onnx",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::write(&path, onnx_rs::encode(&model))?;
    Ok(path)
}

fn check_repeated(options: OnnxOptions) -> Result<()> {
    let path = mixed_model()?;
    let result = (|| {
        let mut model = OnnxRuntime::load(&path, options)?;
        assert!(model.is_prepared());
        for factor in [1_i64, 2, 3] {
            let ids = [factor, factor + 1, factor + 2, factor + 3];
            let half = [f16::from_f32(factor as f32); 4];
            let outputs = model.inference(&[
                TensorView {
                    name: "half",
                    shape: &[4],
                    data: TensorData::F16(&half),
                },
                TensorView {
                    name: "ids",
                    shape: &[4],
                    data: TensorData::I64(&ids),
                },
            ])?;
            let ids2 = outputs.iter().find(|x| x.name == "ids2").unwrap();
            let half2 = outputs.iter().find(|x| x.name == "half2").unwrap();
            assert!(matches!(ids2.view().data, TensorData::I64(v) if v == ids.map(|n| n * 2)));
            assert!(
                matches!(half2.view().data, TensorData::F16(v) if v == [f16::from_f32(2.0 * factor as f32); 4])
            );
        }
        let ids = model.input_mut("ids")?;
        let TensorDataMut::I64(values) = ids.data else {
            unreachable!()
        };
        values.copy_from_slice(&[10, 11, 12, 13]);
        model.run()?;
        assert!(matches!(model.output("ids2")?.data, TensorData::I64(v) if v == [20, 22, 24, 26]));
        Ok(())
    })();
    std::fs::remove_file(path)?;
    result
}

#[test]
fn cpu_prepared_mixed_int64_fp16() -> Result<()> {
    check_repeated(OnnxOptions::cpu())
}

#[test]
#[ignore = "requires a host CUDA GPU and compatible ONNX Runtime libraries"]
fn cuda_graph_replays_mixed_int64_fp16() -> Result<()> {
    check_repeated(OnnxOptions {
        backend: BackendSelection::Require(Backend::Cuda),
        cuda: Some(CudaOptions {
            cuda_graph: true,
            ..CudaOptions::default()
        }),
        ..OnnxOptions::default()
    })
}
