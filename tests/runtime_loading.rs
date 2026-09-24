#![forbid(unsafe_code)]

use native_onnx::{OnnxOptions, OnnxRuntime, Result};
use std::process::Command;

mod common;

#[test]
fn missing_and_invalid_runtime_libraries_report_install_steps() -> Result<()> {
    // ORT is process-global, so each failure case must run before initialization
    // in a fresh process instead of changing process-wide environment in tests.
    let directory = tempfile::tempdir()?;
    let invalid = directory.path().join("invalid-library.so");
    std::fs::write(&invalid, b"not a native library")?;
    for path in [directory.path().join("missing-library.so"), invalid] {
        let output = Command::new(std::env::current_exe()?)
            .args(["--exact", "runtime_failure_child", "--nocapture"])
            .env("NATIVE_ONNX_TEST_RUNTIME", path)
            .output()?;
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

#[test]
fn runtime_failure_child() -> Result<()> {
    let Some(path) = std::env::var_os("NATIVE_ONNX_TEST_RUNTIME") else {
        return Ok(());
    };
    let result = OnnxRuntime::load(
        common::fixture("linear.onnx"),
        OnnxOptions {
            runtime_path: Some(path.into()),
            ..OnnxOptions::cpu()
        },
    );
    let error = result
        .err()
        .expect("invalid runtime must return an error, not panic");
    let message = error.to_string();
    assert!(
        message.contains("Install ONNX Runtime >= 1.27"),
        "{error:#}"
    );
    assert!(
        message.contains("ORT_DYLIB_PATH") && message.contains("PATH"),
        "{error:#}"
    );
    println!("{error:#}");
    Ok(())
}
