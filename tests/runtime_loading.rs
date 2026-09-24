#![forbid(unsafe_code)]

use native_onnx::{OnnxOptions, OnnxSession, Result};
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
            .env("NATIVE_ONNX_TEST_RUNTIME_FAILURE", "1")
            .env("ORT_DYLIB_PATH", path)
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
    let Some(_) = std::env::var_os("NATIVE_ONNX_TEST_RUNTIME_FAILURE") else {
        return Ok(());
    };
    let result = OnnxSession::load(common::fixture("linear.onnx"), OnnxOptions::cpu());
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

#[test]
fn runtime_is_discovered_without_an_explicit_path() -> Result<()> {
    let mut child = Command::new(std::env::current_exe()?);
    child
        .args(["--exact", "runtime_discovery_child", "--nocapture"])
        .env("NATIVE_ONNX_TEST_RUNTIME_DISCOVERY", "1")
        .env_remove("ORT_DYLIB_PATH");
    // An installation outside the loader's default directories only needs its
    // directory on the standard search path, just like CUDA and TensorRT.
    if let Some(path) = std::env::var_os("ORT_DYLIB_PATH") {
        let path = std::path::PathBuf::from(path);
        if path.is_file() {
            let variable = if cfg!(target_os = "windows") {
                "PATH"
            } else if cfg!(target_os = "macos") {
                "DYLD_LIBRARY_PATH"
            } else {
                "LD_LIBRARY_PATH"
            };
            let mut paths = vec![path.canonicalize()?.parent().unwrap().to_owned()];
            if let Some(existing) = std::env::var_os(variable) {
                paths.extend(std::env::split_paths(&existing));
            }
            child.env(variable, std::env::join_paths(paths)?);
        }
    }
    let output = child.output()?;
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

#[test]
fn runtime_discovery_child() -> Result<()> {
    if std::env::var_os("NATIVE_ONNX_TEST_RUNTIME_DISCOVERY").is_none() {
        return Ok(());
    }
    assert!(std::env::var_os("ORT_DYLIB_PATH").is_none());
    let mut model = OnnxSession::load(common::fixture("linear.onnx"), OnnxOptions::cpu())?;
    let output = model.inference(&[native_onnx::TensorView::f32("x", &[1, 16], &[1.; 16])])?;
    assert_eq!(output[0].view().as_f32()?, &[16.; 16]);
    Ok(())
}
