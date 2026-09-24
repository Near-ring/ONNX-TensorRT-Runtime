//! Test private implementation helpers without exposing them in the public API.
//! Source modules are compiled here so the published library needs no test files.
#![deny(unsafe_code)]

use native_onnx::{Backend, Result};

#[path = "../src/cuda_graph.rs"]
mod cuda_graph;
// Only the pure byte-count validator is exercised by this test target.
#[allow(dead_code, unsafe_code)]
#[path = "../src/cuda_transfer.rs"]
mod cuda_transfer;
#[path = "../src/diagnostics.rs"]
mod diagnostics;

#[path = "unit/cuda_graph.rs"]
mod cuda_graph_tests;
#[path = "unit/cuda_transfer.rs"]
mod cuda_transfer_tests;
#[path = "unit/diagnostics.rs"]
mod diagnostics_tests;
