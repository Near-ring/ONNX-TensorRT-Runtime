//! Coordinate ORT global capture with native setup, teardown and inference.
//! Shared access permits concurrent replay; exclusive access protects capture.
use crate::Result;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

// CaptureModeGlobal can affect other host threads, even on another device.
// This gate covers this crate's sessions only, not external CUDA consumers.
static EXECUTION: RwLock<()> = RwLock::new(());

pub(crate) fn exclusive() -> Result<RwLockWriteGuard<'static, ()>> {
    EXECUTION.write().map_err(|_| {
        anyhow::anyhow!(
            "CUDA capture coordinator poisoned by an earlier panic; restart the process"
        )
    })
}

pub(crate) fn shared() -> Result<RwLockReadGuard<'static, ()>> {
    EXECUTION.read().map_err(|_| {
        anyhow::anyhow!(
            "CUDA capture coordinator poisoned by an earlier panic; restart the process"
        )
    })
}

pub(crate) fn teardown() -> RwLockWriteGuard<'static, ()> {
    // Destructors must still free resources after a Rust panic.
    EXECUTION
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
