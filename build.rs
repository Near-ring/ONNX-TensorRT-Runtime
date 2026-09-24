fn main() {
    println!("cargo:rustc-check-cfg=cfg(cuda_platform)");
    if matches!(
        std::env::var("CARGO_CFG_TARGET_OS").ok().as_deref(),
        Some("linux" | "windows")
    ) {
        println!("cargo:rustc-cfg=cuda_platform");
    }
}
