fn main() {
    let mut build = cxx_build::bridge("src/lib.rs");

    #[cfg(target_os = "macos")]
    build.flag("-std=c++14");

    build.compile("rustlib");

    println!("cargo:rerun-if-changed=src/lib.rs");
}
