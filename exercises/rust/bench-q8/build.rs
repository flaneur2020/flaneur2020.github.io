// build.rs
fn main() {
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-search=native=/Users/yazhou/code/ggml/build/src");
}
