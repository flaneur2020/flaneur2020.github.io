#[cxx::bridge]
mod ffi {
    // Rust functions exposed to C++
    extern "Rust" {
        fn add(a: i32, b: i32) -> i32;
    }
}

// Implementations of the Rust functions
fn add(a: i32, b: i32) -> i32 {
    a + b
}

