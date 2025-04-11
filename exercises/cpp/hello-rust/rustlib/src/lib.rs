#[cxx::bridge]
mod ffi {
    // Rust functions exposed to C++
    extern "Rust" {
        fn add(a: i32, b: i32) -> i32;
        fn greet(name: &str) -> String;
    }
}

// Implementations of the Rust functions
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}
