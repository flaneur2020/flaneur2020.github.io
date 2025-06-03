use anyhow::Result;

#[cxx::bridge]
mod ffi {
    extern "Rust" {
        fn fallible1(depth: usize) -> Result<String>;
    }
}

fn fallible1(depth: usize) -> Result<String> {
    if depth == 0 {
        return Err(anyhow::Error::msg("fallible1 requires depth > 0"));
    }
    todo!()
}

