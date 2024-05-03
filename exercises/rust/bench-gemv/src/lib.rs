#![feature(test)]
#![feature(portable_simd)]
#![allow(soft_unstable)]
#![feature(stdarch_neon_dotprod)]

mod qgemv;
mod sgemv;
mod sgemv_strided;
mod util;
