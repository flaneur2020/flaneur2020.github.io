#![feature(test)]
#![feature(portable_simd)]
#![allow(soft_unstable)]

use std::simd::{i32x4, SimdInt};

use half::f16;

#[repr(C, packed)]
#[derive(Debug, Clone)]
pub struct BlockQ8_0 {
    d: f16,       // delta
    qs: [i8; 32], // quants
}

#[link(name = "ggml")]
extern "C" {
    fn ggml_vec_dot_q8_0_q8_0(
        n: i32,               // number of elements
        s: *mut f32,          // result
        bs: usize,            // not used?
        vx: *const BlockQ8_0, // binary of quantized vec x
        bx: usize,            // not used?
        vy: *const BlockQ8_0, // binary of quantized vec y
        by: usize,            // not used?
        nrc: i32,             // always 1?
    );
}

fn vec_dot_q8_ggml(n: i32, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let mut result: f32 = 0.0;
    unsafe {
        ggml_vec_dot_q8_0_q8_0(
            n,
            &mut result as *mut f32,
            0,
            x.as_ptr(),
            0,
            y.as_ptr(),
            0,
            1,
        );
    }
    result
}

fn vec_dot_q8_naive(n: i32, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let mut result: f32 = 0.0;
    for i in 0..n / 32 {
        let mut tmp = 0.0;
        for j in 0..32 {
            tmp += (x[i as usize].qs[j] * y[i as usize].qs[j]) as f32;
        }
        result += tmp * f16::to_f32(x[i as usize].d) * f16::to_f32(y[i as usize].d);
    }
    result
}

fn vec_dot_q8_vectorized(n: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let mut sumf: f32 = 0.0;
    for i in 0..n / 32 {
        let mut sumi: i32 = 0;
        for j in 0..8 {
            let ax = i32x4::from_array([
                x[i].qs[j * 4] as i32,
                x[i].qs[j * 4 + 1] as i32,
                x[i].qs[j * 4 + 2] as i32,
                x[i].qs[j * 4 + 3] as i32,
            ]);
            let bx = i32x4::from_array([
                y[i].qs[j * 4] as i32,
                y[i].qs[j * 4 + 1] as i32,
                y[i].qs[j * 4 + 2] as i32,
                y[i].qs[j * 4 + 3] as i32,
            ]);
            sumi += (ax * bx).reduce_sum();
        }
        sumf += sumi as f32 * x[i].d.to_f32() * y[i].d.to_f32();
    }

    sumf
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};
    extern crate test;
    use test::Bencher;

    // generate a random vector of BlockQ8_0
    fn gen_rand_block_q8_0() -> BlockQ8_0 {
        let mut rng = thread_rng();
        let d: f32 = rng.gen_range(0.0..2.0);
        let mut qs: [i8; 32] = [0; 32];
        for i in 0..32 {
            qs[i] = rng.gen::<i8>();
        }
        BlockQ8_0 {
            d: f16::from_f32(d),
            qs,
        }
    }

    fn gen_rand_block_q8_0_vec(n: usize) -> Vec<BlockQ8_0> {
        let mut v: Vec<BlockQ8_0> = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(gen_rand_block_q8_0());
        }
        v
    }

    #[test]
    fn test_vec_dot_q8() {
        let v1 = vec![
            BlockQ8_0 {
                d: f16::from_f32(1.0),
                qs: [1; 32],
            },
            BlockQ8_0 {
                d: f16::from_f32(1.0),
                qs: [1; 32],
            },
        ];
        let v2 = vec![
            BlockQ8_0 {
                d: f16::from_f32(1.0),
                qs: [2; 32],
            },
            BlockQ8_0 {
                d: f16::from_f32(1.0),
                qs: [2; 32],
            },
        ];
        let result = vec_dot_q8_ggml(64, &v1, &v2);
        assert_eq!(result, 128.0);
        let result = vec_dot_q8_naive(64, &v1, &v2);
        assert_eq!(result, 128.0);
        let result = vec_dot_q8_vectorized(64, &v1, &v2);
        assert_eq!(result, 128.0);
    }

    #[bench]
    fn bench_vec_dot_q8_ggml(b: &mut Bencher) {
        let v1 = gen_rand_block_q8_0_vec(1000);
        let v2 = gen_rand_block_q8_0_vec(1000);
        b.iter(|| vec_dot_q8_ggml(32000, &v1, &v2));
    }

    #[bench]
    fn bench_vec_dot_q8_naive(b: &mut Bencher) {
        let v1 = gen_rand_block_q8_0_vec(1000);
        let v2 = gen_rand_block_q8_0_vec(1000);
        b.iter(|| vec_dot_q8_naive(32000, &v1, &v2));
    }

    #[bench]
    fn bench_vec_dot_q8_vectorized(b: &mut Bencher) {
        let v1 = gen_rand_block_q8_0_vec(1000);
        let v2 = gen_rand_block_q8_0_vec(1000);
        b.iter(|| vec_dot_q8_vectorized(32000, &v1, &v2));
    }
}
