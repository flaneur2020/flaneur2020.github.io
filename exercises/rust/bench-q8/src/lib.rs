#![feature(test)]
#![feature(portable_simd)]
#![allow(soft_unstable)]
#![feature(stdsimd)]

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

pub fn vec_dot_q8_ggml(n: i32, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
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

pub fn vec_dot_q8_naive(n: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let mut result: f32 = 0.0;
    for i in 0..(n / 32) {
        let mut tmp = 0.0;
        for j in 0..32 {
            tmp += (x[i].qs[j] as i32 * y[i].qs[j] as i32) as f32;
        }
        result += tmp * f16::to_f32(x[i as usize].d) * f16::to_f32(y[i as usize].d);
    }
    result
}

pub fn vec_dot_q8_stdsimd(n: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
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

#[cfg(target_arch = "aarch64")]
pub fn vec_dot_q8_neon(n: usize, a: &[BlockQ8_0], b: &[BlockQ8_0]) -> f32 {
    unsafe {
        use std::arch::aarch64;

        let mut sumv = aarch64::vdupq_n_f32(0.0);
        let zerov = aarch64::vdupq_n_s32(0);

        for i in 0..n / 32 {
            let ab = a.get_unchecked(i);
            let bb = b.get_unchecked(i);

            let av0 = aarch64::vld1q_s8(ab.qs.as_ptr());
            let av1 = aarch64::vld1q_s8(ab.qs.as_ptr().add(16));
            let bv0 = aarch64::vld1q_s8(bb.qs.as_ptr());
            let bv1 = aarch64::vld1q_s8(bb.qs.as_ptr().add(16));

            let tmpv = aarch64::vcvtq_f32_s32(aarch64::vaddq_s32(
                aarch64::vdotq_s32(zerov, av0, bv0),
                aarch64::vdotq_s32(zerov, av1, bv1),
            ));
            sumv = aarch64::vmlaq_n_f32(sumv, tmpv, f16::to_f32(ab.d) * f16::to_f32(bb.d));
        }

        aarch64::vaddvq_f32(sumv)
    }
}

#[cfg(target_arch = "aarch64")]
pub fn vec_dot_q8_neon_unrolled(n: usize, a: &[BlockQ8_0], b: &[BlockQ8_0]) -> f32 {
    unsafe {
        use std::arch::aarch64;
        let mut sumv0 = aarch64::vdupq_n_f32(0.0);
        let mut sumv1 = aarch64::vdupq_n_f32(0.0);
        let zerov = aarch64::vdupq_n_s32(0);

        for i in (0..n / 32).step_by(2) {
            let ab0 = a.get_unchecked(i);
            let ab1 = a.get_unchecked(i + 1);
            let bb0 = b.get_unchecked(i);
            let bb1 = b.get_unchecked(i + 1);

            let av00 = aarch64::vld1q_s8(ab0.qs.as_ptr());
            let av01 = aarch64::vld1q_s8(ab0.qs.as_ptr().add(16));
            let av10 = aarch64::vld1q_s8(ab1.qs.as_ptr());
            let av11 = aarch64::vld1q_s8(ab1.qs.as_ptr().add(16));

            let bv00 = aarch64::vld1q_s8(bb0.qs.as_ptr());
            let bv01 = aarch64::vld1q_s8(bb0.qs.as_ptr().add(16));
            let bv10 = aarch64::vld1q_s8(bb1.qs.as_ptr());
            let bv11 = aarch64::vld1q_s8(bb1.qs.as_ptr().add(16));

            // vdotq_s32: dot product of two q registers (128 bit) of signed int 8, output is 4 int32 values in a q register
            // vcvtq_f32_s32: convert a q register (128 bit) from signed int 32 to f32
            // vmlaq_n_f32: multiply an scalar over a q register (128 bit) and accumulate. it seems the compiler will produce a fmul.4s and fadd.4s, not a single fmla.4s
            // vfmaq_f32: multiply and accumulate two q registers (128 bit) of f32, output is a q register (128 bit) of f32, it seems the compiler will produce a fmla.4s
            sumv0 = aarch64::vfmaq_f32(
                sumv0,
                aarch64::vcvtq_f32_s32(aarch64::vaddq_s32(
                    aarch64::vdotq_s32(zerov, av00, bv00),
                    aarch64::vdotq_s32(zerov, av01, bv01),
                )),
                aarch64::vdupq_n_f32(f16::to_f32(ab0.d) * f16::to_f32(bb0.d)),
            );

            sumv1 = aarch64::vfmaq_f32(
                sumv1,
                aarch64::vcvtq_f32_s32(aarch64::vaddq_s32(
                    aarch64::vdotq_s32(zerov, av10, bv10),
                    aarch64::vdotq_s32(zerov, av11, bv11),
                )),
                aarch64::vdupq_n_f32(f16::to_f32(ab1.d) * f16::to_f32(bb1.d)),
            );
        }

        aarch64::vaddvq_f32(sumv0) + aarch64::vaddvq_f32(sumv1)
    }
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
        let v1 = gen_rand_block_q8_0_vec(4);
        let v2 = gen_rand_block_q8_0_vec(4);

        let naive_result = vec_dot_q8_naive(64, &v1, &v2);
        let result = vec_dot_q8_ggml(64, &v1, &v2);
        assert!((result - naive_result).abs() < 1e-2);
        let result = vec_dot_q8_stdsimd(64, &v1, &v2);
        assert!((result - naive_result).abs() < 1e-2);
        let result = vec_dot_q8_neon(64, &v1, &v2);
        assert!((result - naive_result).abs() < 1e-2);
        let result = vec_dot_q8_neon_unrolled(64, &v1, &v2);
        assert!((result - naive_result).abs() < 1e-2);
    }

    #[bench]
    fn bench_vec_dot_q8_ggml(b: &mut Bencher) {
        let v1 = gen_rand_block_q8_0_vec(2000);
        let v2 = gen_rand_block_q8_0_vec(2000);
        b.iter(|| vec_dot_q8_ggml(64000, &v1, &v2));
    }

    #[bench]
    fn bench_vec_dot_q8_naive(b: &mut Bencher) {
        let v1 = gen_rand_block_q8_0_vec(2000);
        let v2 = gen_rand_block_q8_0_vec(2000);
        b.iter(|| vec_dot_q8_naive(64000, &v1, &v2));
    }

    #[bench]
    fn bench_vec_dot_q8_stdsimd(b: &mut Bencher) {
        let v1 = gen_rand_block_q8_0_vec(2000);
        let v2 = gen_rand_block_q8_0_vec(2000);
        b.iter(|| vec_dot_q8_stdsimd(64000, &v1, &v2));
    }

    #[bench]
    fn bench_vec_dot_q8_neon(b: &mut Bencher) {
        let v1 = gen_rand_block_q8_0_vec(2000);
        let v2 = gen_rand_block_q8_0_vec(2000);
        b.iter(|| vec_dot_q8_neon(64000, &v1, &v2));
    }

    #[bench]
    fn bench_vec_dot_q8_neon_unrolled(b: &mut Bencher) {
        let v1 = gen_rand_block_q8_0_vec(2000);
        let v2 = gen_rand_block_q8_0_vec(2000);
        b.iter(|| vec_dot_q8_neon_unrolled(64000, &v1, &v2));
    }
}
