use half::f16;
use rayon::prelude::*;

#[repr(C, packed)]
#[derive(Debug, Clone)]
pub struct BlockQ8_0 {
    d: f16,       // delta
    qs: [i8; 32], // quants
}

pub fn vec_dot_q8_neon(a: &[BlockQ8_0], b: &[BlockQ8_0], n: usize) -> f32 {
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

pub fn qgemv_naive(m: usize, k: usize, a: &[BlockQ8_0], b: &[BlockQ8_0], c: &mut [f32]) {
    for mi in 0..m {
        let mut sum = 0.0;
        for ki in (0..k).step_by(32) {
            let a_offset = (mi * k + ki) / 32;
            let b_offset = ki / 32;
            let mut tmp = 0.0;
            for i in 0..32 {
                tmp += a[a_offset].qs[i] as f32 * b[b_offset].qs[i] as f32;
            }
            tmp *= a[a_offset].d.to_f32() * b[b_offset].d.to_f32();
            sum += tmp;
        }
        c[mi] = sum;
    }
}

// a: m x k
// b: k x 1
// c: m x 1
pub fn qgemv_naive_dot(m: usize, k: usize, a: &[BlockQ8_0], b: &[BlockQ8_0], c: &mut [f32]) {
    for mi in 0..m {
        c[mi] = vec_dot_q8_neon(&a[(mi * k) / 32..], b, k);
    }
}

pub fn qgemv_rayon(_m: usize, k: usize, a: &[BlockQ8_0], b: &[BlockQ8_0], c: &mut [f32]) {
    c.par_iter_mut().enumerate().for_each(|(mi, cmi)| {
        *cmi = vec_dot_q8_neon(&a[(mi * k) / 32..], b, k);
    });
}

pub fn qgemv_rayon_chunked(_m: usize, k: usize, a: &[BlockQ8_0], b: &[BlockQ8_0], c: &mut [f32]) {
    c.chunks_exact_mut(2).enumerate().for_each(|(mi, cn)| {
        cn[0] = vec_dot_q8_neon(&a[((mi + 0) * k) / 32..], b, k);
        cn[1] = vec_dot_q8_neon(&a[((mi + 1) * k) / 32..], b, k);
        //cn[2] = vec_dot_q8_neon(&a[((mi + 2) * k) / 32..], b, k);
        //cn[3] = vec_dot_q8_neon(&a[((mi + 3) * k) / 32..], b, k);
    });
}

pub fn qgemv_rayon_unchecked(_m: usize, k: usize, a: &[BlockQ8_0], b: &[BlockQ8_0], c: &mut [f32]) {
    c.par_iter_mut().enumerate().for_each(|(mi, cmi)| {
        let ab = unsafe { a.get_unchecked((mi * k) / 32..) };
        *cmi = vec_dot_q8_neon(&ab, b, k);
    });
}

pub fn qgemv_rayon_tiled(_m: usize, k: usize, a: &[BlockQ8_0], b: &[BlockQ8_0], c: &mut [f32]) {
    c.chunks_exact_mut(32).enumerate().for_each(|(mi, cn)| {
        let mut tmp = [0.0; 32];
        for j in 0..k / 64 {
            for i in 0..32 {
                let ab = unsafe { a.get_unchecked(((mi * 32 + i) * k) / 32..) };
                let bb = unsafe { b.get_unchecked(j * 2..j * 2 + 2) };
                tmp[i] += vec_dot_q8_neon(ab, bb, 64);
            }
        }
        for i in 0..32 {
            cn[i] = tmp[i];
        }
    });
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use rand::{thread_rng, Rng};
    extern crate test;
    use test::Bencher;

    use crate::util::init_rayon;

    use super::*;

    const M: usize = 3200;
    const K: usize = 8640;

    // generate a random vector of BlockQ8_0
    fn gen_rand_block_q8_0() -> BlockQ8_0 {
        let mut rng = thread_rng();
        let d: f32 = rng.gen_range(0.0..1.0 / 256.0);
        let mut qs: [i8; 32] = [0; 32];
        for i in 0..32 {
            qs[i] = rng.gen::<i8>();
        }
        BlockQ8_0 {
            d: f16::from_f32(d),
            qs,
        }
    }

    fn gen_rand_block_q8_0_vec(elms: usize) -> Vec<BlockQ8_0> {
        assert!(elms % 32 == 0, "elms must be a multiple of 32");
        let blks = elms / 32;
        let mut v: Vec<BlockQ8_0> = Vec::with_capacity(blks);
        for _ in 0..blks {
            v.push(gen_rand_block_q8_0());
        }
        v
    }

    #[test]
    fn test_gemv() {
        let (m, k) = (3200, 3200);
        let a = gen_rand_block_q8_0_vec(m * k);
        let b = gen_rand_block_q8_0_vec(k);
        let mut c1 = vec![0.0; m];
        let mut c2 = vec![0.0; m];

        qgemv_naive(m, k, &a, &b, &mut c1);
        qgemv_naive_dot(m, k, &a, &b, &mut c2);
        assert_relative_eq!(&c1[..], &c2[..], epsilon = 1e-3);
    }

    #[bench]
    fn bench_000_qgemv_naive(bench: &mut Bencher) {
        let a = gen_rand_block_q8_0_vec(M * K);
        let b = gen_rand_block_q8_0_vec(K);
        let mut c = vec![0.0; M];
        bench.iter(|| {
            qgemv_naive(M, K, &a, &b, &mut c);
        });
    }

    #[bench]
    fn bench_001_qgemv_use_dot(bench: &mut Bencher) {
        let a = gen_rand_block_q8_0_vec(M * K);
        let b = gen_rand_block_q8_0_vec(K);
        let mut c = vec![0.0; M];
        bench.iter(|| {
            qgemv_naive_dot(M, K, &a, &b, &mut c);
        });
    }

    #[bench]
    fn bench_002_qgemv_rayon(bench: &mut Bencher) {
        init_rayon();
        let a = gen_rand_block_q8_0_vec(M * K);
        let b = gen_rand_block_q8_0_vec(K);
        let mut c = vec![0.0; M];
        bench.iter(|| {
            qgemv_rayon(M, K, &a, &b, &mut c);
        });
    }

    #[bench]
    fn bench_003_qgemv_rayon_unchecked(bench: &mut Bencher) {
        init_rayon();
        let a = gen_rand_block_q8_0_vec(M * K);
        let b = gen_rand_block_q8_0_vec(K);
        let mut c = vec![0.0; M];
        bench.iter(|| {
            qgemv_rayon_unchecked(M, K, &a, &b, &mut c);
        });
    }

    #[bench]
    fn bench_004_qgemv_rayon_chunked(bench: &mut Bencher) {
        init_rayon();
        let a = gen_rand_block_q8_0_vec(M * K);
        let b = gen_rand_block_q8_0_vec(K);
        let mut c = vec![0.0; M];
        bench.iter(|| {
            qgemv_rayon_chunked(M, K, &a, &b, &mut c);
        });
    }

    #[bench]
    fn bench_004_qgemv_rayon_tiled(bench: &mut Bencher) {
        init_rayon();
        let a = gen_rand_block_q8_0_vec(M * K);
        let b = gen_rand_block_q8_0_vec(K);
        let mut c = vec![0.0; M];
        bench.iter(|| {
            qgemv_rayon_tiled(M, K, &a, &b, &mut c);
        });
    }
}
