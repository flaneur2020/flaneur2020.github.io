use rayon::prelude::*;

use crate::util::init_rayon;

fn dot_product_strided(a: &[f32], b: &[f32], a_stride: usize, n: usize) -> f32 {
    use std::arch::aarch64;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    unsafe {
        let mut sumv0 = aarch64::vdupq_n_f32(0.0);
        let mut sumv1 = aarch64::vdupq_n_f32(0.0);
        for i in (0..n).step_by(8) {
            let tmp = vec![
                *a_ptr.add(i * a_stride),
                *a_ptr.add((i + 1) * a_stride),
                *a_ptr.add((i + 2) * a_stride),
                *a_ptr.add((i + 3) * a_stride),
                *a_ptr.add((i + 4) * a_stride),
                *a_ptr.add((i + 5) * a_stride),
                *a_ptr.add((i + 6) * a_stride),
                *a_ptr.add((i + 7) * a_stride),
            ];
            let av0 = aarch64::vld1q_f32(tmp.as_ptr());
            let bv0 = aarch64::vld1q_f32(b_ptr.add(i));
            let av1 = aarch64::vld1q_f32(tmp.as_ptr().add(4));
            let bv1 = aarch64::vld1q_f32(b_ptr.add(i + 4));
            sumv0 = aarch64::vfmaq_f32(sumv0, av0, bv0);
            sumv1 = aarch64::vfmaq_f32(sumv1, av1, bv1);
        }

        aarch64::vaddvq_f32(sumv0) + aarch64::vaddvq_f32(sumv1)
    }
}

fn dot_product(a: &[f32], b: &[f32], n: usize) -> f32 {
    use std::arch::aarch64;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    unsafe {
        let mut sumv0 = aarch64::vdupq_n_f32(0.0);
        let mut sumv1 = aarch64::vdupq_n_f32(0.0);
        for i in (0..n).step_by(8) {
            let av0 = aarch64::vld1q_f32(a_ptr.add(i));
            let bv0 = aarch64::vld1q_f32(b_ptr.add(i));
            let av1 = aarch64::vld1q_f32(a_ptr.add(i + 4));
            let bv1 = aarch64::vld1q_f32(b_ptr.add(i + 4));
            sumv0 = aarch64::vfmaq_f32(sumv0, av0, bv0);
            sumv1 = aarch64::vfmaq_f32(sumv1, av1, bv1);
        }

        aarch64::vaddvq_f32(sumv0) + aarch64::vaddvq_f32(sumv1)
    }
}

fn copy_strided(
    a: &[f32],
    a_shape0: usize,
    a_shape1: usize,
    a_stride0: usize,
    a_stride1: usize,
    b: &mut [f32],
) {
    for i in 0..a_shape0 {
        for j in 0..a_shape1 {
            b[i * a_shape1 + j] = a[i * a_stride0 + j * a_stride1];
        }
    }
}

fn sgemv_dot_rayon_strided(
    m: usize,
    k: usize,
    a_stride: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    c.par_iter_mut().enumerate().for_each(|(mi, c)| {
        let ac = unsafe { a.get_unchecked(mi * k..) };
        *c = dot_product_strided(&ac, b, a_stride, b.len())
    });
}

fn sgemv_dot_rayon_strided_chunked(
    m: usize,
    k: usize,
    a_stride: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    c.par_chunks_exact_mut(4)
        .enumerate()
        .for_each(|(mi, cp)| unsafe {
            cp[0] = dot_product_strided(a.get_unchecked(mi * k..), b, a_stride, b.len());
            cp[1] = dot_product_strided(a.get_unchecked((mi + 1) * k..), b, a_stride, b.len());
            cp[2] = dot_product_strided(a.get_unchecked((mi + 2) * k..), b, a_stride, b.len());
            cp[3] = dot_product_strided(a.get_unchecked((mi + 3) * k..), b, a_stride, b.len());
        });
}

fn sgemv_dot_rayon_strided_copied(
    m: usize,
    k: usize,
    a_stride0: usize,
    a_stride1: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    c.par_chunks_exact_mut(8)
        .enumerate()
        .for_each(|(mi, cp)| unsafe {
            let mut buf = vec![0.0; 8 * k];
            copy_strided(
                a.get_unchecked(mi * k..),
                8,
                k,
                a_stride0,
                a_stride1,
                &mut buf,
            );
            for j in 0..8 {
                cp[j] = dot_product(&buf[j * k..], b, b.len());
            }
        });
}

fn sgemv_dot_rayon_non_strided(m: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    c.par_iter_mut()
        .enumerate()
        .for_each(|(mi, c)| *c = dot_product(&a[mi * k..(mi + 1) * k], b, b.len()));
}

fn sgemv_naive(m: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..k {
        for j in 0..m {
            c[i] += a[i * m + j] * b[i];
        }
    }
}

// a: k x m, b: k
fn sgemv_naive2(m: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for j in 0..m {
        for i in 0..k {
            c[i] += a[i * m + j] * b[i];
        }
    }
}

fn sgemv_unrolled(m: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for ki in 0..k {
        for mi in (0..m).step_by(4) {
            c[mi] += a[ki * m + mi] * b[ki];
            c[mi + 1] += a[ki * m + mi + 1] * b[ki];
            c[mi + 2] += a[ki * m + mi + 2] * b[ki];
            c[mi + 3] += a[ki * m + mi + 3] * b[ki];
        }
    }
}

fn sgemv_unrolled_simd(m: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for ki in 0..k {
        for mi in (0..m).step_by(4) {
            let av = std::simd::f32x4::from_slice(&a[ki * m + mi..ki * m + mi + 4]);
            let bv = std::simd::f32x4::splat(b[ki]);
            let cv = std::simd::f32x4::from_slice(&c[mi..mi + 4]);
            let cv = cv + av * bv;
            cv.copy_to_slice(&mut c[mi..mi + 4]);
        }
    }
}

unsafe fn sgemv_unrolled_simd_neon(m: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    use std::arch::aarch64;
    for ki in 0..k {
        for mi in (0..m).step_by(4) {
            let av = aarch64::vld1q_f32(a[ki * m + mi..].as_ptr());
            let bv = aarch64::vdupq_n_f32(b[ki]);
            let cv = aarch64::vld1q_f32(c[mi..].as_ptr());
            let cv = aarch64::vfmaq_f32(cv, av, bv);
            aarch64::vst1q_f32(c[mi..].as_mut_ptr(), cv);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{linalg::general_mat_vec_mul, Array1, Array2};
    use rand::{thread_rng, Rng};
    extern crate test;
    use crate::util::init_rayon;
    use test::Bencher;

    const M: usize = 3200;
    const K: usize = 100;

    // generate a random vector of BlockQ8_0
    fn generate_random_vector(len: usize) -> Vec<f32> {
        let mut rng = thread_rng();
        (0..len).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    #[bench]
    fn bench_sgemv_dot_non_strided(bench: &mut Bencher) {
        let a = generate_random_vector(M * K);
        let b = generate_random_vector(K);
        let mut c = vec![0.0; M];
        bench.iter(|| sgemv_dot_rayon_non_strided(M, K, &a, &b, &mut c));
    }

    #[bench]
    fn bench_sgemv_dot_rayon_strided1(bench: &mut Bencher) {
        let a = generate_random_vector(M * K);
        let b = generate_random_vector(K);
        let mut c = vec![0.0; M];
        bench.iter(|| sgemv_dot_rayon_strided(M, K, 1, &a, &b, &mut c));
    }

    #[bench]
    fn bench_sgemv_dot_rayon_strided_copied(bench: &mut Bencher) {
        let a = generate_random_vector(M * K);
        let b = generate_random_vector(K);
        let mut c = vec![0.0; M];
        bench.iter(|| sgemv_dot_rayon_strided_copied(M, K, K, 1, &a, &b, &mut c));
    }

    #[bench]
    fn bench_sgemv_naive(bench: &mut Bencher) {
        let a = generate_random_vector(M * K);
        let b = generate_random_vector(K);
        let mut c = vec![0.0; M];
        bench.iter(|| sgemv_naive(M, K, &a, &b, &mut c));
    }

    #[bench]
    fn bench_sgemv_naive2(bench: &mut Bencher) {
        let a = generate_random_vector(M * K);
        let b = generate_random_vector(K);
        let mut c = vec![0.0; M];
        bench.iter(|| sgemv_naive2(M, K, &a, &b, &mut c));
    }

    #[bench]
    fn bench_sgemv_unrolled(bench: &mut Bencher) {
        let a = generate_random_vector(M * K);
        let b = generate_random_vector(K);
        let mut c = vec![0.0; M];
        bench.iter(|| sgemv_unrolled(M, K, &a, &b, &mut c));
    }

    #[bench]
    fn bench_sgemv_unrolled_simd(bench: &mut Bencher) {
        let a = generate_random_vector(M * K);
        let b = generate_random_vector(K);
        let mut c = vec![0.0; M];
        bench.iter(|| sgemv_unrolled_simd(M, K, &a, &b, &mut c));
    }

    #[bench]
    fn bench_sgemv_unrolled_simd_neon(bench: &mut Bencher) {
        let a = generate_random_vector(M * K);
        let b = generate_random_vector(K);
        let mut c = vec![0.0; M];
        bench.iter(|| unsafe { sgemv_unrolled_simd_neon(M, K, &a, &b, &mut c) });
    }
}
