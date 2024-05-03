use rayon::prelude::*;

use crate::util::init_rayon;

pub fn dot_product(a: &[f32], b: &[f32], n: usize) -> f32 {
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

pub fn dot_product_strided(a: &[f32], b: &[f32], a_stride: usize, n: usize) -> f32 {
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

pub fn sgemv_naive(m: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for mi in 0..m {
        let mut sum = 0.0;
        for ki in 0..k {
            sum += a[mi * k + ki] * b[ki];
        }
        c[mi] = sum;
    }
}

pub fn sgemv_dot(m: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for mi in 0..m {
        c[mi] = dot_product(&a[mi * k..(mi + 1) * k], b, b.len());
    }
}

pub fn sgemv_dot_rayon(m: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    c.par_iter_mut()
        .enumerate()
        .for_each(|(mi, c)| *c = dot_product(&a[mi * k..(mi + 1) * k], b, b.len()));
}

pub fn sgemv_dot_threadpool(m: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let threads = 4;
    let chunk_size = m / threads;

    rayon::scope(|s| {
        for (i, cp) in c.chunks_exact_mut(chunk_size).enumerate() {
            s.spawn(move |_| {
                for j in 0..chunk_size {
                    let mi = i * chunk_size + j;
                    let ac = unsafe { a.get_unchecked(mi * k..(mi + 1) * k) };
                    cp[j] = dot_product(&ac, b, b.len());
                }
            })
        }
    });
}

pub fn sgemv_dot_rayon_unchecked(m: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    c.par_iter_mut().enumerate().for_each(|(mi, c)| {
        let ac = unsafe { a.get_unchecked(mi * k..) };
        *c = dot_product(&ac, b, b.len())
    });
}

pub fn sgemv_dot_rayon_chunked(m: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    c.par_chunks_exact_mut(4).enumerate().for_each(|(mi, cc)| {
        cc[0] = dot_product(&a[(mi * 4 + 0) * k..], b, b.len());
        cc[1] = dot_product(&a[(mi * 4 + 1) * k..], b, b.len());
        cc[2] = dot_product(&a[(mi * 4 + 2) * k..], b, b.len());
        cc[3] = dot_product(&a[(mi * 4 + 3) * k..], b, b.len());
    });
}

pub fn sgemv_dot_rayon_chunked_unchecked(m: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    c.par_chunks_exact_mut(4)
        .enumerate()
        .for_each(|(mi, cc)| unsafe {
            cc[0] = dot_product(&a.get_unchecked((mi * 4 + 0) * k..), b, b.len());
            cc[1] = dot_product(&a.get_unchecked((mi * 4 + 1) * k..), b, b.len());
            cc[2] = dot_product(&a.get_unchecked((mi * 4 + 2) * k..), b, b.len());
            cc[3] = dot_product(&a.get_unchecked((mi * 4 + 3) * k..), b, b.len());
        });
}

pub fn sgemv_dot_tiled1d(m: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let n_tile = k / 4;
    for ki in (0..k).step_by(n_tile) {
        for mi in 0..m {
            unsafe {
                c[mi] += dot_product(
                    a.get_unchecked(mi * k + ki..),
                    &b.get_unchecked(ki..),
                    n_tile,
                );
            }
        }
    }
}

pub fn sgemv_dot_rayon_strided(
    m: usize,
    k: usize,
    a_stride: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    c.par_iter_mut().enumerate().for_each(|(mi, c)| {
        let ac = unsafe { a.get_unchecked(mi * k..) };
        *c = dot_product_strided(&ac, b, 1, b.len())
    });
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
    const K: usize = 8640;

    // generate a random vector of BlockQ8_0
    fn generate_random_vector(len: usize) -> Vec<f32> {
        let mut rng = thread_rng();
        (0..len).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    #[bench]
    fn bench_00_sgemv_naive(bench: &mut Bencher) {
        let a = generate_random_vector(M * K);
        let b = generate_random_vector(K);
        let mut c = vec![0.0; M];
        bench.iter(|| sgemv_naive(M, K, &a, &b, &mut c));
    }

    #[bench]
    fn bench_01_sgemv_ndarray(bench: &mut Bencher) {
        let a = Array2::from_shape_vec((M, K), generate_random_vector(M * K)).unwrap();
        let b = Array1::from_vec(generate_random_vector(K));
        let mut c = Array1::zeros(M);

        bench.iter(|| {
            general_mat_vec_mul(1.0, &a, &b, 0.0, &mut c);
        });
    }

    #[bench]
    fn bench_02_sgemv_dot(bench: &mut Bencher) {
        let a = generate_random_vector(M * K);
        let b = generate_random_vector(K);
        let mut c = vec![0.0; M];
        bench.iter(|| sgemv_dot(M, K, &a, &b, &mut c));
    }

    #[bench]
    fn bench_02_sgemv_dot_tiled1d(bench: &mut Bencher) {
        let a = generate_random_vector(M * K);
        let b = generate_random_vector(K);
        let mut c = vec![0.0; M];
        bench.iter(|| sgemv_dot_tiled1d(M, K, &a, &b, &mut c));
    }

    #[bench]
    fn bench_03_sgemv_dot_rayon(bench: &mut Bencher) {
        let a = generate_random_vector(M * K);
        let b = generate_random_vector(K);
        let mut c = vec![0.0; M];
        bench.iter(|| sgemv_dot_rayon(M, K, &a, &b, &mut c));
    }

    #[bench]
    fn bench_04_sgemv_dot_threadpool(bench: &mut Bencher) {
        let a = generate_random_vector(M * K);
        let b = generate_random_vector(K);
        let mut c = vec![0.0; M];
        bench.iter(|| sgemv_dot_threadpool(M, K, &a, &b, &mut c));
    }

    #[bench]
    fn bench_05_sgemv_dot_rayon_unchecked(bench: &mut Bencher) {
        let a = generate_random_vector(M * K);
        let b = generate_random_vector(K);
        let mut c = vec![0.0; M];
        bench.iter(|| sgemv_dot_rayon(M, K, &a, &b, &mut c));
    }

    #[bench]
    fn bench_06_sgemv_dot_rayon_chunked(bench: &mut Bencher) {
        let a = generate_random_vector(M * K);
        let b = generate_random_vector(K);
        let mut c = vec![0.0; M];
        bench.iter(|| sgemv_dot_rayon_chunked(M, K, &a, &b, &mut c));
    }

    #[bench]
    fn bench_07_sgemv_dot_rayon_strided1(bench: &mut Bencher) {
        let a = generate_random_vector(M * K);
        let b = generate_random_vector(K);
        let mut c = vec![0.0; M];
        bench.iter(|| sgemv_dot_rayon_strided(M, K, 1, &a, &b, &mut c));
    }
}
