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

fn sgemv_dot_rayon_non_strided(m: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    c.par_iter_mut()
        .enumerate()
        .for_each(|(mi, c)| *c = dot_product(&a[mi * k..(mi + 1) * k], b, b.len()));
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
}
