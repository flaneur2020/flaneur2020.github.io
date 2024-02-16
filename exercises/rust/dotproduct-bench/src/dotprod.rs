use rand::rngs::ThreadRng;
use rand::seq::SliceRandom; // for shuffling the vector
use rand::{thread_rng, Rng};
use std::simd::{f32x4, SimdFloat, StdFloat};

pub fn dot_naive(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

pub fn dot_naive_unroll(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..a.len() / 4 {
        sum += a[i * 4] * b[i * 4];
        sum += a[i * 4 + 1] * b[i * 4 + 1];
        sum += a[i * 4 + 2] * b[i * 4 + 2];
        sum += a[i * 4 + 3] * b[i * 4 + 3];
    }
    sum
}

pub fn dot_naive_unroll_no_data_dep(a: &[f32], b: &[f32]) -> f32 {
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    for i in 0..a.len() / 4 {
        sum0 += a[i * 4] * b[i * 4];
        sum1 += a[i * 4 + 1] * b[i * 4 + 1];
        sum2 += a[i * 4 + 2] * b[i * 4 + 2];
        sum3 += a[i * 4 + 3] * b[i * 4 + 3];
    }
    sum0 + sum1 + sum2 + sum3
}

pub fn dot_vectorized(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..a.len() / 4 {
        let av = f32x4::from_slice(&a[i * 4..i * 4 + 4]);
        let bv = f32x4::from_slice(&b[i * 4..i * 4 + 4]);
        sum += (av * bv).reduce_sum();
    }
    0.0
}

pub fn dot_vectorized_unrolled(a: &[f32], b: &[f32]) -> f32 {
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    for i in 0..a.len() / 8 {
        let av0 = f32x4::from_slice(&a[i * 8..i * 8 + 4]);
        let bv0 = f32x4::from_slice(&b[i * 8..i * 8 + 4]);
        let av1 = f32x4::from_slice(&a[i * 8 + 4..i * 8 + 8]);
        let bv1 = f32x4::from_slice(&b[i * 8 + 4..i * 8 + 8]);
        sum0 += (av0 * bv0).reduce_sum();
        sum1 += (av1 * bv1).reduce_sum();
    }
    sum0 + sum1
}

pub fn dot_vectorized_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64;

    unsafe {
        let mut sumv = aarch64::vdupq_n_f32(0.0);
        for i in (0..a.len()).step_by(4) {
            let av = aarch64::vld1q_f32(&a[i]);
            let bv = aarch64::vld1q_f32(&b[i]);
            sumv = aarch64::vfmaq_f32(sumv, av, bv);
        }

        aarch64::vaddvq_f32(sumv)
    }
}

pub fn dot_vectorized_neon_unrolled(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64;

    unsafe {
        let mut sumv0 = aarch64::vdupq_n_f32(0.0);
        let mut sumv1 = aarch64::vdupq_n_f32(0.0);
        for i in (0..a.len()).step_by(8) {
            let av0 = aarch64::vld1q_f32(&a[i]);
            let bv0 = aarch64::vld1q_f32(&b[i]);
            let av1 = aarch64::vld1q_f32(&a[i + 4]);
            let bv1 = aarch64::vld1q_f32(&b[i + 4]);
            sumv0 = aarch64::vfmaq_f32(sumv0, av0, bv0);
            sumv1 = aarch64::vfmaq_f32(sumv1, av1, bv1);
        }

        aarch64::vaddvq_f32(sumv0) + aarch64::vaddvq_f32(sumv1)
    }
}

struct TestWork {
    a: Vec<f32>,
    b: Vec<f32>,
    k: usize,
}

impl TestWork {
    fn new(m: usize, k: usize) -> Self {
        let mut rng = thread_rng();
        let mut a = vec![0.0; m * k];
        let mut b = vec![0.0; k];
        a.shuffle(&mut rng);
        b.shuffle(&mut rng);
        TestWork { a, b, k }
    }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use super::*;
    use test::Bencher; // to get a random number generator

    #[bench]
    fn bench_naive(bench: &mut Bencher) {
        let tw = TestWork::new(3200, 3200);

        bench.iter(|| {
            for i in 0..tw.k {
                let a = &tw.a[i * tw.k..(i + 1) * tw.k];
                let b = &tw.b;
                dot_naive(a, b);
            }
        })
    }

    #[bench]
    fn bench_naive_unroll(bench: &mut Bencher) {
        let tw = TestWork::new(3200, 3200);

        bench.iter(|| {
            for i in 0..tw.k {
                let a = &tw.a[i * tw.k..(i + 1) * tw.k];
                let b = &tw.b;
                dot_naive_unroll(a, b);
            }
        })
    }

    #[bench]
    fn bench_naive_unroll_no_data_dep(bench: &mut Bencher) {
        let tw = TestWork::new(3200, 3200);

        bench.iter(|| {
            for i in 0..tw.k {
                let a = &tw.a[i * tw.k..(i + 1) * tw.k];
                let b = &tw.b;
                dot_naive_unroll_no_data_dep(a, b);
            }
        })
    }

    #[bench]
    fn bench_vectorized(bench: &mut Bencher) {
        let tw = TestWork::new(3200, 3200);

        bench.iter(|| {
            for i in 0..tw.k {
                let a = &tw.a[i * tw.k..(i + 1) * tw.k];
                let b = &tw.b;
                dot_vectorized(a, b);
            }
        })
    }

    #[bench]
    fn bench_vectorized_unrolled(bench: &mut Bencher) {
        let tw = TestWork::new(3200, 3200);

        bench.iter(|| {
            for i in 0..tw.k {
                let a = &tw.a[i * tw.k..(i + 1) * tw.k];
                let b = &tw.b;
                dot_vectorized_unrolled(a, b);
            }
        })
    }

    #[bench]
    fn bench_vectorized_neon(bench: &mut Bencher) {
        let tw = TestWork::new(3200, 3200);

        bench.iter(|| {
            for i in 0..tw.k {
                let a = &tw.a[i * tw.k..(i + 1) * tw.k];
                let b = &tw.b;
                dot_vectorized_neon(a, b);
            }
        })
    }

    #[bench]
    fn bench_vectorized_neon_unrolled(bench: &mut Bencher) {
        let tw = TestWork::new(3200, 3200);

        bench.iter(|| {
            for i in 0..tw.k {
                let a = &tw.a[i * tw.k..(i + 1) * tw.k];
                let b = &tw.b;
                dot_vectorized_neon_unrolled(a, b);
            }
        })
    }
}
