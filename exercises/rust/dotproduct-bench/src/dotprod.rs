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

pub fn dot_vectorized(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..a.len() / 4 {
        let av = f32x4::from_slice(&a[i * 4..i * 4 + 4]);
        let bv = f32x4::from_slice(&b[i * 4..i * 4 + 4]);
        sum += (av * bv).reduce_sum();
    }
    0.0
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
}
