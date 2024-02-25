#![feature(test)]
#![feature(portable_simd)]
#![allow(soft_unstable)]
#![feature(stdsimd)]

use half::f16;

pub fn exp_vec_f32(v: &mut [f32]) {
    v.iter_mut().for_each(|x| *x = x.exp());
}

pub fn init_exp_cache() -> Vec<f16> {
    let mut exp_f16 = Vec::with_capacity(65536);
    for i in 0..65536 {
        let x16 = f16::from_bits(i as u16);
        let x16e = f16::from_f32(x16.to_f32().exp());
        exp_f16.push(x16e);
    }
    exp_f16
}

pub unsafe fn exp_vec_f32_cached(v: &mut [f32], cache: &[f16]) {
    let cache_ptr = cache.as_ptr();
    v.iter_mut().for_each(|x| {
        let x16 = f16::from_f32(*x);
        let x16n = x16.to_bits();
        *x = (*cache_ptr.add(x16n as usize)).to_f32();
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};
    extern crate test;
    use test::Bencher;

    fn gen_rand_vec_f32(n: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    #[bench]
    fn bench_exp_vec_f32(bench: &mut Bencher) {
        let mut v = gen_rand_vec_f32(1024);
        bench.iter(|| {
            exp_vec_f32(&mut v);
        })
    }

    #[bench]
    fn bench_exp_vec_f32_cached(bench: &mut Bencher) {
        let mut v = gen_rand_vec_f32(1024);
        let cache = init_exp_cache();
        bench.iter(|| unsafe {
            exp_vec_f32_cached(&mut v, &cache);
        })
    }
}
