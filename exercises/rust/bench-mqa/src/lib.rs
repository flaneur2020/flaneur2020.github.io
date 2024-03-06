#![feature(test)]
#![feature(portable_simd)]
#![allow(soft_unstable)]
#![feature(stdsimd)]

use half::f16;
use rand::{thread_rng, Rng};

struct Work {
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    q: Vec<f32>,           // n_heads x head_dim
    kcache: Vec<f32>,      // seq_len x n_heads x head_dim
    vcache: Vec<f32>,      // seq_len x n_heads x head_dim
    attn_scores: Vec<f32>, // n_heads x seq_len
    output: Vec<f32>,      // n_heads x head_dim
}

impl Work {
    fn new(seq_len: usize, n_heads: usize, head_dim: usize) -> Self {
        let q = gen_rand_vec_f32(n_heads * head_dim);
        let kcache = gen_rand_vec_f32(seq_len * n_heads * head_dim);
        let vcache = gen_rand_vec_f32(seq_len * n_heads * head_dim);
        let attn_scores = gen_rand_vec_f32(n_heads * seq_len);
        let output = vec![0.0; n_heads * head_dim];
        Self {
            seq_len,
            n_heads,
            head_dim,
            q,
            kcache,
            vcache,
            attn_scores,
            output,
        }
    }

    // kvcache layout 1: seq_len x n_heads x head_dim
    fn multi_head_attention_layout1(&self) -> Vec<f32> {
        let mut attn_scores = vec![0.0; self.n_heads * self.seq_len]; // n_heads, seq_len
        let k_cache_strides = [self.n_heads * self.head_dim, self.head_dim, 1]; // seq_len, n_heads, head_dim
        let k_cache_strides = [k_cache_strides[1], k_cache_strides[0], k_cache_strides[2]]; // n_heads, seq_len, head_dim
        let q_strides = [self.head_dim, 1];
        for h in 0..self.n_heads {
            for pos in 0..self.seq_len {
                let offset_a = h * k_cache_strides[0] + pos * k_cache_strides[1];
                let offset_b = h * q_strides[0];
                attn_scores[h * self.seq_len + pos] = strided_vec_dot(
                    &self.kcache,
                    &self.q,
                    offset_a,
                    offset_b,
                    1,
                    1,
                    self.head_dim,
                );
            }
        }

        let mut output = vec![0.0; self.n_heads * self.head_dim];
        let v_cache_strides = [self.n_heads * self.head_dim, self.head_dim, 1]; // seq_len, n_heads, head_dim
        let v_cache_strides = [v_cache_strides[1], v_cache_strides[2], v_cache_strides[0]]; // n_heads, head_dim, seq_len
        let attn_scores_strides = [self.seq_len, 1]; // n_heads, seq_len
        for h in 0..self.n_heads {
            for i in 0..self.head_dim {
                let offset_a = h * v_cache_strides[0] + i * v_cache_strides[1];
                let offset_b = h * attn_scores_strides[0];
                let stride_a = v_cache_strides[2];
                let stride_b = attn_scores_strides[1];
                output[h * self.head_dim + i] = strided_vec_dot(
                    &self.vcache,
                    &attn_scores,
                    offset_a,
                    offset_b,
                    stride_a,
                    stride_b,
                    self.seq_len,
                );
            }
        }
        output
    }

    // kvcache layout 1: n_heads x seq_len x head_dim
    fn multi_head_attention_layout2(&self) -> Vec<f32> {
        let mut attn_scores = vec![0.0; self.n_heads * self.seq_len]; // n_heads, seq_len
        let k_cache_strides = [self.seq_len * self.head_dim, self.head_dim, 1]; // n_heads, seq_len, head_dim
        let q_strides = [self.head_dim, 1];
        for h in 0..self.n_heads {
            for pos in 0..self.seq_len {
                let offset_a = h * k_cache_strides[0] + pos * k_cache_strides[1];
                let offset_b = h * q_strides[0];
                attn_scores[h * self.seq_len + pos] = strided_vec_dot(
                    &self.kcache,
                    &self.q,
                    offset_a,
                    offset_b,
                    1,
                    1,
                    self.head_dim,
                );
            }
        }

        let mut output = vec![0.0; self.n_heads * self.head_dim];
        let v_cache_strides = [self.seq_len * self.head_dim, self.head_dim, 1]; // n_heads, seq_len, head_dim
        let v_cache_strides = [v_cache_strides[0], v_cache_strides[2], v_cache_strides[1]]; // n_heads, head_dim, seq_len
        let attn_scores_strides = [self.seq_len, 1]; // seq_len, n_heads
        for h in 0..self.n_heads {
            for i in 0..self.head_dim {
                let offset_a = h * v_cache_strides[0] + i * v_cache_strides[1];
                let offset_b = h * attn_scores_strides[0];
                let stride_a = v_cache_strides[2];
                let stride_b = attn_scores_strides[1];
                output[h * self.head_dim + i] = strided_vec_dot(
                    &self.vcache,
                    &attn_scores,
                    offset_a,
                    offset_b,
                    stride_a,
                    stride_b,
                    self.seq_len,
                );
            }
        }
        output
    }

    fn mha_part1(&self) {
        let mut attn_scores = vec![0.0; self.n_heads * self.seq_len]; // n_heads, seq_len
        let k_cache_strides = [self.seq_len * self.head_dim, self.head_dim, 1]; // n_heads, seq_len, head_dim
        let q_strides = [self.head_dim, 1];
        for h in 0..self.n_heads {
            for pos in 0..self.seq_len {
                let offset_a = h * k_cache_strides[0] + pos * k_cache_strides[1];
                let offset_b = h * q_strides[0];
                attn_scores[h * self.seq_len + pos] = strided_vec_dot(
                    &self.kcache,
                    &self.q,
                    offset_a,
                    offset_b,
                    1,
                    1,
                    self.head_dim,
                );
            }
        }
    }

    fn mha_part2(&self) {
        let mut output = vec![0.0; self.n_heads * self.head_dim];
        let v_cache_strides = [self.seq_len * self.head_dim, self.head_dim, 1]; // n_heads, seq_len, head_dim
        let v_cache_strides = [v_cache_strides[0], v_cache_strides[2], v_cache_strides[1]]; // n_heads, head_dim, seq_len
        let attn_scores_strides = [self.seq_len, 1]; // seq_len, n_heads
        for h in 0..self.n_heads {
            for i in 0..self.head_dim {
                let offset_a = h * v_cache_strides[0] + i * v_cache_strides[1];
                let offset_b = h * attn_scores_strides[0];
                let stride_a = v_cache_strides[2];
                let stride_b = attn_scores_strides[1];
                output[h * self.head_dim + i] = strided_vec_dot(
                    &self.vcache,
                    &self.attn_scores,
                    offset_a,
                    offset_b,
                    stride_a,
                    stride_b,
                    self.seq_len,
                );
            }
        }
    }
}

fn strided_vec_dot(
    a: &[f32],
    b: &[f32],
    offset_a: usize,
    offset_b: usize,
    stride_a: usize,
    stride_b: usize,
    k: usize,
) -> f32 {
    let mut sum = 0.0;
    for i in 0..k {
        sum += a[offset_a + i * stride_a] * b[offset_b + i * stride_b];
    }
    sum
}

fn gen_rand_vec_f32(n: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate test;
    use test::Bencher;

    const SEQ_LEN: usize = 100;
    const N_HEADS: usize = 8;
    const HEAD_DIM: usize = 2048;

    #[test]
    fn test_work() {
        let work = Work::new(SEQ_LEN, N_HEADS, HEAD_DIM);
        work.multi_head_attention_layout1();
    }

    #[bench]
    fn bench_multi_head_attention_layout1(b: &mut Bencher) {
        let work = Work::new(SEQ_LEN, N_HEADS, HEAD_DIM);
        b.iter(|| work.multi_head_attention_layout1());
    }

    #[bench]
    fn bench_multi_head_attention_layout2(b: &mut Bencher) {
        let work = Work::new(SEQ_LEN, N_HEADS, HEAD_DIM);
        b.iter(|| work.multi_head_attention_layout1());
    }

    #[bench]
    fn bench_mha_part1(b: &mut Bencher) {
        let work = Work::new(SEQ_LEN, N_HEADS, HEAD_DIM);
        b.iter(|| work.mha_part1());
    }

    #[bench]
    fn bench_mha_part2(b: &mut Bencher) {
        let work = Work::new(SEQ_LEN, N_HEADS, HEAD_DIM);
        b.iter(|| work.mha_part2());
    }
}
