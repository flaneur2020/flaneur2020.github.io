春节假期在做 crabml 的 CPU 推理优化，在做了一些优化后，推理一个 3B 的 Q8_0 模型的性能从 7 tok/s 优化到了 12 tok/s，仍比跑到 14 tok/s 的 ggml 慢一些，不过先记录一下做优化的过程，感觉追平性能还是有戏的。

打开 GGML_PERF=1 来编译 llama.cpp，跑推理时可以看到每个算子的耗时分布：

```
perf_total_per_op_us[             ADD] =   0.160 ms
perf_total_per_op_us[             MUL] =   0.225 ms
perf_total_per_op_us[        RMS_NORM] =   0.210 ms
perf_total_per_op_us[         MUL_MAT] =  63.537 ms
perf_total_per_op_us[           SCALE] =   0.032 ms
perf_total_per_op_us[             CPY] =   0.544 ms
perf_total_per_op_us[         RESHAPE] =   0.078 ms
perf_total_per_op_us[            VIEW] =   0.104 ms
perf_total_per_op_us[         PERMUTE] =   0.052 ms
perf_total_per_op_us[       TRANSPOSE] =   0.026 ms
perf_total_per_op_us[        GET_ROWS] =   0.002 ms
perf_total_per_op_us[   DIAG_MASK_INF] =   0.027 ms
perf_total_per_op_us[        SOFT_MAX] =   0.106 ms
perf_total_per_op_us[            ROPE] =   0.193 ms
perf_total_per_op_us[           UNARY] =   0.274 ms
```

可见 MUL_MAT 占据了耗时的绝对分布。在推理中，这里的 MUL_MAT 基本是 矩阵 x 向量 的相乘，也就是 GEMV 操作。

GEMV 可以比较 naive 拆解为 dot product 的循环：

![[Screenshot 2024-02-25 at 11.56.32.png]]

naive 的 GEMV 和 dot product 可以理解为：

```rust
pub fn gemv_naive_dot(m: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for mi in 0..m {
        c[mi] = vec_dot(&a[(mi * k)..], b, k);
    }
}

pub fn dot_naive(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..b.len() {
        sum += a[i] * b[i];
    }
    sum
}
```

往简单讲，做推理优化的主要内容就是用 SIMD、多线程、GPU 等等手段，来优化上面的这两个循环的执行。
## Q8_0 量化

不过在本地跑模型推理，全精度的 f32 类型不大可能在考虑范围之内，内存不够。所以需要经过量化来减少内存占用，比如 8bit 量化，就是将一个 f32 值压缩到 8bit，这一来只需要四分之一的内存占用。当然也有更激进的 4bit 量化甚至 1 bit 量化，能够节约更多的内存，当然精度有所损失，模型的表现性能也会有相应地下降。

除了单纯的减少内存占用之外，经过量化之后的矩阵乘法所需的内存吞吐也能够大大减少，而 GEMV 被普遍认为是一个 memory bandwidth bound 的操作，经过量化之后，矩阵相乘的性能也会大大提高。

Q8_0 量化是 ggml 中最粗暴的一种量化方式，大约是选择 32 个 f32 数值作为一个块，在块内找出来最大的数值按 f16 精度保存，让块内的每个数去除以最大值，放在 -256 ~ 255 的一个范围。在内存中每个块这样表示：

```rust
#[repr(C, packed)]
#[derive(Debug, Clone)]
pub struct BlockQ8_0 {
    d: f16,       // delta
    qs: [i8; 32], // quants
}
```

反量化的逻辑很简单，把 qs 的每个元素乘以 d 字段就完事了：

```rust
pub fn dequantize(blks: &[Block8_0], out: &mut [f32]) {
    for (blk, i) in blks.iter().enumerate() {
    	let d = blk.d.to_f32();
		for j in 0..32 {
		    let q = blk.qs[j];
		    out[i*32 + j] = q as f32 * d;
	    }
    }
}
```


做 dot product 时不需要将它做反量化后再相乘，可以直接对两个 `&[Block8_0]` 直接计算：

```rust

pub fn vec_dot_q8_naive(n: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let mut result: f32 = 0.0;
    for i in 0..(n / 32) {
        let mut tmp: f32 = 0.0;
        for j in 0..32 {
            tmp += (x[i].qs[j] as i32 * y[i].qs[j] as i32) as f32;
        }
        result += tmp * f16::to_f32(x[i as usize].d) * f16::to_f32(y[i as usize].d);
    }
    result
}
```

已知 crabml 跑的不如 ggml 快，这个 naive 实现一定是距离 ggml 的实现有差距的，可以捞出来 ggml 的实现可以做一个比较。
## benchmark

ggml 中对应的函数是 `ggml_vec_dot_q8_0_q8_0`，是一个 C 函数，可以从 rust 这边调用过来：

```rust
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

pub fn vec_dot_q8_ggml(n: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let mut result: f32 = 0.0;
    unsafe {
        ggml_vec_dot_q8_0_q8_0(
            n as i32,
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
```

它的几个参数好像没有在用到，就不细究分别是干啥的了，在 rust 这头简单包装成一个 `vec_dot_q8_ggml`，可以 benchmark 一下：

```rust
    const TEST_BLOCKS: usize = 1000;
    const TEST_ELEMS: usize = TEST_BLOCKS * 32;

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


    #[bench]
    fn bench_vec_dot_q8_ggml(b: &mut Bencher) {
        let v1 = gen_rand_block_q8_0_vec(TEST_BLOCKS);
        let v2 = gen_rand_block_q8_0_vec(TEST_BLOCKS);
        b.iter(|| vec_dot_q8_ggml(TEST_ELEMS, &v1, &v2));
    }

    #[bench]
    fn bench_vec_dot_q8_naive(b: &mut Bencher) {
        let v1 = gen_rand_block_q8_0_vec(TEST_BLOCKS);
        let v2 = gen_rand_block_q8_0_vec(TEST_BLOCKS);
        b.iter(|| vec_dot_q8_naive(TEST_ELEMS, &v1, &v2));
    }

```

跑 cargo bench 可以看到：

```
test tests::bench_vec_dot_q8_ggml                     ... bench:         703 ns/iter (+/- 50)
test tests::bench_vec_dot_q8_naive                    ... bench:      10,898 ns/iter (+/- 464)
```

差 15 倍！

差距非常可观。
## 优化1：stdsimd

首先想到的优化方法是 SIMD，试试 rust 标准库中的 stdsimd，来将 4 个 int8 组成一个 SIMD 寄存器，然后对 SIMD 寄存器直接相乘，一乘就是乘 4 个：

```rust
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
```

跑下 benchmark，相比 naive 确实有了比较明显的三倍提升，但是距离 ggml 仍差距不少：

```
test tests::bench_vec_dot_q8_ggml                     ... bench:         703 ns/iter (+/- 50)
test tests::bench_vec_dot_q8_naive                    ... bench:      10,898 ns/iter (+/- 464)
test tests::bench_vec_dot_q8_stdsimd                  ... bench:       3,070 ns/iter (+/- 84)
```

看 GGML 代码是各个平台手撸 SIMD 指令，与上面理论上可以跨平台的 stdsimd 有一个明显的不同是，上面的代码使用了 i32x4 这样的 simd 寄存器，而 GGML 中使用了 i8x16 寄存器，这样两轮 SIMD 指令就能算一对 BlockQ8_0 的 dot product，有四倍的差距是很 make sense 的。

在 stdsimd 直接使用 i8x16 寄存器可不可以呢？不幸的是似乎不行，点积运算很简单，先相乘、后相加。不过两个 i8 相乘，按说结果需要一个 i16 寄存器才放得下，这就要求我们起码得用 i16x8 的寄存器用于相乘。

不过 ARM 的 NEON 指令中有一个 vdotq_s32 的指令，能够将两个 i8x16 寄存器做点积，并将结果保存到一个 i32x4 寄存器中，这里就不怕乘法溢出的问题了。

显然 i8x16 寄存器 + vdotq_s32 指令是更高效的实现，我们也手写一把。

## 优化 2：手写 NEON 指令

多提一嘴 ARM 的 NEON 指令集。在 NEON 中，SIMD 寄存器最大可以 128bit，这被称作一个 `q register`，其中 q 是 `quad word` 的缩写，4 个 32 bit 的字，也就是 128 bit。

似乎我们只需要关注 q 寄存器这一种寄存器，既然 SIMD 了就肯定有多大用多大。

利用 q 寄存器，可以做 i32x4 或者 i16x8 或者 i8x16 的计算。在这里我们一个 block 32 个 bytes，通过 i8x16 计算，两轮计算就能算一对 Block 的点积。

参考 ggml 的手写 NEON 指令写一个 rust 版：

```rust

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
```

这段代码中循环一轮迭代就可以计算出一对 Block 的点积。

计算过程大约如下：

1. 将两个 Block 拆成两个 i8x16 寄存器（av0、av1 和 bv0、bv1）
2. 将两个 i8x16 寄存器分别做 `vdotq_s32` 计算变成 i32x4，然后通过 `vaddq_s32` 相加，成为一个 i32x4 寄存器
3. 在乘以 d 之前，通过 `vcvtq_f32_s32` 将 i32x4 转化为 f32x4 类型，指令中的 "cvt" 是 "convert" 的缩写，得到 `tmpv`
4. 通过 `vmlaq_n_f32` 将两个 d 相乘的到的 f32 值，乘以 `tmpv`，并累加到 f32x4 的 `sumv` 寄存器上
5. 最后将 f32x4 的 `sumv` 寄存器通过 `vaddvq_f32` 加为一个 f32


跑一把 bench：

```

test tests::bench_vec_dot_q8_ggml                     ... bench:         703 ns/iter (+/- 50)
test tests::bench_vec_dot_q8_naive                    ... bench:      10,898 ns/iter (+/- 464)
test tests::bench_vec_dot_q8_stdsimd                  ... bench:       3,070 ns/iter (+/- 84)
test tests::bench_vec_dot_q8_neon                     ... bench:         943 ns/iter (+/- 100)
```

提升很明显，快了三倍，但是仍比 ggml 的实现慢 200ns/iter。

不过 ggml 的代码中做了 loop unrolling，我们也跟着做一下。

## 优化3：loop unrolling


```rust
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

            sumv0 = aarch64::vmlaq_n_f32(
                sumv0,
                aarch64::vcvtq_f32_s32(aarch64::vaddq_s32(
                    aarch64::vdotq_s32(zerov, av00, bv00),
                    aarch64::vdotq_s32(zerov, av01, bv01),
                )),
                f16::to_f32(ab0.d) * f16::to_f32(bb0.d),
            );

            sumv1 = aarch64::vmlaq_n_f32(
                sumv1,
                aarch64::vcvtq_f32_s32(aarch64::vaddq_s32(
                    aarch64::vdotq_s32(zerov, av10, bv10),
                    aarch64::vdotq_s32(zerov, av11, bv11),
                )),
                f16::to_f32(ab1.d) * f16::to_f32(bb1.d),
            );
        }

        aarch64::vaddvq_f32(sumv0) + aarch64::vaddvq_f32(sumv1)
    }
}

```

在这个版本中，我们和 GGML 一样，一次性计算两个 Block 的点积，按说相比没有 unroll 的版本，肯定能快一些：

```
test tests::bench_vec_dot_q8_ggml                     ... bench:         703 ns/iter (+/- 50)
test tests::bench_vec_dot_q8_naive                    ... bench:      10,898 ns/iter (+/- 464)
test tests::bench_vec_dot_q8_stdsimd                  ... bench:       3,070 ns/iter (+/- 84)
test tests::bench_vec_dot_q8_neon                     ... bench:         943 ns/iter (+/- 100)
test tests::bench_vec_dot_q8_neon_unrolled            ... bench:         712 ns/iter (+/- 117)
```

但是到这里仍然差 10ns。是因为 rust 相比 C 仍没有那么 zero cost 吗？

## 优化4：loop unrolling + 合并变量

问了一下 GPT4，它给建议说，这里我们定义了 sumv0 和 sumv1 两个变量，有重复，没有必要定义两个。

感觉有道理，这个操作是可以有累加的，何必定义两个变量？

```rust
#[cfg(target_arch = "aarch64")]
pub fn vec_dot_q8_neon_unrolled_single_sum(n: usize, a: &[BlockQ8_0], b: &[BlockQ8_0]) -> f32 {
    unsafe {
        use std::arch::aarch64;
        let mut sumv0 = aarch64::vdupq_n_f32(0.0);
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

            sumv0 = aarch64::vmlaq_n_f32(
                sumv0,
                aarch64::vcvtq_f32_s32(aarch64::vaddq_s32(
                    aarch64::vdotq_s32(zerov, av00, bv00),
                    aarch64::vdotq_s32(zerov, av01, bv01),
                )),
                f16::to_f32(ab0.d) * f16::to_f32(bb0.d),
            );

            sumv0 = aarch64::vmlaq_n_f32(
                sumv0,
                aarch64::vcvtq_f32_s32(aarch64::vaddq_s32(
                    aarch64::vdotq_s32(zerov, av10, bv10),
                    aarch64::vdotq_s32(zerov, av11, bv11),
                )),
                f16::to_f32(ab1.d) * f16::to_f32(bb1.d),
            );
        }

        aarch64::vaddvq_f32(sumv0)
    }
}
```

把 `sumv1` 干掉，只留一个 `sumv0`。

再跑下：

```
test tests::bench_vec_dot_q8_ggml                     ... bench:         703 ns/iter (+/- 50)
test tests::bench_vec_dot_q8_naive                    ... bench:      10,898 ns/iter (+/- 464)
test tests::bench_vec_dot_q8_stdsimd                  ... bench:       3,070 ns/iter (+/- 84)
test tests::bench_vec_dot_q8_neon                     ... bench:         943 ns/iter (+/- 100)
test tests::bench_vec_dot_q8_neon_unrolled            ... bench:         712 ns/iter (+/- 117)
test tests::bench_vec_dot_q8_neon_unrolled_single_sum ... bench:         953 ns/iter (+/- 38)
```

听 GPT4 的建议优化之后，比没 unroll 版反而还慢了！可见 AGI 仍任重道远。

不过仔细想来，其实变慢很 make sense。现代 CPU 的 out of order execution 机制下，会分析寄存器之间的数据依赖关系，如果两组指令之间不涉及任何数据依赖，则可以提供指令级并行（Instruction-Level Parallelism），这里的并行能力受 CPU 核心中闲置的计算单元和寄存器数量所限制。我们干掉了 `sumv1` 变量，两组计算的中间结果都累加到 `sumv0` 上，使得两组完全独立的计算产生了数据依赖，也就无法利用 out of order execution 带来的指令集并行性了，以至于性能退化到了非 loop unroll 版本上。

到这里可以显示出 loop unrolling 提升性能的真正原因在于暴露提高更高的指令集并行的潜力，而不仅仅是减少 branch prediction misses，在计算密集的循环里，branch prediction miss 反而是微乎其微的。

## 看汇编

难道这个性能差异就是来自 rust 的开销？只能看汇编来看看区别了。

GGML 版：

```
00000000000183bc <_ggml_vec_dot_q8_0_q8_0>:
   183bc: 6f00e401     	movi.2d	v1, #0000000000000000
   183c0: 6f00e400     	movi.2d	v0, #0000000000000000
   183c4: 7100801f     	cmp	w0, #0x20
   183c8: 540004cb     	b.lt	0x18460 <_ggml_vec_dot_q8_0_q8_0+0xa4>
   183cc: d2800008     	mov	x8, #0x0                ; =0
   183d0: 53057c09     	lsr	w9, w0, #5
   183d4: 910088aa     	add	x10, x5, #0x22
   183d8: 9100886b     	add	x11, x3, #0x22
   183dc: 6f00e400     	movi.2d	v0, #0000000000000000
   183e0: 6f00e401     	movi.2d	v1, #0000000000000000
   183e4: ad7f0d62     	ldp	q2, q3, [x11, #-0x20]
   183e8: 3cc02164     	ldur	q4, [x11, #0x2]
   183ec: 3cc12165     	ldur	q5, [x11, #0x12]
   183f0: ad7f1d46     	ldp	q6, q7, [x10, #-0x20]
   183f4: 3cc02150     	ldur	q16, [x10, #0x2]
   183f8: 3cc12151     	ldur	q17, [x10, #0x12]
   183fc: 6f00e412     	movi.2d	v18, #0000000000000000
   18400: 4e869452     	sdot
   18404: 4e879472     	sdot
   18408: 4e21da42     	scvtf.4s	v2, v18
   1840c: 7c5de163     	ldur	h3, [x11, #-0x22]
   18410: 1ee24063     	fcvt	s3, h3
   18414: 7c5de146     	ldur	h6, [x10, #-0x22]
   18418: 1ee240c6     	fcvt	s6, h6
   1841c: 1e260863     	fmul	s3, s3, s6
   18420: 4f831040     	fmla.4s	v0, v2, v3[0]
   18424: 6f00e402     	movi.2d	v2, #0000000000000000
   18428: 4e909482     	sdot
   1842c: 4e9194a2     	sdot
   18430: 4e21d842     	scvtf.4s	v2, v2
   18434: 7d400163     	ldr	h3, [x11]
   18438: 1ee24063     	fcvt	s3, h3
   1843c: 7d400144     	ldr	h4, [x10]
   18440: 1ee24084     	fcvt	s4, h4
   18444: 1e240863     	fmul	s3, s3, s4
   18448: 4f831041     	fmla.4s	v1, v2, v3[0]
   1844c: 91000908     	add	x8, x8, #0x2
   18450: 9101114a     	add	x10, x10, #0x44
   18454: 9101116b     	add	x11, x11, #0x44
   18458: eb09011f     	cmp	x8, x9
   1845c: 54fffc43     	b.lo	0x183e4 <_ggml_vec_dot_q8_0_q8_0+0x28>
   18460: 6e20d400     	faddp.4s	v0, v0, v0
   18464: 7e30d800     	faddp.2s	s0, v0
   18468: 6e21d421     	faddp.4s	v1, v1, v1
   1846c: 7e30d821     	faddp.2s	s1, v1
   18470: 1e212800     	fadd	s0, s0, s1
   18474: bd000020     	str	s0, [x1]
   18478: d65f03c0     	ret
```

rust 版：

```
0000000000000c28 <__ZN8bench_q824vec_dot_q8_neon_unrolled17h9f95656789587f05E>:
     c28: d3451408     	ubfx	x8, x0, #5, #1
     c2c: ab401908     	adds	x8, x8, x0, lsr #6
     c30: 54000560     	b.eq	0xcdc <__ZN8bench_q824vec_dot_q8_neon_unrolled17h9f95656789587f05E+0xb4>
     c34: 91008829     	add	x9, x1, #0x22
     c38: 9100886a     	add	x10, x3, #0x22
     c3c: 6f00e400     	movi.2d	v0, #0000000000000000
     c40: 6f00e401     	movi.2d	v1, #0000000000000000
     c44: ad7f0d22     	ldp	q2, q3, [x9, #-0x20]
     c48: 3cc02124     	ldur	q4, [x9, #0x2]
     c4c: 3cc12125     	ldur	q5, [x9, #0x12]
     c50: ad7f1d46     	ldp	q6, q7, [x10, #-0x20]
     c54: 3cc02150     	ldur	q16, [x10, #0x2]
     c58: 3cc12151     	ldur	q17, [x10, #0x12]
     c5c: 6f00e412     	movi.2d	v18, #0000000000000000
     c60: 4e869452     	sdot
     c64: 4e879472     	sdot
     c68: 4e21da42     	scvtf.4s	v2, v18
     c6c: 7c5de123     	ldur	h3, [x9, #-0x22]
     c70: 1ee24066     	fcvt	s6, h3
     c74: 7c5de143     	ldur	h3, [x10, #-0x22]
     c78: 1ee24067     	fcvt	s7, h3
     c7c: 1e2708c3     	fmul	s3, s6, s7
     c80: 4f839042     	fmul.4s	v2, v2, v3[0]
     c84: 4e22d400     	fadd.4s	v0, v0, v2
     c88: 6f00e402     	movi.2d	v2, #0000000000000000
     c8c: 4e909482     	sdot
     c90: 4e9194a2     	sdot
     c94: 4e21d842     	scvtf.4s	v2, v2
     c98: 7d400123     	ldr	h3, [x9]
     c9c: 1ee24064     	fcvt	s4, h3
     ca0: 7d400143     	ldr	h3, [x10]
     ca4: 1ee24065     	fcvt	s5, h3
     ca8: 1e250883     	fmul	s3, s4, s5
     cac: 4f839042     	fmul.4s	v2, v2, v3[0]
     cb0: 4e22d421     	fadd.4s	v1, v1, v2
     cb4: 91011129     	add	x9, x9, #0x44
     cb8: 9101114a     	add	x10, x10, #0x44
     cbc: f1000508     	subs	x8, x8, #0x1
     cc0: 54fffc21     	b.ne	0xc44 <__ZN8bench_q824vec_dot_q8_neon_unrolled17h9f95656789587f05E+0x1c>
     cc4: 6e20d400     	faddp.4s	v0, v0, v0
     cc8: 7e30d800     	faddp.2s	s0, v0
     ccc: 6e21d421     	faddp.4s	v1, v1, v1
     cd0: 7e30d821     	faddp.2s	s1, v1
     cd4: 1e212800     	fadd	s0, s0, s1
     cd8: d65f03c0     	ret
     cdc: 6f00e401     	movi.2d	v1, #0000000000000000
     ce0: 6f00e400     	movi.2d	v0, #0000000000000000
     ce4: 6e20d400     	faddp.4s	v0, v0, v0
     ce8: 7e30d800     	faddp.2s	s0, v0
     cec: 6e21d421     	faddp.4s	v1, v1, v1
     cf0: 7e30d821     	faddp.2s	s1, v1
     cf4: 1e212800     	fadd	s0, s0, s1
     cf8: d65f03c0     	ret
```

clang 和 Rust 两个编译器不愧同为 LLVM，前面装载寄存器的部分汇编指令基本上完全一致。

不过一行行看下来，在乘法这里有区别：

```
     c7c: 1e2708c3     	fmul	s3, s6, s7
     c80: 4f839042     	fmul.4s	v2, v2, v3[0] <-- here
     c84: 4e22d400     	fadd.4s	v0, v0, v2  <-- here
     c88: 6f00e402     	movi.2d	v2, #0000000000000000
     c8c: 4e909482     	sdot
     c90: 4e9194a2     	sdot
```

```
   1841c: 1e260863     	fmul	s3, s3, s6
   18420: 4f831040     	fmla.4s	v0, v2, v3[0] <-- here
   18424: 6f00e402     	movi.2d	v2, #0000000000000000
   18428: 4e909482     	sdot
   1842c: 4e9194a2     	sdot
   18430: 4e21d842     	scvtf.4s	v2, v2
```

rust 版本中在计算点积时，编译出来的是 `fmul.4s` 和 `fadd.4s` 两个指令，而 GGML 版中生成的是一条 `fmla.4s` 指令。

`fmla` 是 "Floating-point Fused Multiply-Add" 的缩写，能够将乘法和加法 fuse 在一起，能减少一个 SIMD 指令执行的开销。

## 优化 5: 使用 Fused Multiply Add 指令

上面的乘法+加法对应的是代码中的 vmlaq_n_f32 调用，似乎它并没有产生 FMA 指令。手工显式指定一下 `vfmaq_f32`：

```rust
#[cfg(target_arch = "aarch64")]
pub fn vec_dot_q8_neon_unrolled_vfma(n: usize, a: &[BlockQ8_0], b: &[BlockQ8_0]) -> f32 {
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


            sumv0 = aarch64::vfmaq_f32( // <- changed from vmlaq_n_f32
                sumv0,
                aarch64::vcvtq_f32_s32(aarch64::vaddq_s32(
                    aarch64::vdotq_s32(zerov, av00, bv00),
                    aarch64::vdotq_s32(zerov, av01, bv01),
                )),
                aarch64::vdupq_n_f32(f16::to_f32(ab0.d) * f16::to_f32(bb0.d)),
            );

            sumv1 = aarch64::vfmaq_f32( // <- changed from vmlaq_n_f32
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

```

再跑一把：

```
test tests::bench_vec_dot_q8_ggml                     ... bench:         704 ns/iter (+/- 19)
test tests::bench_vec_dot_q8_naive                    ... bench:      10,836 ns/iter (+/- 77)
test tests::bench_vec_dot_q8_stdsimd                  ... bench:       3,065 ns/iter (+/- 55)
test tests::bench_vec_dot_q8_neon                     ... bench:         937 ns/iter (+/- 5)
test tests::bench_vec_dot_q8_neon_unrolled            ... bench:         713 ns/iter (+/- 17)
test tests::bench_vec_dot_q8_neon_unrolled_single_sum ... bench:         936 ns/iter (+/- 70)
test tests::bench_vec_dot_q8_neon_unrolled_vfma       ... bench:         702 ns/iter (+/- 10)
```

甚至比 ggml 版稍微还快了点！


## 总结

到这里我们优化就告一段落了，从最初的 naive 版到 neon_unrolled_vfma 版，已快了 15 倍。

整理一下收获最大的地方有：

- rust 标准库中跨平台的 stdsimd 不是很值得投入，最终面向平台的细节，还是需要特定平台的 SIMD 指令来发挥性能潜力
- 没有数据依赖的 loop unrolling 是发挥指令集并行的要义，解除数据依赖的作用远大于 branch prediction 友好着一层


本文的所有代码和 benchmark 脚手架可见于 https://github.com/flaneur2020/flaneur2020.github.io/tree/master/exercises/rust/bench-q8



