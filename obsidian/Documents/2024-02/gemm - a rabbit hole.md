https://bluss.github.io/rust/2016/03/28/a-gemmed-rabbit-hole/

文章作者是 [matrixmultiply](https://crates.io/crates/matrixmultiply) 和 ndarray 的作者。

BLIS 的思路是做一个 4xk 的 input column，计算出来一个 4x4 的结果 matrix。

## A 4-by-4 microkernel for f32

```rust
/// index ptr with index i
unsafe fn at(ptr: *const f32, i: usize) -> f32 {
    *ptr.offset(i as isize)
}

/// 4x4 matrix multiplication kernel for f32
///
/// This does the matrix multiplication:
///
/// C ← α A B + β C
///
/// + k: length of data in a, b
/// + a, b are packed
/// + c has general strides
/// + rsc: row stride of c
/// + csc: col stride of c
/// + if `beta` is 0, then c does not need to be initialized
#[inline(always)]
pub unsafe fn kernel_4x4(k: usize, alpha: f32, a: *const f32, b: *const f32,
                         beta: f32, c: *mut f32, rsc: isize, csc: isize)
{
    let mut ab = [[0.; 4]; 4];
    let mut a = a;
    let mut b = b;

    // Compute matrix multiplication into ab[i][j]
    unroll_by_8!(k, {
        let v0 = [at(a, 0), at(a, 1), at(a, 2), at(a, 3)];
        let v1 = [at(b, 0), at(b, 1), at(b, 2), at(b, 3)];
        loop4x4!(i, j, ab[i][j] += v0[i] * v1[j]);

        a = a.offset(4);
        b = b.offset(4);
    });


    macro_rules! c {
        ($i:expr, $j:expr) => (c.offset(rsc * $i as isize + csc * $j as isize));
    }

    // Compute C = alpha A B + beta C,
    // except we can not read C if beta is zero.
    if beta == 0. {
        loop4x4!(i, j, *c![i, j] = alpha * ab[i][j]);
    } else {
        loop4x4!(i, j, *c![i, j] = *c![i, j] * beta + alpha * ab[i][j]);
    }
}
```


有了这个 micro kernel 之后，只需要 5 个循环：

https://github.com/bluss/matrixmultiply/blob/4e48dffa840f005babeebd04be676e2faa258b73/src/gemm.rs#L78-L176

