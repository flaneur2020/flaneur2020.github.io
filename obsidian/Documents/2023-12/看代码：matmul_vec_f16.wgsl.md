
## References

https://github.com/cryscan/web-rwkv/blob/main/src/shaders/matmul_vec_fp16.wgsl

## 前置

定义了一个 View 结构体来存元信息：

```
struct View {
    stride: vec4<u32>,
    offset: vec4<u32>,
    shape: vec4<u32>,  
};
```

还有这种写法，就是将 workgroup id 这些定义在一个 Input 结构中，然后 `fn matmul(in: Input)` 这样注入：

```

struct Input {
    @builtin(workgroup_id) bid: vec3<u32>,
    @builtin(global_invocation_id) uid: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(local_invocation_index) index: u32,
};
```

定义了三个参数，都是 uniform 类型，感觉 `[B, K, M]` 这种可能舒服点。

```wgsl
@group(0) @binding(0) var<uniform> va: View;                                // [K, M, B]
@group(0) @binding(1) var<uniform> vb: View;                                // [K, N, B]
@group(0) @binding(2) var<uniform> destination: View;                       // [M, N, B]
```

为什么定义成 `uniform`？

A uniform is a blob of data available to every invocation of a set of shaders

具体的数据是通过 `array<vec2<u32>>` 表示的：

```
@group(0) @binding(3) var<storage, read> matrix: array<vec2<u32>>;          // (B, R, C)
@group(0) @binding(4) var<storage, read> input: array<vec4<f32>>;           // (B, T, C)
@group(0) @binding(5) var<storage, read_write> output: array<vec4<f32>>;    // (B, T, R)

```

随后可以定义一个 `unpack4x16float` 把它解出来：

```
fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}
```

有一个 sketch 变量用于保存中间结果：
`
```
const BLOCK_SIZE: u32 = 128u;

var<workgroup> sketch: array<vec4<f32>, BLOCK_SIZE>;
```

## MatMul 本体

block size 是 128。

每次迭代：

- for (var i = index; i < stride; i += BLOCK_SIZE)，其中 BLOCK_SIZE = 128
- 从 matrix 中拿 4x4 的 sub-block
- 从 vector 中拿 4 个元素

read 4 rows from the matrix, each with 4 unpacked floats, forming a 4x4 sub-block


```
    workgroupBarrier();

    reduce_sum(index, 64u);
    reduce_sum(index, 32u);
    reduce_sum(index, 16u);
    reduce_sum(index, 8u);
    reduce_sum(index, 4u);
    reduce_sum(index, 2u);
    reduce_sum(index, 1u);

```

```
fn reduce_sum(index: u32, stride: u32) {
    if index < stride {
        sketch[index] += sketch[index + stride];
    }
    workgroupBarrier();
}
```