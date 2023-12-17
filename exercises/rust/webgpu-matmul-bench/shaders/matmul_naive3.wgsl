struct Meta {
    M: u32,
    N: u32,
    K: u32,
}

@group(0) @binding(0)
var<storage, read> input_a: array<f32>;

@group(0) @binding(1)
var<storage, read> input_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> input_c: array<f32>;

@group(0) @binding(3)
var<uniform> input_m: Meta;

// dispatch workgroup size: (M / 16, K / 16, 1)
// performance: ? flops/s on M1

@compute
@workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {

    // a: (m, n)
    // b: (n, k)
    // c: (m, k)
    let M = input_m.M;
    let K = input_m.K;
    let N = input_m.N;

    let m_val = global_id.x;
    let k_val = global_id.y;
    var sum = 0.0f;
    for (var n_val = 0u; n_val < N; n_val = n_val + 1u) {
        sum += input_a[m_val * N + n_val] * input_b[n_val * K + k_val];
    }
    input_c[m_val * K + k_val] = sum;
}
