struct Meta {
    M: u32,
    N: u32,
    K: u32,
}

@group(0) @binding(0)
var<storage, read> input_a: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> input_b: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> input_c: array<vec4<f32>>;

@group(0) @binding(3)
var<uniform> input_m: Meta;

// dispatch workgroup size: (M / 32, K / 32, 1)
// each workgroup computes a 16x16 block of the output matrix
// vectorized by every 4 elements in a row of C
// performance: 347.4 Gflops/s on M1

@compute
@workgroup_size(8, 8, 1)
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

    let m_idx = global_id.x * 4u;
    let k_idx = global_id.y * 4u;

    var sum0 = vec4<f32>(0.0);
    var sum1 = vec4<f32>(0.0);
    var sum2 = vec4<f32>(0.0);
    var sum3 = vec4<f32>(0.0);
    for (var n_idx = 0u; n_idx < N; n_idx += 1u) {
        let bv = input_b[n_idx * K + k_idx];
        let av = input_a[(m_idx + 0u) * N + n_idx];
        sum0 += av.x * bv;
        sum1 += av.y * bv;
        sum2 += av.z * bv;
        sum3 += av.w * bv;
    }
    input_c[(m_idx + 0u) * K + k_idx] = sum0;
    input_c[(m_idx + 1u) * K + k_idx] = sum1;
    input_c[(m_idx + 2u) * K + k_idx] = sum2;
    input_c[(m_idx + 3u) * K + k_idx] = sum3;
}
