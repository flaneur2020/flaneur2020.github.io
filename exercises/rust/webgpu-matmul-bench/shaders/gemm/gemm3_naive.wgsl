struct Meta {
    M: u32,
    N: u32,
    K: u32,
}

@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> B: array<f32>;

@group(0) @binding(2)
var<storage, read_write> C: array<f32>;

@group(0) @binding(3)
var<uniform> _m: Meta;

// dispatch workgroup size: (M / 16, N / 16, 1)
// each workgroup computes a 16x16 block of the output matrix
// performance: 41 Gflops/s on M1

@compute
@workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {

    // a: (m, k)
    // b: (k, n)
    // c: (m, n)
    let M = _m.M;
    let K = _m.K;
    let N = _m.N;

    let mi = global_id.x;
    let ni = global_id.y;
    var sum = 0.0f;
    for (var ki = 0u; ki < K; ki = ki + 1u) {
        sum += A[mi * K + ki] * B[ki * N + ni];
    }
    C[mi * N + ni] = sum;
}
