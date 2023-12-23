struct Info {
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

@group(0) @binding(3)
var<uniform> info: Info;

// dispatch workgroup size: (M / 8, N / 32, 1)
// - vectorized processing: take 4 columns of B at a time
// performance: 447.66G Gflops/s on M1

@compute
@workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {

    // a: (m, k)
    // b: (k, n)
    // c: (m, n)
    let M = info.M;
    let N = info.N;
    let K = info.K;


    let m_idx = global_id.x;
    let n_idx = global_id.y * 4u;

    var tmp = vec4<f32>(0.0);
    for (var k_idx = 0u; k_idx < K; k_idx += 4u) {
        let b_idx = (k_idx * N + n_idx) / 4u;
        let bv0 = B[b_idx];
        let bv1 = B[b_idx + 1u];
        let bv2 = B[b_idx + 2u];
        let bv3 = B[b_idx + 3u];
        let av = A[(m_idx * N + k_idx) / 4u];
        tmp = fma(vec4(av.x), bv0, tmp);
        tmp = fma(vec4(av.y), bv1, tmp);
        tmp = fma(vec4(av.z), bv2, tmp);
        tmp = fma(vec4(av.w), bv3, tmp);
    }
    let c_idx = (m_idx * N + n_idx) / 4u;
    C[c_idx] = tmp;
}
