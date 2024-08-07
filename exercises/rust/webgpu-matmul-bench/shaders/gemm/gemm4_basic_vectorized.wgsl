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
// performance: 661.10 Gflops/s on M1

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
    let K = info.K;
    let N = info.N;

    let mi = global_id.x;
    let ni = global_id.y * 4u;

    var tmp = vec4<f32>(0.0);
    for (var ki = 0u; ki < K; ki += 4u) {
        let bi = (ki * N + ni) / 4u;
        let bv0 = B[bi + 0u];
        let bv1 = B[bi + N / 4u];
        let bv2 = B[bi + N / 4u * 2u];
        let bv3 = B[bi + N / 4u * 3u];
        let av = A[(mi * N + ki) / 4u];
        tmp = fma(vec4(av.x), bv0, tmp);
        tmp = fma(vec4(av.y), bv1, tmp);
        tmp = fma(vec4(av.z), bv2, tmp);
        tmp = fma(vec4(av.w), bv3, tmp);
    }
    let c_idx = (mi * N + ni) / 4u;
    C[c_idx] = tmp;
}
