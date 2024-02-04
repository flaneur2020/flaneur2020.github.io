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

// this implementation tries to split the work with every 128 threads
// - every 128 threads handles one column of B
// - each thread handles N / 128 rows of B
// - each workgroup handles N columns of B
// - dispatched workgroups: (M, K)
//
// performance: 12G flops/s on M1

var<workgroup> sketch: array<f32, 128>;

@compute
@workgroup_size(128)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {

    // a: (m, k)
    // b: (k, n)
    // c: (m, n)
    let M = _m.M;
    let K = _m.K;
    let N = _m.N;

    let m_val = workgroup_id.x;
    let n_val = workgroup_id.y;
    let chunk_size = K / 128u;
    for (var i = 0u; i < chunk_size; i = i + 1u) {
        let k_val = local_id.x * chunk_size + i;
        sketch[local_id.x] += A[m_val * K + k_val] * B[k_val * N + n_val];
    }
    workgroupBarrier();

    if (local_id.x == 0u) {
        var sum = 0.0f;
        for (var i = 0u; i < 128u; i = i + 1u) {
            sum += sketch[i];
        }
        C[m_val * N + n_val] = sum;
    }
}
