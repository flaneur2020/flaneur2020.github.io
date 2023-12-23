struct Info {
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> B: array<f32>;

@group(0) @binding(2)
var<storage, read_write> C: array<f32>;

@group(0) @binding(3)
var<uniform> info: Info;

var<workgroup> tA: array<f32, 256>;
var<workgroup> tB: array<f32, 256>;

const TILE_N = 16u;

// dispatch workgroup size: (M / 16, N / 16, 1)
// make a 16x16 tile
// each workgroup handles 16x16 elements of C
// not that fast, only 106.16GFLOPS

@compute
@workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    // a: (m, k)
    // b: (k, n)
    // c: (m, n)
    let M = info.M;
    let K = info.K;
    let N = info.N;

    let m_idx = global_id.x;
    let n_idx = global_id.y;
    let tile_row = local_id.x;
    let tile_col = local_id.y;
    var tmp = 0.0f;

    for (var tn = 0u; tn < K / TILE_N; tn += 1u) {
        // each thread loads one element from A and B into tA and tB
        tA[tile_row * TILE_N + tile_col] = A[m_idx * K + tn * TILE_N + tile_col];
        tB[tile_row * TILE_N + tile_col] = B[(tn * TILE_N + tile_row) * N + n_idx];
        workgroupBarrier();

        for (var dot_idx = 0u; dot_idx < TILE_N; dot_idx += 1u) {
            tmp += tA[tile_row * TILE_N + dot_idx] * tB[dot_idx * TILE_N + tile_col];
        }
        workgroupBarrier();
    }

    C[m_idx * N + n_idx] = tmp;
}
