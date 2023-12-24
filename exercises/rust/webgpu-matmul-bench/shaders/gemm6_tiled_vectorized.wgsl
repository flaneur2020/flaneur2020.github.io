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

var<workgroup> tA: array<vec4<f32>, 64>;
var<workgroup> tB: array<vec4<f32>, 64>;

const TILE_N = 16u;

// dispatch workgroup size: (M / 16, N / 16, 1)
// make a 16x16 tile
// each workgroup handles 16x16 elements of C
// not very fast, 422.81 gflops/s
// even slower 200+ gflops/s when setting the tile size as 32x32

@compute
@workgroup_size(16, 4, 1)
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
    let n_idx = global_id.y * 4u;
    let tile_row = local_id.x;
    let tile_col = local_id.y * 4u;
    var tmp = vec4(0.0);

    for (var tile_offset = 0u; tile_offset < K; tile_offset += TILE_N) {
        // each thread loads one element from A and B into tA and tB
        tA[(tile_row * TILE_N + tile_col) / 4u] = A[(m_idx * K + tile_offset + tile_col) / 4u];
        tB[(tile_row * TILE_N + tile_col) / 4u] = B[((tile_offset + tile_row) * N + n_idx) / 4u];
        workgroupBarrier();

        for (var dot_idx = 0u; dot_idx < TILE_N; dot_idx += 4u) {
            let brow0 = tB[(dot_idx * TILE_N + tile_col) / 4u];
            let brow1 = tB[((dot_idx + 1u) * TILE_N + tile_col) / 4u];
            let brow2 = tB[((dot_idx + 2u) * TILE_N + tile_col) / 4u];
            let brow3 = tB[((dot_idx + 3u) * TILE_N + tile_col) / 4u];
            let arow = tA[(tile_row * TILE_N + dot_idx) / 4u];
            tmp += vec4(arow.x) * brow0;
            tmp += vec4(arow.y) * brow1;
            tmp += vec4(arow.z) * brow2;
            tmp += vec4(arow.w) * brow3;
        }
        workgroupBarrier();
    }

    C[(m_idx * N + n_idx) / 4u] = tmp;
}
