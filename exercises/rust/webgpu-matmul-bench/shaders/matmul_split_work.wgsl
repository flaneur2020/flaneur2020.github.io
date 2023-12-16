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

// this implementation tries to split the work with every 128 threads
// - every 128 threads handles one column of input_b
// - each thread handles N / 128 rows of input_b
// - each workgroup handles N columns of input_b
// - dispatched M * K (M * N * K / N) times
//
// performance: 9.6G flops/s on M1

var<workgroup> sketch: array<f32, 128>;

@compute
@workgroup_size(128)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {

    // a: (m, n)
    // b: (n, k)
    // c: (m, k)

    let m_val = workgroup_id.x;
    let k_val = workgroup_id.y;
    let chunk_size = input_m.N / 128u;
    for (var i = 0u; i < chunk_size; i = i + 1u) {
        let n_val = local_id.x * chunk_size + i;
        sketch[local_id.x] += input_a[m_val * input_m.N + n_val] * input_b[n_val * input_m.K + k_val];
    }
    workgroupBarrier();

    if (local_id.x == 0u) {
        var sum = 0.0f;
        for (var i = 0u; i < 128u; i = i + 1u) {
            sum += sketch[i];
        }
        input_c[m_val * input_m.K + k_val] = sum;
    }
}
