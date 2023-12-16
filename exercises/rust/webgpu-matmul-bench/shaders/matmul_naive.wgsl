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

// matmul_naive.wgsl parallelizes over the first dimension of the output matrix,
// - each thread handles one row of the output matrix.
// - splitted m/64 workgroups
// its performance is poor, only 2G flops.

@compute
@workgroup_size(64)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let mi = 64u * workgroup_id.x + local_id.x;
    if mi >= input_m.M {
        return;
    }    

    // a: (m, n)
    // b: (n, k)
    // c: (m, k)

    for (var ki = 0u; ki < input_m.K; ki = ki + 1u) {
        var sum = 0.0f;
        for (var ni = 0u; ni < input_m.N; ni = ni + 1u) {
            let a = input_a[mi * input_m.N + ni];
            let b = input_b[ni * input_m.K + ki];
            sum += a * b;
        }
        input_c[mi * input_m.K + ki] = sum;
    }
}
