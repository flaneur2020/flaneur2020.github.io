@group(0) @binding(0)
var<storage, read_write> input: array<f32>;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x;
    input[gidx] = exp(input[gidx]);
}
