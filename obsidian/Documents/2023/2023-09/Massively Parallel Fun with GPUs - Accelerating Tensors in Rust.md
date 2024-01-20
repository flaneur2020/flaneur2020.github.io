https://getcode.substack.com/p/massively-parallel-fun-with-gpus

一个 exp 的 shader 作为例子：

```wgpu
@group(0) @binding(0)
var<storage, read> input_0: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_0: array<f32>;

@compute
@workgroup_size(64)
fn call(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x;
    output_0[gidx] = exp(input_0[gidx]);
}
```


- shader 约等于 CUDA 中 kernel 的概念、workgroup 相当于 Block 的概念；
- binding 类似 protobuf 的字段标号，外部调用这个 shader 时，可以将 buffer 和字段标号绑定；
- global_id 这个 `vec3<u32>` 相当于 CUDA 中的 threadIdx、blockDim、blockIdx，可以传递一个三元组来表示当前 thread 关心的坐标；

一个完整的流程下来：

> Once you have your shader code, executing the shader proceeds as follows:
> 
> 1. Create any necessary buffers, and copy data to them via functions on `wgpu::Device`.
> 2. Create the `ComputePipeline` for the shader.
> 3. Bind buffers from step 1 to corresponding variables defined in the shader via bind groups. Bind groups are created via `wgpu::Device`.
> 4. Dispatch the shader with a given number of workgroups. This step [is somewhat tedious](https://github.com/kurtschelfthout/tensorken/blob/v0.2/src/raw_tensor_wgpu.rs#L186), and needs a few intermediate objects like a "command encoder" and a "compute pass". The gist is you submit a list of commands to the `Queue`, and get a submission index back.
> 5. Poll the device using this submission index, to learn when execution finishes.

- 向 WGPU 提交任务是通过一个队列，然后可以等待它完成；