https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html

If we define a workgroup as say `@workgroup_size(3, 4, 2)` then we’re defining 3 * 4 * 2 threads or another way put it, we’re defining a 24 thread workgroup.

If we then call `pass.dispatchWorkgroups(4, 3, 2)` we’re saying, execute a workgroup of 24 threads, 4 * 3 * 2 times (24) for a total of 576 threads.

![[Pasted image 20231111220924.png]]
其中紫色的部分是 local_invocation_id 来标识的，蓝色部分是 workgroup_id。

workgroup 的设定好像相当于 CUDA 的 block，block 之间的执行顺序是不可控的，只有在 workgroup 内部可以做线程协调。pip

在每次调用 shader 时，有这几个 builtin variable：

- `local_invocation_id`：workgroup 内部的 thread 的 id，是一个 (x, y, z) 的三元组；
- `workgroup_id`：表示 workgroup 的 id，也是一个 (x, y, z) 三元组；
- `global_invocation_id`：对应每个 thread 的 unique 的 id；
	- global_invocation_id = workgroup_id * workgroup_size + local_invocation_id

`num_workgroups` 是我们在 `pass.dispatchWorkgroups` 中传递的参数。

## Workgroup Size

应该怎么选择 workgroup 的 size？

为什么不直接用 `@workgroup_size(1, 1, 1)`？那么直接通过 `pass.dispatchWorkgroups` 来确定迭代数量就可以了。

在 workgroup 中执行多个线程，要在性能上优于多次独立的 dispatch。

WebGPU 有一些约束，`workgroup_size` 需要避免超过 512，一般建议默认为 64。