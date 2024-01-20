https://surma.dev/things/webgpu/

Currently, WebGPU allows you to create two types of pipelines: A Render Pipeline and a Compute Pipeline

I like to think of Render Pipelines as a specialization/optimization of Compute Pipelines.

considerably understates that these pipelines are physically different circuits in your GPU

With WebGPU, a pipeline consists of one (or more) programmable stages, where each stage is defined by a shader and an entry point. A Compute Pipline has a single `compute` stage, while a Render Pipeline would have a `vertex` and a `fragment` stage.

在 intel 的术语中，最底层是「Execution Unit」，里面有多个「SIMD core」。这意味着它有 7 个 core 在每步（lockstep），总是执行同样的指令。这也是为什么 GPU 性能专家总是避免 if 语句，如果 EU 遇到 if/else，所有的 core 都必须执行两个分支的指令，除非所有的 core 都碰巧走在同一个分支上。对 loop 也是同样，如果一个 core 较早地退出了 loop，它也需要同样等待其他核心执行完毕。

访问内存的时间较长，需要几百个时钟周期。为了利用等待的时间，每个 EU 都尽量地多接工作（oversubscribed with work）。每当一个 EU 进入 idle，它会切换另一个 work item。GPU 会尽量地使 work queue 足够繁忙，来让 EU 总是处于工作中。

![[Pasted image 20231111232507.png]]

多个 EU 会组成一个 SubSlice，每个 SubSlice 能够共享一个小的 Shared Local Memory（64Kb）。

## Workgroups

在传统上，vertex shader 会在每个 vertex 上执行一次，而 fragment shader 会在每个像素上执行一次。

一个 workgroup 中的任务是一起调度的。

So what _is_ the right workgroup size? It really depends on the semantics you assign the work item coordinates. I do realize that this is not really an helpful answer, so I want to give you the same advice that [Corentin](https://twitter.com/dakangz) gave me: “Use [a workgroup size of] 64 unless you know what GPU you are targeting or that your workload needs something different.” It seems to be a safe number that performs well across many GPUs and allows the GPU scheduler to keep as many EUs as possible busy.

## Exchanging data

The `binding` number can be freely chosen and is used to tie a variable in our WGSL code to the contents of the buffer in this slot of the bind group layout. Our `bindGroupLayout` also defines the purpose for each buffer, which in this case is `"storage"`. Another option is `"read-only-storage"`, which is read-only (duh!), and allows the GPU to make further optimizations on the basis that this buffer will never be written to and as such doesn’t need to be synchronized. The last possible value for the buffer type is `"uniform"`, which in the context of a compute pipeline is mostly functionally equivalent to a storage buffer.

### Staging Buffers

<mark>Instead, GPUs have additional memory banks that are accessible to both the host machine as well as the GPU, but are not as tightly integrated and can’t provide data as fast. </mark>Staging buffers are buffers that are allocated in this intermediate memory realm and can be [mapped](https://en.wikipedia.org/wiki/Memory-mapped_I/O) to the host system for reading and writing. <mark>To read data from the GPU, we copy data from an internal, high-performance buffer to a staging buffer</mark>, and then map the staging buffer to the host machine so we can read the data back into main memory. For writing, the process is the same but in reverse.