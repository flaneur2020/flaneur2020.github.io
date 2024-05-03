## Motivation

> the core framework that is making it possible to build hyper-optimized, cross platform and scalable GPU algorithms — and that is the Vulkan Framework.

vulkan 是完全开源的，与 opengl 这种老的框架不同，vulkan 在设计时就将现代的 GPU 架构考虑在内。

vulkan SDK 提供了特别底层的 GPU 访问能力。优势是能力强，但是劣势就是啰嗦。

写一段程序需要 500～2000+ 行的脚手架代码。

pytorch、tensorflow 等框架在对接 vulkan 时，都是大量相似的胶水代码。
## Enter Kompute

kcompute 出于将 vulkan 变得易用。

kcompute 在设计上并不想隐藏任何 vulkan 的概念。但是更加易于上手。

## Writing your first Kompute

![[Pasted image 20240503101040.png]]

Kcompute tensor 对应一份 GPU 中的数据，operation 对应一个操作，记录在 sequence 中。

操作的流程：

1. Create a Kompute Manager to manage resources
2. Create Kompute Tensors to hold data
3. Initialise the Kompute Tensors in the GPU with a Kompute Operation
4. Define the code to run on the GPU as a “compute shader”
5. Use Kompute Operation to run shader against Kompute Tensors
6. Use Kompute Operation to map GPU output data into local Tensors
7. Print your results