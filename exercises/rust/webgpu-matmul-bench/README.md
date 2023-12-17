make a benchmark on matmul shaders.

todo:

- [x] revise the segmem interface to allow reuse the buffer
- [x] add flops estimation
- [ ] add my shader into wgpu-mm and view the benchmark result
- [ ] add a shader with reduce sum pattern: https://gist.github.com/flaneur2020/eebbb23de1ddac69e8a2183a1b0b4f1c


reference:

- https://jott.live/markdown/webgpu_safari
- https://jott.live/code/webgpu_mm.js
