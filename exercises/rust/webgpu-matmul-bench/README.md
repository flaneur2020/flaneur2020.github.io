make a benchmark on matmul shaders.

todo:

- [x] revise the segmem interface to allow reuse the buffer
- [x] add flops estimation
- [x] add my shader into wgpu-mm and view the benchmark result
- [x] compare the correctness with a vanilla matmul
- [x] vectorize the tiled gemm
- [ ] why wgpu-mm's sgemm_3 is faster?  learn wgpu-mm's tfj shader, which the fastest (could reach 900 gflops/s)
- [ ] gemv shaders

reference:

- https://jott.live/markdown/webgpu_safari
- https://jott.live/code/webgpu_mm.js
