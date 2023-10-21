https://github.com/flame/how-to-optimize-gemm/wiki
https://stackoverflow.com/a/54546544
https://slideplayer.com/slide/13539785/
https://slideplayer.com/slide/13539785/
https://github.com/pigirons/sgemm_hsw
https://www.adityaagrawal.net/blog/architecture/matrix_multiplication

GEMM 特指 $C = A \times B + C$ 这种矩阵相乘和连加。


![[Screenshot 2023-10-12 at 11.13.39.png]]

这个是 Goto GEMM 算法。图里值得一提的是颜色，不同的部分放置在哪个层级的 cache 中。能够针对 L2 Cache Miss 做优化。

最重要的是蓝色的这个分片一定要放在 L1 Cache 下，并按照水平的顺序读内存。反观左侧的矩阵可以在分块中按垂直的方向读内存。

最粗糙的 vec_dot 的问题是因为右侧的向量大于 L1 Cache，会在矩阵相乘中反复读取右侧的向量。

最后的 micro kernel 中应当使用 SIMD 进行加速，最好使用 fused mul add。



