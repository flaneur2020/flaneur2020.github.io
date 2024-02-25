#gemm 

GotoBlas 的论文。

![[Screenshot 2024-02-24 at 10.57.59.png]]

通用的 GEMM 可以看做是上述特化子操作组合而来。

- M：Matrix、Both dimensions are large or unknown.
- P：Panel，One of the demension is small.
- B: Block, Both demensions are small.

GEBP 并不是 GEMV，GEMV 中 m 是 1，这里似乎是一个比较小的 block。

![[Screenshot 2024-02-24 at 11.02.40.png]]

关于 GEBP，有这些 Assumptions：

1. $m_c$ 和 $k_c$ 这些维度足够小，完整的A 矩阵，和 B、C 矩阵的 $n_r$ 列（也就是 $B_j$ 和 $C_j$），可以一起放到 cache 中；
2. 如果 $A$、$C_j$、$B_j$ 都在 cache 中，那么 $C_j := AB_j + C_j$ 可以最快地计算出来；
3. 如果 $A$ 在 cache 中，那么它可以一直位于 cache 直到不再需要它为止，不必重复 load 内存；

基于上面的假设， GEBP 的开销可以分摊为从主存到 cache 的数据搬运量。

更新完整的 $C$ 的成本包括：

- $m_c k_c + (2m_c + k_c)n$ 次 memops
- $2m_c k_c n$ 次 flops

优化的算法会想办法最大化 flops 计算与 memops 的比例：

$$
\frac{2m_c k_c n }  { m_c k_c + (2m_c + k_c)n} \frac{flops}{memops}
\quad \text{where} \space k_c << n
$$

也就是最大化：

$$ \frac{2m_ck_c}{2m_c + k_c} $$
因此，合理安排 $m_c$ 和 $k_c$ 作为分块的大小很关键。

此外，$A$ 矩阵应当尽可能地有多少 cache 用多少 cache，并且应当接近于正方形。并且留出来给至少给 $B_j$ 和 $Cj$ 的 cache。

## 4.2 Refinement

问题来了，要选择 $m_c \times k_c$ 的 A 矩阵放匹配哪一层 cache 的大小？

上面的公式显示，越大的 $m_c \times k_c$，计算的密度越高。

L1 cache 往往比较小，可不可以将 $A$ 放到 L2 cache 中，让 $m_c$ 和 $k_c$ 能足够大？

假设 $A$ 在 L2 cache， $Bj$ 和 $C_j$ 可以放到 L1 cache 中。

计算 $A B_j + C_j$  需要 $2m_c k_c n_r$ 次 FLOPS 操作，$m_c k_c$ 次 A 元素从 L2 cache load 到寄存器。

设 CPU 做浮点运算的速率为 $R_{comp}$，设从 L2 load 数据到寄存器的速率为 $R_{load}$。

要让算力跑满，需要做到：

$$
 \frac{2 m_c k_c n_r}{m_c k_c} >= \frac{R_{comp}}{R_{load}}
$$

也就是使 $n_r >= \frac{R_{comp}}{2 R_{load}}$。

一般 CPU 的算力确实大于内存的吞吐几个量级，不过这里是 L2 Cache 的吞吐。

### 4.2.2 TLB

TLB miss 和 cache miss 的区别：cache miss 不会 stall 整个 CPU。


因此需要追加假设：

1. $m_c$ 和 $k_c$ 这些维度足够小，完整的A 矩阵，和 B、C 矩阵的 $n_r$ 列（也就是 $B_j$ 和 $C_j$），可以同时被 TLB 进行寻址，因此在执行计算期间，不会发生 TLB miss；
2. 如果 A 可被 TLB 进行寻址，则 A 的寻址会一直处于 TLB 之中。

### 4.2.3 Packing

A 是一个更大的矩阵的 sub matrix，它本身并不连续，这意味着对它寻址，需要涉及多个 TLB entry。

解决办法就是将 A pack 到一个连续的 work array 中，也就是 $\tilde{A}$。




