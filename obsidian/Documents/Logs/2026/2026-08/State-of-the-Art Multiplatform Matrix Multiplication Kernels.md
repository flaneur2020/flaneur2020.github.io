## Four Levels of Abstractions

- Tile Matmul: 对应硬件：单个 thread block 内的寄存器和 cuda core 进行计算；每个 tile 大约 16x16 这样；
- Stage Matmul：读经 shared memory 和线程块内的 barrier；负责数据搬运，然后使用 tile matmul 进行执行；
- Global Matmul：对应 global memory 和多个 thread block 的协作；处理长 K 维度；K 维度可能非常大（比如 4096），Global Matmul 负责循环，将 K 维度切成很多段；每段交给一个 stage matmul 处理，然后将结果累加起来；决定了数据从全局内存加载的粒度；
- Batch Matmul：对应 GPU 的 grid 和 SM 的调度，处理 batch 维度，将这些 batch 分到不同的线程块上；

从上往下看：

- Batch 决定“哪个 SM 做哪块”
- Global 决定“K 维怎么切分” （如果 K 很大，那么一次将所有数据放到共享内存是不可能的）
- Stage 决定 Global Memory 到 Shared Memory 的搬运
- Tile 决定“寄存器怎么算”

![[Pasted image 20260804224719.png]]

## Double Buffering

```rust
// Stage Matmul 内部的双缓冲
struct PartitionMatmul {
    lhs_tiles: [LhsTile; pm],      // Lhs 一次性加载到寄存器
    rhs_buffer: [RhsTile; 2],      // Rhs 使用双缓冲
    accumulators: [Accumulator; pm * pn],
}

impl PartitionMatmul {
    fn execute(&mut self) {
        // 预加载第一个 Rhs 瓦片
        load_rhs(&mut self.rhs_buffer[0], chunk_0);
        
        for k in 0..pk {
            let current = k % 2;
            let next = (k + 1) % 2;
            
            // 预加载下一个 Rhs 瓦片（异步）
            if k + 1 < pk {
                load_rhs(&mut self.rhs_buffer[next], chunk_k+1);
            }
            
            // 使用当前 Rhs 瓦片计算
            for m in 0..pm {
                for n in 0..pn {
                    execute_tile_matmul(
                        &self.lhs_tiles[m],
                        &self.rhs_buffer[current],
                        &mut self.accumulators[m * pn + n]
                    );
                }
            }
        }
    }
}

```



在做当前 tile 的计算之前，先 load 一下 rhs 的下一个 tile。

似乎可以依赖编译器和运行时，解开这个依赖 load_rhs 的依赖，自动先跑计算，同时 load 下一个 tile。

double buffering 在 stage matmul 和 global matmul 里都存在。

stage matmul 这里是 共享内存 到 寄存器 的延迟。

global matmul 这里是按 k 维度拆分，隐藏的是 global memory 到 shared memory 的延迟；

```
理想情况下的时间线：

周期: 0        400       800       1200      1600
     |---------|---------|---------|---------|
     
加载: [Chunk0加载] [Chunk1加载] [Chunk2加载] [Chunk3加载]
计算:             [Chunk0计算] [Chunk1计算] [Chunk2计算]

重叠:           |←--重叠--→|←--重叠--→|←--重叠--→|
```