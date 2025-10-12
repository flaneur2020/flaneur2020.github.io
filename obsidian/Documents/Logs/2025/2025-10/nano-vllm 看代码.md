

## 问题

- 每次 Decode 是怎样的流程？
- 怎样处理多个 request 的？
- 怎样做抢占的？

### block_table 是什么？

每个 sequence 对应一个 `block_table`。在 `ModelRunner#prepare_block_tables` 中初始化的。

不同的 sequence 的 block_table 的长度不同。比较长的 sequence，占有的 block 数量更多。

`block_table` 扮演了操作系统的页表的作用。

### block_table 怎样解决 prefix reuse？

一个 slot 对应一个 token 的位置。

不同的 sequence 可能会有共享的前缀，因此，不同的 sequence 可能引用到同一组 blocks。

```
序列 A: "The quick brown fox jumps over the lazy dog"
序列 B: "The quick brown fox runs very fast today"

共享的完整block: ["The", "quick", "brown", "fox"]

物理 KV Cache:
Block 0: ["The", "quick", "brown", "fox"]     ← 被序列A和B共享
Block 1: ["jumps", "over", "the", "lazy"]     ← 序列A独有
Block 2: ["dog", -, -, -]                     ← 序列A独有  
Block 3: ["runs", "very", "fast", "today"]    ← 序列B独有

序列 A: block_table = [0, 1, 2]  # 引用: Block 0=2, Block 1=1, Block 2=1
序列 B: block_table = [0, 3]     # 引用: Block 0=2, Block 3=1
```

只有一个 block 完全相同时才会重用。

### slot mapping 是怎样的形状？

slot 可以理解为 kvcache 下一个输出的位置。

Prefill 和 Decode 阶段，slot_mapping 的形状不同。

Prefill 阶段，`slot_mapping` 是一个 list，长度等于所有 sequence 的总 token 数量。Prefill 阶段 kvcache 还没有得到填充。

Decode 阶段，每个序列一个 slot，对应最后一个 token 的位置。slot 指向**新生成 token 的 KV Cache 存储位置。

### 为什么需要 slot_mapping 这个机制？

似乎是为了预计算出来写入的位置。

## 概念

### Sequence

每个序列有一个自己的 block_table。

### Block

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
```

kvcache 分配的最小单元。会通过引用计数来管理内存的生命周期。

BlockManager 管理 Block：

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
```

hash_to_block_id 会用于前缀缓存，似乎是区块链那种，前一个 block 的 hash 加上这个 block 的 hash 计算出来的。

前缀缓存并不会只看 hash，也会检查 token_ids 做进一步确认。

只有满的块，才会计算 hash 作为前缀缓存重用。

