作者 rewrite 了 polars 中的所有的 string/binary 的 data structure。

最初 polars 是 arrow2 crate 的用户。这意味着不能轻易修改 string 类型，因为 arrow2 的目标是完整的遵循 arrow 的 spec。

去年作者 fork 了一部分 arrow2 为 polars-arrow。是对 arrow spec 的一个裁剪实现，适配 polars 的需要。

## 1. Why the change?

arrow spec 中的 string type 有三个 buffer 组成。

几个 string 会连续地存储在一个 `data` buffer 中。然后有一个 offsets 来存它们的偏移。

![[Pasted image 20240807221017.png]]

### 1.1 The good

存储很紧凑，所有的 string 内存分配都在一个 buffer 中（不过难于提前预测 buffer 的大小）。每个 string 只有一个 i64 的额外存储开销。访问 string 需要一个 indirection，不过顺序便利是局部的，对 cache 友好。

### 1.2 The bad

难于提前预料分配的大小，这会导致额外的 memcpy。

此外一大缺点是，在 `gather` （taking rows by index）和 `filter`（taking rows by boolean mask）操作中，时间与 string size 有线性相关。

`group-by`, `join`, `window-functions` 等操作都严重依赖 `gather` 和 `filter`，这部分慢了对 query 性能影响比较大。

## 2. Hyper/Umbra style string storage

对应的解决方案来自 hyper/umbra 系统，也被 arrow spec 采纳：

将 string 对应 16 bytes 的一个 column，称作“view”。

当 string 小于 12 bytes 时，string 的数据 inline 在里面：

1. 4 bytes 的 length
2. 4 bytes 的 string prefix
3. 8 bytes 的剩余内容（zero padded）

当 string 大于 12 bytes 时：

1. 4 bytes 的 length
2. 4 bytes 的 string prefix
3. 4 bytes 的 buffer index
4. 4 bytes 的 string offset

![[Pasted image 20240807223731.png]]

### 2.1 The good

- 对于短的 string，可以整个 inline 进来，不需要再 indirect 访问；
- Because the views are fixed width and larger string buffers can be appended, we can also mutate existing columns.
- 大的 string 可以做到 intern 处理，可以节约很多内存，允许做到 dictionary encoding；
- 对于当前 column 计算出新数据的时候，比如 `filter` 和 `gather` 等可以在常数时间内拷贝出来这些元素，只拷贝 view，buffer 可以不动；
- Many operations can be smart with this data structure and only work on (parts of the) views. We already optimized many of our string kernels with this, and more will follow.
### 2.2 The bad

- 保存一个 string 需要稍多的存储空间：默认的 arrow string 有 8 bytes 的 overhead，而 binary view encoding 对长 string 默认有 16 bytes 的 overhead、对短 string 有 4 + 12 - string len 的 overhead；在作者看来这是值得的；
- 作者认为最大的不足是，需要考虑垃圾回收；当执行 `gather`/`filter` 之后，我们仍需要保留原本的 string，这需要我们想办法来决定何时做垃圾回收。



