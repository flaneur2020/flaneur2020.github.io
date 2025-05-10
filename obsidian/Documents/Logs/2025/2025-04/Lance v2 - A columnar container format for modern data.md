作者据称是为了更好的适应 AI/ML 的 workload。而且有些其他人反馈在做这类场景时，parquet 不大好用。
### Point Lookups

parquet 不大适应 point lookup 场景的原因是它的 encoding 格式不大能做到 “sliceable”，往往需要 load 一整个 row 才能读取一行。

这对 multi-modal workload 不大爽。因为 multi-modal 场景下，值通常都很大。

![[Pasted image 20250426173459.png]]

### Wide Columns

wide column 是指每个值都特别大的情况。

传统数据库中的每个列往往都很小。其中 String 就已经属于最大的了，但是实践中，也通常很小。

在 ML workload 中，我们经常希望储存 tensor 信息，比如 semantic search embedding（4kb 的 CLIP embedding）甚至更大的图像。

![[Pasted image 20250426173700.png]]

### Very Wide Schemas

比如金融（每个 ticker 一个列）；feature store（每个记录可能有几千个 feature）；

parquet 在 projection 等方面有一定帮助，但是仍然需要 load 整个 schema 的 metadata 进来。这对 low latency 要求的 workload 来讲压力较大。

### Flexible Encodings

paquet 在支持新的 encoding 时，往往需要对 file reader 做修改。

### Flexible Metadata

Parquet 的元数据结构限制了编码对列或文件元数据的访问，导致一些编码决策不够理想（例如，字典编码需要将字典存储在每个行组中）。

## 解决

- **编码即扩展 (Encodings are Extensions)**：Lance v2 本身没有内置编码，而是通过扩展来处理。文件读取器和写入器对编码一无所知，编码的细节由插件决定。这使得添加新编码无需修改核心格式，提高了灵活性。
    - **推论 1：没有类型系统**：Lance 格式本身没有类型系统，将列视为页面集合。读取器和写入器负责将这些页面转换为特定类型的数组（例如使用 Arrow 类型系统）。
    - **推论 2：赋能编码开发者**：这使得添加新编码变得容易，有助于管理生态系统碎片化。
- **废除行组 (Abolish Row Groups)**：Lance v2 取消了行组的概念。列中的页面不需要彼此相邻，列写入器在积累足够数据后即可将页面刷新到磁盘。这带来了以下好处：
    - **推论 1：理想的页面大小**：每个列写入器可以配置自己的缓冲区，以匹配文件系统的理想页面大小。
    - **推论 2：解耦 I/O 和计算**：读取时，可以独立于 I/O 大小选择批处理大小，实现 I/O 并行性与计算并行性的解耦。
- **灵活性 (Flexibility)**：Lance 中的列长度可以不同，支持“一次写入一个数组”或“一次写入一个批次”的方式。数据可以放置在页面、列或文件缓冲区中，这为处理非表格数据等新用例提供了可能。
    - **推论 1：“真正的”列投影**：每列的元数据存储在独立的块中，读取单列无需读取其他列的元数据，提高了处理极宽模式的效率。
    - **推论 2：数据与元数据的流动性**：写入器可以选择将字典、跳跃表、区域图等信息存储在列元数据或页面中，提供了更大的灵活性。
    - **推论 3：统计信息即编码元数据**：统计信息（如区域图）被视为编码过程的一部分，可以存储在列元数据中，无需修改核心格式即可支持新的统计信息类型。