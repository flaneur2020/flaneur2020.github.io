vmstorage 并不会直接在存储中保存完整的 `node_cpu_seconds_total` 这样的 series name。

而是像这样的一个：

![[Pasted image 20260920212803.png]]

vmstorage 还需要维护一个 TSID 到 metric 的映射关系，这个就属于 IndexDB 的工作。

像 `sum_over_time(node_cpu_seconds_total{mode="idle"}[5m])` 这样的一个 query，vmstorage 需要做的是，检索出特定 time series 在特定时间范围内的所有 samples。至于 aggregation 计算（比如 `sum_over_time`），是在 vmselect 里做的。

IndexDB 的工作，就是将 human-readable 的 metric 名字，比如 `node_cpu_seconds_total{mode="idle"}` 翻译为内部的 **TSID** 列表。

每个 TSID 大约可以理解成一个数据块。

![[Pasted image 20260920213220.png]]


## How IndexDB is Structured

每个 partition 有自己的 IndexDB。partition 是 UTC 的 **月级**（YYYY_MM）来分的。因此每个 IndexDB 涉及一个月的时间范围。

![[Pasted image 20260920213414.png]]

Partition 也是 retention 的单位。在 retention 时，直接删一个月的数据和 IndexDB。

IndexDB 的每行数据都以一个数字前缀开头，前缀用于表示它是一个什么类型的数据。默认有其中前缀：


|     |                             |         |                                                      |                          |
| --- | --------------------------- | ------- | ---------------------------------------------------- | ------------------------ |
| 1   | Tag -> Metric IDs           | Global  | status=200 -> 67,99,100,120,130                      | 查询入口：从标签找候选 metric       |
| 2   | Metric ID -> TSID           | Global  | `49 -> TSID{metricID=49}`                            | 从逻辑 ID 补全到物理指针           |
| 3   | Metric ID-> Metric Name     | Global  | 49 -> http_request_total{method="Get", status="200"} | 把内部 ID 还原成可读名字           |
| 4   | Deleted Metric ID           | Global  | 49                                                   | 删除标记                     |
| 5   | Date → Metric IDs           | Per Day | 2024-01-01 -> 152                                    | 快速判断某天有没有这个 metric       |
| 6   | Date + Tag → Metric IDs     | Per Day | 2024-01-01 method=GET -> 152,156,201<br>             | 按天的 Tag 到 metric IDs 的列表 |
| 7   | Date + Metric Namec -> TSID | Per Day |                                                      |                          |
> **TSID (Timeseries ID) is technically a wrapper around a metric ID, with a few extra fields.** The metric ID itself is a large, unique number that identifies each timeseries. From the user's perspective, there is not much practical difference between TSIDs and metric IDs. **Since they have a one-to-one mapping, we will use the terms interchangeably for educational purposes.**


```go
type TSID struct {
    AccountID uint32
    ProjectID uint32   // 多租户标识

    MetricGroupID uint64  // 指标组 ID（指标名 __name__ 的 xxhash）
    JobID         uint32  // 第 0 个 tag 的 xxhash
    InstanceID    uint32  // 第 1 个 tag 的 xxhash

    MetricID uint64       // 指标（时间序列）的唯一 ID
}
```

数据是按 TSID 排列的，便于让**同名指标**（如所有 `memory_usage`）在磁盘上聚在一起。

### Part Data of IndexDB on Disk

IndexDB 的组织方式与主存储相似，但是是不同类型的数据。主存储保存的是 sample 和时间戳，而 IndexDB 保存索引条目（items）和辅助数据（lens），便于快速找到这些条目。

```bash
/path/to/vmstorage-data/data/indexdb/
├── 2026_01/                          # Partition IndexDB (YYYY_MM)
│   ├── parts.json                    # List of IndexDB parts for this partition
│   ├── 183A9F12C4D8E001/             # IndexDB part directory
│   │   ├── metadata.json             # Part metadata
│   │   ├── items.bin                 # Index rows payload
│   │   ├── lens.bin                  # Item lengths/offset helpers
│   │   ├── index.bin                 # Block headers
│   │   └── metaindex.bin             # Top-level lookup index
│   └── ...
├── 2026_02/
│   ├── parts.json
│   ├── 183B11AA09F7007C/
│   │   ├── metadata.json
│   │   ├── items.bin
│   │   ├── lens.bin
│   │   ├── index.bin
│   │   └── metaindex.bin
│   └── ...
└── 2026_03/
    └── ...
```

items.bin 文件存储实际的索引记录，比如 Tag 到 metric ID 列表的映射。

lens.bin 储存这些记录的 offset。这样 vmstorage 可以直接跳到 items.bin 的正确字节范围，不用从头扫描。

![[Pasted image 20260920215346.png]]

实践中，items.bin 和 lens.bin 都是按块（Block）写入的。

![[Pasted image 20260920215406.png]]

块的边界记录在 index.bin 中，里面有 block header。

同一个块里的行，如果共享前缀（比如 `http_`），则只存一次。

```
块头: http_
行1: active_requests 199,301
行2: connections_active 720,930,931,932
```

metaindex.bin 指向 index.bin 里的 section。

### 读取的流程

```
┌─────────────────────────────────────────────────────────┐
│                     查询请求                              │
│         http_request_total{status="200"}                │
│              时间范围: 2024-01-01 13:00-14:00            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  找到重叠的 partition  │
              └──────────┬───────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │  按天索引（前缀 6）快速缩小范围   │
        │  6 2024-01-01 status=200 ...   │
        └────────────────┬───────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │  全局索引（前缀 2）补全 TSID     │
        │  2 49 TSID{metricID=49,...}    │
        └────────────────┬───────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │  用 TSID 去主存储读数据块        │
        └────────────────┬───────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │  返回样本给 vmselect 做聚合      │
        └────────────────────────────────┘
```

### Merge 流程

先缓冲在内存，刷成小的 part，然后后台不停把小的 part 合并成大 part。

merge 也是去重、降采样的时机。

```
Storage
 └── Partition (按月, YYYY_MM)
      └── Part (可位于内存或磁盘)
           └── Block (最多 8K 个样本，属于同一条时序)
```

- 每个 part 由**按 TSID 排序**的块组成

```
写入数据
   │
   ▼
内存缓冲（最多 1 秒，-inmemoryDataFlushInterval 可配）
   │
   ▼
内存 part（可被查询搜索）
   │
   ▼
定期持久化到磁盘：data/Small/YYYY_MM/
   │
   ▼
后台 Merge：小 part → 大 part
```

### IndexDB 的 Merge 流程

```
shard_count = cpu_cores * min(16, cpu_cores)
```

- 4 个核 -> 16 个分片
- 每个分片最多 256 个内存块
- 每个块最多 64kb 数据

IndexDB 的合并流程与主存储几乎一致，额外做索引条目的集合合并与去重。

```
分片填满
   │
   ▼
pending blocks（待处理块）
   │
   ├── 每 1 秒周期性刷盘
   └── 或内存中块太多时刷盘
   │
   ▼
读取所有 item → 合并排序 → 写入新的内存 part

```

```
┌─────────────────────────────────────────────────────────┐
│                      写入路径（快）                       │
│  数据 → 内存缓冲(1s) → 内存 part → fsync → parts.json    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    Merge 路径（后台）                     │
│                                                          │
│   小 part ──┐                                            │
│   小 part ──┼──► 合并 ──► 大 part                        │
│   小 part ──┘      │                                     │
│                    ├── 去重                              │
│                    ├── 降采样                            │
│                    ├── 释放已删除时序空间                 │
│                    └── 索引：合并 metricID 列表、排序去重 │
│                                                          │
│   ⚠️ 磁盘空间不足 → 停止合并 → part 增多 → 查询变慢      │
└─────────────────────────────────────────────────────────┘

```