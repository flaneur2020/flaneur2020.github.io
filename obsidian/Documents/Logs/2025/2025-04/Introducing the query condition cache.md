 query condition cache：在 granule level，允许 clickhouse 跳过相当一部分 data。

clickhouse 安 8k 行一个 granule 来组织数据。

![[Pasted image 20250426155659.png]]

## cache 的组织形式

 Cache Key：

- Table ID：表的标识符
- Part Name：数据 Part 的 ID
- Condition Hash：Where 条件的哈希值

Cache Value：

- `matching_masks`: 是一个 bitmap，对应这个 part 中的 granule 列表，每个 granule 对应一个 bit，如果是零，则表示该 granule 中没有相关的数据。

```
table_uuid:     6f0f1c9d-3e98-4982-8874-27a18e8b0c2b  -- Table ID
part_name:      all_9_9_0                             -- Part Name
key_hash:       10479296885953282043                  -- Condition Hash
matching_marks: [1,1,1,0,0,0, ...]                   -- Array indicating matching granules
```

## 优化效果

场景：查找包含“🥨”表情符号的帖子的语言分布

```sql
SELECT
    arrayJoin(CAST(data.commit.record.langs, 'Array(String)')) AS language,
    count() AS count
FROM bluesky
WHERE
    data.kind = 'commit'
    AND data.commit.operation = 'create'
    AND data.commit.collection = 'app.bsky.feed.post'
    AND data.commit.record.text LIKE '%🥨%'
GROUP BY language
ORDER BY count DESC
SETTINGS use_query_condition_cache = false;
```

**启用 Query Condition Cache (SETTINGS use_query_condition_cache = true):**

- 查询耗时：0.055 秒
- 处理行数：1.08 million 行
- 处理数据量：98.42 MB
- 峰值内存使用：102.66 MiB

**关闭 Query Condition Cache (SETTINGS use_query_condition_cache = false)**:

-  查询耗时：0.601 秒
- 处理行数：99.43 million 行
- 处理数据量：9.00 GB
- 峰值内存使用：418.93 MiB

