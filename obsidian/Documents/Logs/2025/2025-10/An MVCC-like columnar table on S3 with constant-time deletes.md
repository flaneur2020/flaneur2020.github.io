tldr：

- 感觉跟 iceberg 还是差不多，就是一个根；
- 用 previous 指针找旧的 manifest；

---

Apache Iceberg 和 Deltalake 都将 metadata 文件和 deletion 标记文件与数据文件分开。

## The delete problem with immutable formats

有三种选择：

1. 重写整个文件，移除掉这一行
2. 将该文件标记为删除，过滤掉删除的行
3. 在单独的文件中记录删除标记，在检索时候过滤

大部分系统都选择第三种办法。iceberg 会在 manifest 文件中管理 data file 和 deletion files 的元信息。deltalake 在 transaction log 中记录 deletion 文件。

## S3 conditional writes for coordination

```
S3 Bucket: mytable/
├── _latest_manifest              ← mutable pointer (CAS only)
│   {
│     "version": 123
│   }
│
├── manifest/v00000123.json       ← immutable snapshot
│   {
│     "version": 123,
│     "previous": 122,
│     "data_files": [...],
│     "tombstones": [...]
│   }
│
├── data/2025/10/04/14/
│   ├── f81d4fae.parquet          ← Parquet file (multiple row groups)
│   ├── a1b2c3d4.parquet          ← Parquet file (multiple row groups)
│   └── ...
│
└── tombstone/2025/10/04/14/
    └── abc123.del                ← delete marker

```

## Manifest structure and snapshot isolation

```json
{
  "version": 123,
  "previous_version": 122,
  "created_at": "2025-10-04T13:45:12Z",
  "schema": {
    "columns": [
      {
        "name": "id",
        "type": "int64"
      },
      {
        "name": "event_time",
        "type": "timestamp[us]"
      },
      {
        "name": "payload",
        "type": "binary"
      }
    ]
  },
  "data_files": [
    {
      "path": "s3://mytable/data/2025/10/04/13/f81d4fae.parquet",
      "size_bytes": 268435456,
      "row_group_count": 60,
      "total_rows": 12000000,
      "min_values": {
        "event_time": "2025-10-04T13:00:00Z",
        "id": 1000000
      },
      "max_values": {
        "event_time": "2025-10-04T13:30:00Z",
        "id": 12999999
      }
    },
    {
      "path": "s3://mytable/data/2025/10/04/13/a1b2c3d4.parquet",
      "size_bytes": 268435456,
      "row_group_count": 60,
      "total_rows": 12000000,
      "min_values": {
        "event_time": "2025-10-04T13:30:00Z",
        "id": 13000000
      },
      "max_values": {
        "event_time": "2025-10-04T14:00:00Z",
        "id": 24999999
      }
    }
  ],
  "tombstones": [
    "s3://mytable/tombstone/2025/10/04/13/abc123.del"
  ]
}
```

读者总是读最新的 `_latest_manifest`。

留一个 `previous` 指针指向旧版本，允许一个 retention window。