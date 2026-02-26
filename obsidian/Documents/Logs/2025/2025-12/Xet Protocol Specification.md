## Overview

Xet Protocol 是一个两阶段过程：

1. Recontruction Query：获取 CAS API 来得到文件的 reconstruction metadata；
2. Data Fetching：结合 reconstruction data 来下载、组装文件；

## Stage 1: Calling the Reconstruction API

### Single File Reconstruction

对于大文件，比较推荐一批一批地进行重建。比如先捞 10GB、下载下来，再捞 10GB 的元信息。

用户必须要有 read 权限的 auth token。

需要使用 HTTP Range 头来指定下载的字节范围。

## Stage 2: Understanding the Reconstruction Response

### QueryReconstructionResponse Structure

```json
{
  "offset_into_first_range": 0,
  "terms": [
    {
      "hash": "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
      "unpacked_length": 263873,
      "range": {
        "start": 0,
        "end": 4
      }
    },
    ...
  ],
  "fetch_info": {
    "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456": [
      {
        "range": {
          "start": 0,
          "end": 4
        },
        "url": "https://transfer.xethub.hf.co/xorb/default/a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
        "url_range": {
          "start": 0,
          "end": 131071
        }
      },
      ...
    ],
    ...
  }
}
```

- offset_into_first_range: 表示第一个数据块中要跳过的字节数，数据必须按完整的块进行下载；
- terms：表示需要下载的块，按顺序排列；`[start, end)` 表示该 xorb 中包含哪些块；
- `fetch_info`：为每个 xorb 哈希提供具体的下载信息；

## Stage 3: Downloading and Reconstructing the File

