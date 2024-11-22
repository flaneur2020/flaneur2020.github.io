## Why

- Linking DuckDB into every extension: 之前的架构下，需要将 duckdb 静态编译到每个 extension 中；新的架构下，扩展只需要调用 function pointer 即可，不需要考虑 linking 了，能够大大减小 extension 的体积；“Passing a c struct” 的做法来自 sqlite；
- 目前的 C++ API Extension 和 Duckdb 的版本密切绑定，在 duckdb 每发一个版本，都需要重新 build；
- Requiring C++ for writing your extensions：有了 C 扩展之后，就不需要用 C++ 了，用 rust 之类都可以；

## Implementation

思路是不依赖动态、静态链接，extension 的 init 方法能够传一个包含所有方法的结构体进来：

```C
typedef struct {
	// ...
	idx_t (*duckdb_data_chunk_get_size)(duckdb_data_chunk chunk);
	duckdb_vector (*duckdb_data_chunk_get_vector)(duckdb_data_chunk chunk, idx_t col_idx);
	// ...
} duckdb_ext_api_v1;
```

初始化时候大约这样：

```C
#include "duckdb_extension.h"

DUCKDB_EXTENSION_ENTRYPOINT(duckdb_connection connection, duckdb_extension_info info, duckdb_extension_access *access) {
    // Register extension stuff here
}
```

