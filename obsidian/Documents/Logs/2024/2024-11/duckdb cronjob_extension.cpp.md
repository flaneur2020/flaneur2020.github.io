```
struct CronTask {
    string id;
    string query;
    string schedule;
    bool active = true;
    std::chrono::system_clock::time_point next_run;
    std::vector<std::pair<std::chrono::system_clock::time_point, string>> execution_history;
    std::mutex mutex;
    cron_expr expr;
};
```

CronScheduler 有 AddTask、GetTaskStatus、RemoveTask、RunScheduler、ExecuteTask 几个方法

## 问题

### 怎么和 duckdb 对接的？

实现了一个 `CronScalarFunction(DataChunk &args, ExpressionState &state, Vector &result)` 

```C++
    auto &query_vector = args.data[0];
    auto &schedule_vector = args.data[1];

    UnaryExecutor::Execute<string_t, string_t>(
        query_vector, result, args.size(),
        [&](string_t query) {
            try {
                auto schedule = schedule_vector.GetValue(0).ToString();
                auto task_id = scheduler->AddTask(query.GetString(), schedule);
                return StringVector::AddString(result, task_id);
            } catch (const Exception &e) {
                throw InvalidInputException("Error scheduling cron task: %s", e.what());
            }
        });
```

调用时，用 `UnaryExecutor::Execute` 执行传入的 callback 函数。

（为什么要需要再 UnaryExecutor 里面？直接调用不行？）

（猜应该是这个 Executor 可以帮助处理返回值的细节）

### 怎样初始化？

```C++
void CronjobExtension::Load(DuckDB &db) {
    scheduler = make_uniq<CronScheduler>(*db.instance);
    if (!scheduler) {
        throw InternalException("Failed to initialize cron scheduler");
    }

    // Register the cron scalar function
    auto cron_func = ScalarFunction("cron", 
                                  {LogicalType::VARCHAR, LogicalType::VARCHAR},
                                  LogicalType::VARCHAR,
                                  CronScalarFunction);
    ExtensionUtil::RegisterFunction(*db.instance, cron_func);

    // Register cron_delete function
    auto cron_delete_func = ScalarFunction("cron_delete",
                                         {LogicalType::VARCHAR},
                                         LogicalType::BOOLEAN,
                                         CronDeleteFunction);
    ExtensionUtil::RegisterFunction(*db.instance, cron_delete_func);



    // Register the cron_jobs table function
    auto cron_jobs_func = TableFunction("cron_jobs", {}, CronJobsFunction, CronJobsBind);
    cron_jobs_func.init_global = CronJobsInit;  // Add this line
    ExtensionUtil::RegisterFunction(*db.instance, cron_jobs_func);
}
```

就是注册了三个函数，对外通过 C 的 ABI 暴露 `cronjob_init` 和 `cronjob_version`：

```
extern "C" {
DUCKDB_EXTENSION_API void cronjob_init(duckdb::DatabaseInstance &db) {
    duckdb::DuckDB db_wrapper(db);
    db_wrapper.LoadExtension<duckdb::CronjobExtension>();
}

DUCKDB_EXTENSION_API const char *cronjob_version() {
    return duckdb::DuckDB::LibraryVersion();
}
```