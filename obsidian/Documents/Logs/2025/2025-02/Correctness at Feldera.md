Feldera 有复杂的 correctness surface，传统的测试金字塔并不适用了。这些包括：

1. 一个增量计算引擎（有严谨的理论基础）
2. 支持复杂的 SQL 种类
3. 能够与多种不同的后端进行交互
4. 能够与不同的外部系统进行交互
5. 有自己的 control plane 能够部署、管理 pipelines；

## **Formally verifying the DBSP theory**

DBSP 是 Feldera 的核心，它能够计算出复杂 query 的增量计算。

DBSP 的优雅之处在于，它是可以组合的：if Feldera can incrementalize a primitive operator within a query, then the composition of that operator into a larger dataflow is always correct。

DBSP 的理论是经过 Lean Theorem Prover 证明的。

证明的正确不能代表实现的正确，怎样保证实现也是正确的呢？

## **Differential testing of the implementation**

> A significant bulk of our testing strategy is therefore to make use of **_differential testing_**. The core idea is to compare two different executions for the same input and compare the results produced by the runs for disparities. We typically call one run the implementation and the other the reference run. The best part of this approach is that it makes the oracle simple: **just compare the outputs produced by the reference and implementation runs.** Now all we need to do is get a vast corpus of test cases (generated or reused from standard suites).

思路是比较两个不同的输入，比较两个结果的输出差异。

这里的 reference run 可以是：

> - A full-blown database like Postgres, SQLite, or DuckDB
> - A query engine like DataFusion
> - Known outputs from any of the above (i.e., golden traces or test cases)
> - two different runs of Feldera itself (e.g. a run with and without injected crashes to test our fault-tolerance layer)
> - SQL with and without optimizations
> - incremental and non-incremental query plans
> - A simplified model of a program

等等

### **Testing the Feldera query engine**

Feldera 和 flink 不同，每个时间点上，一个 feldera pipeline 的结果会和 batch system 完全相同。

这个 guarantee 让测试变得简单。

> Our SQL compiler adapts a [large corpus of test cases](https://github.com/feldera/feldera/tree/c8237d789f5239597944214da007bdfc16597c91/sql-to-dbsp-compiler/SQL-compiler/src/test/java/org/dbsp/sqlCompiler/compiler/sql/postgres) from other open-source projects (Postgres**,** MySql, Calcite). These tests help us validate everything from semantics around the type system (narrowing, widening, casts, overflows, and interplay with ternary logic), to behavior of aggregate functions, arrays, windows, intervals and more. Each test is made of queries and the expected results as displayed by the reference query engine.

集成了大量来自其他开源项目的 test case，帮助检查关于类型系统的所有语义（narrowing、widening、casts、overflows、interplay with ternary logic），以及 aggregate functions、arrays、windows 等等。

feldera 基于的 Apache Calcite。

**DataFusion comparison**。虽然和 datafusion 在语法上有一点差异，在内部的 e2e test 中对 datafusion 也有做一些。

> **SQL Logic Tests.** We also regularly run the [SQL Logic Tests (SLT)](https://github.com/feldera/feldera/tree/main/sql-to-dbsp-compiler) from the SQLite project. Each test presents a SQL query and an md5 checksum of the result. Feldera [passed all 5M+ tests](https://github.com/feldera/feldera/tree/main/sql-to-dbsp-compiler#sqllogictest-test-results) early in our project lifecycle.

包含了 500w 个 test case。

**DuckDB, Snowflake, BigQuery and more.** 针对 duckdb 等 OLAP 系统做对比。

> **SQL Lancer.** [SQL Lancer](https://github.com/sqlancer/sqlancer) is a tool that generates databases and/or queries and provides different oracles to ensure correctness. We’ve recently begun using SQL Lancer to generate test cases as well and it’s already been useful to generate test cases that stress our parser. We've so far ran only the oracle that compares queries with and without optimization but that hasn't revealed any bugs yet -- this is likely because we already do such comparisons ourselves in our SQL compiler's tests.

使用 SQL Lancer 生成 SQL 来对 parser 进行压测。

> **Manually written tests.** Despite all the above approaches, there is still a lot more to test! We still invest quite some time into manual testing via the traditional testing pyramid. One of the most useful pieces of that pyramid is a growing suite of end-to-end tests that exercise a complex test matrix across the different types and aggregates we support (e.g., checking for `argmax` over all types). All of these are validated for correctness against Postgres or other DBs, and the expected behavior noted alongside the test case itself. Here's a small example:

> One of the most useful pieces of that pyramid is a growing suite of end-to-end tests that exercise a complex test matrix across the different types and aggregates we support

通过手工规则生成 SQL 来做 e2e test，执行复杂的 test matrix，并针对 postgresql 比较结果。

### **Testing the platform**

> **Faulty vs fault-free runs.** The idea is simple: run a workload twice, one where you’ve injected crashes into the execution, and another where you haven’t. Compare intermediate and end states, and if they’re not identical, you have a bug.

执行一个 workload 两次，在其中的一个 workload 中注入错误，在另一个 workload 中不注入。比较中间结果和最终结果，如果不一样，那么最后一定是有 bug。

> **Model-based tests for the pipeline-manager.** The pipeline-manager is the control plane through which users define, compile, run and manage pipelines. The pipeline-manager implements a REST API using Postgres as its backing store. It has an API server that issues SQL queries to marshal/unmarshal API objects to and from the database.

```rust
enum StorageAction {
  // ... A lot omitted
  ListPipelines(TenantId),
  GetPipeline(TenantId, String),
  GetPipelineById(TenantId, PipelineId),
  NewPipeline(
	TenantId,
	#[proptest(strategy = "limited_uuid()")] Uuid,
	#[proptest(strategy = "limited_pipeline_descr()")]
    PipelineDescr,
  ),
  NewOrUpdatePipeline(
	TenantId,
	#[proptest(strategy = "limited_uuid()")] Uuid,
	#[proptest(strategy = "limited_pipeline_name()")] String,
	#[proptest(strategy = "limited_pipeline_descr()")]
    PipelineDescr,
  ),
  // ... A lot omitted
}
```

用户的所有交互，都可以认为是 enum 中的一个 Action。线上的生产是在 postgres 中存储的。

作者使用了 proptest 来生成用户的交互历史，同时实现了一个 in-memory 的 pipeline manager 实现作为 reference implementation。

这样比较它们的结果是否一致。