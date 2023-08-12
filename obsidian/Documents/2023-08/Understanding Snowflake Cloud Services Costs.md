https://www.phdata.io/blog/understanding-snowflake-cloud-services-costs/

tldr：
- 好像 cloud service 类似 aws 的 s3 的 api 调用次数费用，但是费用不透明。

## What Are Snowflake’s Cloud Services?
- Cloud Service 包括：
	- Authentication
	- Access Controls
	- Infrastructure Management
	- Metadata Management
	- Security
	- Sharing and collaboration
	- SQL Optimization
	- Transactions

## What is the Snowflake Cloud Services Cost?
- snowflake 会在后台跑这些 cloud service，只要跑，就会消耗 credit
- 每天扣费一次，如果 Cloud Service 的消费小于 Compute Cost 的 10%，则不会扣费
- （好像比较接近 s3 的 api 调用次数费用，但是这部分费用在 snowflake 这边完全不大透明）

## How to Reduce the Snowflake Cloud Services Cost
- Copy Command: When using Snowflake COPY commands, only copy a targeted list of files.
- Reduce Data Definition Language Operations
	- 如果 DDL、clone 等操作涉及大量对象操作，
- Reduce Show Commands
- Bulk/Batch Data Loads：批量操作替代单行的插入
- Rewrite Complex SQL Queries：复杂 Query 编译时间长，也属于 Cloud Service 的开销

