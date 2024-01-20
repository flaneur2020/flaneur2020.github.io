
- BYOC 本身不是一个新概念，在 90 年代有一个 MSP （Managed Service Provider）的概念；MSP 当年面向的客户也是 “want to keep some degree of control and visibility but don’t want to operate the software themselves anymore”
- BYOC 的例子有 Databricks（the modern pioneer of BYOC）、StarTree BYOC、Redpanda BYOC、StreamNative BYOC；
- BYOC 的 promise 包括：
	- 在自己的账号中管理数据，有更低的成本；
	- 更便宜，更低的 TCO；
- 但是 BYOC 会失去：
	- Serverless/ Resource Pooling（赞同）
	- Operational efficiency（不是很赞同）
	- A clear responsibility boundary. （有道理）

## The promise of security

- 作者的观点是 end to end encryption 才是金标准；（但是也就 kafka 这个形态有 e2e encryption 这一说的吧...）