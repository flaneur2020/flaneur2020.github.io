prev: [[2024 - Index]]
## Jan

- Night
	- Snapshot in slatedb
	- Sync Commit in slatedb
	- fizzbee
	- limbo 加测试
	- [[ggblas 看代码]]
- Books
	- Python 量化交易
	- [[OO年代的想象力]]
	- [[文革前的邓小平]]
- Readings
	- data sys
		- [[iceberg FastAppend & MergeAppend]]
		- [[How bloom filters made SQLite 10x faster]]
		- [[Robust External Hash Aggregation in the Solid State Age]]
		- [[Preventing Data Resurrection with Repair Based Tombstone Garbage Collection]]
		- [[DSQL Vignette - Transactions and Durability]]
		- [[Can Applications Recover from fsync Failures?]]
	- quality
		- [[FizzBee Quick Start for TLA+ Users]]
		- [[sdv - Quality Report]]
		- [[Lightweight property-based testing at Row Zero]]
	- [[rustgo - calling Rust from Go with near-zero overhead]]
	- [[So you wanna write Kubernetes controllers?]]

## Feb

- Night
	- flappy Bird RL
	- DQN
	- [[Reinforcement Learning - 策略梯度]]
	- PPO
- Books
	- 动量交易
	- 深度强化学习实战
- Readings
	- RL
		- [[Flappy Bird RL]]
		- [[Deep Reinforcement Learning forFlappy Bird]]
		- [[Log-derivative trick]]
	- Inference
		- [[GPU 异构大模型推理策略 - KTransformers]]
		- [[Throughput is Not All You Need - Maximizing Goodput in LLM Serving using Prefill-Decode Disaggregation]]
	- datasys
		- [[Correctness at Feldera]]

## Mar

- Night
	- backtrader
	- Policy Gradient: Baseline
- Books
	- 动量交易
	- 穿越抑郁的正念之道
- Readings
	- datasys
		- [[Doris - Runtime Filter]]
		- [[trino - Dynamic Filtering]]
		- [[Testing Database Engines via Pivoted Query Synthesis]]
		- [[Efficient Filter Pushdown in Parquet]]
	- Infra
		- [[Building Clickhouse BYOC on AWS]]
	- AI
		- [[My LLM codegen workflow atm]]

## April

- Night
	- slatedb: sync commit
	- https://github.com/Eventual-Inc/Daft/issues/4144
- Books
	- 简单致富
	- 美国困局
- Readings
	- datasys
		- [[Inside ScyllaDB Rust Driver 1.0 - A Fully Async Shard-Aware CQL Driver Using Tokio]]
		- [[Optimizing SQL (and DataFrames) in DataFusion - Part 1]]
		- [[Optimizing SQL (and DataFrames) in DataFusion - Part 2]]
		- [[Introducing the query condition cache]]
		- [[Taking out the Trash - Garbage Collection of Object Storage at Massive Scale]]
		- [[Lance v2 - A columnar container format for modern data]]
	- kube

## May

- Night
	- slatedb: sync commit
	- daft: panic when selecting struct literal
	- opendal: object_store https://github.com/apache/opendal/issues/6171
- Books
	- 价值投资 3.0
- Readings
	- kube
		- [[Binpack Scheduling That Supports Batch Jobs]]
		- [[Gang Scheduling Ray Clusters on Kubernetes with Multi-Cluster-App-Dispatcher (MCAD)]]
		- [[A Deeper Dive of kube-scheduler]]
	- datasys
		- [[Incremental View Maintenance with Datafusion and Iceberg]]
		- [[datafusion-materialized-view 看代码]]
		- [[pocket watch - verifying exabytes of data]]

## Jun

- Night
	- slatedb: sync commit
	- daft: panic when selecting struct literal
	- opendal: object_store https://github.com/apache/opendal/issues/6171
- Books
	- 价值投资 3.0
	- 段永平投资访谈录
	- Gold is a better way
- Readings
	- datasys

## July

- Night
	- slatedb: retention iterator
	- opendal: object_store
	- kiwi-rs
- Books
	- Gold is a better way
	- 资本帝国
- Readings
	- ai infra
		- [[Gateway API Inference Extension - CRD]]
		- [[Gateway API Inference Extension - Scheduler]]
		- [[Gateway API Inference Extension - Data Layer Architecture]]
		- [[Real-world engineering challenges - building Cursor]]
	- rust
		- [[Leaktracer - A Rust allocator to trace memory allocations]]
	- storage
		- [[Using Lightweight Formal Methods to Validate a Key-Value Storage Node in Amazon S3]]


## Aug

- Night
	- slatedb:
		- snapshot API
		- transaction API
	- opendal: object_store
	- kiwi-rs: Engine trait
	- cilium: socket-lb
- Books
	- To B 的本质
	- DeFi与金融的未来
	- [[告别失控]]
- Readings
	- ai infra
		- [[Serving Large Language Models on Huawei CloudMatrix384]]
	- coding
		- [[Unit Testing Principles]]


## Sep

- Night
	- slatedb: transaction API, SSI Conflict Checks
	- kiwi-rs: Engine trait, seperate tokio runtime
	- learn Mooncake
- Books
- Readings
	- ai infra
		- [[DeepkSeek serving on Huawei CloudMatrix384]]
		- [[25x Faster Cold Starts for LLMs on Kubernetes]]
		- [[Mooncake - Store Preview]]
		- [[Mooncake - Transfer Engine]]

## Oct

- Night
	- slatedb: transaction benchmark
	- nano-vllm
	- learn TRPO 
- Writings
	- [[nano-vllm 看代码]]
- Readings
	- llm
		- [[A Survey of Reinforcement Learning for Large Reasoning Models]]
	- ai infra
		- LLM Query Scheduling with Prefix Reuse and Latency Constraints
		- [[Dynamo Disaggregation - Separating Prefill and Decode for Enhanced Performance]]
		- [[How To Reduce Cold Start Times For LLM Inference]]
		- [[]]
	- sys
		- [[Cancelling async Rust]]
		- [[An MVCC-like columnar table on S3 with constant-time deletes]]

## Nov

- Learn
	- nvidia-dra-plugin
	- libkrun
- Readings
	- cloud infra
		- [[GKE network interface at 10 - From core connectivity to the AI backbone]]
		- [[Dynamic Resource Allocation]]
		- [[Understanding RDMA Components in Linux]]
		- [[Data Replication Design Spectrum]]
		- [[CSI 概念]]
	- sys
		- [[libkrun - README]]
		- [[libkrun - 看代码]]
		- [[libkrun - virtio-blk]]
	- ai infra
		- [[Efficient Request Queueing – Optimizing LLM Performance]]
		- [[Prefill and Decode for Concurrent Requests - Optimizing LLM Performance]]

## Dec

 - Learn
 - Readings
	- ai infra
		 - [[RDMA over Ethernet for Distributed AI Training at Meta Scale]]
	- cloud infra
		- [[Under the hood - Amazon EKS ultra scale clusters]]