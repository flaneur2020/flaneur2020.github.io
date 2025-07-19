### cursor by numbers

- 50 工程师、每秒 1M 请求、12 月内有 100x 用户增长、每天企业客户产生 100M 行代码（包括 NVdia、uber、stripe、instacart、shopify、ramp、datadog 等）
- 算上非企业客户，每天有10亿行；
- 每年 5 亿+ 收入；
- index 的规模有几百 tb；

## 1. Tech stack

- cursor 背后的 3 年的 code base 有 25k 个文件、700w 行代码；
- editor 是 vscode 的 fork，因此继承了 vscode 的技术栈；
- cursor 在很早就决定要自己做一个 editor 来掌控交互的体验，而不仅仅是一个 extension
- 而从零开始做一个 editor 的工程量过于庞大，因此选择了 fork vscode 的路线；

### backend

- typescript：大部分业务逻辑是这样写的
- rust：所有的性能敏感的组件都用的 rust，比如 orchestrator
- node api to rust：有一个 nodejs 的 bridge 层，允许从 typescript 中调用 rust 代码，比如 invoking index 逻辑；
- monolith：所有的 backend service 都在一个大的单体中，整体部署；

### 数据库

- turbopuffer：一个多租户的数据库产品 to store encrypted files and the Merkle Tree of workspace，选择它是看中了它的可扩展性，而且不需要和 sharding 搏斗；
- Pinecone：向量数据库

### Data Streaming

- 用的 warpstream

### Tooling

- Datadog
- Pagerduty
- Slack
- Sentry
- Amplitude
- Stripe
- WorkOS
- Vercel
- Linear
- Cursor

### Model Training

cursor 使用了以下产品来做自己的 model，或者 finetune 现有的模型：

- Voltage Park
- Databricks MosaicML
- Foundry

### Physical infrastructure

所有的 infra 都跑在云上。

大多数 CPU infra 都在 aws 上，他们也跑着几千个 nvidia H100 GPU，这里的很多 GPU 都跑在 azure 上。

Inference 是 Cursor 目前最大的 GPU usage，比如 autocomplete。实际上，azure 的 GPU 全部是用于 infererence 的，不包括其他 LLM 相关的工作比如 finetuning 和 training model。

Terraform 是 cursor 用于管理 GPU 和虚拟机等资源的工具。

## 2. How Cursor’s autocomplete works

有一个 low latency sync engine 支持着 “tab model”。

这里的希望能快速地生成，并且最好能小于一秒。

![[Pasted image 20250712151146.png]]

## 3. How Cursor’s Chat works without storing code on the server

cursor 会对代码做索引，并不会直接把代码往服务端放。

> **Search is done using codebase indexes**. Codebase indexes are previously-created embeddings. It tries to locate the embeddings that are best matches for the context using vector search. In this case, the vector search returned two very close results: in server.js, and index.html.

寻找上下文时，通过 code base 索引，根据 embedding 进行搜索。最后在客户端这里，找到相关的代码上下文。

![[Pasted image 20250712152613.png]]
#### Keeping the index up-to-date using Merkle trees

![[Pasted image 20250712152656.png]]每三分钟，cursor 会做一次 index sync。

![[Pasted image 20250712152719.png]]
## 4. Anyrun: Cursor’s orchestrator service

是 rust 写的。

再往后就是收费内容了。 😳