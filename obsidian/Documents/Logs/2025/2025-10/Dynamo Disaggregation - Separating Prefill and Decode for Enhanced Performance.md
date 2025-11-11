分离 Prefill 和 Decode 到不同的 llm 引擎有很多好处。

比如可以为 memory bound 的 decoding 阶段开大一点的 TP，给 computation bound 的 prefill 阶段开小一点的 TP。

也比如，对于较长的 context，将 prefill 发到 prefill engine 去，能够允许 decode 不被 prefill 阻塞。

Disaggreagted 执行有如下几个步骤：

1. Prefill engine 计算出来 prefill phase，生成 kv cache；
2. Prefill engine 将 kv cache 转移到 decode engine；
3. decode engine 计算 decode 阶段；

不过，不是所有的 prefill phase 都需要做 pd 分离，比如 prefill 比较短，或者 decode engine 有足够长的。

## Design

有四个组件：

- Worker：执行 prefill 和 decode request
- Prefill worker：仅执行 prefill request
- Disaggregated Router：确认是本地执行还是远程执行
- Prefill Queue：cache、load balance 远程的 prefill 请求

## Conditional Disaggregation

1. prefill length 没有 prefix cache 的部分，如果超过一定 threshold；
2. prefill queue 中远端的 prefill 请求的数量，小于特定的阈值；

## Prefill Queue

Prefill request 是 computation bound （除了特别短的 prefill），为了保障 ttft，会在专门的 iteration 中执行，没有其他请求组 batch。会在多个 prefill engine 中做负载均衡，dynamo 会放一个全局的 prefill queue，让 prefill worker 去抢活。

这个 prefill queue 是用 nats 做的。

## Efficient KV Transfer

Dynamo 使用 NIXL 来转移 KV cache，直接从 prefill engine 的 VRAM 转移到 decode engine 的 VRAM。

KV transfer 是 non-blocking 的，允许 GPU forward pass 来处理其他请求。

KV block 被分配之后，worker 的 scheduler 会发送 remote prefill request，包含已分配的 KV block 的内存描述符，插到 prefill queue 里。

在 remote prefill 完成之后，worker scheduler 自动将 decode request 加入执行。这允许 worker 在等待 prefill 的同时，仍能执行 decode/prefill 任务。

对于不同的 KV layout 的机器（比如不同的 TP size），dynamo 会应用一个高性能的 kernel 将 kv block 进行转换。