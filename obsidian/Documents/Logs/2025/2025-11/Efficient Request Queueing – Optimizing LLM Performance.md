
### Starting Point: A Bare Inference Engine

- 一个像 vLLM 或者 huggingface TGI 的 inference engine 包括：
	- 一个 worker 干活
	- 一个队列接受到来的请求
	- 一个 scheduler 从队列中取出请求，移动给 worker
- 为什么需要队列？因为 GPU 的任务按 batch 来处理会更高效，这个队列允许 scheduler 从队列中取出多个请求，作为单个 batch 去处理

### Problem: "Power Users" Can Block Other Users

- 当一个 user A 发发送大量请求时，会快速填满整个队列
- 其他 user B、user C 可能就会长时间得不到响应

### Solution: Fair Scheduling

- 可以给每个（用户，模型）单独的队列，scheduler 不会按 FIFO 进行处理所有的请求，而是公平地从每个用户的队列中捞任务

![[Pasted image 20251129210005.png]]

### Possible Extensions

- 除了 request 数量，还可以参考的一个指标是 “process time”
- 较长的 prompt，以及较长的生成输出，也会阻塞其他用户较长时间，不幸的是，我们很难评估生成输出的长度
- 在商业上下文中，一个请求的 “cost” 可以是另一个关键的指标
- 一种潜在的解决方法是，一个用户可以拥有多个不同的队列，不同队列有不同的优先级
- 比如，TNG 的 AI 助手的 chat interface 需要有一个高优先级，另外的一些 use case 比如 batch API 会有一个更低的优先级

### Problem: No Backpressure by Backend Queue

- 如果一个 user A 向后端发送了一大批请求之后，过来了 user C 一个请求
- 这时即使有前面的公平队列，user C 仍然需要等待较长时间
- 比较希望的是，后端的执行队列能够越短越好

### Solution: Fetch Metrics

- 作者需要从 vLLM 的 prometheus 指标中，捞出来一些后端的队列信息
- 调度器的规则：当 队列长度 < 目标值（比如 3） 时，才允许新请求入队
	- 目标队列的长度如果较高，则批次充分利用，执行效率高，代价是新请求延迟大
- 注意，目标队列长度并不反应最大并发量
	- 目标的队列长度如果是 3，意味着有 3 个任务在等待，当前批次中可能有 10 个请求在处理中，总共在处理的任务，可能是 13

### Possible Extensions

- 还可以更精细一些，基于 SLO 指标，比如报告的 TPOT 指标如果超过 150ms，则不处理新的请求
- 对于不同的请求优先级，也可以指定不同的 metric 阈值：比如，对 batch API 的低优先级队列，我们只有在当前 backend queue 中为空时，才调度任务，宁可 GPU 的利用率降低，减少新的更高优先级的请求的延迟的影响
- 也有更多潜在的优化项：如果发现当前的队列长度指标为零，你可以一下子将三个请求发给后端，而不再重复捞指标

### Alternative: Backend-Side Priority Scheduling

- 最近 vLLM 也添加了一个基于优先级的调度策略，允许请求在发送给后端之前标记一个优先级
- 产生的副作用是，低优先级的任务可能会被 evict 出来重新回到 waiting queue

![[Pasted image 20251129212033.png]]

### Can Backend-Side Priority Scheduling Replace All Queues on the LLM-Server Side?

- 仍有一些 caveat：
	- backend priority 特性只在 vLLM 中有，在其他引擎中可能不一定有
	- 在 LLM-Server 层面有一个调度器，有助于基于 tpot 对 scheduling rate 进行比较精细的调整
	- 频繁的 reorder、requeuing 对延迟的影响可能还需要检验一下