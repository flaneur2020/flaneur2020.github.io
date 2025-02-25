LLM 现在的 latency 需求比较多样化了，比如，在 chatbot 中，会预期一个比较快的首 token 响应时间，比如 0.2s，但是 Decoding 的速度不一定需求特别高，能 match 人类的阅读速度即可，而代码补全需要一个更快的 end-to-end 时间。

在这篇 blog 中，作者提出吞吐量未必是最佳的优化目标。作者提出一个 **goodput** 指标，每秒完成的请求次数。

作者开发了 DistServe 能够支持 Prefii/Decode 分离，并集成到了 vLLM 中。

## Background: Throughput vs. Goodput

在生产中，很多应用的延时需求并不一致，常用的 SLO 通常包括：

1. Time to first token latency：衡量 LLM 第一次返回 token 的延迟；
2. Time per output token：每个 token 平均的生成时间（每秒 5 个？）；

![[Pasted image 20250216115549.png]]
> Goodput (P90 TTFT < 200ms and P90 TPOT < 50ms) = maximum request rate per second when at least 90% of requests have both TTFT < 200ms and TPOT < 50ms

## Disaggregating Prefill and Decoding

1. **No interference between prefill and decode** makes both phases faster and easier to attain their respective SLO.
2. **Decoupled resource allocation and parallelism strategy** such that optimization can tailor for prefill and decode separately.
![[Pasted image 20250216120248.png]]![[Pasted image 20250216120302.png]]