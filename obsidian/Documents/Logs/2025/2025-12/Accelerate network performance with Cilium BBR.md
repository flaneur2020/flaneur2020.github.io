传统的 TCP 拥塞控制已经有 30 多年的历史，大多基于“丢包”（loss-based）进行反应。

如果遇到丢包，会使流量异常地大幅减少。

Google 开发的 BBR 算法属于 avoidance based，识别网络瓶颈，自动调整发送频率，最小化队列中的排队延迟。

主动测量网络的瓶颈带宽（BtlBw）和往返传播时延（RTprop）：
- [ ] 
1. BtlBw：数据发送路径上的最大可用带宽
2. RTprop：数据包在不排队的情况下的往返传播时延

主动探测机制：

周期性地改变发送速率，去感知上面两个参数的变化。

- 周期性地（比如 8 个 RTT）将发送速率提高到高于当前估计的 BtlBw（一般是 25%？）持续一段时间（比如 1RTT）
- 周期行地（比如 10s）将发送速率降低（比如 4 个报文 / rtt）

BBR 是状态机驱动的：

1. STARTUP：指数增长发送速率，直到送达率不再显著增长，然后进入 DRAIN 排空队列
2. DRAIN：降低发送速率，排空在 START UP 阶段产生的队列积压
3. PROBE_BW：大部分时间在这个阶段，周期性地探测带宽
4. PROBE_RTT：每 10s 进入一次

不过 BBR 在 k8s 中实现时，会遇到时间戳方面的困难

> Fair Queues (FQ) schedulers need packet timestamps to determine how to pace the traffic to a given rate.
> 
> The Linux kernel would clear packet timestamps as packets traversed network namespace.

Linux内核5.18+版本会保留时间戳，不再清除。