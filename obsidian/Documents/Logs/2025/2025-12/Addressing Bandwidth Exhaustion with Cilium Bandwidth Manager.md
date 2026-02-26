## Congestion Handling

目前 CNI 带宽控制有以下挑战：

1. 在 ingress 侧，CNI 带宽控制插件实现了在 veth 上的 queuing 机制，已经进入到 host 的包会排在队列中，等待 qdisc 才会进入 pod；qdis 是一个令牌桶算法（Token Bucket Filter）；![[Pasted image 20251205135906.png]]
2. 令牌桶大致上对应字节数，一开始 TBF 内部会装满字节，以应对一次突发的流量请求；令牌以恒定的速率（比如 10mb/s）补充，直到桶满为止；TBF 的后果是：它通过等待攒够令牌才发送数据包，从而增加了延迟；这个算法也不是为多核心设计的，虽然网卡支持多队列，但是所有流量都会走到这个令牌桶；最终容易碰到单点串行化瓶颈；
3. 在 egress 侧，CNI 插件需要 work around 一下，避免 TBF 在流量 shaping 上的限制；linux 只能在 egress 侧实现 traffic shaping，流量会被转发到 Intermediate Functional Block 来应用 traffic shaping；为了限速，就得在网络中增加一条![[Pasted image 20251205140357.png]]

总结下：

- 传统用 CNI 插件（如 TBF）限速 Pod 带宽，会人为插入网络跳点，增加延迟、消耗 CPU、浪费现代网卡的多队列性能，得不偿失。  
- 真正的解法是用 **eBPF + XDP** —— 在网卡驱动层直接限速，没有额外跳点、零拷贝、多核并行，不拖慢网络、不耗 CPU。

## Introducing the Timing Wheels

Van Jacobson 是网络界的传奇人物，他合作发明了很多耳熟能详的工具比如 traceroute、BPF-based tcpdump，也对 TCP 协议本身做了很多贡献。比如 Congestion Avoidance and Control 论文对 TCP 的拥塞控制的发展做了很大贡献。

Van 也曾经提到时间轮可以是相比队列更优的 congestion 控制方法。

基本思路是放一个 _Earliest Departure Time_（EDT）时间戳在 packet 中。根据这个时间戳来发送包。

![[Pasted image 20251205141020.png]]
## Cilium's Bandwidth Manager Implementation

1. cilium 观察 pod annotation 得到 bandwidth 信息；
2. cilium agent 将 bandwidth 需求推到 eBPF 数据面；
3. 流量的 enforcement 发生在硬件设备环节，而非 veth；
4. cilium agent 基于用户的 bandwidth policy 需求实现 EDT timestamp；
5. cilium 对多队列有感知，能够自动设置 multi-queue qdisc，根据 fair queue leaf qdiscs
6. fair queue 实现时间轮机制，根据 packet 的时间戳来分散 traffic；

