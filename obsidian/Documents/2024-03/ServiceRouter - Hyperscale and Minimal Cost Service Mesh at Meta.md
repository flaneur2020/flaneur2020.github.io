https://www.micahlerner.com/2024/03/28/servicerouter-hyperscale-and-minimal-cost-service-mesh-at-meta.html

meta 内部有一个 ServiceRouter，与 istio、linkerd、envoy 等组件不同；

1. 能够 embed 在程序内部，减少跑单独的 sidecar 的成本，作者说一个单独 sidecar 形式的 service router 会需要 1750000 个 t4g.small 的实例；
3. service router 好像是巨大 scale 的第一个被讨论的 RPC routing infra；
4. service router 可以处理 sharded stateful 服务
5. 使用 latency ring 的思路来处理跨 region 的流量

## How does the system work?

> There are three main functions of ServiceRouter:

> - Gathering the data that informs how services talk to each other.
> - Distributing that data reliably around the network.
> - Routing a request from a service to another service.

有一个 Routing Infomation Base 的概念。

![[Pasted image 20240330224333.png]]
![[Pasted image 20240330224340.png]]!

![[Pasted image 20240330224421.png]]

![[Pasted image 20240330224559.png]]

有多种部署模式，可以直连，也可以作为代理。

其中 SRLib 就是一个 library。

在跨 region 通信时，SRLib 的直连形式并不最优。这时不如有一个更集中的 proxy 来作为网关来维持长连接，减少频繁 RPC 的开销。

![[Pasted image 20240330224916.png]]

ServiceRouter 也支持完全无法内嵌 SRLib 的语言比如 erlang 来接入这套 infra 的方法。 

## Load Balancing

> One of the most novel features of ServiceRouter is its approach to global load balancing traffic across regions.

> An RPC client uses cross-region RTTs to estimate its latency to different servers. Starting from ring1, if the client finds any RPC server whose latency is within the latency bound for ring i, it filters out all servers in ring i+1 and above, and randomly samples two servers from ring i. If the service has no servers in ring i, it considers servers in ring i+1, and so forth. SR’s default setting maps 


> ServiceRouter integrates another input to the Routing Information Base - the load of a “locality ring”. This data allows ServiceRouter to support functionality like “route X% of traffic to this locality ring until the load of that locality ring exceeds X%, then send traffic to the next locality ring.” This is particuarly useful during incidents, where traffic can spill across multiple regions.
