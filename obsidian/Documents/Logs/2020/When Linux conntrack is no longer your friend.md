https://www.projectcalico.org/when-linux-conntrack-is-no-longer-your-friend/

- conntrack 机制：跟踪所有网络连接；允许跟踪网络流中的包；
- NAT 依赖于 connection tracking，用于改包；比如 kube-proxy 会用 NAT 将包重定向到目标 pod，连接中返回的包也要 NAT 改回来；
- 有状态防火墙，如 calico，使用 conntrack 信息来对响应流量做白名单；

So, where does it break down?

- conntrack 表有可配置的最大上限，如果溢出，连接及会被拒绝或者丢弃 ，导致丢包；
- 对大多数场景来说，表的上限不是问题，然而存在一些场景需要额外关心 conntrack 表：
	- 有极多的并发连接数，如果配置了 conntrack 表的上限 128k，有大于 > 128k 个并发连接的话，必然有问题；
	- 每秒的连接数特别多，也有可能有问题；即使连接比较短命，linux 也会跟踪较长时间，如 120s，这就导致 conntrack table 溢出(128k / 120s = 1092 connections/s)；

A real-world example

- 一个具体的例子，有个 SaaS 客户有一组 memcached 服务器跑在 bare mental 服务器上，每台负载 5w 条短连接；这个量级是超过标准 linux 可以对付的；
- 他们尝试调优 conntrack 配置，增加 table size，减少 timeout；但是这个 tuning 很脆弱，会显著增加 RAM 占用（几 Gb！）；生命周期短的连接，使用 conntrack 没有性能优势（占用 cpu、提高包的延时）；
- 最后他们用了 calico，calico 的 network policy 允许为特定负载绕过 conntrack（doNotTrack）；

What are the trade-offs of bypassing conntrack?