[https://cilium.io/blog/2018/04/17/why-is-the-kernel-community-replacing-iptables/](https://cilium.io/blog/2018/04/17/why-is-the-kernel-community-replacing-iptables/)


- iptables 在过去一致用于 linux 的防火墙和 packet filter 场景；
- iptable 在 20 年前替代掉它的前任 ipchains 时，防火墙功能还比较简单：
	- 保护本地应用避免收到预期外的网络流量 INPUT chain
	- 保护本地应用避免发出预期外的网络流量 OUTPUT chain
	- 过滤收到、路由的网络流量 Forward chain
- 当时的网络速度还比较慢，拨号上网啥的；
- 这是 iptable 的设计背景；
- 各种 acl 的网络规则都顺序地遍历防火墙规则；
	- 依次顺序遍历，随着规则的增加时间线性增长；

The intermediate workaround: ipset
- 随着发展，网速变得越来越快，iptable 的规则膨胀到上千行；
- ipset 允许将多个规则压缩到一个 hash table 中；
- 不幸的是 ipset 并不能解决所有问题，一个例子是 kube-proxy；
	- kube-proxy 使用 iptable 的 -j NAT 来实现 service 的负载均衡；
	- 它为每个 service 安装多个 iptable rule；
	- 据说 service 太多，这个性能就慢慢不行了；
	- 还暴露出一个缺点：不能 incremental update；只能全量刷；
	- 20K 个 k8s service 的话，一下需要刷 160k 个 iptable rule，需花 5 小时；


The rise of BPF
- 过去依赖于内核开发、编译的任务，现在可以通过 BPF 安全的跑起来；
- 应用：
	- cilium：使用 BPF 做 L3~L7 的网络、安全、负载均衡；
	- facebook 在用 BPF / XDP 替换 IPVS；
	- netflix 的 Brenda Gregg 使用 BPF 做性能 profile、tracing；
	- google 在搞 bpfd；
	- cloudflare 在用 bpf 来防御 DDoS攻击；

How has the kernel community reacted?
- 有人在搞把 nftables 转换为 BPF 的东西；