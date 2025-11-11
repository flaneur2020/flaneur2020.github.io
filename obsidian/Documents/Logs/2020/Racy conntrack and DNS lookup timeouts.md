https://www.weave.works/blog/racy-conntrack-and-dns-lookup-timeouts

- iptable 模式下 kube-proxy 会为每个 service 创建几个 iptable 规则搞 nat；
- /etc/resolve.conf 中 nameserver 10.96.0.10 指向的 kube-dns 做 service 是个 cluster IP；
- DNAT in Linux Kernel：修改包出包的地址；

Problem
- 问题出同一个 socket 在不同的线程中出两个 udp 包；
- udp 没有连接和 connection 阶段，因此提前没有 conntrack 表项；
- 会导致一个 race：在 nf_contrack_in 阶段任何一个 packet 都没有找到 conntrack，两个包都创建了同样的 tuple；
- race 的两个包其中的一个会被丢掉；
- gnu c 会同时发起 A 和 AAAA DNS 的查询；其中的一条查询会被 drop 掉，客户端会在超时 5s 后重试；

Mitigations
- workaround 可以是在 dns resolver 中设置一个真实的 dns server IP；
- 可以考虑改 tcp 查询；
- ipvs 也解决不了这个问题，也是一样的 conntrack；
- conntrack 各种 racy；