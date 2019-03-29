---
layout: post
title: "Linux 容器网络笔记"
---

容器网络相比过去虚拟机的网络虚拟化有一些变化，过去虚拟机的网络虚拟化，要模拟的是 NIC 设备，要模拟虚拟网卡的硬件细节。到容器时代，网络虚拟化会更多地重用 linux 自身的网络设备，可以在协议栈的三层路由掉，不需要从最底层走一遍硬件模拟。再就是网络的扩展性问题会更上台面，比如虚拟机时代虚拟网络端口的数量没有容器那么多，虚拟主机的上下线也并不频繁，二层网络方案应对虚拟机绰绰有余，但是对较大规模的容器网络可能便不够了。

## L2 bridging: 接入广播域

二层网络接入较简单，几乎是自发现的，将网络端点通过网桥接入广播域即可。交换机自动将包广播到新发现的网络端点，如果对方端点有响应，则记录到地址学习的数据库 fdb 中，随后便可以通过 MAC 地址做点对点的交换通信。

交换机比较怕端口桥接成环，以太网帧也不像 IP packet 带有 TTL，生命周期有限不那么很怕路由环路，而二层环路会将帧无穷无尽地转发下去。为了避免出环，交换机一般实现 STP 协议，交换拓扑信息生成大家认可的唯一拓扑。

Linux 有 bridge 设备与物理的交换机大致上等价，有 flooding、地址学习、STP 的实现。但也有一点差异，便是交换机可以只专注转发，而 Linux 还需要做包括 socket 栈在内的三层往上的协议栈支持。这里有一个细节是 Linux 不允许为接入 bridge 设备的网络设备配置 IP。因为接入 bridge 设备的网络设备会将二层流量无脑转发给 bridge，它自身是无暇理解响应三层流量的，配 IP 也没有意义。那么问题来了，主机的网络接入设备 eth0 接入了 bridge 设备，不能配置 IP，那么 Linux 主机怎样响应 socket 流量？这时可以将 IP 配置给 bridge 设备，bridge 设备对收到的帧做三层解析，替代 eth0 设备向上做三层协议处理，相当于 bridge 设备有一个隐含的端口，对接到 Linux 的三层栈。

通过 bridge，可以做最简单的独立 ip 的容器网络：

<img src="/images/note-on-linux-container-network/br_ns.png"/>

容器的虚拟网络设备可以 veth pair 穿透 namespace，再透过 bridge 接入广播域。一旦接入广播域，便可以通过对 ARP 广播的响应声明自己的 IP。

### macvlan & ipvlan l2

macvlan 可以理解成对 bridge 的简化，这句话 macvlan 的概括比较好：

> The macvlan is a trivial bridge that doesn’t need to do learning as it knows every mac address it can receive, so it doesn’t need to implement learning or stp. Which makes it simple stupid and and fast. ——Eric W. Biederman

macvlan 允许单个物理设备划分成多个独立 mac 地址的虚拟设备。因为地址是已知的，因此不必地址学习，因为不会产生环，也不必 STP。

<img src="/images/note-on-linux-container-network/macvlan.png"/>

单端口多 MAC 地址可能在网络环境中支持不好，比如有些交换机对单端口的 MAC 地址个数有限制，这是公有云的用户不可控的。

ipvlan L2 稍稍跳到二层往上，将 IP 地址作为区分网络设备的标记，允许一个 MAC 地址背后多个 IP 地址。

<img src="/images/note-on-linux-container-network/ipvlan.png"/>

L2 容器网络的问题很明显：

* 广播范围大，假如 100 台物理机每台 100 个容器，广播的范围便是 100 x 100 + 100；
* 抖动大，容器上下线的频次非物理机/虚拟机可比，每变动一个容器会 ARP 风暴所有主机的所有容器；
* STP 是为了保护 L2 网络免受环路通信的影响，然而在极大规模的 L2 网络下，STP 本身也能成为瓶颈；

## L3 Routing

当 L2 网络遭遇扩展性问题时，必须将网络划分网段以隔离广播域，通过 IP 协议来跨网段通信。 网段内的一台主机，对网段外有多少主机上线下线是可以不 care 的，只需要给目标网段的路由器发包即可。

三容器上下线频次高，但是物理机上下线的频次要少得多。层容器网络方案便是将物理机看作一个网段的网关，外部不需要关心网段内部容器的上下线。至于网关的上下线，通过路由协议或者宿主机上跑的 daemon 来同步路由表。像 flannel 的 host-gw 模式，会在每台物理机上跑 flanneld 侦听 etcd 中路由表的变化，同步到本地路由表：

<img src="/images/note-on-linux-container-network/host-gw.png"/>

## References

* [https://tools.ietf.org/html/rfc1180](https://tools.ietf.org/html/rfc1180)
* [Introduction to Linux interfaces for virtual networking](https://developers.redhat.com/blog/2018/10/22/introduction-to-linux-interfaces-for-virtual-networking/)
* [Calico over an Ethernet interconnect fabric](https://docs.projectcalico.org/v3.5/reference/private-cloud/l2-interconnect-fabric)

