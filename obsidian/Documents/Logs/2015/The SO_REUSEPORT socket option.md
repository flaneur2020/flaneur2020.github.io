[https://lwn.net/Articles/542629/](https://lwn.net/Articles/542629/)
[http://thread.gmane.org/gmane.linux.network/102140/focus=102150](http://thread.gmane.org/gmane.linux.network/102140/focus=102150)

- setsockopt(sfd, SOL_SOCKET, SO_REUSEPORT, &optval, sizeof(optval));
- 第一个使用 SO_REUSEPORT 来 bind 的端口可以被多个进程 reuse；
- 为了防止 hijack，其后 reuse 该端口的进程必须拥有同样的 effective uid；
- 对 tcp 和 udp 都可用；面向 tcp 在连接级别均衡；面向 udp 在报文级别均衡；
- 传统的 SO_REUSEADDR 可以允许绑定多个 udp 连接到同一端口；然而缺少安全验证； 
    

背景

- 作者 Tom 在 Google 任职，他们的场景中每秒会 accept 40,000 个连接；
- 早期的连接均衡方案是使用一个单独的线程来 accept 并做分发；这容易使该线程称为瓶颈，而且该线程位于一个 CPU，与其他 CPU 通信存在开销；
- 如果使用多个线程来 accept，可以解决一定问题，但是 accept() 会使首个被唤醒的线程接受新链接，而这又容易使调度器的 bias 影响，使连接的分布不理想；
- Rick Jones 提到 NIC 的 multiqueue 特性，可以为 NIC 设置多个端口，使 NIC 将特定端口的中断发送给特定的 CPU，而 packet 可以在本 CPU 就地处理；
- Tom 提到这正是他希望的，不过最好绑定一个端口来分发；
    

补丁

- Tom 的补丁按 <源ip，源端口，本机ip，本机端口> 这个四元组来计算哈希；这意味着同一个来源会分派到一个固定的进程上，做有状态的通信会方便一些；
- 不过最初的补丁有 defect，当添加、关闭进程时，哈希会重排，导致握手到一半的连接 reset；