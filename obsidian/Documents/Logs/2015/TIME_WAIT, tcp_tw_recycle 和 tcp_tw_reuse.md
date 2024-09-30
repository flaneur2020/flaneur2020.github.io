[http://stackoverflow.com/questions/8893888/dropping-of-connections-with-tcp-tw-recycle](http://stackoverflow.com/questions/8893888/dropping-of-connections-with-tcp-tw-recycle)
[http://wiki.jokeru.ro/howto-tcp_tw_recycle-negative-effect-on-nat](http://wiki.jokeru.ro/howto-tcp_tw_recycle-negative-effect-on-nat)
[http://vincent.bernat.im/en/blog/2014-tcp-time-wait-state-linux.html](http://vincent.bernat.im/en/blog/2014-tcp-time-wait-state-linux.html)
[http://www.pagefault.info/?p=416](http://www.pagefault.info/?p=416)
[http://serverfault.com/questions/342741/what-are-the-ramifications-of-setting-tcp-tw-recycle-reuse-to-1](http://serverfault.com/questions/342741/what-are-the-ramifications-of-setting-tcp-tw-recycle-reuse-to-1)

- TIME_WAIT: tcp/ip 报文会**乱序到达**，关闭连接之后，可能依然会有报文来到本地；如果在同一个端口上新开了连接，可能会被乱入；为了避免同一个端口的新连接接到来自旧连接的报文，在关闭一个连接之后，等待 IP 报文生命周期的两倍时间，才能重用这个端口；<mark>只有主动关闭连接的一方会进入 TIME_WAIT</mark>；
- TIME_WAIT 带来的问题：Connection Table Slot 占用：内核默认限制了本地临时端口号的范围 net.ipv4.ip_local_port_range ，如果大量短连接，则会把可用的端口号吃光，而发生 **EADDRNOTAVAIL** 报错；内存占用和CPU占用：内存占用比较小倒是；建立新连接分配端口号时，要多扫一下； 
- tcp_tw_reuse: <mark>只影响 outgoing connection；要求双端都开启 tcp_timestamps 选项才可行</mark>；握手时内核会检查 tcp 报文的时间戳，如果大于上一个连接的时间戳，则允许重用 TIME_AWAIT 状态的端口；只要开启 tcp_timestamps， 就可以确保没有问题；但是这要求网络的两端都开启 tcp_timestamps；
- tcp_tw_recycle：<mark>影响 ingoing connection 和 outgoing connection; 更激进，假定对方也开了 tcp_timestamps</mark>；只要握手的序列号递增，就可以重用该端口；但是，如果对方同源 IP 发报文的序列号变化，比如发生回退，则 SYN 报文就会被丢掉；NAT 用户可能会发生丢包：同一个 IP 后面是多个用户，他们会存在不同的序列号，发生回退现象而丢包；<mark>不要开这个选项</mark>；