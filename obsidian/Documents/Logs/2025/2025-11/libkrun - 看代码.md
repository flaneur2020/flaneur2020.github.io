
libkrun 的定位似乎是提供一个 exec 入口，跑单个程序提供沙箱。

libkrun 大概两部分：

1. kvm / hvf 参数包装
2. virtio 设备模拟  

其中 virtio 设备模拟，似乎是来自 firecracker 的移植。

因为只支持 virtio 设备模拟，不支持 vfio-pci，因此不支持 GPU 直通。

**libkrun 的 api**

基本所有的 api 都是服务于配置，除了 krun_start_enter。

退出的话，vmm 进程退出，就等于 guest vm 退出了。

libkrun 会监听 sigint，收到这个信号的时候，会直接转发给 guest。

**创建 vm 的流程**

- krun_create_ctx: ctx 似乎只对应一个 ContextConfig
	- 用户可以针对这个 ctx 做一些配置，比如设置 rootfs 的路径(krun_set_root)、exec 的命令的入口（krun_set_exec）
- krun_start_enter 
- 创建 EventManager
- 加载内核 payload
- 内部主要干活的地方在 build_microvm

**对接 virtio 设备的流程**

- virtio 设备注册（MMIO）
- Guest 读取 virtio 设备特性位
- KVM 会给一个 ioeventfd，来通知消息
- KVM 会捕获 MMIO 访问

vmm 天然就可以访问 guest 的所有内存。不需要专门配置共享内存，只需要让 vmm 知道 guest 中的偏移即可。

那么，vmm 是怎样拿到 guest 中 virtqueue 的 ring buffer 的地址的？

这个是 guest 发起的，guest driver 通过 mmio 寄存器，向 vmm 返回 ring buffer 的起始地址。vmm 可以订阅 kvm 的一个 eventfd，来得到这个 mmio 中的信息。

**virtio-blk 为例**

- Guest 会填充 virtqueue，写入 avail ring，写 NOTIFY 寄存器触发
- host 端会收到 eventfd 的事件通知，消费队列里的任务，写入到 used ring
- 处理后，触发中断通知 guest