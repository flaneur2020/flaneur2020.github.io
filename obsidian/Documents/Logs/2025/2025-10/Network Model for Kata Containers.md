
传统的方法，会通过 veth 设备来打通宿主 namespace 和容器 namespace；

宿主机的设备一般是一个 bridge。

对于 hypervisor based 容器来讲，这个方法会变得不再适用。

kata 的办法是创建一个 TAP 设备到虚拟机中。

kata 容器会使用 TC filter 来从 veth 设备和 TAP 设备之间进行通信。

![[Screenshot 2025-10-22 at 10.38.57.png]]

（宿主机上还是常规的 CNI 网络插件，做 veth pair，kata 做的事情是将容器 namespace 内的 eth0 接到 TAP 设备上，该 TAP 设备再接到 qemu 内的 virtio 网络设备上。）

kata 会在 vm 中创建一个 tap0_data 的 tap 设备，和虚拟机内的 eth0 做一个双向的绑定。

（有一个论文做 benchmark 好像 kata 这个 TC rule + tap 的开销还不小，吞吐可以到 10x 的水平）