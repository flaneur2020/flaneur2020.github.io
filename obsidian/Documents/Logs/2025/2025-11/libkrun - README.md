支持 linux（KVM）和 Mac（HVF）。

它集成了一个 vmm，只保留最少化的 emulated devices，对外提供最小化的 C API。

## Virtio device support

- virtio-console
- virtio-block
- virtio-fs
- virtio-gpu
- virtio-net
- virtio-vsock
- virtio-balloon
- virtio-rng
- virtio-snd

## Networking

### virtio-vsock + TSI

**Transparent Socket Impersonation**，允许 VM 直接连接网络，而不需要一个 virtual interface。

在 VM 中没有网络 interface 时，对 AF_INET 和 AF_INET6 的 TSI 会自动开启。

这个需要一个 customized kernel。

### virtio-net + passt/gvproxy

听起来类似 tap 那套。

## Security model

需要将 VMM 和 guest OS 看做同等的安全领域。要实现隔离，还需要在 host 系统上，对 VMM 进程进行限制。

在 Linux 系统中，这套机制就是 namespace。

### virtio-fs

在使用 krun_set_root 和 krun_add_virtiofs 访问宿主机的 fs 时，libkrun 没有增加任何的额外保护。比如去访问同一 fs 的其他文件，甚至 host 其他文件系统的文件。

> A mount point isolation mechanism from the host should be used in combination with virtio-fs.

另外，在使用 virtio-fs 时，guest 也有可能会消耗光宿主机的磁盘空间、inode 空间等。