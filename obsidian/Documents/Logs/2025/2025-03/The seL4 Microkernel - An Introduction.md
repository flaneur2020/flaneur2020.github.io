
## Chapter 1: What's seL4

是个微内核，也是个 hypervisor。可以跑一个完整的 Linux guest OS。seL4 有一套可靠的通信 channel，guest 和应用程序可以相互通信。

seL4 是 proved correct 的且 secure 的。有精细的权限控制能力。

s3L4 ensures safety of time-critical systems，有实时性保障，能保障 worst-case execution time。这意味着如果 kernel 配置正确，则所有的操作都能在有限时间内完成。

seL4 也是世界上最快的 microkernel。

## Chapter 2: seL4 is a microkernel and a hypervisor, it's not an OS

![[Screenshot 2025-03-26 at 10.37.10.png]]

Linux 有两千万行代码，可以认为 Linux 有一个很大的  Trusted Computing Base。

在 microkernel 的设计中，在特权态的代码是极简的。只有 1w 行代码。

microkernel 几乎没有提供任何 service，只是对硬件的一个简单包装，足够对硬件安全地 multiplex 即可。

microkernel 最多提供个 isolation、sandbox 能力，让程序不相互影响。

而且，它会提供一个 protected procedure call （PPC）的机制，能够允许一个程序去调用另一个沙盒中的程序。

微内核的服务都是来自用户态的组件，这些组件仍大多需要来自其他系统的 port。最重要的组件一般是 driver、网络协议栈和文件系统。

即使和其他 microkernel 相比，seL4 的 api 也是非常底层的。因此，从 seL4 上构造系统通常很困难。

一般的开发者，不会直接在 seL4 上构建系统，而是基于一个 framework 进行开发。

seL4 上面有三套常用的组件框架：microkit、camkes 和 genode。

### 2.3 seL4 is also a hypervisor

可以在 seL4 上跑 VM，在 VM 里跑 linux。

![[Screenshot 2025-03-26 at 21.56.03.png]]

protocol stack 可以与 linux 通过 seL4 的 channel 进行通信。

