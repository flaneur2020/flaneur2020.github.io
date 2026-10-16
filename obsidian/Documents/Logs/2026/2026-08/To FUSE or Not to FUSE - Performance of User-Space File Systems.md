
## 2 FUSE Design

### 2.1 High-Level Architecture

![[Screenshot 2026-08-26 at 13.27.42.png]]

- fuse 的 kernel module 会注册一个 /dev/fuse 的 block device；这个 device 会被用于和 user space 的 fuse daemon 进行通信；
- daemon 可以从 /dev/fuse 读取 FUSE request，将 reply 写回到 /dev/fuse；
- 一些文件系统操作，可能不需要和 userspace 的 fuse daemon 进行通信，比如读取操作中，page 已经 cache 在了内核的 page cache 中，就不需要再发一趟了；

### 2.2 Implementation Details

![[Screenshot 2026-08-26 at 13.36.44.png]]

- kernel driver 与用户态 daemon 的所有交互，都来自于 kernel 发送的 FUSE Request 结构；
- 上面的表格包含了所有的操作类型，可见大部分操作都和 VFS 的操作是对应的；
- 在 fs mount 时，会调用 **INIT** 请求，用于交换一些 protocol version、capabilities 信息交换（比如 READDIRPLUS、FLOCK 等），设置一些参数（比如 FUSE 的 read-ahead size、time granulairty 等）；
- 反过来，在 umount 时，会调用 **DESTROY** 请求，做一些清理；
- 内核会在之前的某个请求不再需要时发送 **INTERRUPT** 请求，比如某个 user process 在等待 READ 时退出了；每个请求会携带一个唯一的 **sequence#** 作为请求标识，会被用于 **INTERRUPT** 用作标识；
- 每个 request 也有一个 **node** ID，标识 inode 号；path 到 inode 的翻译，来自 **LOOKUP** 请求；内核会记录一个 path 到 inode 的 cache；当 dcache 中移除了一个 inode 之后，内核会发送一个 **FORGET** 请求告知 user-space 的 daemon；**BATCH_FORGET** 允许内核单个请求批量 FORGET；
- **OPEN** 请求允许 daemon 分配一个 64 bit 的 file handle 给打开的文件；这个 handle 会返回给内核，之后所有相关的文件操作都会带着这个 handle，用户态的 daemon 可以利用这个 handle 来跟踪些元信息；在每个 open 的文件被关闭时，会调用 **FLUSH**，在没有打开的文件引用这个文件时，会调用 **RELEASE**；
- **OPENDIR** 和 **RELEASEDIR** 的语义与 OPEN/RELEASE 一致；**READDIRPLUS** 能够和 **READDIR** 一样返回多个 dirent，同时也能返回每个 entry 更多元信息，这允许内核 prefill 它的 inode cache；
- 当 kernel 判断一个用户态程序能否有权访问文件时，会发一个 **ACCESS** 请求；不过一般的用户在 mount 时就直接用 default_permissions 按 UNIX 的 attributes 来管理访问控制，这时就不会发 ACCESS 请求了；

![[Screenshot 2026-08-26 at 13.50.33.png]]

- FUSE library 逻辑上可以分为两部分，
	- Low-level API：解析来自 kernel 的请求、处理 reply 的序列化、处理文件系统的配置/挂载；隐藏 kernel 和 userspace 程序的潜在的版本差异；
	- High-level FUSE API：就是上面列的那些 API；
- FUSE 内部有五个队列：interrupts、forgets、pending、processing、background
	- 一个请求同一时刻只会存在于一个队列之中；此外，INTERRUPT 请求只会位于 interrupts 队列、FORGET 请求只会放在 forgets 队列；<mark>同步请求会放在 pending 队列</mark>；
	- 给用户态通知时，会按优先级排序一下；
	- 用户态处理中的请求，会体现在 processing 队列中；
	- background queue 会用在体现异步请求上；一般来讲，只有 read request 会进入到 background 队列；write 操作也会进入到 background queue，不过只有在 writeback cache 启用时；
	- <mark>fuse 默认会限制 background queue 的大小，暴露了一个 max_background 参数来配置，默认只有 12</mark>；只有 background queue 的异步请求少于 12 个时，background 队列里的请求才会放行到 pending 队列；
	- 当系统需要清理这些缓存时（由 `bdflush` 线程触发），FUSE 才会把这些写操作**转化为异步请求**，放入后台队列。
- Splicing and FUSE buffers.
	- 在基本款上，FUSE 进程需要处理 `read()` 请求，并将 write 的 reply 写回给 `/dev/fuse`；每次这样的操作，都需要内核与用户态之间的拷贝，这对 read 和 write 操作来讲是不小的负担；
	- FUSE 也支持利用 linux 内核的 splicing 方法，来在两个内核的 in-memory buffer 之间传输数据，而不必拷贝回用户态；
	- 为了支持 splicing，FUSE 将它的 buffer 表示为两种形态之一：
		- 通过指针指向的 user daemon 中普通的内存；
		- 一段使用 file descriptor 指向的 kernel space 内存；
	- user-space 的文件系统如果支持了 `wirte_buf()` 方法，FUSE 就会从 /dev/fuse 设备上 splice 数据，直接将数据传给这个文件描述符形态的 buffer；
	- FUSE 会在数据超过 1 个页面时，自动对 write 请求进行 splice；read 请求也类似，超过两个页面时，会自动处理；
