- 从内核到 daemon 的请求创建与复制，占用了 4kb 写入时间的 18.9%，而从 daemon 到内核的响应复制，则额外占用了 23.5%；
- 在过去有很多技术用来提升 FUSE 的性能：
	- 通过共享内存的 ring-based bufferer，减少拷贝和 daemon 和内核上下文切换的开销；（rfuse 这个论文，应该没搞起来）
	- 一些提高性能的内核内的 IO 原语；（Extension Framework for File Systems in User Space）
	- 通过 Splice，加速 fs 和 FUSE module 之间的通信；（To Fuse or Not to FUSE: Performance of User-Space File Systems）
	- 让 FUSE module 通过一个基于内核的文件系统进行通信，而不引进 FUSE daemon；（Optimizing Local File Accesses for FUSE-based Distributed Storage）
- 作者做了一个 MemFS 的 dummy 的 FUSE 文件系统，它就直接将内存中的数据拷贝；这样的 fuse 文件系统，也有两次 memory copy；将这个文件系统作为基线，作者认为数据的拷贝确实是开销的一大来源；

## 2.1 The FUSE framework

- FUSE 有两部分，一部分是 FUSE kernel module，与 Linux VFS 对接，对外暴露一个 FUSE 设备（/dev/fuse）；
- 用户态的进程，通过 libfuse 库实现对接；这个库提供了一套底层 API，处理来的请求，通过这个 device 和内核交流；daemon 处理请求的方式，可以是按 stacked 方式在内核中直接请求下层的文件系统，或者与 userspace 的 daemon 对接；
- 一个到达的请求，如果已在 page cache 中，则直接可以返回给 user；没有命中缓存的请求再转给 userspace 的 daemon；
- Default Interface：fuse 模块从 VFS 接受 IO 请求；在 read 和 write 请求中，FUSE 模块会从 userspace 文件系统的 buffer cache 中移动数据，或者从 stacked 文件系统的 page cache 中移动数据；
	- 移动期间，数据会在两个更多的地方暂存，其一是 FUSE 模块的 page cache，其二是进城自己的 buffer cache；
	- 在写入时，如果是 write through 模式，就是同步写的；如果是异步在 write-back 模式中，则看一个 dirty ratio 或者 timeout；
	- 在 direct IO 模式中，内核会 bypass FUSE module 的 page cache，直接从应用程序和 daemon buffer 之间拷贝；
	- 也就是说，默认的模式下，有两次拷贝；
- Splice Interface：
	- Linux 的 splice 能够做到让一个 pipe，从 user buffer 或者文件描述符之间移动数据；管道被视为页指针的环形数组，splice 可以简单地向页面添加额外的指针，即可经过管道传输数据；
	- 这允许 FUSE 跳过应用程序和 page cache 之间拷贝数据；
	- 在读写中，data size 应当大于 1 或者 2 个页面；daemon 应当实现 read/write_buf 函数，来取代常规的 read/write 函数；
	- `_buf` 意味着这是一个通用缓冲区，可以选择文件描述符而非内存地址作为 src 或者目标；如果守护进程检测到支持 splice，它就会创建一个管道，用于在 FUSE 设备与守护进程之间传输；数据传输可以跳过守护进程的缓冲区，直接在管道对端的文件描述符和 stacked 的文件系统的文件之间进行；<mark>这要求 stacked 文件系统支持页面移动语义，从而避免通过管道进行拷贝</mark>。这一来，应用程序缓冲区与 stacked 文件系统之间的路径上，只需要一次拷贝。
- Passthrough
	- 将请求直接转发给 stacked 文件系统；