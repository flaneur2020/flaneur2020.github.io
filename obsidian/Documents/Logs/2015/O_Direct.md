
https://groups.google.com/forum/#!topic/linux.kernel/5ubALxxiQH8%5B1-25%5D
http://www.reddit.com/r/linux/comments/1j7fxn/thanks_linus_for_keeping_o_direct/
http://yarchive.net/comp/linux/o_direct.html

- O_DIRECT 要想能有用，必须配合 aio；而 aio 的存在在 linus 看来只是为了掩盖 O_DIRECT 这个 broken 的接口造成的问题； 
	- 目前 linux 的 aio 只适用于 Direct IO；
- mmap() 的系列操作可以不碰页表；在执行 mmap() 时，内核只修改 vma 而已；在随后的 mwrite() 等操作上，内核会检查 vma 背后绑定的文件描述符，直接走物理内存；可以惰性地留到必要访问内存时再修改页表；
- 很多时候 mmap 的表现并不比阻塞 IO 更好： 
	- Page Table Walking 一般比 memcpy 更慢，而且拷贝意味着省去协调的成本；
	- 阻塞 IO 的 buffer 一般能驻留在 L1 缓存中，memcpy() 会非常快；
- Page Cache 不只是缓存而已，而是扮演多个角色：1. 阻塞 IO 的缓冲区；2. 一个同步的实体，mmap 映射的数据总是一致的；3. 缓存； 
	- 如果只是想跳过缓存，可以使用 posix_fadvise(fd, 0, 0, POSIX_FADV_NOREUSE);
O_DIRECT 在语义上有问题：
- O_DIRECT 只支持固定大小的文件，不能 ftruncate，也不能增加文件长度，这只适用于一个固定容量的场景，把文件当作 raw device； 
	- O_DIRECT 的代码路径完全不同，影响到更合理的接口的实现；
- 一旦使用 O_DIRECT，那么开发者必须自己去 guarantee 数据的一致性，而忘记了这本来是内核的责任；
- linus：如果遇到缓存上的问题，那么应该把缓存刷一致，而不是绕过缓存；
- bad design of O_DIRECT causing the app to have to care about something _else_ it shouldn't care about. 
- If you have issues with caching (but still have to allow it for other things), the way to fix them is not to make uncached accesses, it's to force the cache to be serialized. That's very fundamentally true.
- There's tons of races where an O_DIRECT user (or other users that expect to see the O_DIRECT data) will now see the wrong data