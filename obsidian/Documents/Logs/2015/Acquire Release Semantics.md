[http://preshing.com/20120913/acquire-and-release-semantics/](http://preshing.com/20120913/acquire-and-release-semantics/)
[https://lwn.net/Articles/588300/](https://lwn.net/Articles/588300/)

- a read from memory with "acquire" semantics is guaranteed to happen before any subsequent reads or writes in the same thread.
- A write with "release" semantics will happen (become globally visible) after any preceding reads or writes.
- 但是，并不保证其它线程立即看到结果（可见性）；    
    [https://groups.google.com/forum/#!topic/lock-free/juGMIXTl4-E](https://groups.google.com/forum/#!topic/lock-free/juGMIXTl4-E)
- 满足 Acquire/Release 语义的环境中 Double-Check Locking 应该是安全的

Strong Memory Model

- x86 默认满足 Acquire/Release 语义，为此 x86 也被称作 Strong Memory Model    
    A strong hardware memory model is one in which every machine instruction comes implicitly with acquire and release semantics.
- LoadLoad, LoadStore, StoreStore 不乱序；只有 StoreLoad 可能被乱序；
- x86 环境下的 Double Check Locking 是安全的； 
- 在 PowerPC 平台，lwsync （Light Weight Sync）指令可以保证 LoadLoad, LoadStore, StoreStore 栅栏；而 sync 指令另加一个 StoreLoad 栅栏

LoadStore 栅栏

- 确保不会读到别人未 “flush” 的内容

StoreLoad 栅栏

- 最昂贵的一种栅栏：必须刷掉 Write Buffer
- 一般提供 StoreLoad 栅栏语义的指令，会同时提供所有的栅栏语义
- x86 中，任何 lock 前缀的指令可以视为一个 StoreLoad 栅栏
- jvm 中每个 volatile 写入之后都会跟一个 StoreLoad 栅栏