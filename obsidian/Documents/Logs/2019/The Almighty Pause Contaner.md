https://www.ianlewis.org/en/almighty-pause-container

- 创建一个 parent container，来创建新的 container，然后管理这些 container 的生命周期；
- k8s 中 pause container 扮演了 parent container 的角色；pause container 有两种角色：
	- 解决 namespace sharing；
	- 做 PID 1 来回收僵尸进程；

Sharing namespaces
- 作为 parent 创建下层的 ns；

Reaping Zombies
- linux 中 pid namespace 的进程树中的进程都有个 parent 除了 init 进程即 pid 1；
- 当父进程在子进程完成后不调用wait的syscall时，就会出现生存时间较长的僵死进程。一种情况是，父进程编写得很差，并且简单地忽略了wait调用，或者父进程在子进程之前死亡，而新的父进程没有调用wait。
- 当进程的父进程在子进程之前死亡时，操作系统将子进程分配给“init”进程或PID 1。这意味着，现在当子进程退出时，新的父进程(init)必须调用wait获取它的退出代码，否则它的进程表项将永远保持不变，变成僵尸。

Some Context on PID Namespace Sharing
- Reaping zombies is only done by the pause container if you have PID namespace sharing enabled, and currently it is only available in Kubernetes 1.7+.
- 如果 pid namespace sharing 特性没有开启，则需要每个容器自己回收 zombie 进程；
- PID namespace sharing 允许 pod 中的进程相互发信号；