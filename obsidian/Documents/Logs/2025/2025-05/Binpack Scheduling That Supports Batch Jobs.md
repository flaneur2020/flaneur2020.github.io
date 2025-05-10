## Why Binpack?

k8s 默认采用 `LeastRequestedPriority` 的策略来调度 Pod，优先调度到空闲资源最多的 Node 上，尽量去均匀地调度资源。

但是这个调度策略可能有一些 fragement 出现。

![[Pasted image 20250501185931.png]]

更好的办法是使用一个 binpack 策略：

![[Pasted image 20250501185944.png]]