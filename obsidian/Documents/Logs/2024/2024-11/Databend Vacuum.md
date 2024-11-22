![[Pasted image 20241106173602.png]]

- 每个 snapshot 一个单调递增的时间戳；
- 每个 data file 绑定这个 snapshot 的时间戳，并将这个时间戳编码到文件名中，允许按字典序 list；
- 在执行 vacuum 时，找一个 t1 时刻的 snapshot 作为 gc root，<mark>不需要扫描 t1 时刻之后的 snapshot</mark>。
- 扫描 t1 时刻的 snapshot 引用的文件列表 S1；
- 扫描 t1 时刻之前的所有 data file，若该文件不存在于 S1，则可以清理；

问题：
- 为什么只扫描一个 snapshot 作为 gc root 就可以？
	- t1 时刻之后的 snapshot 不会引用到 t1 时刻的 snapshot 没有引用的更老的文件；

