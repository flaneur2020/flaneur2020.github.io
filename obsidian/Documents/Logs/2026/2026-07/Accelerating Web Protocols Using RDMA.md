
tldr：

- 07 年的老论文；
- 思路大概和 cuObject 差不多：
	1. 两端先建立 RDMA 连接，注册好内存，再发 HTTP 请求
	2. 在 HTTP 的 GET 请求中带一个 rdma header
	3. 服务端完成 rdma 传输后，返回 HTTP 请求响应通知传输完成