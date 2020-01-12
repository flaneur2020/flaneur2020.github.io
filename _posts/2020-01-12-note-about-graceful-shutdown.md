---
layout: post
title: "Graceful Shutdown 考察"
---

Graceful Shutdown 按说是一个 solved problem，尤其是 HTTP 通信，这部分细节都交由应用服务器能完整地处理掉。不过一直没有仔细想过 tcp 长连接场景下优雅停机的实现方式，在这里考察一下。

## gunicorn 的 Graceful Shutdown

先温习一下 HTTP 短连接场景下 Graceful Shutdown 的流程。gunicorn 的 sync 模式是典型的 prefork 模型，启动多个 worker，父进程初始化 listener，使每个 worker 争抢 accept(listener)。

graceful shutdown 的入口在 arbiter.py 的 stop() 方法，会先关闭 LISTENERS，随后向 worker 们传递 term 信号。

gunicorn 的每一种 worker 继承自 worker/base.py 的 Worker 类，Worker 基类中定义了 alive 变量，收到 sigterm 时简单将 alive 变量设置为 False 即完成信号处理。

在 worker/sync.py 的 SyncWorker 中，主流程大约是 `run_for_one()` 中的 `while self.alive:... self.accept(listener)`。如果 alive 变为了 False，处理完当前请求后便不再 accept() 新请求，也即退出主循环。

可以简单总结一下 HTTP 短连接的 graceful shutdown 流程：

1. 关掉 listenfd，便能使前端的负载均衡器感知到这个后端要下掉了，再尝试分发请求会 Connection Refused，可以安全选另一个节点重试；
2. 每个 worker 处理完当前请求后，发现进入退出状态，不再 accept 新请求、退出；
3. 如果 worker 未能在正常时间内退出，再 kill -9 强杀；

## Keep-Alive 的 Graceful Shutdown 流程

按说 Keep-Alive 的 Graceful Shutdown 流程与上面短连接的流程是相似的：关闭 listenfd 之后，等待所有活跃的连接关闭后退出 worker，而 keep-alive 有时间上限，超过时间上限后应退出。

但是 keep alive 存活时间可能非常长，有没有更及时的办法实现退出？

HTTP/1.1 在响应中有定义一个 Connection: close 的 header，向客户端告知这个请求之后，服务端会关闭连接，请客户端另请高明。

这一来可以走这样的流程：

1. 关掉 listenfd 停止接收新连接；
2. 每个 worker 处理完当前请求后，发现进入退出状态，在响应 header 中返回 Connection: close，随后关闭连接；
3. 待连接关闭后，退出 worker；
4. 如果 worker 未能在正常时间内退出，再 kill -9 强杀；

这一流程会存在一个问题：请求前来的时间是不可预期的，如果连接中没有请求前来，则 keep alive timeout 之前走不到 Connection: close 而一直不能退出；如果请求在很久之后才来，这时容易给人 surprise，因为这时执行的仍然是上个版本的老代码。gunicorn 作者提到自己更倾向于开始 graceful shutdown 后，将 idle 状态的连接及时关掉的行为，避免上线一段时间后老代码仍诈尸一下的这种行为。

## grpc 的 Graceful Shutdown

grpc 的 server 对象内建了一个 [GracefulStop()](https://github.com/grpc/grpc-go/blob/master/server.go#L1448) 方法，还没仔细看，不过大约好像是：

1. 关掉 listenfd 停止接收新连接；
2. 通知 worker 停止，向客户端推送 goaway 的 http2 Frame 通知客户端要关闭；
3. 客户端收到 goaway 后关闭连接；
4. 等待所有连接关闭后，退出；

相对于 HTTP/1.1 的 keep alive 的流程，grpc 优势在于 HTTP/2 可以主动推送 GOAWAY 这种控制消息，不用太纠结空闲连接这种情况。改天详细看一下。

## 长连接 RPC 协议的 Graceful Shutdown 流程

个人感觉 RPC 的通信层还是建立在稳定的七层协议上比较稳妥，奈何国内四层长连接通信的 RPC 框架好像比较常见。

四层长连接 RPC 通信的好处是可以在协议层面规定客户端与服务端之间正常的退出行为。假想一个长连接协议中的 graceful shutdown 流程：

1. 关闭 listenfd，使客户端不能前来建立新连接；
2. 服务端从注册中心中反注册自身，客户端收到注册中心推送的变化后，移除关闭老连接；如果担心服务注册信息收敛慢，服务端也可以主动向客户端返回 CLOSED 信息，客户端收到该响应后停止向该连接发送请求，并关闭连接；
3. 服务端发现连接合法关闭时，退出对应的 worker 线程；服务端发现所有连接关闭时，退出自身；
4. 如果连接未在正常时间内由客户端及时关闭，则服务端强制退出。

不管返回 closed 信号还是反注册，这里都是将主动关闭的权力交给了客户端，做到控制流单向流动。

## References

- <https://www.cnkirito.moe/dubbo-gracefully-shutdown/>
- <https://github.com/benoitc/gunicorn/issues/1236>
- <https://serverfault.com/questions/790197/what-does-connection-close-mean-when-used-in-the-response-message>
- <http://dubbo.apache.org/en-us/docs/user/demos/graceful-shutdown.html>
- <https://www.gitdig.com/go-tcpserver-graceful-shutdown/>
