---
date: "2020-01-12T00:00:00Z"
title: "A Study about Graceful Shutdown"
---

Graceful Shutdown is supposed to be a solved problem, especially for HTTP communication, where the details had already been handled properly by the application servers. However, I never thought carefully about how to implement graceful shutdown in the context of long-lived TCP connections. Let me have a study about the details on it.

## Graceful Shutdown in gunicorn

First, let's review the process of Graceful Shutdown in the context of short-lived HTTP connections. gunicorn's sync mode is a text-book prefork model, starting multiple workers, with the parent process initializing the listener, allowing each worker to compete for `accept(listener)`.

The entry point for graceful shutdown is in the `stop()` method of `arbiter.py`, which first closes the LISTENERS and then sends a term signal to the workers.

Each gunicorn worker inherits from the `Worker` class in `worker/base.py`. The `Worker` base class defines an `alive` variable, which is set to `False` when a sigterm is received, completing the signal handling.

In `worker/sync.py`, the main process of `SyncWorker` is roughly `while self.alive:... self.accept(listener)`. If `alive` becomes `False`, the current request is processed, and no new requests are accepted, thus exiting the main loop.

Here's a simple summary of the Graceful shutdown process for short-lived HTTP connections:

1. Close the `listenfd`, so the load balancer in the front end can sense that this backend is being taken down. Any further attempts to distribute requests will result in a Connection Refused, allowing safe retries on another node.
2. Each worker processes the current request, discovers it's in an exit state, stops accepting new requests, and exits.
3. If a worker fails to exit within a normal time frame, it's forcefully killed with `kill -9`.

## Graceful Shutdown Process for Keep-Alive

The Graceful Shutdown process for Keep-Alive should be similar to that of short-lived connections: after closing the `listenfd`, wait for all active connections to close before exiting the worker. Keep-alive has an upper time limit, and should exit after exceeding this limit.

However, keep-alive connections can last a very long time. Is there a more timely way to implement an exit?

HTTP/1.1 defines a `Connection: close` header in the response, informing the client that the server will close the connection after this request, and the client should find another server.

This can be done in the following process:

1. Close the `listenfd` to stop receiving new connections.
2. Each worker, after processing the current request and discovering it's in an exit state, returns a `Connection: close` in the response header and then closes the connection.
3. After the connection is closed, the worker exits.
4. If a worker fails to exit within a normal time frame, it's forcefully killed with `kill -9`.

This process has a problem: the time when a request arrives is unpredictable. If there's no request during the keep-alive timeout, the connection won't reach `Connection: close` and can't exit. If a request comes much later, it can be surprising because the old code from the previous version is still being executed. The gunicorn author mentioned that he prefers to close idle connections immediately after starting Graceful shutdown to avoid the old code from "resurrecting" after a period of time.

## Graceful Shutdown in grpc

The grpc server object has a built-in [GracefulStop()](https://github.com/grpc/grpc-go/blob/master/server.go#L1448) method. Although I haven't looked into it in detail, it seems to be:

1. Close the `listenfd` to stop receiving new connections.
2. Notify the workers to stop and push a `goaway` HTTP/2 Frame to inform the client that the connection will be closed.
3. The client receives the `goaway` and closes the connection.
4. Wait for all connections to close before exiting.

Compared to the keep-alive process in HTTP/1.1, grpc's advantage is that HTTP/2 can actively push control messages like `GOAWAY`, so it doesn't need to worry too much about idle connections. I'll look into it in detail another day.

## Graceful Shutdown Process for Long-Lived RPC Protocols

Personally, I feel that the communication layer of RPC should be built on a stable seven-layer protocol for reliability. However, it seems that four-layer long-lived connection RPC frameworks are more common in China.

The advantage of four-layer long-lived RPC communication is that normal exit behavior between the client and server can be defined at the protocol level. Here's a hypothetical Graceful shutdown process for a long-lived connection protocol:

1. Close listenfd to prevent new clients from establishing connections;
2. The server deregisters itself from the registry, and after clients receive the change notification from the registry, they remove and close old connections. If concerned about slow convergence of service registration information, the server can also actively return a CLOSED message to clients. Upon receiving this response, clients stop sending requests to that connection and close it;
3. When the server detects a legitimate connection closure, it exits the corresponding worker thread; when the server detects that all connections are closed, it exits itself;
4. If the connection is not closed by the client within the normal time, the server forcibly exits.

Whether returning a closed signal or deregistering, the power to actively close is given to the client, ensuring unidirectional flow of control.

## References

- <https://www.cnkirito.moe/dubbo-gracefully-shutdown/>
- <https://github.com/benoitc/gunicorn/issues/1236>
- <https://serverfault.com/questions/790197/what-does-connection-close-mean-when-used-in-the-response-message>
- <http://dubbo.apache.org/en-us/docs/user/demos/graceful-shutdown.html>
- <https://www.gitdig.com/go-tcpserver-graceful-shutdown/>
