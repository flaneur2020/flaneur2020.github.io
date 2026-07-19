TLDR
- 大约相当于 nvidia 做了俩 sdk，一个面向客户端，一个面向服务端，支持这个东西的都是 partner，aws 的 s3 肯定是不支持的；
- 控制面就是 s3 的 GET 接口，会多传递一个 x-amz-rdma-token 来携带 rdma 的内存信息，这个控制面接口会一直阻塞，直到完成传输为止，返回一个空的 body，用于告知客户端完成了传输；
- 服务端接到这个 GET 接口的请求之后，可以调用服务端 sdk 来朝这个 GET 的这个 object 的 path 去写 rdma，对应的客户端的 buffer，和 object path 到对应 buffer 的映射关系，是 sdk 帮助管理的，反正服务端 sdk 只管写就可以；

---


https://docs.nvidia.com/gpudirect-storage/cuobject/cuObjServer-api/index.html

cuObject 能够做到支持 GPU 直接走 rdma 读写 object storage。

提供了两个 library：

1. cuObject client library：与 GPU 应用集成，管理 RDMA 的 data sink or source；作为 cuda toolkit 13.1 的一部分；
2. cuObject server library：与 partner 的对象存储 server 集成，管理 rdma 的 sind or source 传输，与 buffer registration；

![[Pasted image 20260712121949.png]]

- control plane（RESTful API）：修改 partner s3 sdk 中发送请求的回路，附带上 `x-amz-rdma-token` 这样的元信息。
- data plane（RDMA）：使用 RDMA 跨节点传输；

## 1.3. Technical Specifications and Protocols

### 1.3.1. Transport Layer: Dynamic Connection (DC)

与 RC（Reliable Connection）不同，DC transport 不需要每个 client 和 server pair 来建立 connection。

- 客户端不需要知道服务端 server 的 topology 或者数据分布的方式；
- 只有在需要数据传输时才建立连接，可以节约 NIC 资源；
- DC 支持 InfiniBand 和 RoCEv2.

### 1.3.2. RDMA enabled GET and PUT workflow

![[Pasted image 20260712122621.png]]

> The client sends `x-amz-rdma-token` containing RDMA metadata (for example, `00007f7c...`) to the proxy or gateway.
>
> If the transfer is successfully offloaded to RDMA, the proxy responds with `x-amz-rdma-reply`.


### 1.3.3. Data Flow Sequence (GET and PUT operation)

服务端流程：

1. Gateway: parses the tag and, if enabled, instructs specific data nodes to transfer data using RDMA.
2. Data nodes allocate a local buffer and register it for RDMA using cuObject server APIs.
3. The cuObject server APIs synchronously or asynchronously perform IO using the local buffer and remote RDMA tag. The library establishes a DC connection with the client and pushes or pulls the data via RDMA (`RDMA_WRITE` or `RDMA_READ`) directly to or from the client GPU or system memory.
4. Gateway returns HTTP status 200 OK once the RDMA transfer is complete. An RDMA reply tag (for example, x-amz-rdma-reply) is sent to inform the RDMA status.

## 1.4. API Reference for Developers

### 1.4.2. Server Side API (cuObject server library)

- `registerBuffer(const void *ptr, size_t size)`
- `allocHostBuffer(size_t size)`
- `handleGetObject(...)`：执行 RDMA_WRITE
- `handlePutObject(...)`：执行 RDMA_READ

## Example

```C++
cuObjServer server("192.168.1.100", 18515, CUOBJ_PROTO_RDMA_DC_V1);
if (!server.isConnected()) {
    return;
}

uint16_t channel = server.allocateChannelId();
if (channel == INVALID_CHANNEL_ID) {
    return;
}

void* buffer = server.allocHostBuffer(1024 * 1024);
if (buffer == nullptr) {
    server.freeChannelId(channel);
    return;
}

struct rdma_buffer* rdma_buf = server.registerBuffer(buffer, 1024 * 1024);
if (rdma_buf == nullptr) {
    free(buffer);
    server.freeChannelId(channel);
    return;
}

// send rdma data
ssize_t result = server.handleGetObject("request-1",
                                        rdma_buf,
                                        remote_addr,
                                        request_size,
                                        
                                        
                                        rdma_descriptor,
                                        channel);
if (result < 0) {
    // Handle error.
}

server.deRegisterBuffer(rdma_buf);
free(buffer);
server.freeChannelId(channel);
```