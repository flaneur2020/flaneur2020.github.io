## Connection pooling

连接池配置中 pool_max_idle_per_host 是一个比较 critical 的选项；

## Streaming Without Buffering

hyper 实现了 `Body` trait，是一组 chunk 的 async iterator。

```rust
use http_body_util::BodyStream;
use futures_util::StreamExt;

async fn transform_body(
    body: hyper::body::Incoming,
) -> impl hyper::body::Body<Data = Bytes, Error = hyper::Error> {
    let stream = BodyStream::new(body);

    let transformed = stream.map(|result| {
        result.map(|frame| {
            // Transform each chunk here
            // This example just passes through unchanged
            frame
        })
    });

    http_body_util::StreamBody::new(transformed)
}
```

## TCP Tuning

```sh
# Increase socket buffer sizes

sysctl -w net.core.rmem_max=16777216
sysctl -w net.core.wmem_max=16777216

# Increase the backlog queue
sysctl -w net.core.somaxconn=65535
```

