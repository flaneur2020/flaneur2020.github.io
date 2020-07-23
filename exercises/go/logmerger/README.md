LogMerger
=========

多路日志合并工具。

Test coverage: 73.5%

Usage:

```
make && bin/logmerger -c etc/sample.json
```

监控:

```
curl localhost:4004/stats
```

Design
------

LogConsumer: 抽象的消费者，可能消费来自 binlog、kafka 等消息源的日志。
外部通过 LogIterator 读取它收到的消息。

LogIterator 可组合，主要包括：

- ChanIterator: 迭代来自 go channel 中的日志。
- CommitedLogIterator: 过滤消息源只保留 commit 日志。
- MergedIterator: 接受多个 iterator 进行有序归并。
- ThrottledIterator: 根据限流的配置，按时间窗口进行限流。

潜在问题
-------

保全序的要求下， 任一 Iterator 阻塞会使整个 MergedIterator 阻塞。 
初期简单的处理办法可以是:

1. Consumer 遇到任何错误持续重试，如果重试失败则触发报警，记录当前点位，
   退出并重启，根据上次 **合并后** 得到的最后点位继续消费；

2. 使每个消息源产生心跳消息，避免单一来源长时间没有写入，卡住所有；

注意，退出时应先退出 Iterator 的迭代，后退出 Consumer，不然可能导致
顺序错误。