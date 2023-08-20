- visibility timeout 是什么？
	- 一条消息在处理之后，一段时间内不会再被取到
	- 消费者在处理消息之后，需要删除掉消息，避免被其他消费者重复消费
- request attempt ID 是做什么的?
	- [link](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/using-receiverequestattemptid-request-parameter.html)
	- 如果提供了 request attempt ID，下次重试会使用同一个 ID 进行，可以起到去重的作用

Standard Queues
- standard queue 保证 at-least-once，消息可能有乱序，提供 best-effort orderuing
- 如果对顺序有强要求，最好使用 FIFO queue
- sqs 会在多台服务器上冗余保存消息，如果一台服务器不可用，那么这台不可用的服务器上的消息不会被删除，可能会重复收到这条消息，这要求应用程序做到幂等
- Consumer need to need to delete the message after processing else the message will be visible in the queue after visibility timeout.

FIFO Queues
- fifo 的队列需要以 .fifo 结尾
- fifo 队列中的每条消息都需要有一个 message group ID，好像类似 kafka 的 partition key
- 应用程序可以提供一个 unique message deduplication ID 用于去重，如果没有提供，有个默认的 content-based deduplication
- FIFO queue 最多可以有 20,000 条 inflight message，表示客户端收到但是服务端尚未 delete
- 每个 partition 支持每秒最多 30000 条消息
- 接收消息时，会尽量多地返回同一个 message group ID 的消息
- 消费失败的话，可以通过同一个 request attempt ID 进行重试
- 从一个 message group ID 消费到消息之后，除非显式 delete 消息或者消息成为 visible，是不会再重新得到消息的