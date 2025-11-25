firecracker 是 aws lambda 背后的技术。

### Bedrock AgentCore

AgentCore 用于跑 AI agent；

Simon Willson 对 AI agent 有一个定义：An LLM agent runs tools in a loop to achieve a goal.

agent 的每个 session 会给一个自己的 micro vm；

在 session 结束时，这个 vm 就会释放。

agent session 的时间分布不一，有的在毫秒级别，有的会到小时级；

### Aurora DSQL

> Each active SQL transaction in DSQL runs inside its own Query Processor (QPs), including its own copy of PostgreSQL. These QPs are used multiple times (for the same DSQL database), but only handle one transaction at a time.

firecracker 支持 snapshot and restore，能够将 vm 的状态整个保存下来。

当需要 DP 时，就整个恢复一个 vm 出来，这样显著降低了启动时间。

> There’s a bit more plumbing that’s needed to make some things [like random numbers](https://github.com/firecracker-microvm/firecracker/blob/main/docs/snapshotting/random-for-clones.md) work correctly in the cloned VMs[2](https://brooker.co.za/blog/2025/09/18/firecracker.html#foot2).

这个方案下，有几个小细节比如处理随机数的生成。