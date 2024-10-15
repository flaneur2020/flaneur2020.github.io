与 ALB 通过 cookie 实现 sticky ness 不同，NLB 使用一个内置的 5 元组 hash table 来实现不通 backend server 之间的 stickness。

我们通过 DNS 来访问 NLB，会得到两个 round robin 的 NLB endpoints。

60s 之后，客户端连到另一个 NLB 节点之后，stickiness 和 cross-zone loadbalancing 会怎样 work？

> The fact that our NLB is not allowing cross-zone loadbalancing seems to prevent the connection from reaching the same backend every time. The connection enters via NLB endpoint 1 but stickiness has decided that the connection should go to server in AZ 2? Stickiness fails, the disabled cross-zone loadbalancing wins…

## The conclusion

If you don’t allow **cross-zone loadbalancing**, then stickiness is only active within AZ boundaries.

As DNS round-robin could direct a client to a different point of entry after the TTL has expired, strict stickiness is not guaranteed.x