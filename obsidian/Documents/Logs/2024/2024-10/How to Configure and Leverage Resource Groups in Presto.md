presto 的 resource group 是一个 admission control 和 workload 管理机制，能够帮助管理资源的分配。

resource group 能够保证 group 不会超过申请的 quota。

它只在 query 开始时做检查。在 query 执行之后，它就不管了。

presto 的 resource group 可以按一颗树的形式来组织。

### **Understanding Soft and Hard Limits for CPU and Memory**

resource group 可以基于 hard concurrency parameter 来控制 query scheduling。

对于内存，soft limit 会约束一个 group 的 query memory consumption 的总和小于一个特定 limit。

如果 soft limit 达到上限，那么在低于 soft limit 之前不能接新的请求。

对于 CPU，resource group manager 跟踪特定 group 执行 query 花费的时间。如果抵达特定的 threshold，则 group 的 max concurrency 会下降到归零。

对于 CPU，resource group manager 也有一个 cpuQuotaPeriod 定义时间周期。比如一个 group 的 hardCpuLimit 是 1 minute，而 quota period 是 10min，那么抵达上限之后，可以在 10min 窗口后允许执行。

presto 会在每个 query 执行完之后，登记到 usageMillis 中，将 cpuSeconds 累加上去。

### **Key Configuration Parameters**

- cpuQuotaPeriod：相当于多久重置一个用量统计窗口
- hardConcurrencyLimit
- softMemoryLimit, hardMemoryLimit: 
- softCpuLimit, hardCpuLimit：在超过 cpu soft limit 之后，会惩罚 concurrency 值，直到抵达 hardCpu 后，max concurrency 值会归零
- schedulingPolicy: 
	- fair：first-in-first out
	- weighted-fair
	- weighted
	- query-priority：每个 sub group 配置一个 priority；
