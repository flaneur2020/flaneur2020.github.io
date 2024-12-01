预先规划好买单和卖单的价格，然后等价格上下。

## How Does Grid Trading Work?

trader 首先设置一个 reference price。

然后在 reference price 下方 一次性安排一组 buy orders。

然后每个 buy order 配对一个 sell order。

trader 需要设置多个参数，包括最高价、最低价，grid 数量，position sizing 和止损。

其中最高价、最低价通常根据最近的 price range 给出。因此这两个参数比较受最近市场的 volatility 影响。

然后，grid 数量告诉我们计划在这组 price range 中间安排多少个 order。

在设置网格时，需要将 transaction fee 考虑在内。grid 如果太密集，收益就还不如 transaction fee。

### **Favorable scenario**

![[Pasted image 20241125223851.png]]
该策略假定日内价格会上下波动，如果是单向向上或者向下，就不能收益甚至亏损了。而且亏损可能是无限的。

### **Additional parameters**

要避免无限制的亏损，可以加一个 Stop Loss Price。

这个价格应当比 Lowest Price 更低。

当触及止损位后，系统应当自动卖掉所有的头寸。

另一个流行的配置是设置一个 Trigger Price。是指只有当价格到达 trigger price 之上时，才进行交易。

有两种 grid 类型：arithmetic 和 geometric。

使用 arithmetic grid 时，每个 grid 的收益金额（比如 $5）都相等。这种网格更适合更小的 price range。

使用 geometric grid，更适合更大的 price 区间，每个网格的收益比率（比如 1%）相等。

### **When it works, and when it… doesn’t**

除非触及止损，该策略会<mark>随着亏损的增加而增加杠杆</mark>和风险。

这有点像轮盘赌策略，始终押注一种颜色，如果输了就加倍，直到赢（或破产）为止。

现代 position sizing 和资金管理技术通常是相反的思路：在损失之后减少风险，在 profits 之后增加风险。

这不是说 grid trading 不好，只是说它的风险不同。

grid trading 最怕的是单向行情。

## Grid trading performance reporting

假设安排一个日内的网格策略，也就是每天总是会关掉所有的网格。

如果一天价格上涨后，第二天回咯，我们按网格做日内交易仍然会亏损。要盈利，需要在日内有波动存在。