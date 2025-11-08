https://www.youtube.com/watch?v=kb-m2fasdDY
http://highscalability.com/blog/2016/10/12/lessons-learned-from-scaling-uber-to-2000-engineers-1000-ser.html?utm_source=webopsweekly&utm_medium=email

- uber 是一个 winner-takes-all 的市场，要求系统必须 grow fast，你可以走的更慢，显得更有秩序，但是市场不允许，只能与混乱做交易
- 对于 uber 而言，这么多服务是招聘这么多人并保持 productive 的唯一方式，相比 traffic 层面的扩容，团队和特性的扩张更为重要 
	- Scaling the traffic is not the issue. Scaling the team and the product feature release rate is the primary driver.
- 微服务允许绕过人与人之间的沟通；相对于部门之间沟通，部门内宁可自己写新代码，作者不知道这样是否更好，但是确实能让很多工作在没有中心协调的环境中并行起来；

less obvious cost
- 很奇怪的事实是，在一个 super moduler 的系统环境中，会存在这些代价
- everything is a trade off: 即使你没有意识到，也会不停地做 trade off，使用一个方案去解决一个问题，也意味着对其他一些东西的 trade off，而 trade off 的频率比想象中更为频繁
- you can build around problems: 你可以为了新问题去创建新的服务，会倾向于不停地创建新服务去解决问题，却缺少动力去解决现存服务中的现有问题
- you might trade complexity for politics: 解决现存的问题需要 politics，需要与人沟通、考虑合作者的情绪，而工程师开发一个新服务，可以免去这部分的 politics，但却以系统的整体复杂性为代价
- you get to keep your bias: 喜欢 python 的工程师会认为 python rocks，更倾向于去使用 python，却不乐意首先在现有的 code base 中解决问题
languages
- hard to share code
- hard to move between teams
- wiwik: fragment the culture： 多门编程语言会割裂工程师文化
RPC
- RPC 的调用模式往往代表了组织的结构
- HTTP/Rest get complicated: 在一个高速加入新人的环境中，HTTP 作为内部通信协议遇到了很多困难；内网通信需要的往往只是“我想在某个其他位置上执行某函数”，而HTTP 协议首先面向浏览器，作为内网通信协议时容易过于复杂，开发者需要关心 HTTP 的 status、缓存等语义；
- JSON needs a schema
- RPCs are slower than PCs
- wiwiki:  将 datacenter 内的 RPC 看作函数调用，要好过将它看作浏览器交互；
Fanout Issues - Tracing
- fanout 可以导致很多性能问题；
- 假如一个服务的 P99 是 1ms，而 1% 是 1s，对服务自身而言不是很坏；然而存在 fanout 时，撞到慢响应的概率会大大增高，如果 fanout 出 100 个请求，那么 63% 的概率会至少 1s；
- 假如上层对一个服务有大量 fanout，在服务自身的监控图表看来是好的，几乎每个请求都很快；但上层的性能会很糟糕；
logging
- 日志经过建索引会非常有用
- 但是开发者很容易过度地打日志，超过日志系统的负载能力
- 使日志系统支持记账，每月根据建索引的日志量对开发者发账单 
	- The idea is to put pressure back on developers to log smarter, not harder.
- wiwik: 在建立日志时，提供记账机制
Migration
- old stuff has to work
- what happened to immutable
- wiwik: mandate are bad: 强制迁移不好；
Politics
- when you value shipping things in sort like smaller individual components, it can be harder to prioritise what is better for the company, surprising tradeoff;
- Services allow people to play politics
- politics 发生在违背 Company > Team > People 时
Trade Offs
- Everything is a tradeoff
- try to make them intentionally