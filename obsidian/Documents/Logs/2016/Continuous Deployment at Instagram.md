- instagram 每天部署 30~50 次；
Why do it?
- 并不限制每天的上线次数，工程师可以做更多事情；
- 更容易定位到坏的 commit；
- 坏的 commit 可以更快地响应，而不必等待重开上线、找上线管理员；
过去的做法：
- 持续部署之前，开发者 adhoc 地部署；
- 如果想快点上线，那就跑一个小规模 rollout；不然就等着和其他工程师一起去 rollout；
- rollout 过程：先挑一台机器，ssh 上去看日志；没问题之后再去看下一台机器的 rollout；
- 这一切都是基于一个 fabric 脚本来做的，加上一个很基本的数据库和 UI 来记录 rollout 记录；
Canary and testing
- 提供金丝雀发布机制：自动化了 rollout 过程，部署到金丝雀机器上，为开发者 tail 日志，让开发者决定是否执行下一步；
- 然后对金丝雀机器做一些基本的检查：收集金丝雀机器的 http code 数量，要执行下次部署必须达到一定指标，比如少于 0.5% 5xx，或者至少 90% 2xx；
- CI 会对 master 的每个 commit 跑测试，将测试通过的 commit 发布到 Sauron；Sauron 会记录通过测试的 commit 列表，确保部署时部的是通过测试的 commit，而不是最新 commit；
自动化
- 向工程师建议接下来要部署的 commit：从上次部署的 commit 开始，选择尽可能少的 commit，最多不超过三个；如果每个 commit 都通过了 CI，那么每次都会选择下一个 commit；
- 如果部署中 1% 的机器部署失败，那么这一次部署就将视为是失败的；
- 这一来，每次部署视为自动同意了两次：一次 CI、一次金丝雀；
Problems
- Test failure：背着失败的测试去上了线，原因是测试不稳定；经过种种优化将测试时间从 12~15 分钟优化到了 5 分钟内；解决测试基础架构不稳定的问题；
- Backlogs：积压了太多变更；每次只捡一个 commit 尝试部署；当变更积压多了，部署的时间会变长；这时往往人忍不了，强行跳过几个 commit 去部署；这一来失去了一部分持续部署的好处；
- 为了优化 backlog，优化了 commit 选择算法，允许在 backlog 存在时，合并部署多个 commit；会估算每部署一个 commit 的时间（30分钟？），最多每次部署 3 个 commit；
- 导致 backlog 增长的原因之一是基础架构的增长，毕竟仍是 fabric 式的 ssh 连接，最终迁移到了 facebook 的分布式 ssh 系统；
Guiding principles
- 测试：test suite 要快；要有足够的覆盖率，但也不一定需要完美；测试要频繁地执行：code review 时执行、部署前执行、部署后执行；
- 金丝雀：需要一个自动的金丝雀机制；这个机制不需要完美，基本的指标和限制往往就足够好了；
- Automate the normal case：不需要自动化所有操作，只需要自动化最常见的场景即可；一旦遇到问题，就让人工来参与；
- Make people comfortable: 自动化的一大阻碍是，参与者会容易感觉到 disconnected and out of control；自动化系统应该提供足够的可见性，做了什么、正在做什么、出于怎样的状态；
- Expect bad deploys: 坏掉的部署时而发生，但这没关系，只要能快速定位到并回滚就好；
What's next
- Keeping it fast
- Adding canarying: 每次测试通过都自动发布金丝雀接线上流量，检查是否达到指标；
- More data：收集更多指标；
- Improving detection：在金丝雀与所有机器之间，引入更多阶段，比如部署到部分机器上，检查指标之后再部署到全局；