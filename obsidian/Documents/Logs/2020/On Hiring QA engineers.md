- Suddenly, you have a whole QA department with 30+ folks. That is a typical reaction in companies with a track record of problems in the software they ship.
- QA 有多种不同的背景：
	- ex-tester：在开发最后的阶段搞手工测试
	- The Automator: 自动化 test case，并且搞测试框架来优化其他人的工作；
	- The coach：有很多编写、测试软件的经验，目标不是写测试，而是向其他人传授最佳实战，传达写测试的 vision 和策略
	- The infrastructure expert：为编写测试提供 infrastructure、framework 或者其他工具，一些人会管这些叫做 EP
- Should we trust software engineers that don't test their software themselves? I don't think so.
- 在很多公司里，这意味着在 kanban 里增加了一列，现在上到生产之前，需要先经过 QA
- When did everything get so broken that we need a particular part of the process dedicated to quality?
- QA 过程也会引进瓶颈
Solo efforts vs multiplying effects
- 很多组织招聘了这种选手：ex-testers and automation experts.
- 会产生两个问题：ex-tester 没有编写软件和自动化的能力；另一面， automation experts 往往只专精一两项技术，比如 iOS & Android
- 当我们谈到 QA 的招聘，就是： if we're talking about coaches and infrastructure experts, go ahead.
- 这部分工程师扮演起组织中其他成员的 multiplier
- 如果你有 100 位工程师，不需要超过 5% 的人来负责这部分工作
- 如果你认为这些人不够，可能就需要重新想一想，你的自动化和 CI/CD 策略了
The problem of reverse coaching
- It's not easy to find experts in the market.
- Companies tend to avoid significant reorganizations, mainly if they affect technical roles because they're challenging to hire anyway.
- 最终的结果大概率是，SWE 反过来 coaching QA 工程师，而 QA 工程师才应该是屋子里的 expert
- 非但没有 multiplier，反过来 dividing 了
- People that suck up time from your organization because they don't understand modern CI/CD practices, or they don't know how modern distributed systems are impossible to test as a whole, etc
- 最终的形态成了：
	- First and second class citizens in the engineering departments
	- Big QA departments with their objectives and agenda
	- More processes and manual work than necessary
Unit, integration and end-to-end tests
- "Now we only miss the e2e tests, which is QA's job" —, we can't expect our business to go full speed.
- 在导出倒是 distributed system 的环境下，已经没有 e2e test 了，至少不是传统的那种
- If the folks that write a piece of software don't know how to observe it in production, that's a big engineering problem in our teams.
Should I hire QA engineers?
- Hire QA engineers. Not too many. Mostly SWE experts. Don't try to fix your organizational problems by throwing QA power because, in the long run, it won't cut it.