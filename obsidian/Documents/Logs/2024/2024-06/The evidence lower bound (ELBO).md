https://mbernste.github.io/posts/elbo/

- 在一个 latent variable model 中，我们假设观测数据 $x$ 是随机变量 $X$ 的一个 sample。
- 此外我们假设存在另外一个随机变量 $Z$，它们根据联合分布 $p(X, Z; \theta)$ 分布；
- 不幸的是我们数据只是 $X$ 的一个实现，而 $Z$ 是观测不到的；
- 我们的目标是：
	- 根据特定的 $\theta$ 参数，计算后验分布 $P(Z|X;\theta)$
	- 找出来 $\theta$ 的最大似然估计，做到：$\arg\max_{\theta} l(\theta)$
- 其中 $l(\theta)$ 是 log-likelihood 函数：$l(\theta)=\log p(x; \theta) = \log \int_z{p(x, z; \theta)} \, dz$
## What is the ELBO?

- evidence 的含义是 “is just a name given to the likelihood function evaluated at a fixed $\theta$:
	- $evidence = \log p(x; \theta)$”
- 如果我们选择了正确的模型 $p$ 和参数 $\theta$，那么我们会期望观测数据 $x$ 的边际概率较高
- 也就是说，这个 evidence 值是我们选择正确数据模型的“证据”
- 如果我们知道 $Z$ 符合一个 $q$ 的分布，能够满足 $p(x, z; \theta) := p(x|z; \theta) \, q(z)$，那么 evidence lower bound 
