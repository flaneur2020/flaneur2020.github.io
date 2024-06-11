```
in VAE (variantal autoencoder), the math proofs are likely to introduce the probabilities distributions like: p(x), p(z), p(x|z), p(z|x), p(x,z), z is the latent variable.

please describe it with concerete examples.
```

```
how can we get p(z|x) during the training?

what does it means to maximize p(x|z)? 

how to get p(x,z)?

p(x) could be viewed as a mixture of gaussian distribution, right? how does ELBO plays in to guide us optimize p(x)?

why does the ELBO is defined as \log p(x) \ge \mathbb{E}_{q(z|x)} \left[ \log \frac{p(x,z)}{q(z|x)} \right] ?

how does the ELBO can be further decomposed into two terms?

what does \mathbb{E}_{q(z|x)} means? how to calculate it in code?

how does KL divergence corronsponds with expectations?

```

## 怎么理解 p(x)、p(z)、p(z|x)、p(x|z)、p(x, z)？ 

- p(x) 是拿不到的， p(x) 是**优化目标**，最后训练出来的模型的 $p(x)$ 越大越好；
	- Maximizing the ELBO allows us to indirectly maximize the log-likelihood, and thus $p(x)$
	- $\log p(x) = \log \int p(x, z) dz$
	- $\log p(x) = \log \int q(z|x) \frac{p(x, z)}{p(z|x)} dz$
	- ELBO 公式：$\log p(x) \ge \mathbb{E}_{q(z|x)} \left[ \log \frac{p(x,z)}{q(z|x)} \right]$
	- ELBO 又可以拆解为：$\log p(x) \ge \mathbb{E}_{q(z|x)} \left[ \log p(x|z) \right] - KL(q(z|x) || p(z))$
	- ![[Screenshot 2024-06-10 at 22.57.34.png]]
		- 其中 KL 散度总是 >= 0
	- ![[Screenshot 2024-06-10 at 23.04.47.png]]
	- ![[Screenshot 2024-06-10 at 23.08.40.png]]
- p(z|x) 给定一个 x 对应的 z 的分布，是一个高斯分布；<mark>可以在训练 VAE 过程中通过 encoder 神经网络拟合出来</mark>；
- $log p(x|z)$ 可以<mark>通过训练集来算出来</mark>：
	- ![[Screenshot 2024-06-10 at 22.28.11.png]]
	- If $p(x|z)$ is high, it means the reconstruction $\hat{x}$ is very close to $x$.
	- $p(x|z)$ 可以通过 $x$ 与输出的 $\hat{x}$ 计算出来
		- $p(x|z) = \frac{1}{(2\pi\sigma_x^2)^{d/2}} \exp\left(-\frac{1}{2\sigma_x^2} \|x - \hat{x}\|^2 \right)$
		- maximum likelihood
- p(x, z) 是联合分布，等于 $p(z) \cdot p(x|z)$
- p(z) 是一个高斯分布；
## Estimating $𝑝(𝑥)$

- 希望最大化 p(x)，但是 p(x) 是求不出来的；
	- p(x) 可以看做是一个混合高斯分布；
- 退一步求最大化 ELBO: $$\log p(x) \ge \mathbb{E} \left[ \log p(x|z) \right] - KL(q(z|x) || p(z))  $$
- $\mathbb{E}\left[\log p(x|z)\right] = \mathcal{L}_{\text{reconstruction}} = \frac{1}{2\sigma_x^2} \|x - \hat{x}\|^2 + \frac{d}{2} \log(2\pi\sigma_x^2)$

$$
p(x|z) = \frac{1}{(2\pi\sigma_x^2)^{d/2}} \exp\left(-\frac{1}{2\sigma_x^2} \|x - \hat{x}\|^2 \right)
$$

$$
-\log p(x|z) = \frac{d}{2} \log(2\pi\sigma_x^2) + \frac{1}{2\sigma_x^2} \|x - \hat{x}\|^2
$$

## what does $\mathbb{E}_{q(z|x)}$ means?


$$
\mathbb{E}_{q(z|x)} \left[ \log p(x|z) \right] = \int \log p(x|z) q(z|x) dz
$$

## KL Divergence and Expectation

The Kullback-Leibler (KL) divergence can be expressed in terms of expectations. This relationship provides a deeper understanding of what the KL divergence measures and how it can be computed. 

$$\text{KL}(P \| Q) = \mathbb{E}_{P} \left[ \log \frac{P(X)}{Q(X)} \right]$$
