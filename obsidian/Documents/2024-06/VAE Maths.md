## 1. from $\ln p(x)$ to $ELBO$

$$
\begin{align*}
\ln p(x) 
&= \ln p(x) \textcolor{blue}{\underset{=1}{\int q(z|x) \, dz}} \\
&= \int q(z|x)  \ln p(x) \, dz \\
&= \int q(z|x) \ln \textcolor{blue}{\frac{ p(x|z) p(z) }{ p(z|x)}} \, dz  \\
&= \int q(z|x) \ln \frac{p(x|z) p(z) \, \textcolor{blue}{q(z|x)}}{p(z|x) \textcolor{blue}{q(z|x)}} \, dz \\
&= \underbrace{{\int q(z|x) \ln \frac{q(z|x)}{p(z|x)} \, dz}}_{KL \, Divergence} + \int q(z|x) \ln \frac{p(x|z)p(z)}{q(z|x)} dz \\
&= \underbrace{D_{KL} ( q(z|x) || p(z|x))}_{always\, \ge 0} + \int q(z|x) \ln \frac{p(x,z)}{q(z|x)} \, dz \\
&\ge \underbrace{\int q(z|x) \ln \frac{p(x, z)}{ q(z|x)} \, dz}_{ELBO}
\end{align*}
$$

$$
ELBO = \mathcal{L} = \int q(z|x) \ln \frac{p(x, z)}{ q(z|x)} \, dz
$$

where $q(z|x)$ should approximate $p(z|x)$.

## 2. from $ELBO$ to training objectives

$$
\begin{align*}
\mathcal{L} 
&= \int q(z|x) \ln \frac{p(x, z)}{z|x} \, dz \\
&= \int q(z|x) \ln \frac{p(x|z) p(z)}{q(z|x)} \, dz \\
&= \int q(z|x) \ln \frac{p(z)}{q(z|x)} \, dz + \int q(z|x) \ln p(x|z) \, dz \\
&= - \int q(z|x) \ln \frac{q(z|x)}{p(z)} \, dz + \int q(z|x) \ln p(x|z) \, dz \\
&= - D_{KL}(q(z|x) || p(z)) + \mathbb{E}_{z \sim q(z|x)} [\ln p(x|z)] \\
\end{align*}
$$

凑出来两个式子：

1. $q(z|x)$ 和 $p(z)$ 的 KL 散度，对应 divergence loss
2. $\int q(z|x) \ln p(x|z) dz$，等于 $z$ 在 $q(z|x)$ 分布（是一个正态分布）下的期望，对应 reconstruct loss

涉及计算期望时，可以将连续概率的积分改成离散的采样平均：

$$
\begin{align*}
\mathbb{E}_{x \sim p(x)} [f(x)]
&= \int f(x)p(x) dx \\
&\cong \frac{1}{N}\sum^{N}_{i=1} f(x_i), \space x_i \sim p(x)
\end{align*}
$$

代入正态分布的式子 $\ln \mathcal{N}(x; \mu, \sigma) = - \frac{n}{2} \ln(2\pi \sigma ^2)  - \frac{(x-\mu)^2}{2\sigma^2}$ 就约等于最小二乘了。
## 3. put $p(x|z)$ and $q(z|x)$ as Normal distribution

Given finitely many observe samples $x_1, x_2, x_3, ... x_N$ (training data).

$$
\begin{align*}
\max p(x) &\cong \min_{\phi,\theta} (D_{KL}(q_{\phi}(z|x) || p(z))) - \mathbb{E}_{z \sim q_{\phi}(z|x_i)} [\ln p_{\theta}(x|z)] \\
&= \min_{\phi, \theta} D_{KL}(\mu_{\phi}(x), \sigma_{\phi}(x)) \, || \, \mathcal{N}(0,1)) - 
\underbrace{\mathbb{E}_{z \sim N(\mu_{\phi}(x), \sigma_{\phi}(x))}[\ln \mathcal{N}(\mu_{\theta}(z), \sigma_{\theta}(z))]}_{\text{reconstruct loss}}
\\
\text{reconstruct loss} &= \mathbb{E}_{z \sim N(\mu_{\phi}(x), \sigma_{\phi}(x))}[\ln \mathcal{N}(\mu_{\theta}(z), \sigma_{\theta}(z))] \\
&= \frac{1}{N} \sum^N_{i=0} \ln \frac{(x_i - \mu_{\theta}(z_i))^2}{2 \sigma^2}

\end{align*}
$$
