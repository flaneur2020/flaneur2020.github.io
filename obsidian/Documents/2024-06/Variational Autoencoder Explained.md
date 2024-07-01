目标是找到一个学习到生成过程的模型，也就是掌握 $p(x|z)$ 这个分布（假设 $p(z)$ 是已知的正态分布）。一个好的 $p(x|z)$ 能够给训练过程中出现的 $x$ 样本较高的分数。

假设 $p(x|z)$ 的参数是 $\theta$，那么训练的目标就是如下的一个优化问题：

$$
\max_{\theta} \, p_{\theta}(x)
$$

要暴力得到 $p_{\theta}(x)$ 不是不可以，

$$
\begin{align*}
p_{\theta}(x) &= \int_z p(z) p_{\theta}(x|z) \\
&= \sum_{i} p(z_{(i)})p_{\theta}(x|z_{(i)})
\end{align*}
$$

相当于穷举一把 $z$ 的所有样本，这个计算量是巨大的。
## Posterior inference in a latent variable model

可以换一个目标，来优化后验概率 $p(z|x)$。给出来先验 $p(z)$ 和似然 $p(x|z)$，可以计算出来后验 $p(z|x)$。

但是 $p(z|x)$ 也是不可计算的，可以想办法估算 $p(z|x)$，这就是涉及到 Vartional Inference 了。

> Variational inference converts the posterior inference problem into the optimization problem of finding an approximate probability distribution $𝑞(𝑧|𝑥)$ that is as close as possible to $p(z|x)$.

Varational Inference 将后验推断问题转化为一个优化问题，目标是找出来一个估算的概率分布 $q(z|x)$，使它尽可能地接近 $p(z|x)$。

这个优化问题可以写成：

$$
\min_{\phi} \,  KL(q_{\phi}(z|x) \, || \, p(z|x))
$$

KL 散度的公式是这样：$$ KL(q(x) || p(x)) = \int_x q(x) \, log \frac{q(x)}{p(x)} $$
带入进去 $q_{\varphi}(z|x)$ 和 $p(z|x)$：

$$
\begin{align*}
KL(q_{\phi}(z|x) || p(z|x)) 
&= \int_z q_{\phi} (z|x) \, log \frac{q_{\phi}(z|x)}{p(z|x)} \\
&= \int_z q_{\phi} (z|x) \, log \textcolor{red}{\frac{ q_{\phi}(z|x) p(x)}{p(x, z)} } \\
&= \int_z q_{\phi} (z|x) \, log \frac{q_{\phi(z|x)}}{p(x,z)} + \textcolor{red}{\int_z \, q_{\phi} (z|x) \, log \, p(x)} \\
&= -\mathcal{L}(\phi) + log \, p(x)
\end{align*}
$$

为什么 $log p(x) = \int_z q_{\phi}(z|x) \, log \, p(x)？$ 因为 $\int_z \, q_{\phi} (z|x) = 1$ ，把它代入进去了。（为什么等于 1？）

> Since $p(x)$ is independent of $q_{\theta}(z|x)$, minimizing $KL(q_{\theta}(z|x) \,|| \, p(z|x))$ is equivalent to maximizing $\mathcal{L(\phi)}$.

问 GPT4o 说在优化过程中 $\log p(x)$ 是一个常数。因为 $\log p(x)$ 是对 $x$ 的边缘概率，它与 $q_{\phi}(z|x)$ 无关。

$\mathcal{L}(\phi)$ 也就是 $ELBO$。

优化 $\mathcal{L}(\phi)$ 会容易很多，只涉及 $p(x, z) = p(z) p(x|z)$，这里面就没有 intractable integral 了。

$$
\begin{align*}
\mathcal{L}(\phi) 
&= \int_z q_{\phi}(z|x) \, \log \frac{p(x,z)}{q_{\phi}(z|x)} \\
&= \int_z q_{\phi}(z|x) \, \log \frac{p(z)p(x|z)}{q_{\phi}(z|x)}
\end{align*}
$$


最终优化问题相当于：

$$
\max_{\phi} \, \mathcal{L}(\phi)
$$
## Back to the learning problem

上述推导也提出了一种学习 $p(x|z)$ 的方法。

因为 KL 散度一定是大于等于零的，$\mathcal{L}(\phi)$ 其实也是 $\log p(x)$ 的下界。

$$
\mathcal{L}(\phi) \le \log p(x)
$$

最大化 $\mathcal{L}(\theta)$ 等于最大化 $p(x)$。