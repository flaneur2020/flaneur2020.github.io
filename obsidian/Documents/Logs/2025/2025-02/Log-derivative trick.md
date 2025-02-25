https://andrewcharlesjones.github.io/journal/log-derivative.html

> The "log derivative trick" is really a simple application of the chain rule.

## Log derivative trick

假设我们有个函数 $p(x;\theta)$，我们希望得到它的对数求取关于 $\theta$ 的梯度：in

$$
\nabla_{\theta} \log p(x;\theta)
$$

应用 chain rule （链式法则）之后：

$$
\nabla_{\theta} \log p(x;\theta) =
\frac{\nabla_{\theta} p (x; \theta)}{p(x; \theta)}
$$

反过来，可以得到：

$$
\nabla_{\theta}p(x;\theta) = 
p(x;\theta) \nabla_{\theta} \log p(x;\theta)
$$
## Score function estimator

我们想估算函数 $f$ 的期望的梯度：

$$\nabla_{\theta} \mathbb{E}_{p(x;\theta)} \left[ f(x) \right]$$

展开一下：

$$
\begin{align}
\nabla_{\theta} \mathbb{E}_{p(x;\theta)} \left[ f(x) \right]
& = \nabla_{\theta} \int p(x; \theta) f(x) dx \\
& = \int \nabla_{\theta} p(x; \theta) d(x) \qquad \text{(Leibniz rule)}\\
\end{align}
$$

然而，$\nabla_\theta p(x;\theta)$ 并不是一个合法的概率密度，因此不能这样估算期望：

$$
\nabla_\theta \mathbb{E}_{p(x; \theta)} \left[ f(x) \right] \approx
\frac{1}{n} \sum^{n}_{i=1}\nabla_\theta p(x_i; \theta) f(x_i) \qquad \leftarrow\text{ 不可以}
$$

这时就是 log trick 可以应用了：

$$
\begin{align}
\nabla_{\theta} \mathbb{E}_{p(x;\theta)} \left[ f(x) \right]
& = \nabla_{\theta} \int p(x; \theta) f(x) dx \\
& = \int \nabla_{\theta} p(x; \theta) f(x) dx &\quad& \text{(Leibniz rule)} \\
& = \textcolor{blue}{\int p(x;\theta)} \nabla_{\theta} \log p(x;\theta) f(x) \textcolor{blue}{dx} &\quad& \text{(Log derivative rule)} \\
& = \mathbb{E}_{p(x; \theta)} \left[ \nabla_\theta \log p(x;\theta) f(x)\right]
\end{align}
$$
到这里，就可以使用<mark>蒙特卡洛方法</mark>来采样估算期望了：

$$
\mathbb{E}_{p(x; \theta)} \left[ \nabla_\theta \log p(x; \theta) f(x) \right]
\approx \frac{1}{n} \sum^{n}_{i=1} \nabla_\theta \log p(x_i; \theta) f(x_i)
$$

（到这里估算出来了梯度，到工程代码中有 auto grad 的话，一般不会直接操作梯度，仍是转成一个 Loss 函数直接写：


$$
\begin{align}
\textcolor{blue}{\nabla_\theta} Loss(\theta) & = - \frac{1}{n} \sum^{n}_{i=1} \textcolor{blue}{\nabla_\theta} \log p(x_i; \theta) f(x_i) \\
Loss(\theta) & = - \frac{1}{n} \sum^{n}_{i=1} \log p(x_i; \theta) f(x_i) \\
\end{align}
$$
