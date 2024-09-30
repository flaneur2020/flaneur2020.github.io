## References

- https://zhuanlan.zhihu.com/p/566618077
- https://www.zhihu.com/question/618298320/answer/3307885924

## 相关的基础

### 两个均值为零的高斯分布相加

$$
\mathcal{N}(0,\sigma_a^2) + \mathcal{N}(0, \sigma_b^2) = \mathcal{N}(0, (\sigma_a+\sigma_b)^2)
$$

## 加噪声

$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_{t} \textbf{I})
$$
$$
q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})
$$
$$
\begin{align}
x_1 &= \sqrt{\alpha_1}x_0 + \sqrt{1-\alpha_1}\epsilon_1 
&\text{; where}\quad \epsilon_1 \sim \mathcal{N}(0, \textbf{I}) \tag{1} \\
x_2 &= \sqrt{\alpha_2} x_1 + \sqrt{1-\alpha_2}\epsilon_2 \tag{2} \\
&= \sqrt{\alpha_2}(\sqrt{\alpha_1}x_0 + \sqrt{1-\alpha_1}\epsilon_1 ) + \sqrt{1-\alpha_2}\epsilon_2 \tag{3} \\
&= \sqrt{\alpha_1 \alpha_2 } x_0 + \textcolor{blue}{\sqrt{\alpha_2 - \alpha_1\alpha_2} \epsilon_1 + \sqrt{1-\alpha_2}\epsilon_2} \tag{4} \\
&= \sqrt{\alpha_1 \alpha_2 } x_0 + \textcolor{blue}{\sqrt{1 - \alpha_2 + \alpha_2 - \alpha_1\alpha_2} \overline{\epsilon}_2} \tag{5} \\
&= \sqrt{\alpha_1 \alpha_2 } x_0 + \sqrt{1 - \alpha_1\alpha_2} \overline{\epsilon_2} \tag{6} \\
&\,... \\
x_t &= \sqrt{a_1..a_{t}} x_0 + \sqrt{1-\alpha_1...\alpha_{t}} \overline{\epsilon_{t}} \tag{7} \\
&= \sqrt{\overline{a_{t}}} x_0 + \sqrt{1-\overline{\alpha_{t}}} \overline{\epsilon_{t}}
&\text{; where} \quad &\overline{a_{t-1}} = \prod_{t=1}^{T} a_{t} \tag{8} \\
\end{align}
$$

其中 $\epsilon_0, \epsilon_1, ... \epsilon_{t-1}$ 都是采样自 $\mathcal{N}(0, \textbf{I})$。

公式 $(5)$ 相当于相加两个均值都为 0、方差分别为 $\alpha_1-\alpha_0\alpha_1$ 和 $1-\alpha_1$ 的正态分布。

$\alpha_0, \alpha_1, ...\alpha_{t-1}$ 都是小于 1 的， $\sqrt{\overline{a_{t-1}}}$ 最后会接近 0。

## Loss Function

$$
\begin{align}
x_t &= \sqrt{\alpha_{t}} x_{t-1} + \sqrt{1-\alpha_{t}} \epsilon_{t} \\
&= \sqrt{\alpha_t}(\sqrt{\overline{a}_{t-1}} x_0 + \sqrt{1-\overline{a}_{t-1}} \overline{\epsilon}_{t-1})  + \sqrt{1-\alpha_t} \epsilon_t \\
&= \sqrt{\overline{\alpha_t}} x_0 + \sqrt{\alpha_t(1-\overline{\alpha}_{t-1})} \textcolor{blue}{\overline{\epsilon}_{t-1}} + \sqrt{1-\alpha_t} \textcolor{blue}{\epsilon_t}
\end{align}
$$

需要退一步到 $x_{t-1}$ 再代入转化为 $x_0$ 的公式。因为 $\epsilon_t$ 和 $\overline{\epsilon}_t$ 两个采样并不是独立的，而 $\overline{\epsilon}_{t-1}$ 和 $\epsilon_t$ 完全无关。

损失函数针对噪音：

$$
\begin{align}
\mathcal{L} &= ||\epsilon_t - \epsilon_\theta(x_t, t) ||^2 \\
&= || \epsilon_t - \epsilon_\theta(\sqrt{\overline{\alpha_t}} x_0 + \sqrt{\alpha_t(1-\overline{\alpha}_{t-1})} \overline{\epsilon}_{t-1} + \sqrt{1-\alpha_t} \epsilon_t, t)||^2
\end{align}
$$

其中 $\epsilon_t$ 对应采样的到的真实噪声，$\epsilon_\theta(x_t, t)$ 对应一个降噪模型，这个模型的输入是 $x_t$ 的值和时间步 $t$，输出预测的噪声。
## 降低方差

眼下在训练中有四个随机变量：

1. 从所有时间序列中采样一个 $x_0$
2. 从 $1 \sim T$ 中采样一个时间步 $t$
3. 总正态分布 $\mathcal{N}(0, \bf{I})$ 中采样 $\overline{\epsilon}_{t-1}$ 和 $\epsilon_t$

实际训练中随机变量太多方差太大不大好收敛。所以最好用个数学技巧把 $\overline{\epsilon}_{t-1}$ 和 $\epsilon_t$ 合并成一个随机变量。

两个正态分布相加，比如 $\sqrt{\alpha_t-\overline{\alpha}_t} \overline{\epsilon}_{t-1} + \sqrt{1-\alpha_t} \epsilon_t$ 相当于一个正态分布 $\sqrt{1-\overline{\alpha}_t}\epsilon | \epsilon \in \mathcal{N}(0, \bf{I})$。

同理，$\sqrt{1-\alpha_t} \overline{\epsilon}_{t-1} - \sqrt{\alpha_t-\overline{\alpha}_t}\epsilon_t$  也相当于一个正态分布 $\sqrt{1-\overline{\alpha}_t} \omega ｜ \omega \in \mathcal{N}(0, \bf{I})$。

$$
\begin{align}
\sqrt{\alpha_t-\overline{\alpha}_t} \overline{\epsilon}_{t-1} + \sqrt{1-\alpha_t} \epsilon_t  - (\sqrt{\alpha_t-\overline{\alpha}_t} \overline{\epsilon}_{t-1} - \sqrt{1-\alpha_t} \epsilon_t) &= \sqrt{1-\overline{\alpha}_t}\epsilon - \sqrt{1-\overline{\alpha}_t} \omega \\
2 \sqrt{1-\alpha_t} \epsilon_t &= \sqrt{1-\overline{\alpha}_t} (\epsilon - \omega) \\
\epsilon_t &= \frac{\sqrt{1-\overline{\alpha}_t}}{2 \sqrt{1-\alpha_t}} (\epsilon - \omega)
\end{align}
$$
代入到原本的损失函数里：

$$
\begin{align}
\mathcal{L} &= || \epsilon_t - \epsilon_\theta(\sqrt{\overline{\alpha_t}} x_0 + \sqrt{\alpha_t(1-\overline{\alpha}_{t-1})} \overline{\epsilon}_{t-1} + \sqrt{1-\alpha_t} \epsilon_t, t)||^2 \\
&= ||\epsilon_t - \epsilon_\theta(\sqrt{\overline{\alpha}_t} x_0 + \textcolor{blue}{\sqrt{1- \overline{\alpha}_t} \epsilon}, t)||^2 \\
&= || \textcolor{blue}{\frac{\sqrt{1-\overline{\alpha}_t}}{2 \sqrt{1-\alpha_t}} (\epsilon - \omega)}  - \epsilon_\theta(\sqrt{\overline{\alpha}} x_0 + \sqrt{1- \overline{\alpha}_t} \epsilon, t)||^2
\end{align}
$$