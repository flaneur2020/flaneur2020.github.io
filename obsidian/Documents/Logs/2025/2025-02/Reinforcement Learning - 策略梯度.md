
怎样理解增强学习中的策略梯度算法，请整理它背后的 intuition，请注意使用 latex 格式的公式
使用简单易懂的语言，分段落讲解。
请介绍它的公式的原理，尤其是 Policy Gradient Theorem 的推导过程。

## 问题

- Q：策略梯度算法怎样更新的？
- Q：策略梯度算法如果是游戏结束后才能更新梯度吗？
- Q：策略梯度算法的损失函数是多少？

## Notes

### 损失函数

$$
\mathcal{L}(\theta) = -\mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \log \pi_\theta(a_t|s_t) R(\tau) \right]
$$

### 数学前置

#### Lebnizz 定理

#### log derivative trick


### 推导

之前的传统增强学习算法比如 Q-Learning 是针对 Value 函数进行学习，然后再通过价值函数间接地计算最优策略。

策略梯度法则不同，它直接学习策略本身，它直接找一个策略，能在与环境交互中得到最高的期望回报。

策略梯度算法维护一个参数化的策略 $\pi_{\theta}(a|s)$，其中 $s$ 表示当前环境中的状态，对应在状态 $s$ 时，采取 $a$ 动作的概率。

算法的目标是调整参数 $\theta$，使 $\pi_{\theta}$ 最大化期望的回报 $J(\theta)$：

$J(\theta) = \mathbb{E}_{\tau \sim p(\tau|\theta)} [R(\tau)]$

其中 $\tau$ 表示一个轨迹，即状态和动作的序列：$s_0,a_0,s_1,a_1,...,s_T,a_T$。

$p(\tau|\theta)$ 表示在策略 $\pi_\theta$ 下生成轨迹 $\tau$ 的概率。

$R(\tau)$ 表示轨迹 $\tau$ 的累积奖励，$R(\tau) = \sum_{t=0}^{T} r_t$，其中 $r_t$ 是每个时间步 $t$ 中获得的奖励。

（似乎这个在训练中会输入一套完整的轨迹序列 $\tau$ 和相关的奖励值）

要更新策略梯度，公式是这样子：

$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p(\tau|\theta)} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau) \right]$

对于每个轨迹 $\tau$，我们计算每个时间步 $t$ 的 $\nabla_\theta \log \pi_\theta(a_t|s_t)$。 

表示策略在状态 $s_t$ 下采取动作 $a_t$ 的概率对参数 $\theta$ 的影响。 如果 $\nabla_\theta \log \pi_\theta(a_t|s_t)$ 是正的，说明增加参数 $\theta$ 会增加采取动作 $a_t$ 的概率；如果是负的，说明增加参数 $\theta$ 会减少采取动作 $a_t$ 的概率。

我们将每个时间步的 $\nabla_\theta \log \pi_\theta(a_t|s_t)$ 乘以轨迹的累积奖励 $R(\tau)$。 如果 $R(\tau)$ 是正的，说明这个轨迹是好的，我们希望增加采取类似动作的概率；如果 $R(\tau)$ 是负的，说明这个轨迹是坏的，我们希望减少采取类似动作的概率。