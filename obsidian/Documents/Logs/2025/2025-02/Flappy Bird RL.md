https://sarvagyavaish.github.io/FlappyBirdRL/

作者是基于一个网页版的 flappy bird 改的。

强化学习的思路：在做了一个操作之后，能够发现一个新的状态，并基于它得到一个 reward；

Q Learning 是一种 model free 的强化学习算法。

![[Pasted image 20250206221120.png]]

state space 是（小鸟距离右侧下方柱子的距离，小鸟距离右侧下方柱子的高度，是否存活）；

对于每个 state，可以选择两种 action：

1. click
2. do nothing

奖励：

1. 只要存活，就 +1
2. 否则 -1000

learning loop

array Q 初始化为 0，然后总是选择最优的 action；如果遇到平局，则选择什么都不做；

1. 观测 flappy bird 的状态 s，选择针对这个状态，最大化收益的 action；然后让 game engine 执行下一步 "tick"，于是，flappy bird 会进入新的状态 s'
2. 观察这个新的状态 s'，以及对应的 reward；如果小鸟还活着，就 +1，否则 -1000；
3. 基于 Q Learning 规则更新 Q array
 
$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$

这个公式中，$\alpha$ 是学习率，设置为 0.7。

$\gamma$ 是折扣因子。

$max_{\alpha^{'}}Q(s^{'}, a^{'})$ 是新状态 $s'$ 下所有可能动作 $a^{'}$ 的最大 Q 值。