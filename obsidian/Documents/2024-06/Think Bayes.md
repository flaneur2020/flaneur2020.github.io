## Preface

> Many of the methods in this book are based on discrete distributions, which makes some people worry about numerical errors. But for real-world problems, **numerical errors are almost always smaller than modeling errors.**
>
> Furthermore, the discrete approach often allows better modeling decisions, and I would rather have an approximate solution to a good model than an exact solution to a bad model.
> 
> On the other hand, continuous methods sometimes yield performance advantagesfor example by replacing a linear- or quadratic-time computation with a constant-time solution.
>
> So I recommend a general process with these steps:
>
> 1. While you are exploring a problem, start with simple models and implement them in code that is clear, readable, and demonstrably correct. Focus your attention on good modeling decisions, not optimization.
> 2. Once you have a simple model working, identify the biggest sources of error. You might need to increase the number of values in a discrete approximation, or increase the number of iterations in a Monte Carlo simulation, or add details to the model.
> 3. If the performance of your solution is good enough for your application, you might not have to do any optimization. But if you do, there are two approaches to consider. You can review your code and look for optimizations; for example, if you cache previously computed results you might be able to avoid redundant computation. Or you can look for analytic methods that yield computational shortcuts.

## 1 Bayes Theorem

### 1.1 Conditional probability

> A probability is a number between 0 and 1 (including both) that represents **a degree of belief** in a fact or prediction.

> The usual notation for conditional probability is $p(A|B)$, which is the probability of A given that B is true. In this example, A represents the prediction that I will have a heart attack in the next year, and B is the set of conditions I listed.

这个例子中 $P(A|B)$ ，B 中包括作者的性别、年龄、血压等特征，A 是心脏病发作的概率。
### 1.2 Conjoint probability

在 A 和 B 两个事件是独立事件时，满足 $p(A, B) = p(A) p(B)$。这时 $p(B|A) = p(B)$，$A$ 事件是否发生完全不影响 $B$ 事件。

一般来讲，$p(A, B) = p(A) p(B|A)$。
### 1.3 The cookie problem

一号碗有 30 个香草曲奇 10 个巧克力曲奇。二号碗有 20 个香草曲奇 20 个巧克力曲奇。

随机找一个碗，随机拿一个曲奇，这个曲奇是香草的，那么它来自一号碗的概率是多少？

### 1.4 Bayes's theorem

$$p(A|B) = \frac{p(A) p(B|A)}{p(B)}$$

> This example demonstrates one use of Bayes's theorem: it provides a strategy to get from $p(B|A)$ to $p(A|B)$

### 1.5 The diachronic interpretation

另一个理解贝叶斯公式的视角：它为我们提供了一种根据某些数据 $D$，来更新假设 $H$ 的概率的方法。

这被称作 "历时诠释" 。"diachronic" 的意思是跟随时间变化；

> in this case the probability of the hypotheses changes, over time, as we see new data.

$$
p(H|D) = \frac{p(H)p(D|H)}{p(D)}
$$

其中：

- $p(H)$ 称作“先验”（prior），比如 “男女的性别比例”
- $p(H|D)$ 是我们想计算的，得到数据 $D$ 之后 $H$ 发生的概率，这称作“后验”（posterior），比如 “碰到一个抽烟的人，是男性的概率”
- $p(D|H)$ 是 $H$ 发生时数据 $D$ 的概率，称作 “似然” (likelihood)，比如 “男性抽烟的概率”
- $p(D)$ 是归一化常数（normalizing constant），比如 “全人群中抽烟的概率”

### 1.6 The M&M problem

M&M 是一个巧克力品牌，经典版 1994 款是 30% 棕色、20% 黄色、20% 红色、10% 绿色、10% 橙色、10% 褐色。

后来改了颜色，1996 款中 24% 蓝色、20% 绿色、14% 黄色、13% 红色、13% 棕色。

假设一个朋友有两袋 M&M 巧克力，一袋是经典款一袋是新款，他在两袋中分别取了一粒巧克力，一个是黄色一个是绿色。黄色的这粒来自 1994 版的概率是多少？

### 1.7 The Monty Hall problem

Monty Hall 是 Let's Make a Deal 之前的主持人。这个题目来自于 Monty 节目现场：

1. Monty 给你看三个关着的门，每个门后面有一个奖品，一个奖品是车，另外两个门后面是羊；
2. 游戏规则是猜哪个门后面有车，猜对了可以得到车；
3. 你选择打开一个 A 门，另外两个门是 B 门和 C 门；
4. 在打开你选择的 A 门之前，Monty 会在 B 门和 C 门中间有羊的那个门；
5. 然后 Monty 问你，要不要改；

问题就是，你应该改，还是不改？

大部分人觉得改和不改都一样。

实际上，改了的话，赢的概率是 2/3。

站在主持人视角上：

1. 如果嘉宾本来选中了羊（2/3 概率），改选一定赢；
2. 如果嘉宾本来选中了车（1/3 概率），改选一定输；

## Computational Statistics

### 2.1 Distributions

> To represent a distribution in Python, you could use a dictionary that maps from each value to its probability.

PMF：probability mass function，概率质量函数。

## 3 Estimation

### 3.1 The dice problem
 
假如我有一个盒子，里面有 4 面、6 面、8 面、12 面、20 面的骰子 🎲。

假如我随机拿一个骰子，扔出来得到 6，那么拿到其中每个色子的概率分别是多少？

对付这类问题一般可以分三步：

1. Choose a representation for the hypotheses. 
2. Choose a representation for the data. 
3. Write the likelihood function. （每个骰子抽到 6 的概率 🤔）

### 3.2 The locomotive problem

> 铁路公司会按顺序给机车编号，从1到N。某一天你看到一辆编号为60的机车。估计一下铁路公司有多少辆机车。
### 3.7 The German tank problem

## 4 More Estimation

