---
date: "2025-02-25"
title: "Notes on RL: Policy Gradient & Log Derivative Trick"
language: "en"
---

Recently I followed [this tutorial](https://sarvagyavaish.github.io/FlappyBirdRL/) to implement a basic Q-Learning reinforcement learning algorithm for [Flappy Bird](https://github.com/flaneur2020/FlappyBird-RL), which surprisingly worked quite well, reaching over 10,000 points.

Q-Learning essentially estimates a value $Q(s, a)$ for each state-action pair $(s, a)$, and then selects actions based on these $Q$ values. Deep Q-Network (DQN) builds on basic Q-Learning by adding a neural network to estimate $Q$ values, essentially compressing the Q Table using a neural network.

Both Q-Learning and DQN evaluate rewards and select actions based on the estimated value of rewards.

On the other hand, Policy Gradient belongs to a different approach - directly selecting actions $a$ based on the current state $s$. The neural network takes the current state $s$ as input and outputs probabilities for various actions, denoted as $\pi_\theta(a|s)$, where $\theta$ represents the neural network parameters.

Later algorithms used in large language models, such as PPO and GRPO, also belong to the Policy Gradient family.

However, I found the mathematical derivation of Policy Gradient somewhat difficult to understand for me, so I'm taking notes here.

Like the previous notes about machine learning, I'll first list the prerequisite mathematical knowledge needed for the derivation. Later, we'll simply substitute these formulas into the derivation.

## Expectation and Monte Carlo

As mentioned in my previous VAE notes, for continuous random variables, the expectation is calculated as:

$$
\mathbb{E}[x] = \int x p(x) dx
$$

During training, we work with sets of samples. For example, if we have a sample set $X = \{x_1, x_2, \ldots, x_n\}$, the expectation can be estimated as:

$$
\mathbb{E}[x] \approx \frac{1}{n} \sum_{i=1}^{n} x_i, \,\,\,\, x_i \sim p(x)
$$

This method of estimating expectations using sampling is called the **Monte Carlo method**.

Extending this to the expectation of a function $f(x)$:

$$
\begin{align}
\mathbb{E}[f(x)] & = \int f(x) p(x) dx \\
& \approx \frac{1}{n} \sum_{i=1}^{n} f(x_i), \,\,\,\, x_i \sim p(x)
\end{align}
$$

If we can identify a $p(x)$ term in a continuous expectation expression, we can use the Monte Carlo method to estimate the expectation by converting it to a discrete sampling expression.

## Log Derivative Trick

Suppose we have a function $p(x;\theta)$ and we want to find the gradient of its logarithm with respect to $\theta$:

$$
\nabla_{\theta} \log p(x;\theta)
$$

Applying the chain rule, we get:

$$
\nabla_{\theta} \log p(x;\theta) =
\frac{\nabla_{\theta} p (x; \theta)}{p(x; \theta)}
$$

Conversely, we can derive:

$$
\nabla_{\theta}p(x;\theta) = 
p(x;\theta) \nabla_{\theta} \log p(x;\theta)
$$

This technique is called the **Log Derivative Trick**, and it's useful in "Score Function Estimator" scenarios.

## Score Function Estimator

Sometimes, we want to estimate the gradient of the expectation of a function $f$ with respect to $\theta$:

$$\nabla_{\theta} \mathbb{E}_{p(x;\theta)} \left[ f(x) \right]$$

Expanding this:

$$
\begin{align}
\nabla_{\theta} \mathbb{E}_{p(x;\theta)} \left[ f(x) \right]
& = \nabla_{\theta} \int p(x; \theta) f(x) dx \\
& = \int \nabla_{\theta} p(x; \theta) d(x) \qquad \text{(Leibniz rule)}\\
\end{align}
$$

However, $\nabla_\theta p(x;\theta)$ is not a valid probability function, so we cannot use Monte Carlo to estimate the expectation like this:

$$
\nabla_\theta \mathbb{E}_{p(x; \theta)} \left[ f(x) \right] \approx
\frac{1}{n} \sum^{n}_{i=1}\nabla_\theta p(x_i; \theta) f(x_i) \qquad \leftarrow\text{ Not valid}
$$

This is where the log derivative trick comes in. By converting $\nabla_\theta p(x;\theta)$ to $p(x;\theta)\nabla_\theta \log p(x;\theta)$, we can introduce a $p(x;\theta)$ term in the expression:

$$
\begin{align}
\nabla_{\theta} \mathbb{E}_{p(x;\theta)} \left[ f(x) \right]
& = \nabla_{\theta} \int p(x; \theta) f(x) dx \\
& = \int \nabla_{\theta} p(x; \theta) f(x) dx &\quad& \text{(Leibniz rule)} \\
& = \textcolor{blue}{\int p(x;\theta)} \nabla_{\theta} \log p(x;\theta) f(x) \textcolor{blue}{dx} &\quad& \text{(Log derivative trick)} \\
& = \mathbb{E}_{p(x; \theta)} \left[ \nabla_\theta \log p(x;\theta) f(x)\right]
\end{align}
$$

Now we can use the <mark>Monte Carlo method</mark> to estimate the expectation by sampling:

$$
\mathbb{E}_{p(x; \theta)} \left[ \nabla_\theta \log p(x; \theta) f(x) \right]
\approx \frac{1}{n} \sum^{n}_{i=1} \nabla_\theta \log p(x_i; \theta) f(x_i)
$$

This gives us the gradient estimate.

This is the application of the Log Derivative Trick: converting the gradient $\nabla_\theta p(x;\theta)$, which cannot be directly estimated using Monte Carlo, into $p(x;\theta)\nabla_\theta \log p(x;\theta)$, which introduces a probability function $p(x;\theta)$ into the expression, allowing us to use the Monte Carlo method to estimate the expectation. This trick is primarily used in deriving Policy Gradient.

## Policy Gradient

In the Policy Gradient framework, we want to obtain a policy $\pi(a|s;\theta)$ with neural network parameters $\theta$ that takes the current state $s$ as input and outputs probabilities for various actions $a$.

After playing a game, a policy generates a trajectory $\tau$:

$$
\tau = (s_0, a_0, s_1, a_1, \ldots, s_t, a_t)
$$

This trajectory $\tau$ results in a total reward $R(\tau)$, which is the sum of rewards $r(s_t, a_t)$ at each step:

$$
R(\tau) = \sum_{t=0}^{T} r(s_t, a_t)
$$

We want to maximize the expected reward $J(\theta)$ over the entire trajectory:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \right]
$$

To optimize this, we compute the gradient with respect to $\theta$ and perform gradient ascent:

$$
\nabla_{\theta} J(\theta) = \nabla_{\theta} \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \right]
$$

$$
\theta \leftarrow \theta + \alpha \nabla_{\theta} J(\theta)
$$

Expanding this and applying the Leibniz rule and Log derivative trick:

$$
\begin{align}
\nabla_{\theta} J(\theta) 
& = \nabla_{\theta} \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \right] \\
& = \nabla_{\theta} \int \pi_\theta(\tau) R(\tau) d\tau \\
& = \int \nabla_{\theta} \pi_\theta(\tau) R(\tau) d\tau &\qquad& \text{(Leibniz rule)}\\
& = \textcolor{blue}{\int \pi_\theta(\tau)} \nabla_{\theta} \log \pi_\theta(\tau) R(\tau) \textcolor{blue}{d\tau} &\qquad& \text{(Log derivative trick)}\\
& = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_{\theta} \log \pi_\theta(\tau) R(\tau) \right] &\qquad& \\
\end{align}
$$

After applying the Log derivative trick, we get an expectation formula that can be estimated using the Monte Carlo method:

$$
\nabla_{\theta} J(\theta) \approx \frac{1}{n} \sum^{n}_{i=1} \nabla_{\theta} \log \pi_\theta(\tau_i) R(\tau_i)
$$

This derivation is similar to the Score Function Estimator above, but since the neural network outputs probabilities for each action step, we need to expand $\pi_\theta(\tau)$:

$$
\pi_\theta(\tau) = p(s_0) \prod_{t=0}^{T} \pi_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t)
$$

Where $p(s_0)$ is the probability of the initial state, $\pi_\theta(a_t|s_t)$ is the probability of each action step, and $p(s_{t+1}|s_t, a_t)$ is the probability of state transition.

$$
\nabla_{\theta} \log \pi_\theta(\tau) = \textcolor{blue}{\nabla_{\theta} \log p(s_0)} + \sum_{t=0}^{T} \nabla_{\theta} \log \pi_\theta(a_t|s_t) + \textcolor{blue}{\nabla_{\theta} \log p(s_{t+1}|s_t, a_t)}
$$

This expression has two interesting properties:

1. Using the properties of logarithms, multiplication becomes addition, making computation more convenient;
2. Both $\log p(s_0)$ and $\log p(s_{t+1}|s_t, a_t)$ are independent of $\theta$, so their gradients are 0;

Therefore, we can simplify to:

$$
\nabla_{\theta} \log \pi_\theta(\tau) = \sum_{t=0}^{T} \nabla_{\theta} \log \pi_\theta(a_t|s_t)
$$

Substituting this back into the expectation formula:

$$
\nabla_{\theta} J(\theta) \approx \frac{1}{n} \sum^{n}_{i=1} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_\theta(a_t|s_t) R(\tau_i)
$$

This gives us the gradient formula for Policy Gradient.

However, in practice, we typically don't directly manipulate gradients. In an auto-grad framework, we can simply write the gradient descent loss function to optimize:

$$
\begin{align}
Loss(\theta) & = - J(\theta) \\
& \approx - \frac{1}{n} \sum^{n}_{i=1} \sum_{t=0}^{T} \log \pi_\theta(a_t|s_t) R(\tau_i) \\
\end{align}
$$

## Final Thoughts

Overall, the derivation of the Policy Gradient formula isn't particularly long. It primarily applies the Log Derivative Trick to introduce a probability function, and then uses the Monte Carlo method to estimate the expectation.

However, I've heard that this simplest form of Vanilla Policy Gradient has rather poor training performance. Game trajectories are inherently sparse, and the variance in rewards is quite high, making Monte Carlo estimation of expectations rather unstable. I'll continue to explore improved algorithms built on this foundation, such as PPO and GRPO.

## References

- https://andrewcharlesjones.github.io/journal/log-derivative.html
- https://davidmeyer.github.io/ml/log_derivative_trick.pdf