
![[Screenshot 2024-06-25 at 12.32.43.png]]

![[Screenshot 2024-06-25 at 12.19.54.png]]



![[Screenshot 2024-06-25 at 12.31.53.png]]

- 图一算出来 $\log p(x) >= L^v$，KL 散度 $KL(q(z|x) || p(z|x))$ 好像没啥用，知道它总是 $\ge 0$ 就可以了
- 图二继续推 $L^v$，最后反正推出来 $-DK(q(z|x) || p(z)) + \mathbb{E}_{q(z|x^(i))} (\log (p(x^{(i)} | z)))$
- 图三通过采样来估算 $\mathbb{E}_{q(z|x^(i))} (\log (p(x^{(i)} | z)))$