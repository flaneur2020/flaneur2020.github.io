与 RNN 自带顺序不同，Transformer 需要为序列额外引入一个顺序编码（Positional Encoding），它的维度和 embedding 向量的维度相同，与 embedding 向量加在一起来引进顺序信息。

不过 Transformer 原始论文中的 Positional Encoding 看起来像是魔法：

$$
\begin{equation}
PE(k) = \begin{bmatrix}
sin(k) \\
cos(k) \\
sin(\frac{k}{10000^{1/d}}) \\
cos(\frac{k}{10000^{1/d}}) \\
sin(\frac{k}{10000^{2/d}}) \\
cos(\frac{k}{10000^{2/d}}) \\
\vdots \\
sin(\frac{k}{10000^{ (d - 2) / 2 d}}) \\
cos(\frac{k}{10000^{ (d - 2) / 2 d}})
\end{bmatrix}
_{d \times 1}
\end{equation}
$$

代入到公式是这样写：

$$
    \begin{align}
	PE(k,2j) &= sin(\frac{k}{10000^{2j/d}}),\\
	PE(k,2j+1) &= cos(\frac{k}{10000^{2j/d}}).
	\end{align}
	
$$
为什么 sin 和 cos 函数能够表示顺序？

## 时钟

能代入的一个 Intuition 是，我们看时间用的时钟就是 sin、cos 函数表示位置的一个实例。

在时钟里，秒针、分针、时针就代表了一个位置向量，03:01:00 作为一个位置显然是大于 02:44:30 的。

如果用时钟作为位置编码的话，就对应到一个 6 维的向量，因为使用三角函数表示角度的话，需要一个二维向量 $[sin(x) \space cos(x)]$，三个角度就得用一个 6 维向量来表示：

$$
\begin{equation}
PE_{clock}(k) = \begin{bmatrix}
sin(2 \pi k) \\
cos(2 \pi k) \\
sin(2 \pi k / 3600) \\
cos(2 \pi k / 3600) \\
sin(2 \pi k / 3600 * 12) \\
cos(2 \pi k / 3600 * 12) \\
\end{bmatrix}
_{6 \times 1}
\end{equation}
$$
在秒针转动时，分针与时针也会轻微转动，不过秒针、分针、时针的频率会按依次递减。比如秒针每转动 3600 圈，分针才会转完一圈，分针转完 12 圈，时针转完一圈。

如果要增加维度，就加入天、星期、月、年等更高的单位就可以了。

不过时钟这个例子有点不完美的地方是时钟每维的增长幅度不大规律。假设有一个外星时钟是不同维度皆为固定的 3600 倍数周期：

$$
    \begin{align}
	PE(k,2j) &= sin(\frac{k}{3600^{2j}}),\\
	PE(k,2j+1) &= cos(\frac{k}{3600^{2j}}).
	\end{align}
	
$$

到这里和 Transformer 的公式已经很接近了，不同在于这里的频率是指数递减，而 Transformer 的公式在指数项中除以了一个 $d$：$sin(\frac{k}{3600^{2j/d}})$。如果不除 $d$ 的话，位置向量维度间频率的衰减太快，容易使得位置向量中除了前两个位置有数以外，其他维度都是接近于零，真正参与位置编码的维度就太少了。

## references

- https://notesonai.com/Positional+Encoding
- https://kazemnejad.com/blog/transformer_architecture_positional_encoding/