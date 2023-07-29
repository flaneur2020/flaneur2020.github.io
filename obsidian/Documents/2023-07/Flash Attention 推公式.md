## SRAM、HBM 和 Kernel Fuse

![[Flash Attention - gpu memory hierarchy.png]]

GPU 的 SRAM 好像跟 CPU 的 Cache 是一回事，SRAM 很小但是比主存快十好几倍，做计算前需要将数据从主存捞到 SRAM 然后计算。

GPU 中的操作一般封装为 Kernel（计算核），每个 Kernel 是一个简单的算子，可以在一次循环中将多个算子 Fuse 起来，这样有最高的执行效率。

Flash Attention 的出发点就是怎样使 Attention 计算尽量地利用起来 SRAM，并尽量通过 Kernel Fuse，把计算局部在尽量少的迭代中，因为一次迭代就是一轮内存访问，最终的时间就无限接近于 HBM 的吞吐 * 迭代次数。

在 Attention 计算中，Q K 等矩阵乘法是容易并行局部执行的，但是 Softmax 是一个例外，计算 SoftMax 所需的中间结果往往需要迭代完才能知道，Softmax 的计算就成为了整个计算中访问内存的瓶颈部分。

## 数值安全的 Softmax

原始的 softmax 的公式是：

$$
 softmax(x_i) = \frac{ e ^ {x_i} }{ \sum_{j=1}^{N} e ^ {x_j} }
$$

非常大的 $e ^ {x_i}$ 会比较容易产生 overflow，比如 float16 最大值是 65536，如果 $x \ge 11$，就会溢出。为了应对这个问题，一般工程上都会做一个 ”数值安全“ 处理，使每个 $x_i$ 减去 $x$ 中的最大值 $m$：
	
$$ 
softmax(x_i) =
  \frac{ e ^ { x_i } }{ \sum_{j=1}^{N} e ^ { x_j }  } = 
  \frac{ e ^ { x_i  - m } }{ \sum_{j=1}^{N} e ^ { x_j - m }  }
$$

求数值安全的 Softmax 值需要三步迭代：

迭代一：

$$
m_i \gets max(m_{i-1}, m_i)
$$

迭代二：

$$
d_N = \sum^{N}_{i=1} e^{ x_i - m_N }
$$
迭代三：

$$
a_i \gets \frac{e^{x_i - m_N}} { d_N }
$$

## Online Softmax

上述三次迭代的算法中，我们并不能将第二步和第三步迭代给 fuse 到一起，因为第二步依赖着 $m_N$，在第一步完成前得不到这个信息。

在 Nvidia 的《Online normalizer calculation for softmax》论文中找到一个很不错的递推公式，思路是定义一个单独的序列 $d^{'}$：

$$
d^{'}_i = 
\sum_{j=1}^{i} 
e ^ { x_j - m_i }
$$

易知对于 N，$d_N = d_N ^ {'}$ 。

它可以变成一个 $d^{'}_{i-1}$ 的增量计算：

$$
\begin{align}
	d^{'}_i & = \sum_{j=1}^{i} e ^ { x_j - m_i } \\
			& = 
				\left(
				\sum_{j=1}^{i-1}
				e^{x_i-m_i}
				\right) +
				e^{x_i-m_i}
				\tag{1}
			\\
			& = 
				\left(
				\sum^{i-1}_{j=1}
				e^{x_j -m_{i-1}}
				\right)
				e^{m_{i-1}-m_i} + 
				e^{x_i-m_i}
				\tag{2}
			\\
	        & = 
		        d^{'}_{i-1} 
		        e ^ {m_{i-1} - m_i} + 
		        e ^ { x_i - m_i }
			    \tag{3}
\end{align} 
$$

推导上就是凑一个 $\sum^{i-1}_{j=1}e^{x_j-m_{i-1}}$ 出来，把它替换成 $d^{'}_{i-1}$。

到这里计算 Softmax 需要两轮迭代：

迭代一：

$$ 
	  \begin{aligned}
	   m_i & \gets max(m_{i-1}, x_i) \\
	   d^{'}_i & \gets d^{'}_{i-1} e ^ {m_{i-1} - m_i} + e ^ {x_i - m_i} 
	  \end{aligned} 
$$
迭代二：

$$ a_i \gets \frac{ e ^ {x_i - m_N} }{ d_N^{'} } $$
有没有办法缩小为一次遍历？ 

## Flash Attention

不幸的是，对于 softmax 计算来说，答案是 no。softmax 的输出结果是一个向量，只要想拿这个结果就不可能不遍历一遍。

但是在 Self-Attention 算法中，最终的目标不是 Attention score 的 $A$ 矩阵，而是等于 $A \times V$ **加权求和**后得到的 $O$ 矩阵，有没有办法找到一趟得到 $O$ 的算法？数学真神奇，Flash Attention 他们还真推出来了这个递推公式。

$O$ 矩阵中的一行是 $V$ 和 Softmax 结果的加权求和：

$$ o_i \gets \sum^{i}_{j=1}( \frac{ e^{x_j-m_i} }{ d_N } V[j,:] ) $$

利用上面 Online Softmax 一样的技巧，单独引进一个 $o^{'}$ 序列，让它利用局部的 $m_i$ 和 $d^{'}_{i}$ 参与计算：

$$ o^{'}_i \gets \sum^{i}_{j=1}( \frac{ e^{x_j-m_i} }{ d^{'}_i } V[j,:] ) $$
易知对于 N，$o_N$ 等于 $o^{'}_N$。

然后就是在这个公式中，想办法凑一个 $\sum^{i-1}_{j=1}(\frac{e^{x_j-m_{i-1}}}{d^{'}_{i}}V[j,:])$ 出来替换为 $o^{'}_{i-1}$：

$$
	\begin{align}
	o_i^{'} & = 
				\sum^{i}_{j=1}
				\frac{ e^{x_j-m_i} }{ d^{'}_i }
				V[j,:]
			\\
			& = 
				\left(\sum^{i-1}_{j=1}\frac{e^{x_j-m_i}}{d^{'}_{i}}V[j,:]\right) + 
				\frac{ e^{x_i-m_i} }{ d^{'}_i } V[i,:]
				\tag{1}
			\\
			& = 
				\left( 
				\sum^{i-1}_{j=1}
				\frac{ e^{x_j - m_{i}} }{ d^{'}_{i-1} }
				\frac{ d^{'}_{i-1} }{ d^{'}_{i}}
				V[j:]
				\right) + 
				\frac{ e^{x_i-m_i} }{ d^{'}_i } V[i,:]
				\tag{2}
			\\
			& = 
				\left( 
				\sum^{i-1}_{j=1}
				\frac{ e^{x_j - m_{i-1}} }{ d^{'}_{i-1} }
				\frac{ e^{m_{i-1} } }{ e^{m_i} }
				\frac{ d^{'}_{i-1} }{ d^{'}_{i}}
				V[j:]
				\right) + 
				\frac{ e^{x_i-m_i} }{ d^{'}_i } V[i,:]
				\tag{3}
			\\
			& = 
				\left( 
				\sum^{i-1}_{j=1}
				\frac{ e^{x_j - m_{i-1}} }{ d^{'}_{i-1} }
				V[j:]
				\right)
				\frac{ e^{m_{i-1} } }{ e^{m_i} }
				\frac{ d^{'}_{i-1} }{ d^{'}_{i}} + 
				\frac{ e^{x_i-m_i} }{ d^{'}_i } V[i,:]
				\tag{4}
			\\
		    & = 
		        o^{'}_{i-1}
		        \frac{d^{'}_{i-1} e^{m_{i-1}-m_i}}{d^{'}_{i}} + \frac{e ^ {x_i-m_i}}{ d^{'}_i}V[i,:]
		        \tag{5}
	\end{align}
	
$$

有了这个公式，计算 Attention 只需要一轮迭代了：

$$
\begin{align}
  m_i & \gets max(m_{i-1}, x_i) \\
  d^{'}_i & \gets d^{'}_{i-1} e ^ {m_{i-1} - m_i} + e ^ {x_i - m_i} \\
  o^{'}_i & \gets o^{'}_{i-1} \frac{d^{'}_{i-1} e^{m_{i-1}-m_i}}{d^{'}_{i}} + \frac{e ^ {x_i-m_i}}{ d^{'}_i}V[i,:]
\end{align}
$$

不过到这里仍有一个局限，就是它需要顺序跑，不能利用 GPU 按分块并行跑，接下来需要做的事情就是做分块（Tiled）。

## Flash Attention (Tiled)

tbd

## References
- https://zhuanlan.zhihu.com/p/621272925
- Online normalizer calculation for softmax
- [[From Online Softmax to FlashAttention]]

