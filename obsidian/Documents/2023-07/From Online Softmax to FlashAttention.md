- attention 的公式：$X = Q K^{T}, A = softmax(X),  O = AV$
- 有没有办法让 softmax 能够 <mark>associative</mark>？

## 2 (Safe) Softmax
- softmax 的公式：$softmax({ x_1, ..., x_n }) = \left\{  \frac{ e ^ {x_i} }{ \sum_{j=1}^{N} e ^ {x_j} } \right\}$
- 非常大的 $e ^ {x_i}$ 会比较容易产生 overflow，比如 float16 最大值是 65536，如果 $x \ge 11$，就会溢出。
- 为了应对这个问题，一般会做一个 ”数值安全“ 版的 softmax：
	- $\frac{ e ^ { x_i } }{ \sum_{N}^{j=1} e ^ { x_j }  } = \\ \frac{ e ^ { x_i  - m } }{ \sum_{N}^{j=1} e ^ { x_j - m }  }$
	- 其中 $m = max(x_j)$ 。

## 3 Online Softmax
- 上述 3 pass safe softmax 算法中，我们并不能将第二步和第三步给 fuse 到一起，因为第二步依赖着 $m_N$，在第一步完成前，得不到这个信息
- 弄一个单独的序列 $d^{'}_i = \sum_{j=1}^{i} e ^ { x_j - m_i }$，它可以变成一个 $d^{'}_{i-1}$ 的增量计算：
	- $$
	\begin{split}
		d^{'}_i & = \sum_{j=1}^{i} e ^ { x_j - m_i } \\
				& = \left(\sum_{j=1}^{i-1}e^{x_i-m_i}\right) + e^{x_i-m_i}\\
				& = \left(\sum^{i-1}_{j=1} e^{x_j -m_{i-1}}\right) e^{m_{i-1}-m_i} + e^{x_i-m_i} \\
	            & = d^{'}_{i-1} e ^ {m_{i-1} - m_i} + e ^ { x_i - m_i }
	\end{split} 
	$$

- 可知对于 N，$d_N = d_N ^ {'}$
- 2 pass online softmax 算法：
	- 迭代1： $$ 
	  \begin{aligned}
	   m_i & = max(m_{i-1}, x_i) \\
	   d^{'}_i & = d^{'}_{i-1} e ^ {m_{i-1} - m_i} + e ^ {x_i - m_i} 
	  \end{aligned} 
	  $$
	- 迭代2：$$ a_i = \frac{ e ^ {x_i - m_N} }{ d_N^{'} } $$
- 到这里这个算法需要遍历两次来完成 softmax 计算，有没有办法缩小为 1 次遍历？

## 4 FlashAttention
- 不幸的是，对于 softmax 计算来说，答案是 no 
- 但是在 Self-Attention 算法中，最终的目标不是 score matrix $A$，而是等于 $A \times V$ 的 $O$ matrix，有没有办法找到一趟得到 $O$ 的算法？
- 传统的 Attention 算法：
	- 迭代 1：同上面一样，计算出 $d^{'}_N$ 和 $m_N$
	- 迭代 2：$$
	  \begin{aligned}
	    a_i & = \frac{e^{x_i}-m_N}{d^{'}_N} \\
	    o_i & = o_{i-1} + a_i V[i,:]
	  \end{aligned}
	  $$
- 上述迭代 2 的公式可以合并为：$o_i = \sum^{i}_{j=1}( \frac{ e^{x_j-m_i} }{ d^{'}_i } V[j,:] )$
- 利用第三节的技巧，可以推公式推成递推的：
	- $$
	\begin{align}
	o_i^{'} & = \sum^{i}_{j=1}\frac{ e^{x_j-m_i} }{ d^{'}_i } V[j,:] \\

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
到这里仍有一个局限，就是它需要顺序跑，不能利用 GPU 按分块并行跑。