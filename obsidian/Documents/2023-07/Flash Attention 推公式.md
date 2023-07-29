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

不要这个 $m$ 是一个全局状态，需要在遍历完 $x$ 之后才可以得到它，而计算出最后的 softmax 值，必须依赖这个前置的 $m$ 才能做后续的计算。能不能把 $x$ 拆分成小块，让 softmax 操作可以分开跑？
 

## References
- https://zhuanlan.zhihu.com/p/621272925
- Online normalizer calculation for softmax
- [[From Online Softmax to FlashAttention]]

