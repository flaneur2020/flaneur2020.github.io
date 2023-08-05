## Positional Encoding
- 公式
	- $$
    \begin{align}
	PE(k,2j) &= sin(\frac{k}{10000^{2j/d}}),\\
	PE(k,2j+1) &= cos(\frac{k}{10000^{2j/d}}).
	\end{align}
	$$
- 维度越往后，三角函数曲线的频率越低
	- ![[Pasted image 20230805215526.png]]

## 11.6.3.1. Absolute Positional Information
- 以二进制数字为例：
	- 0 in binary is 000
	- 1 in binary is 001
	- 2 in binary is 010
	- 3 in binary is 011
	- 4 in binary is 100
	- 5 in binary is 101
	- 6 in binary is 110
	- 7 in binary is 111
- 在二进制表示中，高位 bit 的频率比低位 bit 的频率更低
- 三角函数的 heatmap 中也有类似的性质，越高位的维度，频率越低，而且 float 表示相对于二进制表示的空间效率更高；
	- ![[Pasted image 20230805215850.png]]

## Relative Positional Information
- 上述的位置编码也有助于 model 能够学习到相对位置
- 因为对于任意一个固定位置的偏移 $\delta$，在 $k+\delta$ 的位置上都可以表示为针对 $k$ 位置的一个线性变换
- 这个线性变换可以通过数学来表示：
	- 使 $w_j=1/10000^{2j/d}$
	- 三角函数两角和公式：
		- $sin(A+B)=sin(A)cos(B)+cos(A)sin(B)$
		- $cos(A+B)=cos(A)cos(B)-cos(A)cos(B)$
	- 任意一对 $(p_{i,2j},p_{i,2j+1})$ 对于任意固定的偏移 $\delta$ 均可以线性映射到 $(p_{i+\delta,2j},p_{i+\delta,2j+1})$

$$
\begin{align}
& 
\begin{bmatrix} 
cos(\delta w_j) \space & 
sin(\delta w_j) \\
-sin(\delta w_j) \space &
cos(\delta w_j)
\end{bmatrix}
\begin{bmatrix}
p_{i,2j} \\
p_{i,2j+1}
\end{bmatrix} \\

= & \begin{bmatrix}
cos(\delta w_j) sin(iw_j) +
sin(\delta w_j) cos(iw_j) \\
-sin(\delta w_j) sin(iw_j) +
cos(\delta w_j) cos(iw_j)
\end{bmatrix} \\

= & \begin{bmatrix}
sin((i+\delta)w_j) \\
cos((i+\delta)w_j)
\end{bmatrix} \\

= & \begin{bmatrix}
p_{i+\delta,2j} \\
p_{i+\delta,2j+1}
\end{bmatrix}
\end{align}
$$