![[Pasted image 20230826113536.png]]

![[Pasted image 20230826113931.png]]


strides 翻译为「步长」。

每个 tensor 是一个连续的内存，在没有取行或列时，每维长度的 sizes （比如 `(D,H,W)`）和 strides （比如 `(H*W,W,1)`）是一个静态的换算关系。

比如 sizes 内容为 `(4, 2, 3)`，那么 strides 对应 `(6, 3, 1)`。第一维每进一，等于走 6 步。**

![[Pasted image 20230826114059.png]]

通过 stride 可以很容易抽出 tensor 中的行与列。

`tensor[1, :]` 对应取出第 1 行。得到的新 tensor 的 sizes 为 `(2)`，strides 为 `(1)`，offset 为 `2`。

![[Pasted image 20230826114709.png]]

取出列的话，比如 `tensor[:, 0]`，得到新的 tensor 的 sizes 为 `(2)`，strides 为 `(2)`，offsets 为 0。