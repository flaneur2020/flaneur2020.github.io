https://jdhao.github.io/2019/07/10/pytorch_view_reshape_transpose_permute/

view() vs reshape()
- view() 好像是一个更老的函数，可以按另一个 shape 来返回一个 tensor；
- 如果修改新的 tensor 中的数据，老的 tensor 的数据也会随之修改；
- reshape() 在 0.4 中引进的，reshape() 会尽量按 no-copy 地返回 tensor；Contiguous 的 input 和兼容的 strides 可以免拷贝；

view() vs transpose()
- view() 只能针对 contiguous 的 tensor 进行变换；
- transpose() 也可以对非 contiguous 的 tensor 做变换；
- 对一个非 contiguous 的 tensor 执行 view() 会报错  [non-contiguous error](https://github.com/pytorch/pytorch/issues/764)

transpose() and permute()
- transpose 只能改两个维度，而 permute() 可以改任意维度

But what does contiguous mean?
- 可以用  [`is_contiguous`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.is_contiguous) 来判断
- As I understand, `contiguous` in PyTorch means if the neighboring elements in the tensor are actually next to each other in memory. Let’s take a simple example:

```fallback
x = torch.tensor([[1, 2, 3], [4, 5, 6]]) # x is contiguous
y = torch.transpose(0, 1) # y is non-contiguous
```