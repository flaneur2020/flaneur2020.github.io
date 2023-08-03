https://kazemnejad.com/blog/transformer_architecture_positional_encoding/#the-intuition

Ideally, the following criteria should be satisfied:

- It should output a unique encoding for each time-step (word’s position in a sentence)
- Distance between any two time-steps should be consistent across sentences with different lengths.
- Our model should generalize to longer sentences without any efforts. Its values should be bounded.
- It must be deterministic.

## Proposed method

- 位置信息不是一个数值，而是一个 d-dimension 的 vector；

## The intuition

![[positional-encoding-binary-sample.png]]

如果希望通过二进制来表示位置，会相当于上图的样子。

<mark>LSB 的 bit 会在每底层一个数字时变化一次，右边第二个 bit 会在每两个数字递增时变化一次，以此递推</mark>。

但是在 float 世界中，使用上面的 binary 表示会浪费空间。因此，可以使用它的 float 替代品：Sinusoidal 函数。