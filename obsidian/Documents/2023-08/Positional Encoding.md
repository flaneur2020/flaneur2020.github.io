- 使用时钟作为 positional encoding 的问题：求距离不方便，小时维度和分钟维度动同样的幅度，看到的距离是一样的， 但是体现不出来距离的倍数差异
	- **2. Relative Positions**: The encoding allows the model to easily learn to attend by relative positions, since for any fixed offset `k`, `PE(p+k)` can be represented as a linear function of `PE(p)`

1. It should output a unique encoding for each time-step
2. Distance between any two time-steps should be consistent across sequences with different lengths.
3. Our model should generalize to longer sentences without any efforts. Its values should be bounded.
4. It must be deterministic.

- relative position 的性质意味着 LLM 可以顺着当前的位置找出特定 relative 位置的相关词汇？


## references

- https://notesonai.com/Positional+Encoding
- https://kazemnejad.com/blog/transformer_architecture_positional_encoding/