
https://github.com/ml-explore/mlx-examples/blob/main/llms/llama/llama.py

## 问题

### k 和 v 是怎样加入 kv cache 的？


```python
        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.n_heads, L, -1])

        keys, values = map(repeat, (keys, values))
```

会先按 repeat 比例，将 key 和 value 物理拼接一把。

按说这样 grouped query attention 的话，kv cache 里会有些重复的 key value 值，repeat 的倍数就是 `n_head / n_kv_head`。**会不会浪费空间**？

然后会 transpose 一下：

```python
    queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
```

transpose 之后，q 的 shape 是 `(B, n_heads, L, head_dim)`，L 似乎是目前的序列长度，L 个 token。这里它的值应该是 1。

加入 key_cache 是通过 concatenate 操作：

```python
    keys = mx.concatenate([key_cache, keys], axis=2)
    values = mx.concatenate([value_cache, values], axis=2)
```

看 concatenate 里面似乎就是简单的追加到 buffer 末尾

加入 kv cache 的内容似乎是 transpose 过后的？

repeat 操作发生在 q、k 这种向量上，是物理的拼接，所以不需要在从 kv cache 中取出来东西后，再做 repeat ，但还是会做 transpose。
### kv cache 的 shape 是什么？

是 `(B, n_heads, L, head_dim)`。

和 TensorRT 里的 kv cache 的 shape 是大致一样的： https://nvidia.github.io/TensorRT-LLM/gpt_attention.html
### 怎么求 scores 的？

```python
scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
```

queries 的 shape 是 `B, n_heads, 1, head_dim`。

keys 经过 transpose 之后的 shape 是 `(B, n_heads, head_dim, L)`

相乘后，相当于每个 head 得到 L 个 scores：`(B, n_head, 1, L)`。

batch matmul 的语义似乎是最后两个维度当做 matrix。
### concatenate 是怎么按第 2 个维度拼接的？

tbd
## todo

- crabml 可以将 repeats 这个操作干掉，换成更暴力的物理拼接。
- 确认一下 concatenate 的语义，怎么按第 2 个维度拼接的。
## 看代码

```python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads
        self.repeats = self.n_heads // self.n_kv_heads
        self.scale = self.args.head_dim**-0.5
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        self.rope = nn.RoPE(
            args.head_dim, traditional=args.rope_traditional, base=args.rope_theta
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.n_heads, L, -1])

        keys, values = map(repeat, (keys, values))

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output), (keys, values)

```


