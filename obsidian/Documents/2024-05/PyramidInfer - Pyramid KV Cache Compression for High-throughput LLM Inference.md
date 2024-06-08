![[Screenshot 2024-05-25 at 10.03.58.png]]

## 3.1 Inference Context Redundancy

在推理中，只有最后一个 token 用于预测下一个 token，作者假设在训练阶段，现有的 context 中 key 和 value 中会保存冗余的信息，但这部分冗余对推理没有用。这被称作 Inference Context Redundancy 假设。
### 3.1.1 Pivotal Context

> This selected proportion consists of the important keys and values with the top-p attention weights, denoted as the Pivotal Context (PvC).

有最高的 top-p 的 attention weight 的 key 和 value，称作 Pivotal Context（PvC）。

> as the layer becomes deeper (larger index), we find that the influence of shorter PvC tends to be smaller

越高层越越可以减少 kv cache 的 size。
### 3.1.2 Discussion

**How does the model gather information to predict the next token?**

生成下一个 token，可以认为是结合 attention weights，从上下文中捞信息。

在较浅的层中，关于上下文的信息往往和所有的 token 都有关，随着层变深，只有少部分 key 和 value 对下一个 token 有贡献。

在推理过程中，键和值存储了两类信息：

1. 用于预测下一个标记的信息（即当前标记的上下文信息）。
2. 为未来的标记提供的上下文信息（即整个序列的上下文信息）。

作者说训练阶段的 teacher forcing 的性质，non-PvC 的 key value cache 对预测下一个 token 的不那么大。
## 3.2 Recent Attention Consistency

> the PvCs selected by these weights are suitable for predicting $x_{n+1}$ but not always suitable for future $x_{t>n+1}$

适合预测下一个 token 的 PvC，不一定对下下一个 token 好使。

作者的目标是找到一种共享的 PvCs，这些 PvCs 可以用来预测多个未来的标记，而不仅仅是下一个标记而已。
### 3.2.1 PvC Consistency

> We convert this goal to finding if there exist keys and values that are frequently attended by subsequent tokens.

目标定义为，**现存的哪些 key 和 value 更频繁地被后续的 token 使用到**。

定义一个相对距离，上下文 token $x_i$ 相对最后一个 token $x_n$ 有多远，这被称作 Recent Ratio $d = \frac{n-i}{n}$。

作者将 $0 < d < 30\%$ 视为 recent sequence $S_r$，$d >= 30\%$ 是为 context sequence $S_c$。

作者计算 $S_r$ 中对 $S_c$ 的注意力权重。他们想确认 $S_c$ 中是否总是被最近的序列 $S_r$ 中的 token 关注。

对每个层中每个 $S_r$ 的 token，作者选择出 top 80% 的 attention weights 作为它们的 PvC。

> We set the keys and values with top-80% attention weights of the last token (d = 0) as the PvC selection baseline.

> we want to measure how much the overlap will be that the PvCs of recent tokens are consistent with the PvC of the last token.

作者希望统计出来，<mark>最近的一组 token 的 PvC 和最后一个 token 的 PvC 有多少重叠</mark>。

最近序列 $S_r$​ 中的标记与最后一个 token 选择的 PvCs 有平均 86% 的重叠。看来**确实存在一些共享的 PvCs，这些 PvCs 总是受到后续 token 的关注**。

虽然共享的 PvCs 在提供上下文信息时有一定作用，但它们并不能完全满足未来 token 预测的需求。

> Fortunately, the PvC selections from recent tokens have high consistency and we can integrate multiple tokens to select the shared ones

recent tokens 的 PvC 选择有比较高的一致性，可以拿多个 token 来选出来共享的 PvC。

### 3.2.2 Discussion

**Why do the deeper layers tend to have lower PvC overlap ratios?**

因为更深的 layer 有 context redundancy，只有一少部分 key 和 value 有比较高的权重能被总是选择为 PvC。

在深的 layer 中，只有少量的 key 和 value 拥有较高的权重，并总是被选为 PvCs。

有幂律分布的性质？

**Context information is mostly stored in the shared PvCs.**

作者发现上下文信息主要保存在共享的 PvC 中。

**The Association between ICR and RAC**

ICR: Inference Context Redundancy

RAC: Recent Attention Consistency

RAC. 可能是因为在处理语言或序列数据时，最近的上下文信息往往更为相关和重要，因此模型会倾向于关注这些高权重和重要的键和值。

> The insight behind these two power law distributions is the same. <mark>The high redundancy in deeper layers indicates that most of the keys and values are useless for inference.</mark>

**Further Verification of ICR about the Role of Non-PvCs**

> we have to verify the non-PvCs are redundant because they carry the information of predicting the tokens next to themselves instead of context information.

> The non-shared PvCs are also assigned high attention weights by the current token, which means they are useful for predicting the token next to the current token

> these three parts of the KV cache: 
> 
> 1. The shared PvCs are the keys and values that subsequent tokens collectively pay attention to.
> 2. The non-shared PvCs seldom appear in nonshared PvCs of other tokens. It means that non-shared PvCs are mostly highly interested in by the current token, with less attention from subsequent tokens. They are mainly used to predict the token next to themself in a teacher-forcing way, which is especially useful in training. 
> 3. Among the non-PvCs, a significant portion is occupied by non-shared PvCs of other tokens.

## 4 Layer-wise PvC Selection

![[Screenshot 2024-05-25 at 13.56.19.png]]