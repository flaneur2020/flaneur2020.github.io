
> Small-scale models have also shown a rapid increase in performance, but these gains are largely derived from increasing the length of training. This approach only scales log-arithmically with dataset size, and the latest small models require up to 15T tokens to improve the state of the art by less than 1-2%.

> Yes, these continued improvements provide evidence that small models are still under-trained. In this work, we explore alternatives to improve small models performance without solely increasing training length. <mark>One solution is to improve the quality of infomation received by the next token prediction task with a richer objective.</mark>

> In particular, we focus our efforts on knowledge distillation, <mark>which replaces the one-hot vector seen at each token with the distribution of potiential next tokens computed from a large model.</mark> This approach is often used to reduce the training time of smaller models by giving them richer gradients. In this work, we instead train for large quantities of tokens with distillation in order to simulate training beyond the number of available tokens. Concretely, we use a larger language model as a teacher to train small models, namely 9B and 2.6B models. On a quantity of tokens that is more than 50x compute-optimal quantity predicted by theory. Along with the models trained with distillation, we also release a 27B model trained from scratch for this work.

## 2. Model Architecture

与 Gemma 1 的差异包括：

> **Local Sliding Window and Global Attention**: We alternate between a local sliding window attention and global attention in every other layer. The sliding window size of local attention layers is set to 4096 tokens, while the span of global attention layers is set to 8192 tokens.

隔层交替使用 local sliding window attention 和 global attention。

> **Logit soft-capping**: Following Gemini 1.5, we cap logits in each attention layer and the final layer such that the value of the logits stays between −soft_cap and +soft_cap. More specifically, we set the logits as `logits ← soft_cap ∗ tanh(logits/ soft_cap)`.

> **Post-norm and pre-norm with RMSNorm**

> **Grouped-Query Attention**: 27B 和 9B 的模型都使用的 GQA，num_groups 为 2 。

## 3. Pre-training

训练 27B 用了 13T 的 tokens。9B 模型用了 8T、2.6B 模型用了 2T。
### 3.2 Knowledge Distillation

> Given a large model used as a teacher, <mark>we learn smaller models by distilling from the probability given by the teacher of each token $x$ given its context 𝑥 , i.e.,  $P_T(x | x_c)$</mark>. More precisely, we minimize the negative log-likelihood between the probabilities from the teacher and the student:
> 
> $$ min_{P_s} \sum_{x} - P_T(x|x_c) \log P_S (x|x_c) $$

其中 $P_T$ 是 teacher 的概率、$P_S$ 是学生的概率。

## 4. Post Training

> - Supervised Fine Tuning
> - RLHF
> - Model Merging: We average models from experiments run with different hyperparameters.
> - Data filtering
> - Formatting: 

## 5. Ablations




