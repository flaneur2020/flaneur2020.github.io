TLDR：

1. 仅需 24GB 显卡和 136 GB 内存就可以跑起来 Deepseek V2 模型。
2. llama.cpp 把一些特定的层挪到 GPU 的办法，对 MoE 不大合适；
3. MoE 大部分参数都是摸鱼的，但是占用宝贵的显存；
4. 作者的思路是<mark>把 MLA 这种计算密度高的操作先放在 GPU，摸鱼多的 Routed Expert 参数，都放在内存里用 CPU 跑</mark>；
5. deepseek v3 的激活参数 37B，共享网络参数 16.5B、Expert 参数 20.5B（每个 token 激活 8 个专家，每个专家约 2.56B 参数）
6. 这意味着，GPU 里放共享网络参数 16.5B 就可以，24GB 显存的 4090 就能跑起来；

## Offload MoE Experts

对模型参数总量巨大，但是每次激活参数较少的 MoE 模型来讲，异构推理会挺有优势。

不过 llama.cpp 只支持把一些特定的层挪到 GPU、把 kv cache 挪到 CPU 这样的简单策略。

这对 deepseek 这种模型就不大好办：

7. MLA 算子的计算量大；
8. 按层 off load 占的显存也不少，MoE 的话闲置的就会摸鱼；

作者测试下来发现在 llama.cpp 开了 offload 也不会怎么变快。

作者提出了基于计算强度的 offload 策略。优先将计算强度高的计算放到 GPU（MLA > Shared Expert > Routed Expert），直到 GPU 放不下为止。

