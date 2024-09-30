## The challenges of large-scale model training

- **Hardware reliability**
- **Fast recovery on failure**
- **Efficient preservation of the training state**
- **Optimal connectivity between GPUs**

## Innovating across the infrastructure stack

> We also pivoted by modifying the [Grand Teton](https://engineering.fb.com/2022/10/18/open-source/ocp-summit-2022-grand-teton/) platform that was developed using NVIDIA H100 GPUs, increased the TDP of the GPUs to 700W, and moved to HBM3 on the GPUs. Since we did not have time to change the cooling infrastructure, we had to remain in an air-cooled environment. The mechanical and thermal designs had to change to accommodate this, and that triggered a validation cycle to support a large-scale deployment.

### Network

> There are two leading choices in the industry that fit these requirements: RoCE and InfiniBand fabrics. Both of these options had tradeoffs. On the one hand, Meta had built RoCE clusters for the past four years, but the largest of those clusters only supported 4K GPUs. We needed significantly larger RoCE clusters. On the other hand, Meta had built research clusters with InfiniBand as [large as 16K GPUs](https://ai.meta.com/blog/ai-rsc/). However, those clusters were _not_ tightly integrated into Meta’s production environment, nor were they built for the latest generation of GPUs/networking. This made for a difficult decision of what fabric to build with.
>
> So we decided to build both: [two 24k clusters](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/), one with RoCE and another with InfiniBand. Our intent was to build and learn from the operational experience. These learnings will inform the future direction of GenAI fabrics. We optimized the RoCE cluster for quick build time, and the InfiniBand cluster for full-bisection bandwidth. We used both InfiniBand and RoCE clusters to train [Llama 3](https://ai.meta.com/blog/meta-llama-3/), with the RoCE cluster used for training the largest model. Despite the underlying network technology differences between these clusters, we were able to tune both of them to provide equivalent performance for these large GenAI workloads

弄了 RoCE 和 infiniband 两个集群，都能跑起来。