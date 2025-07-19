> But I still thought of myself as a scale maximalist: all that really mattered, I thought, was training bigger models on more data. Anything else (read: reasoning models) appeared to be a coping mechanism, just a way to get by while we wait for the hardware needed to train bigger models.

> RL isn’t just a way to give models more compute. **RL training really is teaching models something** _**different**_**, a way to use compute to generate better answers** given finite model capacity. Through RL, models are clearly learning something that they’re not getting from pretraining.

## Two waves of AI scaling

> The AI research-into-production cycle moves through a few distinct phases. First, we as a community identify a new learning paradigm. Second, we find the correct datasets for training and design evaluations to know when our models are getting better, and by how much. And third, we scale it to all hell.

1. 找到一个新的 paradigm
2. 找出来正确的 dataset 和 design evaluation 方法
3. scale 到极限

## The murky path to RL scaling starts with _data_

> We’ve identified a new paradigm: learning to _reason_. But reasoning models are [in their GPT-3 era](https://www.mechanize.work/blog/the-upcoming-gpt-3-moment-for-rl/): they’re trained on small datasets to do a narrow selection of tasks. We have a brittle proof-of-concept in the reasoning models of 2025. These models have achieved state-of-the-art scores on a small number of tasks, mostly expert-level math and coding questions.
