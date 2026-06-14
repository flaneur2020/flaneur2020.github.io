## How to think about scaling to multiple GPUs

大约就是相当于一个模型，replicate 到 N 个 GPU 上，每个 GPU 上训练得到一定的 gradient，将这些 gradient 合并回同一个 model 参数上。

## Training a Model

模型一次训练的步骤：

1. Create batches of data.
2. At each step, feed one batch through the model.
3. Get logits, pass through softmax for scores, calculate loss.
4. Call loss.backward() to compute gradients.
5. Call optimizer.step() to update weights.

其中第 5 步，负责将权重更新回模型。

## Wait, isnt this just Data Parallelism?

pytorch 之前就有一个 DP 模块，做的事情在概念上都是一样的。区别在实现方面：

1. DP 使用单个进程（因为 GIL 的限制）；将输入 scatter 给 GPU、计算 loss、广播回 gradient；这有很多开销，因为 main GPU（rank 0）会成为通信的瓶颈；
2. DDP 给每个 GPU spawn 一个进程；每个进程有自己的 interpreter，拥有 optimizer；没有中心的 coordinator；gradient 会通过 all-reduce 在反向传播中间同步，通信与计算实现能实现 overlap；

## Gradient Synchronisation and All-Reduce

![[Pasted image 20260614180715.png]]

每个 GPU 最后会得到完整的同样的 gradient，最后传递给 optimizer.step()，每个 replica 保持同步；

### Ring All-Reduce

![[Pasted image 20260614180807.png]]

 naive 的 all0-reduce 会给单个节点较大的压力；每个 gpu 要么是中心化的 coordinator，要么与每个其他的 GPU 进行通信，创建 $O(N^2)$ 的通信压力；

Ring 的 all reduce 能够避免这种 bottleneck。N 个 GPU 组成了一个逻辑上的 ring。每个 GPU 只与它的邻居通信：

1. 每个进程独立计算 gradient；
2. 每个进程将 gradient 传递到下一个进程，并将从前一个进程中得到的 gradient 继续向下传递；在循环 N 次后，每个进程都将拥有同样的所有 gradient；

（这个 ring 似乎相当于转了整个一圈）

## Basic Terminology

### World Size

涉及整个 distributed job 的所有进程数。

如果有两台机器、每个机器 8 张卡，world size = 16。

```
torch.distributed.get_world_size()
```

### Rank

一个 unique identifier（0 到 world_size - 1）分配给每个进程。`rank 0` 会认为是 master 进程。

```
torch.distributed.get_rank()
```

### Local Rank

单台机器里的 process index。比如 8 卡的一个机器，local rank 的取值为 0～7。

```
int(os.environ['LOCAL_RANK'])
```

### Master Address & Port

所有进程都需要知道 master 进程（rank 0）的地址，才能初始化整个组。

需要的环境变量主要是 `MASTER_ADDR` 和 `MASTER_PORT`。

### Backend

在 Nvidia GPUs 上，就是 NCCL。CPU 的话，可以使用 GLOO 或 MPI。

## Launching with torchrun

```
torchrun --nproc_per_node=8 train.py
```

它会自动启动 8 个进程，每个进程设置自己的 `RANK`, `LOCAL_RANK`, `WORLD_SIZE` 等环境变量。