
最近想学习一下工业级的推理引擎的设计，不过 vLLM 和 SGLang 似乎都已经发展的比较复杂了，读起来大概会比较吃力。

前段时间看到一个[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，它的代码比较精简，但是也有一套完整的 Page Attention 和 Scheduler 的实现，国庆假期学习了一下，很有意思，在这里记录一下自己的理解。

## 代码结构

nano-vllm 的结构非常干净。除了内置一个 Qwen3 的模型结构定义，主要逻辑都在 `engine/` 目录，核心文件就这几个：

- `llm_engine.py`：对外暴露 `generate` 接口；接收请求交给 scheduler 处理。
- `sequence.py`：把请求封装为 `Sequence`，跟踪每个请求的生命周期和内存映射关系；
- `scheduler.py`：管理请求队列，处理请求的调度和抢占；
- `block_manager.py`：管理 KV Cache 内存块的分配和释放，以及按 hash 进行复用；
- `model_runner.py`：执行模型前向推理，scheduler 每一轮调度，均对应一次执行；

## 基本概念

### Page Attention

vLLM 快速流行的关键之一就是 Page Attention。nano-vllm 的核心设计也以此为基底，可以把它理解成 vLLM 的一个教学版实现。

在推理中，对内存管理的需求很直接：

1. 显卡的 KV Cache 总空间是固定的，除了模型权重等固定占用外，剩余显存都应当留给 KV Cache。怎样分配和管理这部分内存？
2. 用户请求的长度长短不同，每个请求对应的 KV Cache 也长短差异极大，如何在同一个 batch 内支持这些不同长度的请求？

Page Attention 会把整个用于 KV Cache 的内存切成固定大小的 Block，并维护“序列位置 → Block”的映射。这样做有几个好处：

1. 全局管理 KV Cache 相关的内存，通过分配表来管理 KV Cache 内存 Block 的分配与释放；
2. 允许为不同请求灵活拼装不同数量的 Block；
3. 可以在前缀一致时复用 Block，减少重复计算和存储。

### Sequence 和 Block

在 nano-vllm 的 Page Attention 实现中，两个最重要的概念是 `Sequence` 和 `Block`。

nano-vllm 的每个用户请求都会被封装成一个 `Sequence` 对象。

`Sequence` 对象的字段如下：

```python
class Sequence:
    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING  # RUNNING, FINISHED
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
```


在这些字段里，最核心的是 `block_table`：这是一个整数列表，按序列位置保存对应的 KV Cache 内存的 Block ID。另一个重要的字段是 `status`，用于跟踪请求生命周期中的状态流转。

其他字段则基本是一些周边的信息。

在 nano-vllm 中，一个 Block 的大小等于 256 个 embedding；例如一个请求累计 1000 个 token，就会占用 4 个 Block。

`Block` 的结构更简单：

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
```

每个 Block 有唯一 ID、引用计数、`hash` 和 `token_ids` 四个字段。其中：

- `block_id` 和 KV Cache 物理内存块对应，用于地址映射，Sequence 中的 `block_table` 中所指向的，就是这里的 Block ID。
- `ref_count` 用来追踪共享：多个序列复用同一前缀时，引用计数大于 1。
- `hash` 用于前缀复用（Prefix Cache）的快速命中，稍后还会再提。

画个图：

``` dot
// ========================================
// Version 1: Clear Mapping Diagram (Simplified)
// ========================================
digraph SequenceBlockMapping {
    rankdir=TB;
    node [fontname="Arial"];
    edge [fontname="Arial", fontsize=10];
    
    // Sequence object
    seq [label=<
        <table border="1" cellborder="0" cellspacing="0" cellpadding="6">
            <tr><td bgcolor="#ADD8E6" colspan="2"><b>Sequence 42</b></td></tr>
            <tr><td align="left">seq_id:</td><td>42</td></tr>
            <tr><td align="left">status:</td><td>RUNNING</td></tr>
            <tr><td align="left">tokens:</td><td>[101, 234, ..., 891]</td></tr>
            <tr><td bgcolor="#FFFACD" align="left"><b>block_table:</b></td>
                <td bgcolor="#FFFACD"><b>[3, 7, 12]</b></td></tr>
        </table>
    >, shape=plaintext];
    
    // Mapping table
    mapping [label=<
        <table border="1" cellborder="1" cellspacing="0" cellpadding="6">
            <tr><td bgcolor="#FFE4B5" colspan="3"><b>block_table Mapping</b></td></tr>
            <tr><td><b>Logical Block</b></td><td><b>→</b></td><td><b>Physical Block</b></td></tr>
            <tr><td>0</td><td>→</td><td>3</td></tr>
            <tr><td>1</td><td>→</td><td>7</td></tr>
            <tr><td>2</td><td>→</td><td>12</td></tr>
        </table>
    >, shape=plaintext];
    
    // Physical memory pool
    pool [label=<
        <table border="1" cellborder="1" cellspacing="0" cellpadding="4">
            <tr><td bgcolor="#FFA500" colspan="8"><b>GPU Physical Memory Pool</b></td></tr>
            <tr>
                <td bgcolor="#D3D3D3">[0]<br/>Other</td>
                <td bgcolor="#D3D3D3">[1]<br/>Other</td>
                <td>[2]<br/>Free</td>
                <td bgcolor="#E0FFFF"><b>[3]</b><br/>Seq42</td>
                <td>[4]<br/>Free</td>
                <td bgcolor="#D3D3D3">[5]<br/>Other</td>
                <td>[6]<br/>Free</td>
                <td bgcolor="#E0FFFF"><b>[7]</b><br/>Seq42</td>
            </tr>
            <tr>
                <td bgcolor="#D3D3D3">[8]<br/>Other</td>
                <td>[9]<br/>Free</td>
                <td bgcolor="#D3D3D3">[10]<br/>Other</td>
                <td>[11]<br/>Free</td>
                <td bgcolor="#E0FFFF"><b>[12]</b><br/>Seq42</td>
                <td colspan="3"></td>
            </tr>
        </table>
    >, shape=plaintext];
    
    // Connections
    seq -> mapping [label="contains", penwidth=2, color=blue];
    mapping -> pool [label="locates blocks", penwidth=2, color=red];
}

```

### Prefill 和 Decode

推理系统通常分为 Prefill 和 Decode 两个阶段：

- **Prefill 阶段**：计算用户输入的 prompt，把 KV Cache 填满该有的前缀。Prefill 算力密度高、序列长度长，通常一次只处理 1 个请求，多请求就 FIFO 排队。
- **Decode 阶段**：在已有 KV 的基础上增量生成，每次只长出 1 个 token。只跑一个 token 是远远吃不满 GPU 的计算密度的，因此往往需要把多个请求合成一个 Batch 一起跑。

``` dot
// ========================================
// KV Cache Perspective: Prefill vs Decode
// ========================================
digraph KVCachePrefillDecode {
    rankdir=LR;
    node [fontname="Arial"];
    edge [fontname="Arial", fontsize=10];
    
    label="";
    labelloc=t;
    fontsize=14;
    
    // Prefill Phase
    subgraph cluster_prefill {
        label="Prefill Phase: Fill KV Cache for Entire Prompt";
        style=rounded;
        color=blue;
        penwidth=2;
        
        // Sequence A
        p_seq_a_before [label=<
            <table border="0" cellborder="1" cellspacing="0" cellpadding="4">
                <tr><td colspan="6">Before Prefill</td></tr>
                <tr><td bgcolor="#ADD8E6"><b>Seq A</b></td><td>Empty</td><td>Empty</td><td>Empty</td><td>Empty</td><td>Empty</td></tr>
            </table>
        >, shape=plaintext];
        
        p_seq_a_after [label=<
            <table border="0" cellborder="1" cellspacing="0" cellpadding="4">
				<tr><td colspan="6">After Prefill</td></tr>
                <tr><td bgcolor="#ADD8E6"><b>Seq A</b></td>
                    <td bgcolor="#FFD700">K₁V₁</td>
                    <td bgcolor="#FFD700">K₂V₂</td>
                    <td bgcolor="#FFD700">K₃V₃</td>
                    <td bgcolor="#FFD700">K₄V₄</td>
                    <td bgcolor="#FFD700">K₅V₅</td>
                </tr>
            </table>
        >, shape=plaintext];
        
        p_seq_a_before -> p_seq_a_after [label="Process all 5 tokens", penwidth=2, color=blue];
    }

    // Decode Phase
    subgraph cluster_decode {
        label="Decode Phase: Incrementally Append 1 token to KV Cache";
        style=rounded;
        color=green;
        penwidth=2;
        
        // Before decode
        d_before [label=<
            <table border="0" cellborder="1" cellspacing="0" cellpadding="4">
                <tr><td colspan="7" bgcolor="#D3D3D3"><b>Before Decode</b></td></tr>
                <tr><td bgcolor="#ADD8E6"><b>Seq A</b></td>
                    <td>K₁V₁</td><td>K₂V₂</td><td>K₃V₃</td><td>K₄V₄</td><td>K₅V₅</td>
                    <td></td>
                </tr>
                <tr><td bgcolor="#FFE4B5"><b>Seq B</b></td>
                    <td>K₁V₁</td><td>K₂V₂</td><td>K₃V₃</td>
                    <td colspan="3"></td>
                </tr>
                <tr><td bgcolor="#E0FFFF"><b>Seq C</b></td>
                    <td>K₁V₁</td><td>K₂V₂</td>
                    <td colspan="4"></td>
                </tr>
            </table>
        >, shape=plaintext];
        
        // After one decode step
        d_after [label=<
            <table border="0" cellborder="1" cellspacing="0" cellpadding="4">
                <tr><td colspan="7" bgcolor="#90EE90"><b>After Decode Step</b></td></tr>
                <tr><td bgcolor="#ADD8E6"><b>Seq A</b></td>
                    <td>K₁V₁</td><td>K₂V₂</td><td>K₃V₃</td><td>K₄V₄</td><td>K₅V₅</td>
                    <td bgcolor="#FFD700">K₆V₆</td>
                </tr>
                <tr><td bgcolor="#FFE4B5"><b>Seq B</b></td>
                    <td>K₁V₁</td><td>K₂V₂</td><td>K₃V₃</td>
                    <td bgcolor="#FFD700">K₄V₄</td>
                    <td colspan="2"></td>
                </tr>
                <tr><td bgcolor="#E0FFFF"><b>Seq C</b></td>
                    <td>K₁V₁</td><td>K₂V₂</td>
                    <td bgcolor="#FFD700">K₃V₃</td>
                    <td colspan="3"></td>
                </tr>
            </table>
        >, shape=plaintext];
        
        d_before -> d_after [label="Generate 1 token\nfor all sequences\n(batched together)", penwidth=2, color=green];
    }
}

```

由于 Prefill 和 Decode 两个阶段的特点不同，Scheduler 需要做一些协调工作：

- 跟踪请求处理状态：尚未 Prefill 的先补 Prefill；
- 从已 Prefill 完成的请求里，组一个尽可能大的 Decode batch；

此外，Scheduler 还是内存分配/释放的发起者：

- Prefill 前要分配足够的 Block；
- Decode 时随着序列变长需要根据 KV Cache 的增长而继续分配 Block；
- 内存吃紧时，也需要做些驱逐来腾点空间出来。

反过来，驱逐与重跑，也属于 Scheduler 的协调工作，这些就都在状态机中进行协调。

## 主要流程

### 请求调度

从调度视角看，可以把一次完整的处理拆成「调度轮次」。每一轮调度，Scheduler 做决策、准备输入，然后由 `model_runner` 执行一次前向推理。

一轮调度循环的过程大致如下：

1. 接收新请求，封装为 `Sequence`，状态初始为 WAITING。
2. 对尚未 Prefill 的请求，尝试进行 Prefill：
   - 计算需要的 Block 数量；若可分配，则分配并执行 Prefill 前向；
   - Prefill 完成后，该序列进入可 Decode 的集合；
1. 若存在可 Decode 的序列，按根据一定策略组一个 batch，执行一次 Decode：
   - 每个序列生成 1 个新 token，更新 `last_token`、`num_tokens`；
   - 命中 EOS 或达到 `max_tokens` 的，标记 FINISHED；
1. 若内存不足以支撑下一次 Prefill/Decode，需要触发驱逐。

可见，在推理的调度中 Sequence 以及相关的几个队列是第一公民，调度相关的状态就都保存在里面。一个完整的请求，经过 Prefill、Decode、释放等状态的迁移，就都跟踪在 Sequence 及队列中。

``` dot
digraph SchedulerOverview {
      graph [rankdir=TB, bgcolor="#fbfbfb", nodesep=0.7, ranksep=0.9, fontname="Roboto"];
      node  [shape=box, style="rounded,filled", fontname="Roboto", fontsize=10, penwidth=1.3, color="#3c3c3c"];
      edge  [color="#555555", penwidth=1, arrowsize=0.9, fontname="Helvetica", fontsize=10];

      // Actors: colors group by owner
      Engine  [label="LLMEngine.add_request\ncreate Sequence", fillcolor="#ffe6a7", color="#d39a00"];
      WaitQ   [label="WAITING deque",                         fillcolor="#d8f3dc", color="#6abf83"];
      RunQ    [label="RUNNING deque",                         fillcolor="#d8f3dc", color="#6abf83"];
      Admit   [label="schedule(): prefill / decode\ncapacity & KV gate", fillcolor="#cfe2ff", color="#4e6cb6"];
      Preempt [label="preempt()\nfree blocks → WAITING front",          fillcolor="#cfe2ff", color="#4e6cb6", penwidth=1.6];
      Post    [label="postprocess()\nappend token",                    fillcolor="#cfe2ff", color="#4e6cb6"];
      Finish  [label="FINISHED\nrelease KV blocks",                    fillcolor="#cfe2ff", color="#4e6cb6"];
      GPU     [label="ModelRunner.run\nGPU batch",                     fillcolor="#e5d1ff", color="#7a57c1"];

      Engine -> WaitQ  [label="enqueue"];
      WaitQ  -> Admit  [label="dispatch request", color="#3a7a57", fontcolor="#2f5b41"];
      RunQ   -> Admit  [label="next decode cycle", color="#3a7a57", fontcolor="#2f5b41"];

      Admit  -> RunQ   [label="admit & allocate", color="#4e6cb6", fontcolor="#30518f"];
      Admit  -> Preempt[label="cannot allocate / append", color="#4e6cb6", fontcolor="#30518f"];
      Preempt -> WaitQ [label="requeue front", color="#4e6cb6", fontcolor="#30518f"];

      RunQ   -> GPU    [label="batch → GPU"];
      GPU    -> Post   [label="logits"];
      Post   -> Finish [label="EOS or max tokens", color="#4e6cb6", fontcolor="#30518f"];
      Post   -> RunQ   [label="continue decoding", color="#4e6cb6", fontcolor="#30518f"];
  }

```


### Block 分配与复用

在 Block 的分配与管理方面，Page Attention 也有点像 Linux 系统这样，所有的空闲内存都当作 Page Cache 用起来。在装载完模型参数之后，所有还剩下的显存也都当作 KV Cache 来用起来。

在初始化时，`BlockManager` 会初始化来跟踪起来 `free_block_ids`、`used_block_ids` 乃至 `hash_to_block_id`。

其中 `hash_to_block_id` 是内存复用的基础。它会为每个 Block 计算出哈希，每当填满一个 Block 时，都可以查一下有没有哈希相同的 Block，就不需要再单独分配 Block 了，增加引用计数即可。

值得一提的是，Block 的 hash 并不是单纯根据 Block 的数据计算出来的，也会结合前一个 Block 的哈希，形成一整个 Prefix Hash。

``` dot
digraph PrefixSharingExample {
    rankdir=TB;
    node [fontname="Arial"];
    edge [fontname="Arial", fontsize=10];
    
    label="Prefix Caching: Sharing Blocks via Hash";
    labelloc=t;
    fontsize=14;
    
    // Request A
    subgraph cluster_req_a {
        label="Request A";
        style=rounded;
        color=blue;
        
        req_a [label=<
            <table border="0" cellborder="1" cellspacing="0" cellpadding="4">
                <tr><td bgcolor="#ADD8E6" colspan="4"><b>Token Sequence A</b></td></tr>
                <tr>
                    <td bgcolor="#90EE90">[101, 234]</td>
                    <td bgcolor="#90EE90">[567, 891]</td>
                    <td bgcolor="#FFE4B5">[112, 223]</td>
                    <td bgcolor="#FFE4B5">[334]</td>
                </tr>
                <tr>
                    <td>Hash: 0x1a2b</td>
                    <td>Hash: 0x3c4d</td>
                    <td>Hash: 0xaaa1</td>
                    <td>Hash: 0xbbb2</td>
                </tr>
            </table>
        >, shape=plaintext];
    }
    
    // Request B
    subgraph cluster_req_b {
        label="Request B";
        style=rounded;
        color=green;
        
        req_b [label=<
            <table border="0" cellborder="1" cellspacing="0" cellpadding="4">
                <tr><td bgcolor="#E0FFFF" colspan="4"><b>Token Sequence B</b></td></tr>
                <tr>
                    <td bgcolor="#90EE90">[101, 234]</td>
                    <td bgcolor="#90EE90">[567, 891]</td>
                    <td bgcolor="#FFC0CB">[445, 556]</td>
                    <td bgcolor="#FFC0CB">[667]</td>
                </tr>
                <tr>
                    <td>Hash: 0x1a2b</td>
                    <td>Hash: 0x3c4d</td>
                    <td>Hash: 0xccc3</td>
                    <td>Hash: 0xddd4</td>
                </tr>
            </table>
        >, shape=plaintext];
    }
    
    // Shared blocks
    subgraph cluster_shared {
        label="Shared KV Cache Blocks (Same Hash)";
        style=rounded;
        color=orange;
        
        shared1 [label="Block 0\n[101, 234]\nHash: 0x1a2b", shape=box3d, fillcolor=lightgreen, style=filled];
        shared2 [label="Block 1\n[567, 891]\nHash: 0x3c4d", shape=box3d, fillcolor=lightgreen, style=filled];
    }
    
    // Private blocks
    private_a [label="Block 2 (A)\n[112, 223]\nHash: 0xaaa1", shape=box3d, fillcolor=lightyellow, style=filled];
    private_a2 [label="Block 3 (A)\n[334]\nHash: 0xbbb2", shape=box3d, fillcolor=lightyellow, style=filled];
    
    private_b [label="Block 2 (B)\n[445, 556]\nHash: 0xccc3", shape=box3d, fillcolor=lightpink, style=filled];
    private_b2 [label="Block 3 (B)\n[667]\nHash: 0xddd4", shape=box3d, fillcolor=lightpink, style=filled];
    
    // Connections
    req_a -> shared1 [label="share", penwidth=2, color=green, style=dashed];
    req_a -> shared2 [label="share", penwidth=2, color=green, style=dashed];
    req_a -> private_a [penwidth=2, color=blue];
    req_a -> private_a2 [penwidth=2, color=blue];
    
    req_b -> shared1 [label="share", penwidth=2, color=green, style=dashed];
    req_b -> shared2 [label="share", penwidth=2, color=green, style=dashed];
    req_b -> private_b [penwidth=2, color=green];
    req_b -> private_b2 [penwidth=2, color=green];
    
    // Note
    note [label="Same prefix → Same hash → Share blocks\nHash includes previous hash → Chain dependency", 
          shape=note, fillcolor=lightyellow, style=filled];
    
    shared1 -> shared2 [label="hash chain", color=red, penwidth=2, constraint=false];
}


```

### 抢占与驱逐

凡是有调度器概念的地方，就总是能见到 “驱逐” 字样存在。

在推理场景的驱逐，似乎只有一个出发点就是 KV Cache 的内存不够用了。这时可以驱逐一些请求出去，连带着 KV Cache 释放出来。

Prefill 阶段的驱逐会比较简单，只要把 waiting 队列中占用 KV Cache 的请求拿掉，腾出空间足够当前的 Prefill 任务完成就可以了。

Decode 阶段会随着 Decode 的进行，KV Cache 变得越来越长。这时如果分配 Block 失败，则触发抢占。这时就得从 running 队列中踢掉任务，从而腾出空间，给其余的请求使用。

``` dot
digraph PreemptionSimplified {
    rankdir=LR;
    node [fontname="Arial"];
    edge [fontname="Arial", fontsize=10];
    
    label="";
    labelloc=t;
    fontsize=14;
    
    // Before preemption
    subgraph cluster_before {
        label="Sequences Running, Memory Full";
        style=rounded;
        color=red;
        penwidth=2;
        
        running_before [label=<
            <table border="0" cellborder="1" cellspacing="0" cellpadding="6">
                <tr><td bgcolor="#90EE90" colspan="4"><b>Running Queue (3 sequences)</b></td></tr>
                <tr><td><b>Seq</b></td><td colspan="3"><b>Blocks</b></td></tr>
                <tr>
                    <td bgcolor="#ADD8E6">Seq A</td>
                    <td bgcolor="#ADD8E6">[0]</td>
                    <td bgcolor="#ADD8E6">[1]</td>
                    <td colspan="1"></td>
                </tr>
                <tr>
                    <td bgcolor="#FFE4B5">Seq B</td>
                    <td bgcolor="#FFE4B5">[2]</td>
                    <td bgcolor="#FFE4B5">[3]</td>
                    <td bgcolor="#FFE4B5">[4]</td>
                </tr>
                <tr>
                    <td bgcolor="#E0FFFF">Seq C</td>
                    <td bgcolor="#E0FFFF">[5]</td>
                    <td bgcolor="#E0FFFF">[6]</td>
                    <td colspan="1"></td>
                </tr>
            </table>
        >, shape=plaintext];
        
        memory_before [label=<
            <table border="0" cellborder="1" cellspacing="0" cellpadding="4">
                <tr><td bgcolor="#FFA500" colspan="8"><b>GPU Memory Pool</b></td></tr>
                <tr>
                    <td bgcolor="#ADD8E6">[0]<br/>A</td>
                    <td bgcolor="#ADD8E6">[1]<br/>A</td>
                    <td bgcolor="#FFE4B5">[2]<br/>B</td>
                    <td bgcolor="#FFE4B5">[3]<br/>B</td>
                    <td bgcolor="#FFE4B5">[4]<br/>B</td>
                    <td bgcolor="#E0FFFF">[5]<br/>C</td>
                    <td bgcolor="#E0FFFF">[6]<br/>C</td>
                    <td bgcolor="#D3D3D3">[7]<br/>Other</td>
                </tr>
                <tr><td colspan="8" bgcolor="#FF6B6B" align="center"><b>❌ No free blocks!</b></td></tr>
            </table>
        >, shape=plaintext];
        
        running_before -> memory_before [style=invis];
    }
    
    // After preemption
    subgraph cluster_after {
        label="Sequences Running, Memory Available";
        style=rounded;
        color=green;
        penwidth=2;
        
        running_after [label=<
            <table border="0" cellborder="1" cellspacing="0" cellpadding="6">
                <tr><td bgcolor="#90EE90" colspan="4"><b>Running Queue (2 sequences)</b></td></tr>
                <tr><td><b>Seq</b></td><td colspan="3"><b>Blocks</b></td></tr>
                <tr>
                    <td bgcolor="#ADD8E6">Seq A</td>
                    <td bgcolor="#ADD8E6">[0]</td>
                    <td bgcolor="#ADD8E6">[1]</td>
                    <td bgcolor="#FFD700">[5]</td>
                </tr>
                <tr>
                    <td bgcolor="#FFE4B5">Seq B</td>
                    <td bgcolor="#FFE4B5">[2]</td>
                    <td bgcolor="#FFE4B5">[3]</td>
                    <td bgcolor="#FFE4B5">[4]</td>
                </tr>
            </table>
        >, shape=plaintext];
        
        memory_after [label=<
            <table border="0" cellborder="1" cellspacing="0" cellpadding="4">
                <tr><td bgcolor="#FFA500" colspan="8"><b>GPU Memory Pool</b></td></tr>
                <tr>
                    <td bgcolor="#ADD8E6">[0]<br/>A</td>
                    <td bgcolor="#ADD8E6">[1]<br/>A</td>
                    <td bgcolor="#FFE4B5">[2]<br/>B</td>
                    <td bgcolor="#FFE4B5">[3]<br/>B</td>
                    <td bgcolor="#FFE4B5">[4]<br/>B</td>
                    <td bgcolor="#FFD700">[5]<br/>A<br/>(new)</td>
                    <td bgcolor="#90EE90">[6]<br/>Free</td>
                    <td bgcolor="#D3D3D3">[7]<br/>Other</td>
                </tr>
                <tr><td colspan="8" bgcolor="#90EE90" align="center"><b>✓ 1 free block!</b></td></tr>
            </table>
        >, shape=plaintext];
        
        running_after -> memory_after [style=invis];
    }
    
    memory_before -> running_after [label="Evict Seq C", penwidth=3, color=red, fontsize=12];
}

```

## 小结

nano-vllm 提供了一套最小可行的多请求调度与 KV Cache 管理实现，代码结构清晰，对理解 Page Attention 和 LLM Scheduler 确实是很好的素材。后面有空可以在这个基础上接着捋一下 vLLM 的流程。