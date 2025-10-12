
最近想学习一下工业级的推理引擎的设计，不过 vLLM 和 SGLang 似乎都已经发展的比较复杂了。前段时间看到一个[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，它的代码比较简短，但是也有一个完整的 page attention 和 scheduler 的实现，国庆假期学习了一下，很有意思，在这里记录一下。

感兴趣的点主要在：

- 怎样处理多个 request 的？
- 怎样抢占的？
- 怎样管理 kv cache 的内存的？

## 代码结构

nano-vllm 的代码结构非常简洁，除了一个内置的 qwen3 模型的结构定义，主要代码在 engine/ 目录中，只有这几个文件：

- llm_engine.py
- sequence.py
- scheduler.py
- block_manager.py
- model_runner.py

### `llm_engine.py`

`LLMEngine` 是 LLM 推理引擎的入口，主要对外提供 `generate()` 这个接口，将 prompt 转换为 `Sequence` 通过 `add_request()` 传入 scheduler 进行处理。

### `sequence.py`

每个请求会被封装为 `Sequence` 对象，在 Sequence 中，除了请求的状态，也关联管理着每个请求的 `block_table` 也就是逻辑上的 KV Cache 块到物理内存的映射关系。

### `scheduler.py`

`Scheduler` 会将请求在 waiting、running 队列中管理，暴露 `add()` 接收新的请求，每次调度 `schedule()` 向前走一步推理，并在内存紧张时，通过 `preempt()` 管理驱逐。

### `block_manager.py`

管理 kv cache 内存块的分配与释放。

### `model_runner.py`

真正的执行层，对应 scheduler 每一步调度，执行一步推理。

似乎主要对应 pytorch 的集成，像 `BlockManager` 中管理的块，会对应到 pytorch 的 Tensor 中来体现真实的内存。

nano-vllm 甚至做了多卡的多进程推理的支持，有一个简单的共享内存发送指令、event 通知协同等待的机制。

## 主要流程

### 请求调度

``` dot
digraph SchedulerOverview {
      graph [rankdir=TB, bgcolor="#fbfbfb", nodesep=0.7, ranksep=0.9, fontname="Helvetica"];
      node  [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, penwidth=1.3, color="#3c3c3c"];
      edge  [color="#555555", penwidth=1.2, arrowsize=0.9, fontname="Helvetica", fontsize=10];

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

### 内存管理