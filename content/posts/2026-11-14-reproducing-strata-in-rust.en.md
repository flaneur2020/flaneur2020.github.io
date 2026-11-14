---
date: "2026-11-14"
title: "Reproducing Strata in Rust"
language: "en"
---

Strata is an on-device inference framework I came across a while ago. It can run a 100B+ Qwen 3.8 Flash Next model on a single GPU at tens to over a hundred tok/s, enough to genuinely run some agent tasks. A year ago, this would have been completely unimaginable.

My day job is related to LLM infrastructure, but I don't get to work directly with inference frameworks very often. So I wanted to reproduce Strata in Rust to get a feel for it; my gaming PC at home is also relatively easy to debug on. To be honest, the main barrier to working on inference is still the cost of the hardware environment. Now that there's proof that a model of this 100B+ size can run on a single GPU, this is also a good entry point for getting hands-on with this kind of work.

After more than two weeks of tinkering, I can barely match Strata's performance on my own machine. The code is at: <https://github.com/flaneur2020/naiwa-rs>

Locally I'm running the Coder variant of Qwen 3.8 Flash Next. It cuts the number of expert parameters in half, so it's around 70–80B in total. Performance-wise, prefill is around 1,000 tok/s, and decode TPOT is around 36.8 ms, i.e. about 27.2 tok/s. Just looking at generation speed, that's quite sufficient.

The hardware is an RTX 4070 with 12 GiB of VRAM, an AMD Ryzen 3600, and 64 GiB of DDR4 memory. While debugging, I found this configuration isn't ideal for local LLMs. With more CPU cores, AVX512, and DDR5, it would definitely run faster.

Because of budget constraints, the models I mainly had access to were DeepSeek V4.1 Flash and GPT-6.2-Sol. If I had paid for Claude models, I guess it wouldn't have taken me this long—but the tinkering is part of the process. Using cheaper models still leaves some of the fun of hitting bottlenecks and overcoming them...

These days, simply documenting technology doesn't mean much anymore. This blog post is just a record of how I felt during this vibe-coding process—written entirely by hand, with no agent involvement.

## Qwen 3.8 Flash Next and Strata

Qwen 3.8 Flash Next is a very sparse model. Its core body totals 125B parameters, most of which are expert parameters, but only 6B parameters are activated. It has 512 experts per layer, and only 10 are activated per layer. In addition to the model body, there are 51B of Ngram embedding parameters. Altogether, that comes to roughly 170B.

Even at just under 170B in total, quantization alone still isn't enough to run it on consumer hardware. But this model has two advantages:

1. It is genuinely sparse: only a small fraction of expert parameters are activated.
2. Although the Ngram parameter count isn't small, they don't need to be loaded into memory; each token only requires one lookup from the SSD.

Strata's approach is to put effort into expert scheduling, keeping only the relevant, active experts in VRAM. In theory, with 10 experts per layer and each expert only four or five megabytes, even a consumer GPU with only 12 GiB of VRAM can hold them. However, the experts on the GPU have a cache-like, predictive nature. If there is a cache miss or a prediction failure, the missed experts are run directly on the CPU; running two or three billion activated parameters on the CPU is indeed feasible. The delicate part here is overlapping CPU computation and GPU computation.

## Forking mistral.rs

mistral.rs already has an existing implementation of Qwen 3.8 Flash Next. So in our repository, we simply depend on the mistral.rs crate and assemble a server, turning off all models other than Qwen 3.8 Flash Next to reduce some of the context we need to deal with.

When it comes to implementing expert scheduling, we need to make some forks so that we can put the expert-scheduling logic in our own repository.

Roughly, we insert a trait like this:

```rust
+/// Routed experts of one MoE block.
+pub trait RoutedExperts: Send + Sync {
+    /// `xs: [batch, seq, hidden]` on the layer device; `topk_weights` / `topk_ids`:
+    /// `[tokens, top_k]` from the model's router. Returns `[tokens, hidden]` on the layer device.
+    fn forward(&self, xs: &Tensor, topk_weights: Tensor, topk_ids: &Tensor) -> Result<Tensor>;
+}
+
+/// Supplies routed experts for MoE blocks at load time.
+pub trait RoutedExpertsProvider: Send + Sync {
+    /// Return `Some` to own this block's routed experts, or `None` to let the model load them.
+    fn build(&self, ctx: ExpertLoadContext<'_>) -> Result<Option<Arc<dyn RoutedExperts>>>;
+}
```

I saw that the Agent just used a global-variable registry for injection. It's a bit wild, but there's no doubt that this keeps the change surface small. This fork isn't meant to go upstream, so we just do whatever keeps the change surface smallest.

Besides the expert change surface, another thing that needs to be added in the fork is support for Q2_0 and various i-Quant formats, such as IQ4_XS, IQ4_NL, IQ3_XXS, IQ2_S, and IQ3_S. Strata's quantization uses different quantization formats for different layers. Without support for these formats, you won't even get to inference—it will error out while loading the model.

Wiring up these quantization formats is grunt work: you need to write both the CPU and CUDA operators. But the work doesn't seem to require much intelligence; I just used ds v4.1 flash and had them all written in an afternoon.

## Prefill

Prefill and decode differ quite a bit in their access patterns for expert weights.

During prefill, almost every expert gets touched, so pruning experts doesn't mean much; each layer needs to touch almost all experts.

But in my experience, getting high performance during prefill is actually relatively simple: you just need to overlap loading all experts for the next layer into GPU memory with the computation of the current layer.

Transferring all the expert parameters for the next layer takes a non-trivial amount of time. But one advantage of prefill is that we can build larger batches so that the compute density overlaps with the transfer time. Even when going through all the layers takes on the order of 3 seconds, with batches of 3,000, overall prefill performance is still 1k tok/s.

![](/images/2026-11-14-naiwa-prefill-overlap-1.png)

However, within this overlap, the computation of the first layer still has a blocking wait. And VRAM needs to have room for two layers' worth of expert parameters.

One option to consider here is increasing the overlap depth. The rough idea is to split a layer into two or even four pieces. In this overlap pipeline, you compute the parameters for one subset of experts at a time, halving both the transfer volume and the compute volume per stage. That also lets you stack the pipeline deeper—for example, we can prefetch the expert weights needed for the next two stages in advance:

![](/images/2026-11-14-naiwa-prefill-overlap-2.png)

At first glance this idea makes a lot of sense, but it didn't seem to improve the benchmark noticeably. My guess is that once the batch is large enough, the time spent on compute-intensive work exceeds the time spent on memory transfers, so splitting things up yields little benefit and instead adds some launch overhead.

## Decode

I spent much more time on decode than on prefill—there were unexpectedly many details.

The idea for decode is also straightforward: after the expert router produces the list of experts, check whether the expert is in the GPU's expert cache. If so, prefer running it on the GPU; otherwise, run it on the CPU.

That's the basic idea. But during iteration I added an extra rule: for some experts that are not in the GPU's expert cache, also transfer them to the GPU over PCIe and compute them there. After all, while the GPU is waiting for the CPU results, its compute capacity is still fairly idle.

On top of that, I considered adding a readahead mechanism to preload the next layer's experts in advance. In the end, it looked roughly like this:

![](/images/2026-11-14-naiwa-decode-pipeline.png)

After the expert router splits the work, there are three paths:

1. The GPU computes the experts already in VRAM.
2. For missed experts, copy a small subset into GPU memory and compute them on the GPU.
3. The remaining experts are computed in place on the CPU and the results are returned to the GPU.

This flow is unexpectedly synchronization-heavy. From what I can tell, Strata doesn't use native CUDA streams for the transfer. Instead, it uses a doorbell mechanism: while the GPU is waiting for the CPU, it spin-reads a shared-memory location. When the CPU finishes its work, it writes the result back to shared memory and changes an atomic variable there to tell the GPU to exit the spin loop.

Another reason Strata does this is that it records the entire decode step into a CUDA Graph, so one decode only needs one CUDA launch. During CUDA Graph execution there is naturally no other CUDA communication, so a shared-memory doorbell mechanism is a lightweight way to get high responsiveness.

At one point I wanted to learn this full CUDA Graph recording approach. But after a week of tinkering, I found that doing CUDA Graphs requires starting from relatively low-level CUDA primitives. If any piece of the code does something like candle's Tensor `.to_vec()` to copy data back to the CPU, it can't be recorded. The conclusion from all this is that if you want to record CUDA Graphs, it's best to design for them from day one. Trying to add them underneath halfway through is painful and too invasive, so I eventually gave up on that part. I did borrow the doorbell mechanism, though, because busy polling like this should help reduce latency.

But after implementing all this, I was still a bit short of Strata's performance; TPOT was consistently 2–3 ms worse.

Because I was trying to save money, I went through a long frustrating period here. While using DeepSeek V4.1 Flash to optimize this, I often got stuck in a loop like this:

1. ds v4.1 flash would give a few hypotheses and feasible directions, then list a table-like checklist.
2. I'd work on the most likely item on the checklist.
3. To do that work, I'd first need to solve A/B/C/D, step by step.
4. After finishing all the steps, the benchmark showed no gain, so I'd mark it as "dead."
5. Then I'd prompt it again to analyze the current state, and it would list a few more possible directions.
6. The loop would continue until another "dead."
7. When I prompted it to analyze again, it would bring up the same hypotheses from the very beginning.

Since we had already decided not to switch to CUDA Graphs, the only option was to squeeze performance out of eager mode bit by bit. If the gap really came from CUDA Graphs, there was no way to go further.

At my wit's end, I turned to GPT 6 Astral, and the picture suddenly became much clearer. The most useful prompt I found in this process was "please attribute this in detail" rather than a direct "optimize performance." To do the attribution, the agent inserts a lot of instrumentation into the code. With detailed attribution, optimization became much simpler, and at that point I could use GPT-6.2-Sol for the work and save a lot of money.

In the end I made roughly these optimizations:

1. Start the CPU expert computation before the DMA transfer of missed experts to the GPU; this improved things by about 0.9–1 ms.
2. Reduce lookahead-swaps from 2 to 1 to reduce expert-cache thrashing. It seems that promoting more experts to the GPU doesn't improve the hit rate much; this saved about 0.9 ms.
3. Move expert routing to pure GPU computation instead of passing it through the CPU (it shouldn't have gone through the CPU in the first place, but that's how it ended up when vibe coding); this saved about 1.5 ms.
4. Change CPU expert computation from rayon's default thread pool to a busy-polling spin-task thread pool, and pin threads to physical cores; this saved about 1–2 ms.

After these rounds of optimization, I finally matched Strata's speed on this machine (around 36.8 ms). If I had used a SOTA model earlier, I probably could have solved this optimization problem sooner. But without those round trips, I think my understanding of how deep this problem is—and of what SOTA models are capable of—would still be somewhat disconnected.

## Thoughts

In the decode flow, it's easy for the human mind to draw a dataflow diagram that makes perfect sense. But at the micro level, execution often contains details that are hard to show in that high-level diagram. At that point, there's no other way besides careful attribution and benchmarking. And attribution and benchmarking are grueling trial-and-error work. Without a powerful agent, it's really hard to imagine how much time it would take to grind through this by hand!

On the other hand, even a powerful AI won't necessarily produce optimal code in one shot purely by intuition. It also has to first write something that runs, then build its own performance-analysis harness, do attribution, and run benchmarks. It has to go through the same trial-and-error process. So, as far as the whole process goes, it isn't necessarily a case of one prompt and a wish coming true. Objectively speaking, there's still a little fun left in observing the experiment. I think if I come across an interesting project later, I can still use a similar approach to reproduce the experiment and see how it goes.
