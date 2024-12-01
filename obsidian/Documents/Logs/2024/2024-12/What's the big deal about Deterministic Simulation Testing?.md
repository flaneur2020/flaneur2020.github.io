
隔离掉一切分布式系统中的随机性，使多个系统在通信中，使用单线程的方式，控制一切随机性；

然后针对这个单线程程序应用 property test，同时注入错误；

在 foundationdb、antithesis、tigerbeetle、polar signals、warpstream 等系统中应用了。

## Randomness and time

不确定性的一大来源是随机数；

DST 假设系统中的随机数都来自一个统一的 seed；

这个 seed 可以有 simulator 管理，如果遇到错误，则允许重放；

另一个来源是时间依赖。

DST 不是说程序不能依赖时间，而是说，程序需要一个受控的时钟；

这意味着，<mark>需要一些 dependency injection </mark>的办法，来管理时钟和随机数。

## Converting an existing function

```python
class Backoff:
  def init:
    this.rnd = rnd.new(seed = time.now())
    this.tries = 0

  async def retry_backoff(f):
    while this.tries < 3:
      if f():
        return

      await time.sleep(this.rnd.gen())
      this.tries++
```

修改后：

```python
# retry.psuedocode
class Backoff:
  def init(this, time):
    this.time = time
    this.rnd = rnd.new(seed = this.time.now())
    this.tries = 0

  async def retry_backoff(this, f):
    while this.tries < 3:
      if f():
        return

      await this.time.sleep(this.rnd.gen())
      this.tries++
```

然后可以写一个 simulator：

```python
# sim.psuedocode
import "retry.pseudocode"

sim_time = {
  now: 0
  sleep: (ms) => {
    await future.wait(ms)
  }
  tick: (ms) => now += ms
}

backoff = Backoff(sim_time)

while true:
  failures = 0
  f = () => {
    if rnd.rand() > 0.5:
      failures++
      return false

    return true
  }
  try:
    while sim_time.now < 60min:
      promise = backoff.retry_backoff(f)
      sim_time.tick(1ms)
      if promise.read():
        break

    assert_expect_failure_and_expected_time_elapse(sim_time, failures)
  catch(e):
    print("Found logical error with seed: %d", seed)
    throw e
```

<mark>simulator 本身会依赖随机性</mark>。

会生成 seed，并允许重放。

### A single thread and asynchronous IO

多线程的确定性只能通过操作系统、emulator 或者 hypervisor 层面来控制。

Antithesis 或者 Hermit 这样的系统，允许将多线程代码透明地转换为单线程代码。

这几个系统有些限制：1. 错误注入的能力不算强；2. 不能在 mac 上跑；3. 不能再 arm 上跑。

我们的 DST 不一定需要依赖它们，但是我们需要允许使我们的代码在单线程执行。而且，不能用 blocking IO，这会导致  concurrency bug 不能被发现。

简单说就是要求：

1. 单线程
2. 异步 IO

Go 这种程序，在 Polar Signals 他们通过将 go 编译到 wasm，实现了单线程的保证。

不过即使在单线程中，go runtime 仍然会随机地调度 goroutine。

因此 Polar Signals 他们选择了 fork go 的 runtime，来通过环境变量来控制这里调度的随机性。

Resonate 使用了另一种方式，但是也挺脏。

总的来说，在作者看来 Go 在做 DST 方面不是一个好的环境。

rust 没有内置的 async IO，主要是基于 tokio 做的。tokio 的老铁做了一个兼容 tokio 的 simulator 实现，移除了所有的 nondeterminism。

但是不是完全的成功，这个 repo 被一个叫做 termoil 且标记为“this is very experimental” 的 tokio-rs 项目替代。（它支持网络错误注入但不支持磁盘错误注入）

tokio 是一个大型项目，有很多二级依赖，所有都要做到 non-determinism 也确实有点难度。

不过，Pekka 演示了做一个为 simulation test 而设计的 rust 的 async runtime 的可行性。

## A distributed system

这个会更难一点，相当于把多个 node 编码在一个进程中跑。

```python
# sim.pseudocode
import "distsys-node.pseudocode"

seed = if os.env.DST_SEED ? int(os.env.DST_SEED) : time.now()
rnd = rnd.new(seed)

while true:
  sim_fd = {
    send(fd, buf) => {
      # Inject random failure.
      if rnd.rand() > .5:
         throw Error('bad write')

      # Inject random latency.
      if rnd.rand() > .5:
        await time.sleep(rnd.rand())

      n_written = assert_ok(os.fd.write(buf))
      return n_written
    },
    recv(fd, buf) => {
      # Inject random failure.
      if rnd.rand() > .5:
         throw Error('bad read')

      # Inject random latency.
      if rnd.rand() > .5:
        await time.sleep(rnd.rand())

      return os.fd.read(buf)
    }
  }
  sim_io = {
    open: (filename) => {
      # Inject random failure.
      if rnd.rand() > .5:
        throw Error('bad open')

      # Inject random latency.
      if rnd.rand() > .5:
        await time.sleep(rnd.rand())

      return sim_fd
    }
  }

  all_ports = [6000, 6001, 6002]
  nodes = [
    await distsys-node.start(sim_io, all_ports[0], all_ports),
    await distsys-node.start(sim_io, all_ports[1], all_ports),
    await distsys-node.start(sim_io, all_ports[2], all_ports),
  ]
  history = []
  try:
    key = rnd.rand_bytes(10)
    value = rnd.rand_bytes(10)
    nodes[rnd.rand_in_range_inclusive(0, len(nodes)].insert(key, value)
    history.add((key, value))
    assert_valid_history(nodes, history)

    # Crash a process every so often
    if rnd.rand() > 0.75:
      node = nodes[rnd.rand_in_range_inclusive(0, 3)]
      node.restart()
  catch (e):
    print("Found logical error with seed: %d", seed)
    throw e
```

每个节点跑在一个单独的 port 上。

## Other sources of non-determinism

大多数 CPU 操作都是确定性的，但是一些 CPU 指令并不是，此外，也有一些系统调用有不确定性。以及 malloc 也有一定不确定性。

> If we [ignore](https://antithesis.com/blog/deterministic_hypervisor/) Antithesis, people doing DST seem not to worry about these smaller bits of nondeterminism. Yet it's generally agreed that DST is still worthwhile anyway. The intuition here is that every bit of non-determinism you can eliminate makes it that much easier to reproduce bugs when you find them.
>
> Put another way: determinism, even among DST practitioners, remains a spectrum.