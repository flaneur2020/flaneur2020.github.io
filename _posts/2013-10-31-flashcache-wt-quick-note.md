---
layout: post
title: "flashcache-wt Quick Note"
---

flashcache 是 facebook 开源的 ssd 存储产品，它基于内核的 devicemapper 机制，允许将 ssd 设备映射为机械存储设备的缓存，堆叠成为一个虚拟设备供用户读写，从而在一定程度上兼顾 ssd 的高速与机械存储设备的高容量，更加经济高效地支撑线上业务。

flashcache 支持三种缓存策略：

* 回写(Write Back)：修改内容之后，并不立即写入后端设备
* 写透(Write Through): 修改内容时写入后端设备，同时也更新前端设备中的缓存块
* 环写(Write Around): 修改内容时，先写入后端设备，同时使前端设备中对应的缓存块失效

为便于分析，先只看下仅支持写透缓存的一个“分支”：flashcache-wt。它最简单、安全，但整体效率相对而言偏低，它也是一种易失性缓存，在重启机器、卸载设备之后缓存的内容会全部失效。

## 基本使用

确保内核已启用 DeviceMapper，编译内核模块并安装之后：

* 创建缓存设备: `flashcache_wt_create /dev/cachedev /dev/sda1 /dev/hda1`
* 删除缓存设备: `dmsetup remove cachedev`

使用者需要注意建立缓存设备之后，就不应该再对 /dev/sda1 和 /dev/hda1 两个设备直接读写了。

## 缓存的组织

虽然数据会被写入 ssd 设备，但缓存的组织信息一律保存在内存中，即 cache_c 对象。它里面保存了基本的配置选项、统计信息、对两个设备的引用、每个缓存块的相关信息(cache, cache_state)、 以及缓存组中 FIFO 数组(虽然名字叫做 LRU)的下标。

flashcache 中缓存的盘块的默认大小为 4kb，按照多路组相连的形式组织缓存块，每组含有 512 个缓存块，简单按照取模分组。缓存的查找，就是对后端设备块号取模得到组号，然后凭偏移遍历组中的缓存块。

组内按照 FIFO 策略清理，为每个组维护当前头部的下标，每当有新写入，后移下标，并在触顶时归零。

## 缓存块的状态

每个缓存块的状态有：

* `INVALID`：不可用，即缓存块的初始状态
* `VALID`：可用
* `INPROG`：正在(从后端设备)读取中
* `CACHEREADINPROG`：正在读取缓存块
* `INPROG_INVALID`：当前有在进行中的操作，待执行完毕之后会变成 INVALID

## 缓存块的读写

DeviceMapper 提供了一套 bio 转发的机制，flashcache 作为它的框架之下的 Target Driver，通过 dm_register_target() 提供初始化函数、析构函数，以及最重要的块映射策略函数：flashcache_wt_map()，在每次设备发生读写时触发。它做的事情很简单，首先判断操作如果为环写操作，则向后端设备发送无缓存的写操作同时使缓存失效；然后根据操作类型分派到 cache_read() 或者 cache_write()。最后返回 DM_MAPIO_SUBMITTED 表示操作成功。

`cache_read()` 是一个异步操作，它会首先调用 `cache_lookup()` 依据内存中的 `cache_state` 和 `cache` 两个数组查找合适的缓存块，并在必要时按照  FIFO 策略回收（修改对应 `cache_state` 为 `INVALID`），不会发生与存储设备交互。如果找到可用的缓存块，则设置缓存块状态为 `CACHEREADINPROG`，并创建 `kcached_job` 对象，将它作为 `dm_io_async_bvec()` 函数的 context 参数传入，用以跟踪一轮异步 io 操作的生命周期；否则尝试使缓存块无效，如果缓存块为 `INPROG` 状态，则对后端设备启动一轮 uncached IO，读完之后不更新缓存；最后对于确认不在缓存中的块，调用 `read_cache_miss()` 启动 IO 操作，并约定在读取完毕之后更新缓存。

读写完毕之后，DeviceMapper 会调用 `dm_io_async_bvec()` 中指定的回调函数，即 `flashcache_wt_io_callback()` 与 `flashcache_wt_uncached_io_callback()`。它们会依据 `kcached_job` 的信息，指导下一步操作。

`flashcache_wt_uncached_io_callback()` 做的事情比较简单：使对应缓存块失效；调用 `bio_endio()`；释放 `kcached_job` 对象。留意最后的这两行代码：

        if (atomic_dec_and_test(&dmc->nr_jobs))
                wake_up(&dmc->destroyq);

在 flashcache 退出路径的开始，都会调用 `kcached_client_destroy()`，等待来自 `destroyq` 的事件。目的是为了保证在 flashcache 退出之前处理完进行中的所有 `kcache_job`，避免有 IO 操作没有收到 `bio_endio()`。

`flashcasche_wt_io_callback()` 会根据操作的不同，执行一定的 io 转发操作。比如读写后端设备(`READSOURCE`/`WRITESOURCE`)之后， 将内容写入缓存块。需要注意的是 `flashcache_wt_io_callback()` 是在 softirq 上下文中执行，不可以直接发送 io 操作。为此 flashcache 安排了一个 work queue，即 `_kcached_work`，用以真正转发 io 操作。

