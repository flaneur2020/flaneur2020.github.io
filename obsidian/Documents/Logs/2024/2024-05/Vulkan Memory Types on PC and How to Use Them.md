https://asawicki.info/news_1740_vulkan_memory_types_on_pc_and_how_to_use_them

#vulkan

VkMemoryHeap 对应一些物理内存，包括显存或者 RAM。

VkMemoryType 属于一个 heap，对应一些 VkMemoryPropertyFlags，其中包括：

1. VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT：表示该内存限定在只有 device 可以访问；DEVICE_LOCAL 是一个 hint，表示这段内存通过 GPU 访问会更快；
2. VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT：表示你可以在 host 侧通过 `vkMapMemory` 来映射出来一段 CPU 可以直接访问的内存，不需要额外的 `vkCmdCopyBuffer`。
3. VK_MEMORY_PROPERTY_HOST_CACHED_BIT：只有当 HOST_VISIBLE 也开启时，该 bit 才生效；这个 flag 并不影响内存的访问模式，只提示说这个内存可以在 CPU 侧允许 CPU cache。
## 1. The Intel way

```
Heap 0: DEVICE_LOCAL  
  Size = 1,849,059,532 B  
  Type 0: DEVICE_LOCAL, HOST_VISIBLE, HOST_COHERENT  
  Type 1: DEVICE_LOCAL, HOST_VISIBLE, HOST_COHERENT, HOST_CACHED
```

intel 的集成显卡，CPU 和 GPU 可以访问同一块内存。

所有的显存都 HOST_VISIBLE。

> Type 0 without `HOST_CACHED` flag is good for writing through mapped pointer and reading by the GPU, while type 1 with `HOST_CACHED` flag is good for writing by the GPU commands and reading via mapped pointer.

How to use it:

可以直接从磁盘 load 资源进来。不需要创建单独的 staging copy、单独的 GPU copy、单独的 transfer command。
## 2. The NVIDIA way

对于独立显卡，两种 memory heap 更合理：一个 `DEVICE_LOCAL` 来表示显存，另一个表示系统内存。

```
Heap 0: DEVICE_LOCAL  
  Size = 8,421,113,856 B  
  Type 0: DEVICE_LOCAL  
  Type 1: DEVICE_LOCAL  
Heap 1  
  Size = 8,534,777,856 B  
  Type 0  
  Type 1: HOST_VISIBLE, HOST_COHERENT  
  Type 2: HOST_VISIBLE, HOST_COHERENT, HOST_CACHED
```

Nvidia 倾向于将不同种类的资源（depth-stencil textures, render targets, buffers）放在不同的内存 block 中，所以需要在 `VkMemoryRequirements::memoryTypeBits` 中过滤一把自己需要的类型。

How to use it: 需要通过 HOST_VISIBLE 的内存来建一个 staging copy，从磁盘装载到里面，然后发一个 `vkCmdCopyBuffer` 来将它们转到 `DEVICE_LOCAL` 内存中。

## 3. The AMD Way

可能允许 CPU 通过一个普通的 `void*` 指针来访问特定的显存。这个特性被称作 `Base Address Register`（BAR）。

这部分内存可以是独立的一段 memory heap，同时满足 `DEVICE_LOCAL` 和 `HOST_VISIBLE`。

AMD 已经支持了很多年，NVIDIA 也在最近的 driver 中开始支持它。

```
Heap 0: DEVICE_LOCAL  
  Size = 8,304,721,920 B  
  Type 0: DEVICE_LOCAL  
Heap 1  
  Size = 16,865,296,384 B  
  Type 0: HOST_VISIBLE, HOST_COHERENT  
  Type 1: HOST_VISIBLE, HOST_COHERENT, HOST_CACHED  
Heap 2: DEVICE_LOCAL  
  Size = 268,435,456 B  
  Type 0: DEVICE_LOCAL, HOST_VISIBLE, HOST_COHERENT
```

第三个堆只有 256MB。它的 heap 的 memory type 是 `DEVICE_LOCAL`，但是也满足 `HOST_VISIBLE`。

这段内存在显卡中，但是也可以允许 CPU 来映射访问。CPU 写入这段内存时需要经过 PCIe 总线，可以认为是比较慢的。它没有 `HOST_CACHED` flag，意味着这段内存在 CPU 侧没有 cache，因此最好只用来顺序写入。

How to use it: 这个 256mb 的内存，可以用来从 CPU 到 GPU 传递数据，而不必经过显式的 `vkCmdCopy`。这个适合用于每个帧都变化的数据。不过需要注意显卡驱动可能也会利用这部分内存，因此并不是完整的 256mb 内存都是可用的。

## The SAM way

SAM （Smart Access Memory）是 AMD 的 marketing 团队造的术语，它通常被叫做 Resizable BAR 或者 ReBAR。

这个特性允许 CPU 不止访问 256 mb 的内存，而是完整的显存。

这个特性出来的比较晚，因此需要比较新的主板和 CPU（Ryzen 5000 系列）、更新 BIOS、一个新的显卡（比如 Radeon 6000 系列），安装最新的驱动，并在 BIOS 中开启这个特性。

在这之后，所有的 DEVICE_LOCAL 内存都可以通过 `HOST_VISIBLE` 访问。

```
Heap 0  
  Size = 25,454,182,400 B  
  Type 1: HOST_VISIBLE, HOST_COHERENT  
  Type 3: HOST_VISIBLE, HOST_COHERENT, HOST_CACHED  
  Type 5: HOST_VISIBLE, HOST_COHERENT, AMD-specific flags...  
  Type 7: HOST_VISIBLE, HOST_COHERENT, HOST_CACHED, AMD-specific flags...  
Heap 1: DEVICE_LOCAL | MULTI_INSTANCE  
  Size = 17,163,091,968 B  
  Type 0: DEVICE_LOCAL  
  Type 2: DEVICE_LOCAL, HOST_VISIBLE, HOST_COHERENT  
  Type 4: DEVICE_LOCAL, AMD-specific flags...  
  Type 6: DEVICE_LOCAL, HOST_VISIBLE, HOST_COHERENT, AMD-specific flags...
```

这意味着：不是 256mb 的小内存了，所有的内存都可以访问。

> A side note: If you have a texture that changes frequently, possibly writing it directly on the CPU via mapped pointer and reading on a GPU can be faster than doing `vkCmdCopy*`, even if it means the image has to use `VK_IMAGE_TILING_LINEAR`. This is what [DXVK](https://github.com/doitsujin/dxvk) (a Direct3D implementation over Vulkan) is doing for textures created with `D3D11_USAGE_DYNAMIC` flag – or at least it did when I checked it some time ago, if I remember correctly. As always, it is best to implement multiple approaches and measure which one works faster.

对于一个改动频繁的 texture 对象，可能使用 BAR 从 CPU 直接修改会比 `vkCmdCopy` 还快点儿。

> Please note there might also be a memory type that is `DEVICE_LOCAL` but not `HOST_VISIBLE`. Whether it works faster than the `HOST_VISIBLE` one and so it makes any sense to use it, or it is left just for backward compatibility, is not clear to me. When not sure, better to select a memory type with less additional flags than the required/desired ones, or a memory type just higher on the list.

有时可能仍有一个 `DEVICE_LOCAL` 但不 `HOST_VISIBLE` 的 heap 存在，作者怀疑是为了兼容性吧。

## 5. The APU way

和 Intel 的集成显卡类似，但是 vulkan 堆的内存的使用方式是相反的。

```
Heap 0  
  Size = 3,855,089,664 B  
  Type 0: HOST_VISIBLE, HOST_COHERENT  
  Type 0: HOST_VISIBLE, HOST_COHERENT, HOST_CACHED  
Heap 1: DEVICE_LOCAL  
  Size = 268,435,456 B  
  Type 0: DEVICE_LOCAL  
  Type 1: DEVICE_LOCAL, HOST_VISIBLE, HOST_COHERENT
```

`DEVICE_LOCAL` 的内存只有 256mb。

如果分配内存时候只用了 `DEVICE_LOCAL` 内存，那么会遇到麻烦。

要支持这种 GPU，你需要分配那些非 `DEVICE_LOCAL` 的内存，并假设它跑的不会比 `DEVICE_LOCAL` 慢。

最好通过 `vkGetPhysicalDeviceProperties` 检查一下 device type 是不是 `DISCRETE_GPU`。

> What it means: Despite using the same physical system RAM, Vulkan memory is divided into 2 heaps and multiple types. Some of them are not `DEVICE_LOCAL`, some not `HOST_VISIBLE`. What is worse is that the `DEVICE_LOCAL` heap doesn’t span the entire RAM. Instead, it is only 256 MB.

> How to use it: You will get into trouble on such platforms if your application tries to fit all resources needed for rendering in `DEVICE_LOCAL` memory, e.g. by creating critical resources like render-target, depth-stencil textures and then streaming other resources until heap size or budget is reached. Here, 256 MB will probably not be enough to fit even the most important, basic resources, not to mention meshes and textures needed to render a pretty scene. To support this GPU, you need to fall back to non-`DEVICE_LOCAL` memory types with your resources and assume they don’t work much slower than `DEVICE_LOCAL`. To detect that, possibly you can call `vkGetPhysicalDeviceProperties` and check if `VkPhysicalDeviceProperties::deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU` not `DISCRETE_GPU`.
