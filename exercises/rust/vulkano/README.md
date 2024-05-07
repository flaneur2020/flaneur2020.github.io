TODO
- [x] boilerplate in vulkan
- [x] run a compute shader
- [x] a compute shader to add two arrays
- [ ] add a staging buffer: staging buffer is also needed while copying data from host to device


Staging Buffer:

```
    let buf1 = Buffer::from_iter(
        compute.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        // Iterator that produces the data.
        0..65536u32,
    )
    .unwrap();
```

The above buffer is actually a staging buffer.
