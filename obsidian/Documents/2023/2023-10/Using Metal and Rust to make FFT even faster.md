https://blog.lambdaclass.com/using-metal-and-rust-to-make-fft-even-faster/

苹果的 Metal 有自己的 Shading 语言 Metal Shading Language。

本文作者根据 rust 的 Metal-rs 库来介绍基本的 MSL 用法。
## Metal Structures

Metal Thread Hierarchy

- Metal 的 thread structure 和 CUDA 很相似。
- 每个 thread 在一个 grid 中，比如一个 2D 的 grid，坐标系是 (x, y)。
- Thread 通常按照 thread group 进行组织，进一步拆分为 Warps 或者 SIMD groups。
- Warp 中的线程会针对不同的数据并发地执行同一组指令。如果其中一个线程出现 _diverge_ 也就是 if 分支，整个 warp 都会执行到两个分支并影响性能。

Metal Device

- metal API 的核心是 Device，对应一个 GPU 设备；

Command Queue

- Device 接受的指令都是通过 Command Queue 下发的，它会保序执行；

Command Buffers

- buffer 主要作为 GPU 中函数和计算的存储；

Pipeline State

 - 对应特定命令的 GPU 状态；

Encoders

- 每种命令，都需要一个对应的 encoder；
- encoder 会取出函数所需要的参数，加上 pipeline state 的参数；

![[Pasted image 20231017130749.png]]

## Programming in MSL and Rust

```c
[[kernel]]
void dot_product(
  constant uint *inA [[buffer(0)]],
  constant uint *inB [[buffer(1)]],
  device uint *result [[buffer(2)]],
  uint index [[thread_position_in_grid]])
{
  result[index] = inA[index] * inB[index];
}
```

```rust
let device: &DeviceRef = &Device::system_default().expect("No device found");
```

```rust
let lib = device.new_library_with_data(LIB_DATA).unwrap();
let function = lib.get_function("dot_product", None).unwrap();
```

```rust
let pipeline = device
    .new_compute_pipeline_state_with_function(&function)
    .unwrap();
```

```rust
let length = v.len() as u64;
let size = length * core::mem::size_of::<u32>() as u64;

let buffer_a = device.new_buffer_with_data(
    unsafe { mem::transmute(v.as_ptr()) }, // bytes
    size, // length
    MTLResourceOptions::StorageModeShared, // Storage mode
);

let buffer_b = device.new_buffer_with_data(
    unsafe { mem::transmute(w.as_ptr()) },
    size,
    MTLResourceOptions::StorageModeShared,
);
let buffer_result = device.new_buffer(
    size, // length
    MTLResourceOptions::StorageModeShared, // Storage mode
);
```

```rust
let command_queue = device.new_command_queue();

let command_buffer = command_queue.new_command_buffer();

let compute_encoder = command_buffer.new_compute_command_encoder();
compute_encoder.set_compute_pipeline_state(&pipeline);
compute_encoder.set_buffers(
    0, // start index
    &[Some(&buffer_a), Some(&buffer_b), Some(&buffer_result)], //buffers
    &[0; 3], //offset
);
```


```rust
let grid_size = metal::MTLSize::new(
    length, //width
    1, // height
    1); //depth

let threadgroup_size = metal::MTLSize::new(
    length, //width
    1, // height
    1); //depth;

compute_encoder.dispatch_threads(grid_size, threadgroup_size);
```


```rust
compute_encoder.end_encoding();
command_buffer.commit();
command_buffer.wait_until_completed();
```