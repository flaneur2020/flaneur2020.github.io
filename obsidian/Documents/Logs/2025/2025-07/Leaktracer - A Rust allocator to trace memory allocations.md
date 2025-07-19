跟踪程序的内存分配和回收，允许用户 dump 分配的记录，然后分析它。
## The idea behind Leaktracer

- 跟踪所有的内存分配和释放
- 给用户一定自由度，自定义怎么 dump 分配信息
- 能够特别简单集成到现有的程序中，最好一行就可以
## The implementation

LeaktracerAllocator 实现 GlobalAlloc trait。

```rust
pub struct LeaktracerAllocator;

impl LeaktracerAllocator {

    pub const fn init() -> Self {
        LeaktracerAllocator
    }

}

unsafe impl GlobalAlloc for LeaktracerAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
```

### Track allocations by modules

会遇到三个问题：

1. How to get the module and function name at runtime.
2. How do we store the allocations.
3. How can we prevent the allocator from recursively calling itself when we allocate memory for the allocations.

比较棘手的是第三个，因为我们如果在 `alloc` 方法中申请了任何内存，就会产生无限循环。

解决方法很简单，就是定义一个全局变量：

```rust
thread_local! {
    static IN_ALLOC: Cell<bool> = const { Cell::new(false) };
}

impl LeaktracerAllocator {
    // ...
    /// Returns whether the allocation is an external allocation.
    ///
    /// With **external allocation**, we mean that the allocation is not requested by the allocator itself,
    /// but rather by the user of the allocator.
    ///
    /// This is determined by checking if the `IN_ALLOC` thread-local variable is set to `false`.
    fn is_external_allocation(&self) -> bool {
        !IN_ALLOC.get()
    }

    /// Enters the allocation context, marking that an allocation is being made.
    fn enter_alloc(&self) {
        IN_ALLOC.with(|cell| cell.set(true));
    }

    /// Exits the allocation context, marking that the allocation is done.
    fn exit_alloc(&self) {
        IN_ALLOC.with(|cell| cell.set(false));
    }

  // ...

}
```

然后，只有确认 IN_ALLOC 为 false 时，才记录 trace：

```rust
unsafe impl GlobalAlloc for LeaktracerAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        // if the allocation is not null AND the allocation is external, trace the allocation
        if !ptr.is_null() && self.is_external_allocation() {
            self.trace(layout, AllocOp::Alloc);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if !ptr.is_null() && self.is_external_allocation() {
            self.trace(layout, AllocOp::Dealloc);
        }
        unsafe { System.dealloc(ptr, layout) };
    }
}
```

要得到 module 名字的话，可以通过 `backtrace` 来得到。

但是直接从 backtrace 中得到的 module 信息不大靠谱，太乱了。

作者想要关注的是，我们从哪个 mod 中分配的内存。要得到这个信息，只能我们提供一组 mod 列表给 allocator。

SYMBOL_TABLE 需要用户提供，也是一个全局变量。

为了统计每个 mod 的内存占用，需要一个带锁的对象，这个对象中也会有内存分配出现。也需要注意死锁。

最终的效果大约是这样：

```rust
use leaktracer::LeaktracerAllocator;

#[global_allocator]
static ALLOCATOR: LeaktracerAllocator = LeaktracerAllocator::init();

fn main() {
    leaktracer::init_symbol_table(&["my_app", "my_lib"]);
    leaktracer::with_symbol_table(|table| {
        for (name, symbol) in table.iter() {
            tracing::info!(
                "Symbol: {name}, Allocated: {}, Count: {}",
                symbol.allocated(),
                symbol.count()
            );
        }
    })?;

}
```