
row zero 是作者在做的一个世界最快的 spreedsheet 产品。

修改了一个 cell 之后，产生的变化会有很多，作者采用了 property based test。

## Property based testing

基本思路是：

1. 从 input 中采样出来一个输入
2. 在 test 中执行操作
3. 根据形式来判断结果的正确性，或者参考某个 reference implementation
4. 重复很多很多次 

有很多框架可以做这件事情，不过多数都可以追溯到 haskell 的 quick check 库。

这些库能够：

1. 自动生成 sample input
2. 缩小 input 集合，形成最小集
3. 确定性地重放

## Verifying a spreadsheet

有三种 property based testing：

1. test against a reference model for blackbox equality
2. test against a set of general properties or invariant
3. test for crashes

（好像类比到 kv 引擎的话，可以对着一个 btree 做等价性验证）

### Blackbox reference model testing

这是最强大的一种模式。

将实现与一个参考实现进行对比，一般来讲，参考实现会比真正的实现慢得多，或者缺些功能。

可以对着 spreadsheet 做一些随机的插入、更新、删除。与一个哈希表做对比。

### Invariant testing

感觉相当于一些 assert。

### Crash Testing

> Copy paste. Update cell. Double-click drag. Undo. Redo. Filter range. Insert pivot table. Delete all. New sheet. Just every combination of actions imaginable.

## Nuts and bolts: diving into some code

作者没有在使用 proptest 这样的 framework。

```rust
#[test]
fn foo_proptest() {
    let mut rng = rand::thread_rng();
    for action_count in 1..4 {
        for _ in 0..10_000 {
            let seed = rng.gen::<u64>();
            eprintln!("let seed = {seed};");
            let mut rng = ChaChaRng::seed_from_u64(seed);
            // Set up and run an instance of the test
        }
    }
}
```

