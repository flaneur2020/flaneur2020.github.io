https://github.com/facebook/rocksdb/wiki/Merge-Operator

## Why

经常有一些场景是，要读出来一个值，并做一些处理，然后再 Put 回去。

可不可以把这些操作内置到 rocksdb 里面？

（有点 redis 的 lua script 的意思？）

## What

在 rocksdb 中增加了一个通用的 Merge 操作，允许在 rocksdb 中做 read-modify-write semantics。

Merge 操作做了这些事情：

1. 封装 read-modify-write 的语义到一个抽象的接口中
2. 允许用户避免额外的 Get() 开销
3. 允许做后端优化，来确定在不改变语义的前提下怎样组合 operand；
4. 在某些时候允许将成本均摊到增量更新操作中，提高效率

# How to Use It

###  Overview of the Interface

新增了一个名为 MergeOperator 的 interface。

这个 interface 能够告知 rocksdb 怎样针对 base-values （Put/Delete）组合增量更新操作。

也提供了一个 AssociativeMergeOperator 来针对结合律的 Merge 操作。

```C++
    // The Associative Merge Operator interface.
    // Client needs to provide an object implementing this interface.
    // Essentially, this class specifies the SEMANTICS of a merge, which only
    // client knows. It could be numeric addition, list append, string
    // concatenation, ... , anything.
    // The library, on the other hand, is concerned with the exercise of this
    // interface, at the right time (during get, iteration, compaction...)
    class AssociativeMergeOperator : public MergeOperator {
     public:
      virtual ~AssociativeMergeOperator() {}

      // Gives the client a way to express the read -> modify -> write semantics
      // key:           (IN) The key that's associated with this merge operation.
      // existing_value:(IN) null indicates the key does not exist before this op
      // value:         (IN) the value to update/merge the existing_value with
      // new_value:    (OUT) Client is responsible for filling the merge result here
      // logger:        (IN) Client could use this to log errors during merge.
      //
      // Return true on success. Return false failure / error / corruption.
      virtual bool Merge(const Slice& key,
                         const Slice* existing_value,
                         const Slice& value,
                         std::string* new_value,
                         Logger* logger) const = 0;

      // The name of the MergeOperator. Used to check for MergeOperator
      // mismatches (i.e., a DB created with one MergeOperator is
      // accessed using a different MergeOperator)
      virtual const char* Name() const = 0;

     private:
      ...
    };
```

```C++
     void Merge(...) {
       if (key start with "BAL:") {
         NumericAddition(...)
       } else if (key start with "HIS:") {
         ListAppend(...);
       }
     }
```

### Other Changes to the client-visible interface

client 需要首先定义一个 AssociativeMergeOperator，rocksdb 会在它认为需要执行 merge 时调用它。

```C++
    // In addition to Get(), Put(), and Delete(), the DB class now also has an additional method: Merge().
    class DB {
      ...
      // Merge the database entry for "key" with "value". Returns OK on success,
      // and a non-OK status on error. The semantics of this operation is
      // determined by the user provided merge_operator when opening DB.
      // Returns Status::NotSupported if DB does not have a merge_operator.
      virtual Status Merge(
        const WriteOptions& options,
        const Slice& key,
        const Slice& value) = 0;
      ...
    };

    Struct ColumnFamilyOptions {
      ...
      // REQUIRES: The client must provide a merge operator if Merge operation
      // needs to be accessed. Calling Merge on a DB without a merge operator
      // would result in Status::NotSupported. The client must ensure that the
      // merge operator supplied here has the same name and *exactly* the same
      // semantics as the merge operator provided to previous open calls on
      // the same DB. The only exception is reserved for upgrade, where a DB
      // previously without a merge operator is introduced to Merge operation
      // for the first time. It's necessary to specify a merge operator when
      // opening the DB in this case.
      // Default: nullptr
      const std::shared_ptr<MergeOperator> merge_operator;
      ...
    };
```

（调用 Merge() 似乎就是一个 merge 操作的触发）