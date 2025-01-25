
tldr
- 这套手法只适合没有函数调用的短片段
- cgo 会给 c 单独开一个栈，一旦有函数调用，就还是得走 cgo
## Why not CGO

cgo 会给 C 单独开一个 stack；

它会在 go stack 中 defer 调用来防 panic；

这导致，不大适用于 small hot function 的地方。

## Linking it together

可以使用 rust code 生成汇编代码，然后使用汇编的方式来试用它。

这不需要介入到 IR 层，go 在 1.3 之后，能够同时将 code 和高级汇编编译到机器码；