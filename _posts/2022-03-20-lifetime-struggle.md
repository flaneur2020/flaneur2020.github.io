---
title: lifetime 恐惧自救
layout: post
---

lifetime annotation 语法让我感到比较别扭的地方是：

1. 它不是真正的类型，因为不能像真的类型那样实例化，但是它可以像传递一个真实类型一样传递到泛型的类型参数中，而且也确实是实打实的 Subtyping 有协变逆变
1. 它也可以是类似 Trait 那样作为类型约束，除了可以约束其他的 lifetime annotation 比如 `'a: 'b`，也可以和普通的类型做约束比如 `T: 'a`

这个设定还能够接受。不过除此之外，lifetime 时不时跟其他一些奇怪的语法一同出现，每到这时候仍总是感到恐惧。

想克服一下对 lifetime 的恐惧：1. 面对一下；2. 整理分析一下。

粗略整理了一把不懂的代码之后，发现 lifetime 相关不懂的语法好像主要集中在跟各种泛型参数乃至 trait 约束放在一起时候。

找找几个例子详细看一下。

## Ref<’a, T: 'a>

这个例子来自 [https://carols10cents.github.io/book/ch19-02-advanced-lifetimes.html#lifetime-bounds-on-references-to-generic-types](https://carols10cents.github.io/book/ch19-02-advanced-lifetimes.html#lifetime-bounds-on-references-to-generic-types)：

``` rust
struct Ref<'a, T>(&'a T);
```

据说会报这个错（不过我没在本地复现成功，大概 rust 编译器新增加了省略规则）：

``` rust
error[E0309]: the parameter type `T` may not live long enough
 --> src/lib.rs:1:19
  |
1 | struct Ref<'a, T>(&'a T);
  |                   ^^^^^^
  |
  = help: consider adding an explicit lifetime bound `T: 'a`...
note: ...so that the reference type `&'a T` does not outlive the data it points at
 --> src/lib.rs:1:19
  |
1 | struct Ref<'a, T>(&'a T);
  |                   ^^^^^^
```

为什么需要对这里的泛型参数增加 lifetime 约束？

因为 T 可能是一个包含引用的结构体，比如 `Lexer<’a>`，虽然有这个 Lexer 的 ownership，但是它里面有对外的引用；也可能它就是一个引用类型比如 `&PlainText`。在这时，`T: ‘a` 可以用于声明 T 这个类型的生命周期一定在 `‘a` 的范围之内。

## Cow<’a, B: ‘a + ToOwned + ?Sized>

再看一个标准库里的 Cow 的例子：

``` rust
pub enum Cow<'a, B> 
where
    B: 'a + ToOwned + ?Sized, 
 {
    Borrowed(&'a B),
    Owned(<B as ToOwned>::Owned),
}
```

跟前一个例子一样，这里有对泛型参数中的 B 指定 `‘a` 来约束它的生命周期在 `’a` 之内。

这个代码里主要是 `?Sized` 这个东西看不懂，先查一把。

`Sized` trait 意思是类型在编译器可以知道长度，所有的类型参数都有一个默认的 `Sized` bound。`?Sized` 表示放开这个 bound，允许接受非定长的类型。非定长的类型主要来自 Slice 和 Trait Object，比如 `dyn MyTrait` 和 `[u8]` 这种。非定长的类型不能在栈上保存，比如：

``` rust
// Can't be stored on the stack directly
struct MySuperSlice {
  info: u32,
  data: [u8],
}
```

但是非定长类型的引用仍是定长的，可以在栈上传递，比如 `&’a B`、`&dyn MyTrait`, `&[u8]`。`Cow<’a, B>` 中的 B 之所以允许 `?Sized`，是因为 enum 的 Borrowed 部分的内容是对 B 的引用。如果没有声明 `?Sized` 的话，就不能对 `[u8]` 使用 `Cow<>` 了。

## box_disaplayable<’a, T: Display + ‘a>

``` rust
use std::fmt::Display;

fn box_displayable<T: Display>(t: T) -> Box<dyn Display> {
    Box::new(t)
}
```

这段代码为什么不能编译通过？

跟前面的例子一样，原因还是 T 可能是包含引用的结构体，也可能是个对引用类型实现的 trait 比如 `impl Display for &MyType`。在 move 时，需要确保它 move 后仍在 lifetime 范围内。

``` rust
error[E0310]: the parameter type `T` may not live long enough
 --> src/lib.rs:4:5
  |
3 | fn box_displayable<T: Display>(t: T) -> Box<dyn Display> {
  |                    -- help: consider adding an explicit lifetime bound...: `T: 'static +`
4 |     Box::new(t)
  |     ^^^^^^^^^^^
  |
note: ...so that the type `T` will meet its required lifetime bounds
 --> src/lib.rs:4:5
  |
4 |     Box::new(t)
  |     ^^^^^^^^^^^
```

## for<’a>

[http://zderadicka.eu/higher-rank/](http://zderadicka.eu/higher-rank/) 这个例子比较好，我有一个 `ChuckSum` trait，里面包了一个 `calc` 的方法，有 Xor 和 Add 两种算法实现：

``` rust
trait Checksum<R:Read> {
    fn calc(&mut self, r:R) -> Vec<u8>;
}

struct Xor;

impl <R:Read> Checksum<R> for Xor {
    fn calc(&mut self, mut r:R) -> Vec<u8> {
        let mut res: u8 = 0;
        let mut buf = [0u8;8];
        loop {
            let read = r.read(&mut buf).unwrap();
            if read == 0 { break }
            for b in &buf[..read] {
                res ^= b;
            }
        }
        
        vec![res]
    }
}

struct Add;

impl <R:Read> Checksum<R> for Add {
    fn calc(&mut self, mut r:R) -> Vec<u8> {
        // skipped
    }
}
```

很合理。有多种不同的算法，均取满足 `Read` trait 的参数，计算 checksum。

根据参数中指定的算法来计算一个文件的 checksum：

``` rust
fn calc_file_with_checksum(_path: String, mut checksumer: impl Checksum<&[u8]>) -> Vec<u8> {
    let buf = "blah blah blah".to_string().into_bytes();
    checksumer.calc(&buf)
}

```

这个会报错：

``` rust
╰─$ rustc hrtb.rs                                                                                                                                                                                                                                                1 ↵
error[E0106]: missing lifetime specifier
  --> hrtb.rs:25:73
   |
25 | fn calc_file_with_checksum(_path: String, mut checksumer: impl Checksum<&[u8]>) -> Vec<u8> {
   |                                                                         ^ expected named lifetime parameter
   |
help: consider introducing a named lifetime parameter
   |
25 | fn calc_file_with_checksum<'a>(_path: String, mut checksumer: impl Checksum<&'a [u8]>) -> Vec<u8> {
   |                           ++++                                               ++

error: aborting due to previous error

For more information about this error, try `rustc --explain E0106`.
```

按照提示再改一下：

``` rust
error[E0597]: `buf` does not live long enough
  --> hrtb.rs:27:21
   |
25 | fn calc_file_with_checksum<'a>(_path: String, mut checksumer: impl Checksum<&'a [u8]>) -> Vec<u8> {
   |                            -- lifetime `'a` defined here
26 |     let buf = "blah blah blah".to_string().into_bytes();
27 |     checksumer.calc(&buf)
   |     ----------------^^^^-
   |     |               |
   |     |               borrowed value does not live long enough
   |     argument requires that `buf` is borrowed for `'a`
28 | }
   | - `buf` dropped here while still borrowed

error: aborting due to previous error

For more information about this error, try `rustc --explain E0597`.
```

为什么还会报错 borrowed value does not live long enough 呢？

因为按这个 lifetime 约束，calc 的参数的 lifetime 需要比 `‘a` 大，但是 buf 这个变量的生命周期是小于 `‘a` 的，因此报错了。

引用类型可以是 Generic Trait 的类型参数，这本身是没毛病的，不过有了引用就需要有 lifetime，rust 过去在一个函数中出现的 lifetime annotation 只能都来自于函数的 lifetime generic parameter 的声明。如果使用函数的 lifetime ‘a 来标注泛型参数中的引用，就会出现上面的报错。

使用函数的 lifetime ‘a 来标记 `Checksum<&[u8]>` 中的引用，从生命周期的角度上讲并不正确。`Checksum<&[u8]>` 中引用的 lifetime 只跟它的调用点有关，两个不同的调用点，它的 lifetime 会有所不同。

可以做一个 work around，就是单独定义一个函数，通过函数的 lifetime 标记确保引用的生命周期一致，也就没有问题了：

``` rust
fn calc_checksum<'a>(buf: &'a [u8], mut c: impl Checksum<&'a [u8]>) -> Vec<u8> {
    c.calc(buf)
}
```

每个类似的调用点都单独封装一个函数的做法比较丑。为此 rust 引进了 HRTB （Higher Rank Trait Bound）语法，也就是这个 `for <’a>`。这样改：

``` rust
fn calc_file_with_checksum(_path: String, mut checksumer: impl for<'a> Checksum<&'a [u8]>) -> Vec<u8> {
    let buf = "blah blah blah".to_string().into_bytes();
    checksumer.calc(&buf)
}
```

表示它的 lifetime 与 foo 函数的 lifetime 标识无关，在每个具体的调用点上具体绑定，也就不需要只是因为 lifetime 的原因而单独定义一个函数了。

## 总结

结构体与函数的 lifetime 标识还比较容易理解，但是泛型的参数仍有可能代入引用类型，这是很容易忘记的一个地方。此外，在定义泛型结构体或者 Trait 时是没有办法预见泛型参数的 lifetime 的，在这时候比较容易遇到意外，总结下来：

__定义结构体时，泛型参数要不要加上 lifetime 约束？__

泛型参数会容易在心智上默认它是一个 owned 类型，如果它与引用沾上关系，需要理解到这个泛型参数即使是一个 owned 类型，里面仍可能有引用的字段存在，也就会与 lifetime 有关，如果有编译报错，就考虑一下是不是需要为它增加 lifetime 约束，使泛型参数的 lifetime 满足结构体的 lifetime 约束。

__在使用的 Trait 里，有没有泛型参数可能传入引用？__

这类参数通过调用所在的函数的 lifetime 进行约束是没有意义的，它需要的 lifetime 一定会比函数的 lifetime ‘a 要小，也就满足不了函数的 lifetime 的约束。这时可以通过 `for<’a>` 对 Trait 类型本身做约束，函数本身可以不需要 lifetime 参数。

## References

- [https://carols10cents.github.io/book/ch19-02-advanced-lifetimes.html](https://carols10cents.github.io/book/ch19-02-advanced-lifetimes.html)
- [https://ttys3.dev/post/rust-trait-lifetime-bounds/](https://ttys3.dev/post/rust-trait-lifetime-bounds/)
- [https://doc.rust-lang.org/nomicon/hrtb.html](https://doc.rust-lang.org/nomicon/hrtb.html)
- [https://doc.rust-lang.org/nomicon/exotic-sizes.html](https://doc.rust-lang.org/nomicon/exotic-sizes.html#:~:text=Rust%20supports%20Dynamically%20Sized%20Types,DSTs%20are%20not%20normal%20types)
- [https://medium.com/nearprotocol/understanding-rust-lifetimes-e813bcd405fa](https://medium.com/nearprotocol/understanding-rust-lifetimes-e813bcd405fa)
- [https://github.com/pretzelhammer/rust-blog/blob/master/posts/common-rust-lifetime-misconceptions.md](https://github.com/pretzelhammer/rust-blog/blob/master/posts/common-rust-lifetime-misconceptions.md)
- [http://zderadicka.eu/higher-rank/](http://zderadicka.eu/higher-rank/)