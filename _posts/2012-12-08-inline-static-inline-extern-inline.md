---
layout: post
title: "inline, static inline, extern inline"
---

记得以前看过的，不过extern inline的语义全然记不起来了，还得多把东西写下来才行呢。

## Inlining in GNU89

inline并非标准委员会的创作，最早给出实现的GNU89，在一段时间内暂且成为事实上的标准：

+ **inline**

  函数有可能会被展开，但总保留一个未展开的函数，其符号对外部可见，因此不可以将这样的函数放到头文件里，不然链接时会报重名的错误。

+ **static inline**

  函数可能会被展开，但总保留一个未展开的函数，这个函数在不同的编译单元中会存在多个副本，但其符号仅在当前编译单元可见。

+ **extern inline**

  函数可能会被展开，且不会保留一个未展开的函数，当函数没有展开时，会调用某编译单元(_可以是当前编译单元_)中的同名函数，也就是要求人肉在某.c文件中重复一遍。

到这里的共同点是inline都仅仅是对编译器的一个提示，到具体情景中要展开还是不展开，就要看编译器的心情了。但是如果不展开，就调那只未展开的函数作为后备措施，所以static与extern其实影响的是"备用"的那只函数，而与函数的真正展开与否无关。

## But things have changed in C99

在C99中，inline第一次被标准化，不过语义与GNU89相比已经有了比较大的差异：

+ **inline**

  与GNU89的extern inline有点像，函数有可能会被展开，但不会保留一个未展开的函数了，需要人肉在_另一个_编译单元中提供一个同名函数。

+ **extern inline**

  与GNU89的inline比较像，函数有可能会被展开，但总保留一个未展开的函数，且符号对外部可见，也是只能存在于一个编译单元中。于是也不能把extern inline放到头文件里面了。

+ **static inline**

  同GNU89兼容。

## Sample

[https://gist.github.com/4238509](https://gist.github.com/4238509)

## Conclusion

还是只用static inline吧。

## Reference

+ http://stackoverflow.com/questions/216510/extern-inline
+ http://clang.llvm.org/compatibility.html#inline

