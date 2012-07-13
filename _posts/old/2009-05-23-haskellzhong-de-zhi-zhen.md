---
layout: post
title: "haskell中的指针"
tags: 
- FP
- haskell
- trick
- "备忘"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

标题党了，阿门。其实更像C++，java之类语言中的引用。这就是Data.IORef中的IORef，使用它可以创建真正mutable的变量。

在do-notation中可以写这样的代码：
<pre lang="haskell">
do {
	…
	x< - return x+1;
}
</pre>

很像变量，但其实不是真正的mutable，因为它本质上就是(\x -> ..) x+1 ，两个x不在同一个scope，内存中没有值发生改变。

而Data.IORef则是在内存中搞一块地方，提供对这一块内存中输入输出的函数，并通过IO monad将它隔离开，这样就有点真正的变量的意思啦。大致就这几个函数：
</pre><pre lang="haskell">
newIORef :: a -> IO (IORef a) --创建一个新的IORef（废话～，不过有时也觉得haskell挺”面向对象”的…）

readIORef :: IORef a -> IO a
writeIORef :: IORef a -> a -> IO ()
</pre>

太直白了，一个读一个写，都是IO操作，只不过操作的对象不是标准输入输出，而是内存里的一个值。值得一提的是，haskell的惰性求值让它只有在必要的时候才会读取，这样一来在传递参数的时候就不会访问值所在的内存，就像C中传递一个指针那样节约。唉，发明haskell的那个委员会真是一群天才。

Ps：学习haskell到熟悉基本的语法之后，强烈推荐那个《make yourselves a scheme in 48 hours》的教程！通过用haskell实现一个完整的scheme，对于monad，类型系统的理解绝对会大有好处!
