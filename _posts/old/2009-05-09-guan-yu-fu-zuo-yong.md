---
layout: post
title: "关于副作用"
tags: 
- FP
- haskell
- monad
- "笔记"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

所谓三人成虎，monad就是了。

先烈告诉我们，玩haskell，不用学monad；玩monad，不用学范畴论。当年不懂事啊，正被functor态射范畴这堆鬼东西折磨的半死的时候，突然就骂了出来：“妈的，这还是编程么？数学的拉去见学院派大胡子去！”

嗯，咱是程序员。学院派大胡子去死。嗯。

12Dec06
16:20:11 [Botje] monads have to be the singly most tutorialized feature _EVER_

好吧，这里只谈副作用（不过貌似monad还真避不开）。只是我自己的理解，不保证正确。大牛和小白请直接略过。

纵然大牛们反复重申monad不是为副作用而生，但相信大部分人一定是从IO那里才认识的monad。那么什么才是副作用呢？有个装逼（装纯？）的概念，那就是“引用透明”(reference transparency)。如果拿固定的参数调用一个函数，得到的结果一定是相同的。如果拿相同的参数调用同一个函数两次而得到的结果不一样，那么这个函数就是有副作用的。

如ruby的gets函数，它没有参数，返回一个string。每次执行（参数都是为空）返回的结果不一定一样（决定权在你的输入了），所以说它就没有引用透明。

看下Haskell中getLine函数的类型：
<pre lang="haskell">
getLine :: IO String</pre>

和gets不一样，它返回的类型是IO String而非直接的String。先想一下，如果它的返回类型是String会怎样？你就可以把getLine函数塞到四处都是，可以让每个函数的结果都与输入有关。好吧，引用透明呢？见大胡子去了。

IO是个类型构造子，仅仅是把后面的String包起来。如果你输入“Alpaca”，它就返回IO “Alpaca”。怎样从中取出”Alpaca”来呢？像List，Maybe之类就直接上模式匹配就可以了，而所谓模式匹配就是匹配类型构造子。你可以let (x:xs)=alist\，但不可以let (IO str)=getLine。因为IO类型构造子在模块中没有被导出，换句话说，它是“私有”的，就像OO中把构造函数私有一样，只有那个模块里面的函数才可以访问它。这样一来在外面的你就不能从getLine中取值了。那该怎么办？里面有>>=。它可以把你的操作塞到里面去，可就是不让你从把里面的值带到别处。
<pre lang="haskell">
getLine >>= (\str -> {-- your operation here--})
</pre>
在这里>>=的实现大致就是这样（假设>>=只为IO存在）：
<pre lang="haskell">
>>= :: IO a -> (a -> IO b) -> IO b
IO a >>= f = f a
</pre>
它可以把函数串成一条链，并限制函数f的返回类型必须还得是IO。这样，与外界交互的一切代码就都被迫标上了个IO的标记。不是从外面取值，而是往外面传操作，操作完了再把它包好原样扔回去。而在haskell的纯洁世界中，外面的值是完全无法访问的。
