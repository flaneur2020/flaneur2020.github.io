---
layout: post
title: "Haskell：可变状态的命令式语言"
tags: 
- FP
- haskell
- monad
- PL
- "翻译"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

作者：Neil
翻译：ssword
原文：<a href="http://neilbartlett.name/blog/2007/04/11/haskell-an-imperative-language-with-mutable-state/">http://neilbartlett.name/blog/2007/04/11/haskell-an-imperative-language-with-mutable-state/</a>

Haskell是一门惰性的纯函数式语言，也就意味着其中没有可变的变量。呃，有，没有？看下这个Factorial的haskell实现吧：

<pre lang="haskell">
fact n = runST (do
    r < - newSTRef 1
    for (1,n) (\x -> do
       val < - readSTRef r
       writeSTRef r (val * x))
    readSTRef r)
</pre>
嗯，我可没说这样写好。实际上，这样写很糟糕，一行纯函数式代码就可以漂亮的搞定同样的工作。不过请注意下这代码与命令式语言是如何的相像，如对“可变变量”r的破坏性更新。再贴一下，与同等的C代码做个对比：
<pre lang="haskell">

fact n = runST (do               | int fact(int n) {
     r < - newSTRef 1             |    int r = 1;
                                 |    int i;
     for (1,n) (\x -> do         |    for(i=1;i< =n;i++) {
         val <- readSTRef r      |
         writeSTRef r (val * x)) |       r = r * i;
                                 |    }
     readSTRef r)                |    return r;
                                 | }

</pre>

瞧，你几乎可以把C的代码一对一地翻译成haskell。不过得承认，haskell的语法要猥琐些。

这一切是如何做到的呢？答案就是ST monad，有了它就可以写出有内部状态更新而对外仍保持纯粹的算法。runST函数是亮点，它创建了个初始的空状态，然后执行一系列的状态转换，到最后再销毁它。fact依然是Int->Int的纯函数，依然保留了引用透明。

另一个亮点是那个“for”，看起来跟个关键字一般，不过它本质就是一普通函数，定义如下：

<pre lang="haskell">
for :: (Int,Int) -> (Int -> ST s ()) -> ST s ()
for (i,j) k = sequence_ (map k [i..j])
</pre>

同理，我们也可以定义出foreach，if，while等等。有haskell这般强大的表达力，没必要再将控制流的操作定义为语言的关键字；我们甚至可以根据自己的需求来发明新的控制流程。

不过你看出来了没？没人会正二八经地像上面那样实现factorial。而且，我们真的希望如这般曲解haskell，以取悦从C过来的老程序员么？

是的，问题就在这里。有这项技巧，是为那些已知难以使用函数式风格或递归搞定的算法提供的折衷。同样，很多算法需要用到数组-----有限的内存和固定的访问时间-----使其更易于处理。在ST monad中实现数组很容易，但在传统的纯函数式代码中就要困难多了。（UPDATE：Cale说，纯代码中实现不可变的数组还是很容易的，只有可变的数组实现其来才困难）。

恩，ST monad和STRef表示了外部不可见的局部状态，那么全局的可变状态呢？我们可以用IO monad和IORef做到。当然，在用IO的时候就没有像runST那样回到纯代码的函数了[1]。

对我，这引发的思考就是：haskell是真正的纯函数式语言吗？我觉得不是-----它超越了纯函数式。它是既允许命令式编程，又使用强大类型系统来把有副作用的代码与纯代码分离。这一点，令它强大无比。

[1] 这不是绝对…其实有个方法：unsafePerformIO。不过我觉得应该辩证地讨论它。
