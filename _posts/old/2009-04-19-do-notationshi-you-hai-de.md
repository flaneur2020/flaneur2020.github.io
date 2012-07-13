---
layout: post
title: "Do-notation有害论"
tags: 
- FP
- haskell
- monad
- "翻译"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

作者：<a href="http://syntaxfree.wordpress.com/">Dr. Syntaxfree</a>
翻译：ssword
原文：<a href="http://syntaxfree.wordpress.com/2006/12/12/do-notation-considered-harmful/">http://syntaxfree.wordpress.com/2006/12/12/do-notation-considered-harmful/</a>

16:20:11 [Botje] Monads可以算是_有史以来_被复述最多的特性了。
<em>monads have to be the singly most tutorialized feature _EVER_</em>
16:20:17 [monochrom] 为啥这样数学的抽象工具吸引了这么多科普作家，却少有数学的抽象方式讲解？
<em>Why would such a mathematical, abstract tool attract so many popular science authors who do not explain the tool in its mathematical, abstract term?</em>
(from #haskell@irc.freenode.net)

Monads无疑就是haskell中最显眼的特性了，作为处理这堆<a href="http://research.microsoft.com/en-us/um/people/simonpj/Papers/marktoberdorf/">乌合之众</a>（输入输出，并发，异常，以及跨语言调用）和产生一定副作用的标准工具，每个haskell程序员都得在一定程度上面对monad，而在许多人看来，monad无非就是一件以纯洁性和引用透明之名而套上的皮毛罢了。

谈起monads之难，诚然有无数的理由 — 它那邪恶并罕见的名字，深奥的范畴论，命令式世界（无处不在的副作用，脆弱的内置抽象机制）和函数世界（无处不在的抽象，副作用则须经深奥的数学机制）的文化冲突。不过我觉得还有一个大门槛阻碍我们理解monads：它那极致的语法糖。

Do-notation给monadic风格编程提供了一种伪命令式的观感。许多有名的教程都是直接拿Monadic风格的IO做开篇，这让人觉得它就是haskell为与外面有时序的世界交流而内置的一个命令风格模式。而do-notation — 在IO的上下文中引入 — 掩盖了monad在haskell中实现的本质就是一个简单的类型类，并让人忽略了它作为计算模型的存在。

显然，好东西都在Haddock文档中关于<a href="http://cvs.haskell.org/Hugs/pages/libraries/base/Control-Monad.html">Control.Monad</a>的那部分，但要发现它，我们得先放下do-notation才行。

关于函数定义从do到“bind”notation之间的转换有几条简单规则，它们讲解起来很容易，并且已经有了<a href="http://web.mit.edu/ghc/www/users_guide/syntax-extns.html">文档</a>。不过在这里我对语法的转换规则不感兴趣，而要以bind-notation的方式重新讲解IO的基础 — 这样monad风格的结构就可以更清楚地显示出来了。

最重要的两个IO操作应该就是putStr和getLine了，它们差不多就相当于lisp/scheme中的print和read，Basic中的PRINT和INPUT之类。作为一门纯函数式、严格类型的语言，Haskell中的这些操作应该都是用可标明类型的函数来表示。

我们先看看putStr的类型：
<pre lang="haskell">
putStr :: String -> IO ()
</pre>
（我们假定读者已经有阅读类型声明的能力）显然，putStr是取一个字符串做参数，而它的返回结果可以读作“外部世界(Outside World)” — 实际上，要不是这个蹩脚的表达式，OutsideWorld完全可以看作是IO的别名。我们再看看getLine：
<pre lang="haskell">
getLine :: IO String
</pre>
getLine没有参数，只是从外部世界得到一个字符串。作为一个来自外界的字符串，在处理上有一定限制 — 这时就该monad出场了。通过monadic的结构来处理IO，我们可以使用几个简单的函数来没有引用透明地操作getLine所得的变量。

第一个函数是“bind” — 简单起见，由中缀运算符 >>= 表示。它的类型是：
<pre lang="haskell">
(>>=) :: forall a b . m a -> (a -> m b) -> m b
</pre>
“Bind”取一个monadic值（我们这里是一个IO String），一个将一个纯值（如String）转为一个monadic值的函数做参数，并返回一个monadic值。关于它用法的一个例子：
<pre lang="haskell">
shout = getLine >>= (putStr . map toUpper)
</pre>
Bind的第一个参数是一个IO String类型的monadic值，第二个参数是函数(putStr.toUpper)，它取一个字符串做参数并返回一个IO “coin”。如你所料，shout的类型就是个外部世界的值 — 即一个 IO “coin”。
<pre lang="haskell">
shout :: IO ()
</pre>
定义monad第二个基本函数是return。它的类型为：
<pre lang="haskell">
return :: (Monad m) => a -> m a
</pre>
例如，
<pre lang="haskell">
superTwo = return "Two"
</pre>
很简单。
<pre lang="haskell">
superTwo :: (Monad m) => m String
</pre>
有了这两个函数，就可以定义出一个完整的monad了。其他所有的monadic函数都是用它俩定义出来的。一个严格的monad必须满足以下的三个数学性质。
<ol>
<li>	(return x) >>= f == f x</li>
<li>	m >>= return == m</li>
<li>	(m >>= f) >>= g == m >>= (\x -> f x >>= g)</li>
</ol>
由此看出，我们可以用haskell完整地定义出monad — 这就表明了它不是一个内置的特性，而是一个抽象的数学结构。就像rings，borelians，quaternions之类的抽象数学结构一样得以榨干haskell的表达能力：
<pre lang="haskell">
class Monad m where
    (>>=) :: forall a b . m a -> (a -> m b) -> m b
    return :: a -> m a
</pre>
在haskell的入门学习中，你可能已经用了几个monad的实例，如列表（[a]），Maybe还有，是的，IO。每个特定实例的“bind”和“return”实现都可以在一个<a href="http://www.nomaware.com/monads/html/index.html">monad教程</a>中找到。免得再写另一个monad教程，从现在开始，我们就专注在IO monad上。

通过满足以上规则的(>>=)和return可以构造出许多有用的操作 — 这在<a href="http://cvs.haskell.org/Hugs/pages/libraries/base/Control-Monad.html">Haddock文档</a>中Control.Monad那部分里有很多讲解。我们这里还得研究另一个函数 — bind的一个变体，忽略了返回结果的第一个参数（传递给下一个计算），这样一来我们就可以简单地将不相干的操作连续起来。这个函数的类型是：
<pre lang="haskell">
(>>) :: (Monad m) => m a -> m b -> m b
</pre>
根据描述，(>>)实现起来非常的简单：
<pre lang="haskell">
x >> y = x >>= (\_ -> y)
</pre>
譬如，这就是连续两个putStr操作的方法（要知道putStr对前一个putStr返回的()并不感兴趣）。
<pre lang="haskell">
example = putStr "Print me! \n" >> putStr "Print me too! \n"
</pre>
现在我们可以写个monad风格IO的简单例子，用bind notation:
<pre lang="haskell">
greet = getLine >>= (putStr . ("You're "++) . (++" years old! \n")) >> putStr "Congratulations! \n"
</pre>
这样，IO String里的内容就应用在了这个函数里：
<pre lang="haskell">
\x-> ((putStr . ("You're "++) . (++" years old! \n")) >> putStr "Congratulations! \n") x.
</pre>
继续推导：
<pre lang="haskell">
(\x-> ((putStr . ("You're "++) . (++" years old! \n")) x) >>= (\_ -> putStr "Congratulations! \n")
</pre>
这就连续了两个print操作。通过这个数学结构描述连续行为时，haskell中的语法糖可以允许你在一个伪命令式（不是真的命令式）的风格中忽略掉那堆复杂的括号套括号、lambda抽象、point-free表达式和符号关联。
<pre lang="haskell">
greet = do {
    age < - getLine;
    putStr (”You’re “++age++”years old! \n”);
    putStr (”Congratulations! \n”);
}
</pre>
不管这命令式的外表，它并非命令式的变量赋值是毋庸置疑的：只不过是一个把monadic计算（这里是从“外部世界”读值）结果保存到一个符号中的便捷语法罢了，这样我们就可以在后面再处理它，而不必纠结那大块的lambda表达式。

到现在明智的读者已经认清了do-notation的肤浅本质，可以继续他<a href="http://www.cs.utah.edu/~hal/docs/daume02yaht.pdf">basic haskell</a>或<a href="http://www.nomaware.com/monads/html/index.html">monad</a>的学习了。更重要的是，他会在以后面对<a href="http://www.cs.uu.nl/~daan/parsec.html">monadic parser</a>这样复杂的数学结构时理解到do-notation的真正含义。

实际上，出于智慧与理性，我建议大家在每个“toy”中都应该使用bind notation，善求知的haskell程序员的学习，都是按照自己的思路去理解，直到那大块的函数将大脑的栈塞满为止。理解简单的IO问题并不是很重要，而深入理解连续的副作用操作在传统的命令式语言与函数式语言中IO函数组合的不同才是学习haskell编程思想的重中之重，这对摒弃那种“查手册的机器人”的编程习惯也是大有好处的。

</pre>
