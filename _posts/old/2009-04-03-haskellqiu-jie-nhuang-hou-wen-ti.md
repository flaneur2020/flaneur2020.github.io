---
layout: post
title: "haskell求解n皇后问题"
tags: 
- FP
- haskell
- "笔记"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

八皇后问题大家都已耳熟能详, 貌似没有赘述的必要.

看了lich_ray同学的这篇<a href="http://www.javaeye.com/post/421913?page=1">用 Python 秒掉八皇后问题!</a>, 对list comprehension的表达能力有了新的认识. 他的楼下们几乎列出了所有语言的八皇后求解实现, 其中就有albertlee大牛的haskell版本:
<pre lang="haskell">import Control.Monad
import Control.Monad.Writer
import Data.List

diagonal (x1,y1) (x2,y2) = x1 + y1 == x2 + y2
                        || x1 - y1 == x2 - y2

nqueens n = execWriter $ f [1..n] 1 []
    where f [] _ ps = tell [ps]
          f cs r ps = forM_ cs $ \c ->;
                          unless (any (diagonal (r,c)) ps) $
                              f (delete c cs) (r + 1) ((r,c):ps)
main = print $ nqueens 4</pre>
我愚钝, 没看懂. 自己写一个吧, 解题思路无非都是大同小异, 直接把lich_ray同学的话copy过来:
<blockquote>归纳法定义：
　　什么是归纳法定义？回忆一下经典的求级数例程是怎么写出来的。我们根据数学归纳法得到：
　　f (0) = 1
　　f (n) = 1 + f (n - 1)
　　然后把这些抄成编程语言的形式。对于函数 queens 也是这样，我们要确定这个函数的递归下界和递推表示。
　　递归下界很好办，就是 queens(n,0) （此处 n 忽略，因为不影响结果）的输出结果。对于一个没有列数的棋盘，只有一个解，空解 []；同时，输出的解集也只有这一个元素，为 [[]]（下面的数学定义使用了集合表示代替列表）。
　　f (*,0) = {{}}
　　递推表示是什么呢？我们可以在纸上画画图，不难发现，对于一个解，你画出的最后一个位置就是在前面已画出的少一个棋子的格局的基础上再加一个位置安全的棋子。设这个“加”函数为 g (x,y)，“安全”函数为 s (x,y)。那么
　　f (n,m) = {g (x,y) | x ∈ [0, n], y ∈ f (n,m-1) s (x,y) = true}</blockquote>
我初学haskell, 写的很难看. 思路和lich_ray同学的貌似并不完全一致, 不过有一点是毋庸置疑的, 那就是归纳法. 一个m列n行的"皇后问题"可以看作是给一个已经放n个皇后的m列n行的棋盘加一个皇后, 到最后归纳到一行的棋盘, 那就有m种放法. 按照bonus的说法, 就是"Usually you define an edge case and then you define a function that does something between some element and the function applied to the rest." , 这貌似就是递归求解的基本思路了.

求解的queue函数貌似需要两个参数, 两个Int值表示列数和行数, 皇后问题的一个解中只要能表示出所有皇后的所在的列就行了, 放在一个[Int]中, 而queue会有很多解, 所以queue的返回类型应该为[[Int]]. queue m (n-1)所得的结果中有的不能再放皇后了啊, 怎么办? filter之留下还能放的: filter putable $ queue m (n-1). 好了, 剩下的都是可以放皇后的, 都给它放上: concatMap put $ filter putable $ queue m (n-1) . 为什么concatMap呢? 因为put会有很多种结果啊. 好的, 就这么一句:
<pre lang="haskell">queue m n = concatMap put $ filter putable $ queue m (n-1)</pre>

接下来写出putable和put的实现就好了, 想一下, 什么能样的棋盘是putable的? 有安全的地方的棋盘就是putable的, 即(safe_places xs /= []). 该怎样put呢? 在现有的棋盘上所有安全的地方放一个皇后, 把每种情况都放在一个list里就行了: (map (:xs) $ safe_places xs). 两个函数都用了一个safe_places xs , 貌似有重复调用, 效率可能会低些? 呃, 不过据说haskell是有引用透明的, 以相同的参数调用同一函数时貌似会有相应的优化.

最后实现safe_places就行了, 不扯了, 上代码
<pre lang="haskell">
module Main where
import Data.List

queue :: Int -> Int -> [[Int]]
queue m 1 = [ [x] | x  <- [1..m] ]
queue m n = concatMap put $ filter putable $ queue m (n-1)
    where
        putable xs = (safe_places xs /= [])
        put xs = map (:xs) $ safe_places xs
        safe_places xs = [1..m] \\ (concatMap (\(x,y) -> [x-y,x,x+y]) $ zip xs [1..])
</pre>
