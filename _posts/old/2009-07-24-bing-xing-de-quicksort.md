---
layout: post
title: "并行的quicksort"
tags: 
- FP
- haskell
- parallel
- trick
- "备忘"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

haskell搞并行还是挺方便的。纯函数式没状态嘛，这儿锁啊钥匙啊什么的都不用管了。有个Control.Parallel，里貌似只有两个函数，一个par，表示两个并行计算；一个pseq，表示连续计算。参数都是两个名字。像处理个分治算法啥的，就再合适不过了。于是再次膜拜惰性求值，写起来真的太奇特了。

<pre lang="haskell">
module Main where
import Control.Parallel

--the classic one
qsort :: (Ord a) => [a] -> [a]
qsort [] = []
qsort (x:xs) =
 (qsort lt) ++ [x] ++ (qsort gt)
 where
  lt = filter (<x) xs
  gt = filter (>=x) xs


--the parallel one
psort :: (Ord a) => [a] -> [a]
psort [] = []
psort (x:xs) =
 sorted_lt `par` sorted_gt `pseq` (sorted_lt ++ [x] ++ sorted_gt) ----unbelieveable,isn't it? :>
 where
  sorted_lt = psort $ filter (<) xs
  sorted_gt = psort $ filter (>=x) xs


main = do {
 list <- return [5000,4999..1];
 print $ qsort list;
 print $ psort list;
}
</pre>

profile下，给ghc添加一个编译选项 -prof -O：
<pre>
ghc --make -prof -O -auto-all quicksort.hs
</pre>
执行程序，加一个选项 +RTS -p，它会在本目录下生成一个quicksort.prof文件
<pre>
quicksort +RTS -p
</pre>

quicksort.prof文件的部分内容：
<pre>
 Mon Jul 20 16:40 2009 Time and Allocation Profiling Report  (Final)

    quicksort +RTS -p -RTS

 total time  =        6.86 secs   (343 ticks @ 20 ms)
 total alloc = 1,401,627,416 bytes  (excludes profiling overheads)

COST CENTRE                    MODULE               %time %alloc

qsort                          Main                  69.1   49.9
psort                          Main                  30.9   49.9

</pre>

可见在这台双核的机器上，性能提高了一半多 :)
ps:测试的数据貌似不是很好（按说该用个随机数列），不过知道有这回事就行了~
