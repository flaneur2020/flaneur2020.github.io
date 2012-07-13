---
layout: post
title: "haskell的排列组合"
tags: 
- haskell
- "备忘"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

从chinaunix的fp板上看到的, 摘过来
原载自 <a href="http://bbs2.chinaunix.net/thread-1289053-1-2.html">http://bbs2.chinaunix.net/thread-1289053-1-2.html</a>
作者为drunkedcat和MMMIX

组合:
<pre lang="haskell">
combination :: [a] -> [[a]]
combination [] =  [[]]
combination (x:xs) =  (map (x:) (combination xs) )++ (combination xs)
</pre>

排列:
<pre lang="haskell">
permutation :: Eq a => [a] -> [[a]]
permutation [] = [[]]
permutation xs = concatMap (\x -> map (x:) $ permutation (delete x xs)) xs
</pre>
