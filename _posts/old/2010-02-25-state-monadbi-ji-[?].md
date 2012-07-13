---
layout: post
title: "State Monad笔记一"
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

去年听王猫猫讲过之后就一直心安理得再也没看过Monad，以至于到昨天还不知State Monad为何物...囧

关于State Monad，似乎见过的教材都是拿随机数作例子，拿Monad跟一堆let对比，满眼的二元组绕啊绕啊，我给你个Monad让它不那么绕（存疑）～其实什么东西理解不了，往往就是因为不知道它怎么用（比如复变函数 ="=）

换个例子，像那个筛掉一个List中重复元素的nub函数多好....在命令式语言下边大约是这样：

<pre lang="python">
def nub(xs)
    result=[]
    for x in xs
        if not result.include? x
	     result << x
    return result
</pre>

若在函数式语言下边，用迭代大约是这样：

<pre lang="haskell">
inub :: (Eq a) => [a] -> [a]
inub = inub' [] 
inub' rs [] = rs
inub' rs (x:xs) 
    | x `elem` rs    = inub' rs xs
    | otherwise     = inub' (rs++[x]) xs
</pre>

inub'这个函数中有个rs参数，就相当于前面那个result变量。它的值会随着迭代发生改变，不同之处就是ts是引用透明的没有副作用。每次函数调用就是一个状态，状态就是一个值。

不喜欢给函数多加个参数怎么办？State Monad可以做到。在State Monad里面有两个函数可以使用：put，设置当前状态的值；get，获得当前状态的值。就相当于给函数加了一个（只有一个）可变的全局变量，在调用别的函数或者递归时这个值可以保留。

State Monad版本（我没觉得好看 ="=）：

<pre lang="haskell">
mnub :: (Eq a) => [a] -> [a]
mnub xs = evalState (mnub' xs) []
mnub' [] = get
mnub' (x:xs) = do {
    rs <- get;
    if x `elem` rs then (do {
        mnub' xs;
    })
    else (do {
        put $ (rs++[x]);
        mnub' xs;
    });
}
</pre>

原理大约就是do-notation中的每个语句都是用一个wrapper的类型包起来再不断往下传递么，而State的wrapper就是个类型为(a,s)的二元组（外加一层lambda？），每个语句的返回值放到s里，那个全局的值则放在a里...具体还有点迷糊，哪天推导下再说好了  >_<
