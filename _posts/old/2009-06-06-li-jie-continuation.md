---
layout: post
title: "理解continuation"
tags: 
- continuation
- FP
- "笔记"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

当时小，不懂事，初学函数式编程时一头就扎的就是continuation和monad，一直搞不明白就一直放着。上个星期在ShiningRay老师的这篇《<a href="http://shiningray.cn/continuations.html">简介延续“Continuation”</a>》里看到：
<pre lang="python">
def call_cc(f,c):
     f(c,c)
</pre>
貌似很泪流满面的样子，没想到这东西有这么简单。之后在图书馆里写了一个小时的伪代码，貌似搞明白了。按照Continuation Passing Style(cps)的设定，函数没有返回值，而是用函数的运算结果调用函数最后一个参数，很简单吧。不过那些特性又是从何而来呢，像保存状态尾递归优化之类？现在想想，之前之所以不理解，大概就是因为不会写cps的递归吧，其实动手写的话也就那回事。国父孙先生不是说么，知难行易。

如下伪代码，求前n个自然数的和：
<pre lang="python">
def sum(n):
    if (n==0): 0
    else :
        n+sum(n-1)
</pre>
换成等价的前缀形式(if就懒得改了，知道这回事就好)：
<pre lang="python">
def sum(n):
	if (n==0): return 0
    else
        return add(n, sum(sub(n,1)))
</pre>
好，换成cps：
<pre lang="python">
def sum(n,c):
    if (n==0): c(0) #“延续”下去，而不是返回
    else:
        sub n, 1, (\x1 -> #一个lambda，匿名函数
        sum x1, (\x2 -> #递归在这里
        add x2 ,n ,c))
</pre>
再一个迭代的例子：
<pre lang="python">
def iter_sum(n, r):
     if (n==0) : return r
     else : iter_sum(n-1, r+n)
</pre>
换成前缀：
<pre lang="python">
def iter_sum(n,r):
     if (n==0): return r
     else : iter_sum(sub(n, 1), add(r,n))
</pre>
cps:
<pre lang="python">
def c_iter_sum(n,r,c):
    if (n==0): c(r)
    else:
        sub n, 1, (\x1 ->
        add r, n, (\x2 ->
        c_iter_sum(x1, x2, c))) #尾递归
</pre>
可以看出，cps中引入了些自由变量，把函数求值的结果放在堆里而非栈中，于是就可以实现不限深度的递归。不过这样貌似增加了垃圾收集的负担，没有银弹哇。

contiuation中只有“延续”而没有“返回”，每一步“延续”都是对后面函数的调用，通过一条函数与函数的链构成了顺序结构。ShiningRay老师貌似说过，一个函数调用就是一个状态。callcc貌似就是把“延续”的下一个函数保存，从而保存了状态。Smalltalk的那个Seaside框架貌似就是把contiuation作为对象持久化，在下次访问时找到这个持久化的contiuation将状态还原，从而模拟出仿桌面程序开发的效果。貌似要比asp.net那WebForm的实现更加自然些。
