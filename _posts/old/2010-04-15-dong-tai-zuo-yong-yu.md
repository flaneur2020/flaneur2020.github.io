---
layout: post
title: "动态作用域"
tags: 
- PL
- "笔记"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

好像刚弄明白动态作用域是怎么回事，拿个例子：

<pre lang="javascript">
var a=0; //global
def f1():
    a=1
def f2():
    var a; //local
    f1()

f2()
puts a
</pre>

若是静态作用域，f2调用f1，f1修改全局变量a的值为1，输出1。这个多自然...
可是动态作用域就...输出0，全局变量a的值没有变化。

f2不是调用了f1么，f1不是改变a的值么...是啊，f2调用的那个f1改变的是f2那个局部变量a的值。也就是说，同一个函数在不同的环境下调用会有不同的行为。

好吧我正在纳闷为什么会有这种东西存在过...囧
