---
layout: post
title: typedef unsigned long VALUE;
tags: 
- C
- ruby
- VM
- "笔记"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

因为swdpress.cn的死掉严重打击积极性，好久没更新了。此文纯凑数。

ruby中除了Fixnum，其余所有类型的值都是引用传递。读《<a href="http://code.google.com/p/rhgchs/">ruby hacking guide</a>》时看到如下的定义：

<pre lang="c">
typedef unsigned long VALUE;
</pre>

同lua那union+tag不一样，ruby中所有类型的值都存放在一个VALUE中，而没有tag指明其类型。如果是Fixnum，就把数值直接放在VALUE里；如果是其它类型，则存放其地址（unsigned long和C中指针类型的长度一致）。不过万一作为Fixnum的VALUE指向的地址与其他对象有重叠怎么办？它又怎么区别数值和地址呢？

这用到了一点tricks，C的struct在内存中都是以4字节对齐，因此ruby中所有对象的地址都偶数。在表示Fixnum时，ruby就将C的int值左移一位再加一，使其看起来总是个奇数，这样就不会与ruby的其它对象有重叠。

<pre lang="c">
 123  #define INT2FIX(i) ((VALUE)(((long)(i))<<1 | FIXNUM_FLAG))
 122  #define FIXNUM_FLAG 0x01
(ruby.h)
</pre>

判断一个VALUE是Fixnum还是地址，只需判断VALUE是奇数还是偶数就行了。

表示true,false和nil是这样：

<pre lang="c">
 164  #define Qfalse 0        /* Ruby's false */
 165  #define Qtrue  2        /* Ruby's true */
 166  #define Qnil   4        /* Ruby's nil */
(ruby.h)
</pre>

都是偶数。像刚才说的，它们不会被当作是地址了么？这就用到了另一个trick：进程虚拟地址空间的前一部分都是不可访问的。因而0,2,4的地址上不会存在ruby对象。

PS: 感谢FX大大提醒，这套方法有个标准的名字叫做“tagged pointer” :)
