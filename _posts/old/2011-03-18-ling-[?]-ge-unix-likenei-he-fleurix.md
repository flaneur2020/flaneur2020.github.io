---
layout: post
title: "另一个unix-like内核, Fleurix"
tags: 
- ASM
- C
- Kernel
- Unix
- "杂碎"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
  _wp_old_slug: ""
---

该是在去年五月份，照着网上的教程写了几行helloworld，简简单单的int 13h放在virtualbox里打印出来一段红色的“screw you guys all fucked up~”。随后从这段不得体的汇编开始，慢慢地文件系统、内存管理、输入输出、进程管理等等初具轮廓，到现在一个相对完整的内核，不觉已过了九个月。时间就是个见证成长的东西 :)

<a href="https://github.com/fleurer/fleurix">https://github.com/fleurer/fleurix</a>

37个系统调用，七千行C，二百多行汇编。没有管道，没有swap，也不是基于POSIX，各种特性是能删即删，能简即简。不过也算完成了它的设计目标，那就是跑起来。Fleurix已经有了：

<ul>
<li> minix v1的文件系统。原理简单，而且可以利用linux下的mkfs.minix，fsck.minix等工具。
<li> fork()/exec()/exit()等等。a.out的可执行格式，实现了写时复制与请求调页。
<li> 信号。
<li> 一个纯分页的内存管理系统，每个进程4gb的地址空间，共享128mb的内核地址空间。至少比Linux0.11中的段页式内存管理方式更加灵活。
<li> 一个简单的kmalloc()(可惜没大用上)。
<li> 一个简单的终端。
</ul>

<img src="http://i.min.us/im2snS.jpg"></img>
<img src="http://i.min.us/ikheqK.jpg"></img>

硬伤就是，没有硬盘分区，内存也写死了128mb，恐怕无法在真机上运行。

<hr />

编译环境: ubuntu
工具: rake, binutils(gcc, ld), nasm, bochs, mkfs.minix

<pre code="bash">
git clone git@github.com:Fleurer/fleurix.git
cd fleurix
rake
</pre>

<hr />

坦白地说，现在编写内核比起Linus的时代已经容易太多了。有模拟器，有工具链，网上也有相当多的资料可以参考，更有Unix世界的先贤更留下的宝库。但是现在又有谁肯再花十年时间让内核从婴儿长大成人呢？就是有人肯做，也只是重复造轮子罢了，非为智者所取。

那么，Fleurix的意义在哪里呢？

实际应用不合适；说可供大家学习又未免太自以为是。但是对我自己而言，如果不写Fleurix，在这九个月里总不至于去四处搜集Unix的老书，打印论文，理解每个函数的细节，写下这些代码，甚至耽误勾搭妹子。要总是功利地看待得与失的话，又何曾不是得不偿失呢。

因此，何必把一些嘴上喊的事情看的太重。把结果看作是过程的一步分，体会这个过程就够了。尤其是在查阅资料中，居然能够找到跟Unix世界的先贤交谈的感觉，就此，我已心满意足。
