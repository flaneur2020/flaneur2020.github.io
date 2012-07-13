---
layout: post
title: "早期unix同步机制的简单笔记"
tags: 
- Kernel
- Unix
- "笔记"
status: publish
type: post
published: true
meta: 
  _wp_old_slug: ""
  _edit_last: "2"
---

在写那篇关于buffer cache的post时还没有意识到，B_BUSY+B_WANTED+sleep/wakeup，即为早期unix中一套通用的同步机制了。只要存在在不同进程间共享的数据，不管并行还是并发，就都免不了得考虑竞态条件和同步问题。在处理某个对象之前都先给它上锁，防止其他进程碰它。比方前面说的B_BUSY，改叫B_LOCK没准更直白。

早期Unix内核中没有通用的内存分配器，不同类型的对象单独静态分配一个固定长度的数组，inode，super block，file，buf乃至proc等等皆如此。所以有关对象分配释放的代码都是十分相似，而且分配即缓存，统一到一类简单的机制之下。可是反过来看就是把功能相近的代码重复了n遍，也是模块化和重用性不好的体现吧。这也正是现代unix所努力的方向，其结果之一即solaris/linux中的伙伴系统（用于申请内存）和slab（用于对象的缓存）。

<strong>sleep/wakeup</strong>

早期的unix一般都是非抢占的内核，内核态的进程不可以被其它进程打断，不过可以自愿放出控制权（swtch）。一个常见的情景就是，在申请资源的时若该资源的空间已经用尽，就让这个等资源的进程睡眠（sleep）把控制权交给其它进程，等某进程释放了资源则唤醒所有等待该资源的进程（wakeup）。拿偏理论的操作系统书上的说法，sleep/wakeup即早期unix的同步原语了。

其函数原型如下：
void sleep(uint chan, int pri)；
void wakeup(uint chan)；

chan是事件通道（event channel）的缩写，而pri用来指定进程在苏醒那一刻的优先级。像刚才说的比如等待一个空闲的inode，就直接sleep(&ino, PRIINO);把inode对象的地址当作chan即可。

<strong>锁</strong>

有了sleep/wakeup当作同步原语，再给相应的对象加上两个flag就相当于实现了一个简单的锁。上锁即为了防止其它进程碰它，所以获取对象时即上锁，比如iget,namei,getfs,getblk等等。若获取对象时它正在被其它进程使用，就sleep等它释放。这里需要留意一下，就是同一个资源若在释放前在同一进程里重复申请，就会睡眠而永远都不会被唤醒，比如：

bp = getblk(rootdev, 1);
bp = getblk(rootdev, 1); // hangs forever

所以就得小心，资源用完一定记得释放或unlock。不像用户态的程序malloc不free、open不close似乎也没什么大碍，在这里忘记释放的下场就是华丽丽一个死锁。
