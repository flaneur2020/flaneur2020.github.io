---
layout: post
title: "关于GC的FAQ"
tags: 
- C
- GC
- MM
- "翻译"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

翻译：ssword
原文：http://www.iecc.com/gclist/GC-faq.html

这是GC邮件列表的FAQ草稿，欢迎评论、标注或贡献内容。它暂时分为三(?)部分，大约是一般问题，技巧及算法，语言接口和高级论题。以后内容要是多了，这些内容可以重新组织便于查阅。

<!--more-->

我们也提供了这些内容的文本文件，即GC-algorithms.txt, GC-lang.txt, and GC-harder.txt。

这里一些内容的学术气可能要差些，而更偏重于传道（当然具体的内容不能落下）。关于垃圾收集多好多好之类，言辞越简洁越好。

<strong>Common questions
常见问题</strong>

<strong>What is garbage collection?
什么是垃圾收集？</strong>
垃圾收集可以是语言运行时的一部分，也可以是个附加库。它们能够与编译器、硬件或操作系统打交道，自动检查出程序中不再使用的内存，并将其回收利用。这也称作“自动存储（内存）回收”。

<strong>Why is it good?
它好在哪？</strong>

手工内存管理既浪费程序员的生命，也容易引入错误。相当多的程序都难免于内存泄漏，使用了错误处理或线程的程序尤其如此。

未使用过垃圾收集的朋友可能难以察觉它的第二大好处，那就是不必纠结于内存管理的细节（“谁来回收这块内存”），从而简化了程序组件（子程序、库、模块、类）间的接口。

你可以在 Object-Orientation FAQ -- 3.9) Why is Garbage Collection A Good Thing?了解更多内容

<strong>Is garbage collection slow?
垃圾收集会不会很慢？</strong>

并非如此，现代的垃圾收集器的速度几乎可与手工内存管理(malloc/free或new/delete)相媲美。垃圾收集可能不如特定程序中相应的allocator快，不过为保证手工allocator正常工作而添加的额外代码（如引用计数）往往会使得程序比垃圾收集更慢，共享资源的多线程程序中尤其如此。

正如开始所说，内存已经足够便宜，垃圾收集可以应用到非常大的堆上（比如1GB）。活动内存如果足够的大，暂停时间依然是个问题。不过对于现代的垃圾收集器而言，其暂停时间通常也就0.1秒，对人类的交互几乎可以不计。要是小块活动内存，暂停时间就更少了：0.01秒以下。

<strong>Can I use garbage collection with C or C++?
我可以在C\C++中使用垃圾收集吗？</strong>

应该可以。对于存在遗留代码的C和C++可能还差些，不过现代的垃圾收集器（测试良好，高效，无暂停）几乎已经支持了一切。了解更多请看GC, C, and C++ 。

<strong>Does garbage collection cause my program's execution to pause?
垃圾收集会不会让我的程序暂停？</strong>

不必。有不一的算法允许垃圾收集并行化、增量化甚至“实时化”。比如C\C++下边就有增量式的垃圾收集。

<strong>Where can I get a C or C++ garbage collector?
怎样搞到C\C++的垃圾收集？</strong>

Boehm-Weiser collector
http://www.hpl.hp.com/personal/Hans_Boehm/gc/ or
ftp://parcftp.xerox.com/pub/gc/gc.html
Great Circle from Geodesic Systems  or 800-360-8388 or http://www.geodesic.com
Kevin Warne  or 800-707-7171

<strong>Folk myths
坊间传闻</strong>
<ul>
	<li>GC一定不如手工内存管理快</li>
	<li>GC一定会让程序暂停</li>
	<li>手工内存管理就不暂停</li>
	<li>GC与C\C++水火不容</li>
</ul>
<strong>Folk truths
其实…</strong>
<ul>
	<li>大部分动态创建的对象其实少有与其它对象的关联，通常其引用数就是1。</li>
	<li>大部分对象的生存期都很短。</li>
	<li>对象大小、生存期呈爆炸式分布，而不是正态分布。</li>
	<li>VM很重要</li>
	<li>缓存很重要</li>
	<li>不成熟的优化乃万恶之源。</li>
</ul>
<strong>Tradeoffs
权衡</strong>
<ul>
	<li>严格式 vs. 保守式</li>
	<li>移动/压缩 vs. 无移动</li>
	<li>暂停 vs. 增量 vs. 并行</li>
	<li>分代 vs. 无分代</li>
</ul>
<strong>GC, C, and C++</strong>

<strong>What do you mean, garbage collection and C?
你说什么，C语言的垃圾收集？</strong>

可以引入一个垃圾收集器自动管理内存，从而代替malloc和free的手工申请或释放。通常的做法是将malloc替换为垃圾收集器的allocator，而free则替换为一个空函数。比如X11就是如此。

也可以令free依然生效，不过垃圾收集器就弱化为一个防弹衣的存在，堵住潜在的内存泄漏。这一做法也是有了很多应用，并且工作良好。好处是方便程序员手工管理内存，而程序员顾不到的地方，就交给垃圾收集。这不一定比空free风格的快，不过可能让堆变得小点。

<strong>How is this possible?
如何做到的？</strong>

C兼容的垃圾收集可以知道指针在什么地方（例如"bss","data"和堆栈里面），保留在堆中的数据结构可以让它们很方便就能找出哪段数据是可能的指针。来个搜索遍历所有可以访问的指针，剩下没被访问的就是垃圾。

<strong>This doesn't sound very portable. What if I need to port my code and there's no garbage collector on the target platform?
这个听起来移植性好像不怎么样。我需要将代码移植到没有垃圾收集器的平台上该怎么办？</strong>

有些代码一定是平台相关的，但是大多操作系统都有足够的功能，所以C的垃圾收集其几乎可以在所有平台移植。一般而言，只要有垃圾收集的实现，移植性就不是问题。在Boehm-Weiser的移植还不多的时候，我曾经自己移植过两次，其时我对操作系统的底层接口还不甚熟悉。

<strong>Won't this leave bugs in my program?
这不会给我程序引入bug吧？</strong>

这看你怎么想了。垃圾收集器可以解决程序员的很多问题，从而可以把精力放在其他地方，使得工作完成的更轻松。某种意义上讲，这跟浮点算法和虚拟内存的初衷是一致的。它们被发明出来都是为了解决些折腾程序员的乏味问题（如比例运算、换出页到硬盘）。没有FP和VM支持也可以写代码来实现相应的功能，不过只要有这功能可用，人们就不会自己写。一样的道理。

如果程序中用了垃圾收集风格的代码再扔掉垃圾收集器，内存泄漏的bug是肯定的。同样，如果一个程序用了FP（或VM）相关的代码，再卸掉浮点单元和MMU，bug也是一定的。

实际上，许多使用malloc和free的程序里都有内存泄漏，使用个垃圾收集器反而可以减少程序的bug。这可比手工跟踪内存再手工修复利索多了，要是跟踪发现内存泄漏出在三方库，还根本没办法修复。

<strong>Can't a devious C programmer break the collector?
程序员可不可以把这收集器玩崩溃?</strong>

当然可以。不过大多数人应该更喜欢研究些正事，而不是整天想着把自己的工具玩坏掉。收集器需要正在内存空间中的指针，所以想玩坏就有办法。例如双向链表中的翻转指针技巧就不能用——这指针长得不像指针。如果程序把指针写到文件中，呆会再读出来，没准就崩了，因为这些指针指向的内存很可能已经被回收了。没大有程序会这么玩，所以对大多数程序而言，这不是问题。C中一般的（合法的）指针运算都是没问题的。

某技术团队在考虑使用GC时想到一个问题，那就是使用pointer mangling技巧可能会搞出“不透明”的指针。所谓pointer mangling，就是三方库中的指针与一固定的随机数异或下，这一来三方库中的数据只有按照特定的接口才可以访问。这个不兼容保守式GC，也不兼容Ansi C标准的严格编译...甚至会迷惑内存泄漏的跟踪器（跟保守式GC用的技术一样）。不过即便如此，它们依然是合法的程序。

Insert more questions here -- send them to

<strong>What does a garbage collector do about destructors?
垃圾收集器如何对待析构函数？</strong>

析构函数就是对象被释放时候执行的代码，它的用途之一就是来手工回收内存，比如递归地回收对其它对象的引用。垃圾收集器里本没有析构函数的必要：如果一个对象是垃圾，它引用的所有对象就都是垃圾，自然会被收集到。

利用析构函数还可以做点其他事情。有两个应用比较典型：

与外部环境相关的对象的状态。比如文件相关的对象：当一个文件对象被回收时，垃圾收集器应该能保证能够刷新缓冲区、关闭文件，并将文件相关的资源返还给操作系统。

再就是程序需要保留一组在别处引用的对象。程序可能需要知道对象的功能，而不阻止它被收集。

解决这问题有很多方法：

有些系统是“built in”的垃圾收集，它就可以对外部引用的资源有所了解，处理起来也就心里有数
有不少垃圾收集系统有个“弱引用(weak pointer)”的概念，就是不被垃圾收集器认作引用的指针。如果一对象是个弱引用，那么就可以被GC收集。弱引用可以用来实现对象的list之类的数据结构。
再就是，java里可能会这样保护外部资源R：

<pre lang="java">
class ClientR {
   CRWeak wr;
   // delegate all methods to wr;
   ClientR() {
     wr = new CRWeak(this);
   }
}
 
class CRWeak extends WeakReference {
  static ReferenceQueue rq = new ReferenceQueue();
  static {
         Thread th = new CRCleaner(rq);
         th.setDaemon(true);
         th.start();
  }
 
  CRWeak(Object x) {
     super(x, rq);
  }
  ExternalResource r;
  // delegated methods from ClientR
}
 
class CRCleaner extends Thread {
  ReferenceQueue rq;
  CRCleaner(ReferenceQueue rq) { this.rq = rq; }
  public void run() {
         while (true) {
           CRWeak x = (CRWeak) rq.remove();
           // Release x.r
         }
  }
}
</pre>

当没有对象引用ClientR时候，这块内存就回收了，而对它的弱引用则移动到了各自的引用队列。清扫线程可以保证这些外部资源的回收。

许多GC系统都有“析构函数”的概念。垃圾收集器在回收一对象的时候，会执行该对象的一个函数来执行必要的清理操作。这会比较复杂，因为有些问题需要考虑：
<ul>
	<li>对象是在什么时候才会被收集？它并不像看起来这么简单，这对一些资源吃紧的应用尤为棘手。</li>
	<li>析构函数该在哪个线程、资源或者上下文下边执行？</li>
	<li>对象的交叉引用该怎么办？</li>
	<li>如果一个析构函数使得对象不再是垃圾，又该怎么办？</li>
</ul>

<a href="http://www.iecc.com/gclist/GC-faq.html#For%20more%20information"><strong>For more information</strong></a>

<strong>Contributors (so far)
贡献者（目前）</strong>

David Gadbois
Charles Fiterman
David Chase
Marc Shapiro
Kelvin Nilsen
Paul Haahr
Nick Barnes
Pekka P. Pirinen
GC FAQ table of contents
Techniques and algorithms
Language interfaces
Difficult topics
Silly stuff
