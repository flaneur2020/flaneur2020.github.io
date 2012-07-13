---
layout: post
title: "unix笔记：the Buffer Cache"
tags: 
- C
- FS
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

不知这个该怎么译，缓冲缓存？ = =

Buffer Cache在Unix中不只一套缓存机制，更是系统访问块设备中间不可或缺的一层。在早期的Unix中大约可以扮演三个角色：中断请求队列、访问缓冲、高速缓存。简单起见，这里拿Unix V6讨论。

buffer相关的各种信息保存在buffer header中，而其中缓存的数据也就是buffer body分布于内存中固定的一块区域（大约是物理内存的10%），保持一对一的关系。每一块buffer body的大小必须是设备中块的整数倍。

<!--more-->

<h3>buffer pool的结构</h3>
数据结构绝对比算法重要。
<pre lang="c">4520: struct buf
4521: {
4522:        int     b_flags;                    /* see defines below */
4523:        struct  buf *b_forw;            /* headed by devtab of b_dev */
4524:        struct  buf *b_back;            /*  "  */
4525:        struct  buf *av_forw;           /* position on free list, */
4526:        struct  buf *av_back;           /*     if not BUSY*/
4527:        int     b_dev;                  /* major+minor device name */
4528:        int     b_wcount;               /* transfer count (usu. words) */
4529:        char    *b_addr;                /* low order core address */
4530:        char    *b_xmem;                /* high order core address */
4531:        char    *b_blkno;               /* block # on device */
4532:        char    b_error;                /* returned after I/O */
4533:        char    *b_resid;               /* words not transferred after error */
4534:
4535: } buf[NBUF];

4551: struct devtab
4552: {
4553:        char    d_active;               /* busy flag */
4554:        char    d_errcnt;               /* error count (for recovery) */
4555:        struct  buf *b_forw;            /* first buffer for this dev */
4556:        struct  buf *b_back;            /* last buffer for this dev */
4557:        struct  buf *d_actf;            /* head of I/O queue */
4558:        struct  buf *d_actl;            /* tail of I/O queue */
4559: };

4566: struct  buf bfreelist;</pre>
先无视上面粘贴的代码...

主要的数据结构是三个链表。它们都是双向的循环链表，好处就是插入删除起来方便：

<strong>Free List：</strong>
表示当前所有可以分配的buffer，用于其分配。使用Last Recently  Used算法，若是分配一定是在链表的头部取出，若是释放一般都是放到链表的尾部（若错误发生，仍把buffer放回头部）。链表的头部在 bfreelist(4566)，buffer之间由av_forw和av_back连接。若说“Buffer Pool”，这就是了。

<strong>hash queue：</strong>
每个设备一个，里面装了与该设备相关的Buffer（已经缓存的(B_DONE)、或等待读取(B_BUSY)的）等等，用于查找缓存。在v6时只是一个简单的链表，查找就是穷举；到System V好像改成了一个哈希表。同一个buffer可以同时存在于free list和hash  queue，但是标有B_BUSY的buffer与是否在Free List是互斥的（notavail,4999）。
（前面说我们这里拿v6做讨论，可是hash queue...不知道这里叫啥好，不过system V里是叫hash queue的，功能貌似一样...先这么叫着吧 TvT）

<strong>I/O queue：</strong>
也是每个设备一个，表示该设备的请求队列。链表的头部和尾部在devtab的b_actf和b_actl项中指出。buffer若在IO Queue中就肯定是BUSY的，也就肯定不在Free List中，于是使用freelist用的av_forw和av_back作链表的指针。在I/O  queue中的buffer仍在hash queue中。

《unix操作系统设计》中的图没有考虑IO queue，大概是因为没有打算讨论设备驱动吧。于是画了一个，不知是否恰当。

<a href="http://www.fleurer-lee.com/wp-content/uploads/2010/10/thebuffercache-hashqueue.png"><img title="thebuffercache-hashqueue" src="http://www.fleurer-lee.com/wp-content/uploads/2010/10/thebuffercache-hashqueue-300x166.png" alt="" width="480" height="250" /></a><br/>

<h3>buffer的分配/释放</h3>
早期Unix很少在内核中使用动态分配的内存，而固定长度的数组＋free list是大受欢迎的解决方案。其实开开心心malloc/free的现代人不也是喜欢内存池么...

<strong>getblk(dev, blkno)</strong>
这个名字貌似容易给人一个错误印象，其实getblk函数并不会读取设备的块。它用于分配buffer（或许...叫getbuf多好？）：依据设备号和 blkno查找对应的buffer是否存在于hash queue，存在就直接返回这个buffer；不存在，就从freelist中取出一个新的buffer。

好吧上面一句是大白话，具体起来有五种情景：
<ol>
	<li>在hash queue找到了对应的buffer，这个buffer也正好free。</li>
	<li>在hash queue中没有找到这个buffer，于是到freelist中分配buffer。</li>
	<li>在hash queue中没有找到这个buffer，而从freelist中得到的buffer被标记为“延迟写入”(B_DELWRI)，得先把这个块写入磁盘，再找其它的buffer。</li>
	<li>在hash queue里没有找到这个buffer，而freelist也是空的。就把bfreelist标记为B_WANTED，让进程睡眠，等待有新的free buffer。</li>
	<li>在hash queue中找到了对应的buffer，不过这个buffer正在忙（B_BUSY）。跟情景4一样也是让进程睡眠，把这个buffer标记为B_WANTED，等它被free。</li>
</ol>
<strong>brelse(buf)</strong>
buffer release。释放一个buffer，即将其放回freelist。上面getblk的情景4和情景5都是在资源短缺的情况下让进程睡眠等待新资源的释放，而brelse正是唤醒那些睡眠进程的负责人。

注意，使用brelse释放buffer前必先将其上锁（B_LOCK）。
<h3>从设备中读/写块</h3>
bread(dev, blkno)
不知道这是block read还是buffer read的缩写，不过这才是真正用来读取块的函数。先用getblk判断是否缓存，有就返回getblk的结果。没有就发起一个IO请求并让进程睡眠，到IO完成时被唤醒。睡眠、唤醒，这就是阻塞式IO中“阻塞”的由来。

这个的代码比较短，贴过来。
<pre lang="c">4754: bread(dev, blkno)
4755: {
4756:        register struct buf *rbp;
4757:
4758:        rbp = getblk(dev, blkno);
4759:        if (rbp->b_flags&B_DONE)  //判断这个buffer是否可用
4760:                return(rbp);
4761:        rbp->b_flags =| B_READ;    //若不可用，就准备一个IO请求
4762:        rbp->b_wcount = -256;
4763:        (*bdevsw[dev.d_major].d_strategy)(rbp);  //发起一个IO请求
4764:        iowait(rbp);   //等待IO完毕
4765:        return(rbp);
4766: }</pre>
写块(bwrite)与读块差不多，略过吧。
<h3>IO请求队列</h3>
上面bread的代码中，(*bdevsw[dev.d_major].d_strategy)(rbp)这句就 是添加IO请求的操作。这与调用每个设备的驱动相关，不过都会插入到设备的IO队列。若请求队列为空，就立即把请求交给硬件，然后进入睡眠等待硬件中断的发生。一旦发生中断，中断处理程序就负责把数据读入请求队列头部的buffer，将其移出队列并标记为B_IODONE，唤醒等待这个buffer的进程。若队列中有后继，就继续给硬件交请求。
<h3>Buffer Cache机制的优点</h3>
<ul>
	<li>减少了读取磁盘的频率，自不必多说。</li>
	<li>统一了访问块设备的接口，使得文件系统可以独立于设备。</li>
</ul>
<h3>缺点</h3>
<ul>
	<li>磁盘的内容和内存中的内容不一定同步，也就是我们强制关机导致文件系统杯具的原因。</li>
	<li>若是连续写入一个大文件，而这个文件可能再也不会访问，却仍占据了大量的缓存。</li>
	<li>需要复制两次数据：一次是从设备到内核，第二次是从内核到用户空间。诚然第二次复制要比第一次复制快的多，不过文件若比较大，仍会是比较大的开销。</li>
</ul>
Note：现代的Unix系统中，缓存的管理好像融入到了Paging子系统中。Buffer Cache仍然保留，不过仅用来缓存superblock, inode之类常用的对象。
