---
layout: post
title: "简介硬盘分区"
tags: 
- ASM
- "翻译"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
  _wp_old_slug: ""
---

作者：Denis A Nicole
翻译：fleurer
原文：http://www.hpcc.ecs.soton.ac.uk/~dan/filesystems/partition.html

(99年的文章，好像有点老了)

<!--more-->

== Introduction ==

本教程无耻的参考了这个文档http://www.diskwarez.com/articles/par.htm，新引入的任何错误由我本人负责（译者注：翻译引入的错误由我负责 ="=）。本文假定你会使用任意一个raw sector editor，比如diskedit.com。

传统的硬盘依据其物理结构直接按cylinder,head,sector（CHS）进行寻址。到现代的IDE/ATA设备中每个磁道（track）不再是固定数目的扇区(sector)，其寻址机制随之变更：直接按照一个从0开始的扇区号即可，更简单了。为向前兼容，设备增加了一个将CHS转为绝对扇区号的翻译单元，依然允许虚拟CHS方式进行寻址。其翻译过程可以通过HD-MBOOT以及IDEATA等工具进行修改或检查。

注意，Cylinder，Head，Sector的下标分别以0,0,1为起始。扇区号是个例外。

过去最后一个物理柱面通常是保留用于IBM的磁盘诊断，不过可能已经过时。

有个杯具，BIOS接口中Cylinder,Head,Sector的上限分别是1024,256和63，然而IDE/ATA接口的上限是65536,16和256。这一来要不在中间加个转换，就只能出现木桶原理各取最小的一环1024,16,63，每扇区512字节，每张硬盘最大也就504Mb了。为周旋这一限制，现代的BIOS会对CHS进行转换，让设备看起来像是磁头数加倍，且柱面数减半（1024为止）。

(译者注：上面说的磁头数*2同时柱面数/2的变换机制，即LARGE模式。在BIOS中好像可以设置)

还有杯具，那就是DOS不能处理超过255的磁头数，对于超过4Gb的设备，BIOS最多只能把磁头数转换到255。这就把磁盘的大小限制为8Gb。

对于大于8Gb的磁盘，其C,H,S返回值为16383,16,63，表示这个CHS值不正确。

超过8Gb就只能通过首地址为0的逻辑扇区号（LBA）进行寻址。Windows NT4 SP4和Linux等系统都能直接按这种方式寻址IDE/ATA硬盘，不过Windows 95和Windows 98这样依赖BIOS中Int 13h扩展的系统就不那么靠谱了。

==主引导记录(MBR)==

主引导记录即硬盘的第一个扇区（C,H,S=0,0,1）。它为引导代码提供了446kb的空间，后跟4个分区表项（每个16字节），最后以一个0xAA55（小端序）做魔数。这四个表项相对于扇区的偏移分别是0x1be, 0x1ce, 0x1de, 0x1ee。

<pre>
             +--------------------------------+
             |                                |
             |                                |
            /\/             code             /\/
             |                                |
             |                          +-----|
             |                          |     |
             +--------------------------------+
             |table 1    |    table 2   |     |
             +--------------------------------+
             |table 3    |    table 4   |55|aa|
             +--------------------------------+

                       figure 1 - MBR
</pre>

每个分区表项包含了分区的起始及结束地址（CHS格式）、开始的扇区号（LBA格式）、扇区数。以及分区类型和一个可引导标志。

CHS地址是BIOS INT 13h的格式，柱面号在扇区号的那个字节里占着两个高位。其中的bit与int 13h时几个寄存器内的值相同。

<pre>

  BYTES    0      1 - 3      4      5 - 7     8 - 11     12 - 15
       +----------------------------------------------------------+
       |      |           |      |         |           |          |
       | BOOT | START-CHS | TYPE | END-CHS | ABS-START | NUM-SECS |
       |      |           |      |         |           |          |
       +----------------------------------------------------------+

                       Figure 2 - Partition Table

The CHS format is:
       +-----------------+-----------------+-----------------+
       | 7 6 5 4 3 2 1 0 | 7 6 5 4 3 2 1 0 | 7 6 5 4 3 2 1 0 |
       +-----------------+-----------------+-----------------+
       | H H H H H H H H | C C S S S S S S | C C C C C C C C |
       +-----------------+-----------------+-----------------+
       | DH              | CL              |CH               |
       +-----------------+-----------------+-----------------+

                        Figure 3 - CHS Format
</pre>

第一行是每字节的Bit位，第二行表示CHS值（C=cylinder,H=head,S=sector），最后一行是BIOS调用相关的几个寄存器。cylinder在CL中占有高二位，可得如下公式：

C,H,S = ((CL&0xC0)<<2+CH),DH,(CL&0x3F).

ABS-START和NUM-SECS都是四字节的小端数据。

不用的分区表项最好全部清零，将TYPE项设为0也可以。

引导分区的BOOT项为0x80且唯一，其它一律为0。

起始或结束地址若是超过CHS的限制，就设为1023,255,63表示它是错误的。

MBR中只允许一个FAT文件系统。若需要更多，抑或需要四个以上的分区，只能通过一个扩展分区（DOS EXTended partition）实现。

每个分区通常以一个柱面为边界：C,H,S=??,0,1。MBR和扩展分区中的首个分区则通常都是以一个磁头为界：C,H,S=??,1,1。

扩展分区的首个扇区的布局与MBR类似，只不过只有两个分区表项：第一项即其中的第一个分区，可选的第二项指向第二个分区，构成一个链表。

(译者注：某种意义上说，扩展分区并不是真正的分区，其存在只能通过逻辑分区体现。逻辑分区的第一个扇区叫做EBR, 即Extended Boot Record。与MBR中固定的四个分区表项不同，EBR中只有两个分区表项，EBR之间构成一个链表，所以理论上可以分无限的逻辑分区。)

<pre>
Partition Types

   Partition  Fdisk                                          Starting in
   Type       Reports      Size                  FAT Type    version
   ---------------------------------------------------------------------
   01         PRI DOS      0-15 MB               12-Bit      MS-DOS 2.0
   04         PRI DOS      16-32 MB              16-Bit      MS-DOS 3.0
   05         EXT DOS      0-2 GB                n/a         MS-DOS 3.3
   06         PRI DOS      32 MB-2 GB            16-bit      MS-DOS 4.0
   07         ----  Windows NT NTFS [if Boot partition then < 4Gb] ----
   0E         PRI DOS      32 MB-2 GB            16-bit      Windows 95  *
   0F         EXT DOS      0-2 GB                n/a         Windows 95  *
   0B         PRI DOS      512 MB - 2 terabytes  32-bit      OSR2        *
   0C         EXT DOS      512 MB - 2 terabytes  32-bit      OSR2        *
   82         ---------------- Linux Swap ------------------------
   83         ---------------- Linux EXT2 (native) ---------------           
                               [in part from Microsoft KB article Q69912]

		* Types 0C..0F Use the ABS-START and NUM-SECS fields and thus
                  can extend beyond the 8Gb boundary.
</pre>


==引导扇区==

引导的代码位于MBR，LILO什么的就是了。微软提供了一个fdisk /mbr，可以将其标准的引导代码插入mbr。它只用了分区表中START-CHS一项，所以引导分区必须位于硬盘的前8Gb。

同MBR一样，每个活动分区的首个扇区中也有引导代码；在微软的系统中可以使用format命令插入这段代码。引导扇区的末尾同样也是魔数AA55h(小端序，先55h后AAh)。它们都是被装到0000:7c00，不过一般它们都会把自己挪到别处。
