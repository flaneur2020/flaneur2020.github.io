---
layout: post
title: "折腾分区表小记"
tags: 
- "备忘"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

就在五一放假回家前的两个小时把分区表给弄没了，囧。

侥幸心理，装Ubuntu11.04时候用了老版本的initrd.gz与vmlinuz，引导alternate的iso没问题，但是安装到一半的时候报错，重启就发现grub已经没了。"恢复下分区表就行了吧..."于是万恶的侥幸心理再次得势，拿一张xp安装盘的分区表修复工具扫描并修复了下，进windows下载来<a href="http://tw.archive.ubuntu.com/ubuntu/dists/natty/main/installer-i386/current/images/hd-media/">正确的initrd.gz与vmlinuz</a>，回去继续装11.04。到分区的时候发现几个linux分区都让不认ext4的分区表修复工具给搞没了，整个成了一大块不可用的剩余空间。想起来/home里面几个星期的翻译稿和代码竟都忘了commit，于是冷汗...

万幸还没有格式化。经weizhong大哥指点，宽了些心，该是可以找回来的。

尝试了一些工具，最后还是使用了<a href="http://www.cgsecurity.org/wiki/TestDisk">testdisk</a>，它可以扫描整个硬盘，分析出来分区信息。也能在windows下使用。

一开始没仔细看<a href="http://www.cgsecurity.org/wiki/TestDisk_Step_By_Step">教程</a>，反复扫描了好几遍，期间又把分区表清了n回， ＞﹏＜。现在想想还能找回来，真是够侥幸的。留意的地方就是，扫描完毕的时候它会列个表出来，需要自己按左右键设置分区的信息(主分区/用来引导的主分区/还是逻辑分区)，都<a href="http://www.cgsecurity.org/mw/images/Set_partition_to_recover.gif">绿了</a>才好。如果扫描到的分区不全，就选择Deeper Search，它会扫描出分区表的几种可能情况，自己需要选择出正确的分区，确保无误之后再Write。

折腾了一天，最后终于把/home找了回来。代价是把windows的D盘搞丢了，不过估计该比较容易找回。

下次就不一定能这么幸运了，以后装系统什么的千万得对数据上心才是。越来越怕折腾了 &gt;&lt;

<hr/><!--more-->

<strong>后记：</strong>
刚刚看了下<a href="http://www.fleurer-lee.com/2010/10/25/%E7%AE%80%E4%BB%8B%E7%A1%AC%E7%9B%98%E5%88%86%E5%8C%BA/">以前的译文</a>:
<blockquote>
每个分区通常以一个柱面为边界：C,H,S=??,0,1。MBR和扩展分区中的首个分区则通常都是以一个磁头为界：C,H,S=??,1,1。

扩展分区的首个扇区的布局与MBR类似，只不过只有两个分区表项：第一项即其中的第一个分区，可选的第二项指向第二个分区，构成一个链表。
</blockquote>

估计这就是testdisk的原理了吧。扫描整个硬盘，在磁头的边界处看看是不是有扩展分区，然后顺着扩展分区的链表，遍历其下的逻辑分区，看看superblock是否正确。当然实际的情况要复杂的多。
