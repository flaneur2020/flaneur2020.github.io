---
layout: post
title: "折腾grub小记"
tags: 
- grub
- "备忘"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

蛋疼用PQMagic改了下/swap分区的大小，重启发现mbr没了。

拿手机google下之后，找了张windows me的光盘（初一时候买的盘还能用...orz）引导进入dos，fdisk /mbr，windows原地满血复活...满血，是的，windows又把mbr占了...

接着装个wingrub，折腾半天找回了原先的menu.lst，重启，接着是：

<pre lang="sh">
Error 2: Bad file or directory type
</pre>

而且在grub命令下无法列出ext3分区的文件，当时冷汗天杀的PQMagic没把我盘格了吧...回头想下，感情这个wingrub好像不认ext3...

下载个live-CD，进去发现可以读那个分区。apt-get一个grub，进入命令行之后：

<pre lang="sh">
root (hd0, 4)
setup (hd0)
</pre>

提示成功。重启，原先的grub回来了...
