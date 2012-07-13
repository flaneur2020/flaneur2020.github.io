---
layout: post
title: "写一个bootsector"
tags: 
- ASM
- Kernel
- OS
- "备忘"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

看的这个教程 http://share.solrex.org/WriteOS/

仿照《自己动手写操作系统》的格式，除却gas确实很囧的因素，大体上还是不错的。万事开头难，在开头的bootloader这里卡了好久的说...不过动手写一下也不过半小时的功夫 TvT

机器在启动的时候可能是遍历每个磁盘分区，若有发现第512字节位置是个0xAA55的魔数，就认为这是个引导的分区了。然后就把它的前512字节装入内存，从0x7c00位置开始执行。这就是最简单的引导方式了好像。

我们把汇编代码编译成一个512bytes的二进制文件，再把它放到一个软盘的映像里就好。

boot.S
<pre lang="asm">
[bits 16]                       ;real mode
[org 0x7c00]                  ;put code start at 0x7c00
[section .text]

_start:
    mov     ax, cs            ; init seg registers
    mov     ds, ax
    mov     es, ax
    call    _print_str      

_loop: 
    jmp     _loop                             ; forever loop

_print_str:
    mov     ax, str            
    mov     cx, len    
    mov     bp, ax
    mov     bx, 0x000c
    mov     dl, 0
    mov     ax, 0x1301
    int     0x10                              ;int 0x10, just as manual says
    ret

str: db      "screw you guys all fucked up~",10,13
len: equ     $-str

times 510-($-$$) db 0                   ; fill the rest with 0
dw 0xAA55                                  ; magic number
</pre>

编译之

<pre lang="shell">
nasm -f bin boot.S
</pre>

生成一个boot.bin，下一步搞个软盘镜像

<pre lang="shell">
dd if=boot.bin of=boot.img bs=512 count=1
dd if=/dev/zero of=boot.img skip=1 seek=1 bs=512 count=2879
</pre>

ls -l 下，大约会是这样
<pre lang="shell">
-rwxr-xr-x 1 ssword ssword     512 2010-05-26 21:28 boot.bin
-rw-r--r-- 1 ssword ssword 1474560 2010-05-26 21:28 boot.img
-rw-r--r-- 1 ssword ssword     399 2010-05-26 21:28 boot.S
</pre>

然后打开virtualbox，设置软驱映像为boot.img。启动虚拟机就可以看到一个可爱的"screw you all"什么的了 >v<

update: dd好像没什么必要，它只是个灵活的拷贝工具而已？只是填字节的话，自己写脚本的效果也是一样的。
