---
layout: post
title: "GCC内联汇编的笔记"
tags: 
- ASM
- C
- "笔记"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

起比VC风格的内联汇编，GCC的确实要别扭些，一开始要不看手册肯定一头雾水。
<pre lang="c">int foo = 10, bar = 15;
asm volatile("addl  %%ebx,%%eax"
            :"=a"(foo)                //output constraint
            :"a"(foo), "b"(bar)      //input constraint
            :                            //clobbered registers(ignored)
);</pre>
指令大家都明白，不过:"=a"(foo)这样的语法就古怪了。:后面的东西好像叫做约束，指明了输出和输入中用到的变量和寄存器。第一个的"=a"(foo)是输出的约束，就表示汇编执行完毕后foo=a。后面的"a"(foo)是输入的约束，表示汇编执行前的a=foo。这一来C和汇编就可以在约束下边交换数据了。

刚才这个a就是表示分配eax寄存器。

各种约束还挺多的...
<table>
<tr>
<th>a,b,c,d</th>
<th>对应eax,ebx,ecx,edx</th>
</tr>
<tr>
<th>S,D</th>
<th>对应esi,edi</th>
</tr>
<tr>
<th>I</th>
<th>常数</th>
</tr>
<tr>
<th>q</th>
<th>eax,ebx,ecx,edx中静态分配一个</th>
</tr>
<tr>
<th>r</th>
<th>eax,ebx,ecx,edx,esi,edi中静态分配一个</th>
</tr>
<tr>
<th>m</th>
<th>内存定位</th>
</tr>
<tr>
<th>A</th>
<th>同时分配eax和ebx，形成一64位的寄存器</th>
</tr>
<tr>
<th>i</th>
<th>一个编译时确定的立即数。好像ljmp指令的第一个参数就必须得是立即数，比如ljmp $0x80, $label。如果ljmp ax, $label就绘出现一个“Error: suffix or operands invalid for 'ljmp'的错误”</th>
</tr>
</table>
为什么要这么难看的语法呢...我猜这东西最早应该是给编译器而不是给人类设计的吧，比起VC风格的内联汇编，它可以得到更多关于变量和寄存器的信息，编译器分配起寄存器来可以心里有数，不用怕自作聪明的人类把事情都搞乱掉。
