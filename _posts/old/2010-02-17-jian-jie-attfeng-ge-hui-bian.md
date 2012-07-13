---
layout: post
title: "简介AT&T风格汇编"
tags: 
- ASM
- gas
- "翻译"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

作者：vivek
翻译：ssword
原文：<a href="http://sig9.com/articles/att-syntax">http://sig9.com/articles/att-syntax</a>

本文粗谈一下gas(1)的汇编语法，即AT&T风格汇编。初次接触很可能会觉得它别扭，不过若有其他汇编语言的基础，稍事了解即可快速上手。我假设你熟悉INTEL风格汇编——也就是INTEL手册中的那种风格。方便起见，我就用NASM（Netwide Assembler）来做比对。

gas属于GNU Binary Utilities(binutils)，也是GCC的一个后端。对编写较长的汇编程序而言它并非首选，不过对于类Unix系统的内核级hacking，它就无可替代了。选择AT&T风格使得gas饱受争议，人们总说它只是GCC的后端，而对开发者不友好。INTEL风格汇编的教众也认为，它在可读性及编译上几乎是令人窒息。尽管如此，有一点必须了解：很多操作系统都选择了gas作为底层代码的汇编器。
<!--more-->

<strong>基本形式</strong>

AT&T汇编程序的结构与其他汇编大同小异，伪指令、标签、指令—即最多带三个操作数的助记符。要说AT&T汇编的不同，最显眼的地方就是它操作数的顺序。

例如，一个简单的数据移动指令在INTEL风格下边是这个样子：

<pre lang="asm">
mnemonic	destination, source
</pre>

在AT&T风格下边则是这样：

<pre lang="asm">
mnemonic	source, destination
</pre>

一部分人（包括我）觉得这种格式更贴切。接下来说说AT&T风格指令中的操作数。


<strong>寄存器</strong>

每个IA-32架构寄存器的名字必须以'%'作前缀，如%al,%bx,%ds,%cr0，等等。

<pre lang="asm">
mov	%ax, %bx
</pre>

如上的mov指令就是把一个16位寄存器ax中的值移动到另一个16位寄存器bx中。


<strong>字面量</strong>

每个字面量必须以'$'为前缀。 例如：

<pre lang="asm">
mov	$100,	%bx
mov	$A,	%al
</pre>

第一个指令是把值100移动到寄存器bx中，第二个指令是把一个字节A移动到AL寄存器中。下面这个指令就是错误的：

<pre lang="asm">
mov	%bx,	$100
</pre>

怎么说呢，这条指令是要把寄存器bx的值移动给一个字面量，显然不靠谱。


<strong>内存寻址</strong>

在AT&T风格中，内存引用起来是这个格式：

<pre lang="asm">
segment-override:signed-offset(base,index,scale)
</pre>

按你寻址的需求，其中的部分可以省略

<pre lang="asm">
%es:100(%eax,%ebx,2)
</pre>

注意下，基地址及偏移中的数不带前缀'$'。拿几个例子和对应的NASM风格做个比较应该好些：

<pre lang="asm">
GAS memory operand			NASM memory operand
------------------			-------------------

100					[100]
%es:100					[es:100]
(%eax)					[eax]
(%eax,%ebx)				[eax+ebx]
(%ecx,%ebx,2)				[ecx+ebx*2]
(,%ebx,2)				[ebx*2]
-10(%eax)				[eax-10]
%ds:-10(%ebp)				[ds:ebp-10]
</pre>

实例：

<pre lang="asm">
mov	%ax,	100
mov	%eax,	-100(%eax)
</pre>

第一个指令是把寄存器AX中的值移动到数据段寄存器（默认）偏移100的内存位置，第二个指令是把寄存器eax中的值移动到[eax-100]。


<strong>操作数大小</strong>

有时指明操作数的大小是必须的，尤其是移动字面量到内存。例如这个指令：

<pre lang="asm">
mov	$10,	100
</pre>

这里只说了把值10移动到内存偏址100处，而没有说值的大小。在NASM中，这通过给操作数后面跟个byte/word/dword之类的关键词来指明。而在AT&T风格里，是通过指令中b/w/l之类的后缀指明。如：

<pre lang="asm">
movb	$10,	%es:(%eax)
</pre>

把值为10的一个字节移动到内存地址[ex:eax]，另如：

<pre lang="asm">
movl	$10,	%es:(%eax)
</pre>

把值为10的一个长整数移动到同一位置。

再几个例子：

<pre lang="asm">
movl	$100, %ebx
pushl	%eax
popw	%ax
</pre>


<strong>控制流程</strong>

jmp,call,ret等指令可以转移程序的执行位置。在同一代码段中跳转，是近距跳转(near)。若是跳转到不同的代码段，就是远程跳转(far)。可用的跳转地址可以来自相对偏移（label）、寄存器、内存以及段偏移指针。相对偏移通过label指明，如下：

<pre lang="asm">
label1:
	.
	.
  jmp	label1
</pre>

使用寄存器或者内存的值做地址的操作数必须加个前缀'*'。若是远程跳转，必须加个'l'作前缀，如‘ljmp’，‘lcall’等等。例如：

<pre lang="asm">
GAS syntax			NASM syntax
==========			===========

jmp	*100			jmp  near [100]
call	*100			call near [100]
jmp	*%eax			jmp  near eax
jmp	*%ecx			call near ecx
jmp	*(%eax)			jmp  near [eax]
call	*(%ebx)			call near [ebx]
ljmp	*100			jmp  far  [100]
lcall	*100			call far  [100]
ljmp	*(%eax)			jmp  far  [eax]
lcall	*(%ebx)			call far  [ebx]
ret				retn
lret				retf
lret $0x100			retf 0x100
</pre>


段偏移指针按下面的格式指明：

<pre lang="asm">
jmp	$segment, $offset
</pre>

例如：

<pre lang="asm">
jmp	$0x10, $0x100000
</pre>

记住这些很快就能上手了。要了解gas的更多细节，不妨参阅这个<a href="http://sourceware.org/binutils/docs-2.16/as/index.html">文档</a>。
