---
layout: post
title: "lua指令执行相关的几个数据结构笔记"
tags: 
- lua
- VM
- "笔记"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

对lua的upvalue机制一直很好奇，但两三年下来对它的了解只限于paper中的那张插图，于细节则毫不知情。直到最近几天晚上下班无聊看lua源码，也算是回头来补补这个小坑了。

某某机制肯定是针对某问题而发，对upvalue机制而言，问题不外乎就是闭包的实现。对lua来讲，又可以分成三个子问题：

1. 怎样处理外部变量的生存周期（funargs问题）
2. 将闭包的实现得简洁（Common Lisp的闭包实现就比整个lua还长）
3. 尽量高效（深层嵌套的外部变量，该怎样做到高效地查找）

<i>本来打算在一篇里写完，发现基础的内容比较容易喧宾夺主，分两篇好了。在这里先记录一下lua VM执行相关的几个数据结构，不一定全，足够接下来理解upvalue机制即可。</i>
<!--more-->

<h3>预备</h3>

lua中值的表示先掠过。没有什么黑魔法，过一遍refman就足够了。

先看闭包的表示：

<pre lang="c">
#define ClosureHeader \
	CommonHeader; lu_byte isC; lu_byte nupvalues; GCObject *gclist; \
	struct Table *env

typedef struct CClosure {
  ClosureHeader;
  lua_CFunction f;
  TValue upvalue[1];
} CClosure;


typedef struct LClosure {
  ClosureHeader;
  struct Proto *p;
  UpVal *upvals[1];
} LClosure;
</pre>

其中CClosure是为C接口提供(在此掠过)，LClosure就是一般的lua闭包的内部表示了。人肉把ClosureHeader展开，这样好看些：

<pre lang="c">
typedef struct LClosure {
  CommonHeader;
  lu_byte isC; //表示它是C函数还是lua函数
  lu_byte nupvalues; //表示upvalue(也就是外部变量)的数目。至于普通的lua函数，就是nupvalues为0的闭包；
  struct Table *env; //指向全局变量的表，以数字(而不是字符串形式的名字)为索引。
  GCObject *gclist; //为lua GC中的链表，链起lua中所有可回收的对象；
  struct Proto *p; //函数原型，这个留后面说
  UpVal *upvals[1]; //这里有点黑魔法，这里貌似是长度为1的数组，实际上在为LClosure申请内存时，lua会根据upvalue的数目来调整LClosure的长度，见luaF_newLclosure(lfunc.c, 33行)。而*upvals[]这个数组的真正长度，就与nupvalues是相等的。
} LClosure;
</pre>

如果精力有限，可以只留意两个地方：其一是*p，函数原型，就是闭包中各种“静态的”“万年不变”的东西，比如opcode，变量的个数、名字等等信息；其二，就是*env和*upvals，分别表示着闭包中的全局变量和外部变量，那局部变量呢？在栈上。

对FP感兴趣的人们喜欢说“OO is Poor Man's Closure”，实际上闭包也确实有着对象的特征的。想想，一个类放在那里，我们不能直接拿来用，想用它就必须得创建这个类的实例。同样，一个函数原型放在那里，我们也不能直接调用它，要调用它，也得先有一个它的“实例”。“实例”比“原型”多了什么？全局变量、局部变量和外部变量。程序中的指令都是“死”的，“活”的是数据。数据可能在堆里，可能在栈上，这些数据的集合，就是程序当前的状态。

<h3>lua_State</h3>

看lua源码的话一定会注意到，几乎所有函数的开头都是一个lua_State *L。在最早的lua实现中，lua_State是没有的，栈啦全局变量表啦都是全局的变量，写起来可以少打些字，问题是不可重入，不能支持多线程。引入lua_State之后，曾经的全局变量都放到了这里面，每个线程(也就是coroutine)便对应着一个lua_State*。

<pre lang="c">
/*
** `per thread' state
*/
struct lua_State {
  CommonHeader;
  lu_byte status;
  StkId top;  /* first free slot in the stack */
  StkId base;  /* base of current function */
  global_State *l_G;
  CallInfo *ci;  /* call info for current function */
  const Instruction *savedpc;  /* `savedpc' of current function */
  StkId stack_last;  /* last free slot in the stack */
  StkId stack;  /* stack base */
  CallInfo *end_ci;  /* points after end of ci array*/
  CallInfo *base_ci;  /* array of CallInfo's */
  int stacksize;
  int size_ci;  /* size of array `base_ci' */
  unsigned short nCcalls;  /* number of nested C calls */
  unsigned short baseCcalls;  /* nested C calls when resuming coroutine */
  lu_byte hookmask;
  lu_byte allowhook;
  int basehookcount;
  int hookcount;
  lua_Hook hook;
  TValue l_gt;  /* table of globals */
  TValue env;  /* temporary place for environments */
  GCObject *openupval;  /* list of open upvalues in this stack */
  GCObject *gclist;
  struct lua_longjmp *errorJmp;  /* current error recover point */
  ptrdiff_t errfunc;  /* current error handling function (stack index) */
};
</pre>

StkId top; StkId base; StkId stack_last; StkId stack;都是指向求值栈的不同位置。lua5不是寄存器机了，那求值栈里有什么？我还没有调试，不过通过ChunkSpy来看，栈底是函数中的常量，其余都用来存放局部变量和临时变量。那么，函数的返回地址之类放在哪里？

在CallInfo *base_ci; 这边，lua将函数的返回地址、调用者的栈指针以及函数返回值的个数都存在CallInfo这个结构里。可以把CallInfo *base_ci也看作是一个栈，深度是nexeccalls(luaV_execute()中定义的一个局部变量)。不妨留意lvm.c中OP_CALL和OP_RETURN两个指令中luaD_precall()和luaD_poscall()两个函数对它的处理。

lua5是寄存器机，但对C的接口依然是老样的堆栈机，这点挺有意思。一开始不理解lua为什么改到寄存器机，lua的设计原则不是简单么？寄存器机又是公认比堆栈机更难实现。其实仔细想想，多出来的难点无非是寄存器分配算法，给编译器实现寄存器分配算法自然很难，但是lua这里有一点天生省力的地方，那就是它的“寄存器”可以是认为是无限的(255个?)，而像图着色这样复杂到坑爹的算法，都是针对及其有限寄存器的分配而言，此“寄存器分配”非彼“寄存器分配”。至于C接口依然是堆栈机的样子，则可以这样想：压栈就是分配了一个寄存器，弹出就是释放了。

<h3>global_State</h3>

struct lua_State上边还有一个struct global_State，其中五分之四的字段都是针对GC而设，GC部分还没仔细看，在这里先掠过。不过你可以把lua的GC看作是一个可以暂停的状态机。

<h3>指令执行</h3>

lua虚拟机中的指令很少，执行都在lvm.c中的luaV_execute(lua_State *L, int nexeccalls)中。可以认为调用它之前，lua虚拟机已经初始化好了求值栈和一个初始的函数。

先记这些，感兴趣的还是upvalue机制，下篇再记。
