---
layout: post
title: "简介多任务的实现"
tags: 
- ASM
- C
- Kernel
- "翻译"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

原文：<a href="http://hosted.cjmovie.net/TutMultitask.htm">http://hosted.cjmovie.net/TutMultitask.htm</a>
翻译：fleurer

<!--more-->

恩，打算给你的操作系统实现多任务？可以说，多任务是现代操作系统里面最重要的元素之一──没多任务你没脸见人。只跑Bash的linux也有多任务一说──比如debain下边你可以使用Alt+F1,F2,F3,F4切换虚拟终端。不过话说回来，只有一颗cpu的电脑为什么能同时执行10件任务？

答案就是——不为什么，只不过看着像而已。它们切换的速度非常之快，以至于人可以觉得它们是同时执行的。当然也有双核和多核，不过那就不是本文讨论的范畴了。即使有双核多核，像一台普通的的Windows NT(包括XP)内核机器，没有用户登录、没有程序执行，也有约100条线程同时执行。

我们先列些名词好了：
<ul>
	<li><strong>线程</strong> - 进程的一个分子，可以同时执行。比如你玩的一个游戏是个进程，而这个游戏里面的背景音乐、键盘事件、3D绘图则是独立的线程。</li>
	<li><strong>进程 </strong>- 在电脑上运行的一个完整程序，有自己的地址空间（通常使用分页实现）。</li>
	<li><strong>调度算法</strong> - 选出下一个要执行任务的方法。可以简单如Round Robina，也可以考虑上优先级，使一进程可以优先得到充足的时间片。调度算法与任务切换的实现无关。</li>
	<li><strong>基于栈的任务切换</strong> - 切换任务的方法之一。按照这个方法，在发生切换时我们把一些信息都"PUSH"到进程的栈里，于是只需要切换一个栈把用到的东西（eax, ebx, ds, es）都POP出来即可。这个方法比基于硬件的切换（这里不作讨论）更快，已经几乎是切换的首选。</li>
	<li><strong>Round Robin</strong> - 调度算法的一种，可以选出下一个执行的任务。它的实现很简单：把所有的进程（或线程）列出来放到一个表里，反复轮询之，公平分配时间片。</li>
</ul>
接下来动手吧。添加多任务，你的OS准备好了吗？我这里选择了最简单的方法（Round Robin，上面有介绍），需要：一个内存管理器（memory manager, 只要物理内存就够了）；正确设置的IDT；PIT（可编程中断定时器，译者注） IRQ的hook；在保护模式之下。

我们首先得有一个结构体来表示每个进程的信息。简单起见，先这样：
<pre lang="c">typedef struct{        //Simple structure for a thread
 unsigned int esp0;   //Stack for kernel
 unsigned int esp3;   //Stack for process
} Thread;

Thread Threads[2];     //Space for our simple threads. Just 2!
int CurrentTask = -1;  //The thread currenlty running (-1 == none)</pre>
还得有个地方来存放线程信息。不过最好事先搞清楚——这个教程已是尽可能的简化了，我们没有让进程在Ring 3下执行，因为那样一来就得考虑TSS——我不想掉这个大坑。Beyond_Infinity同学在一篇类似的教程里用考虑了这个，如果感兴趣不妨一读。

在考虑创建新任务的方法之前，我先说下如何切换任务。其实很简单。

在收到来自PIC的IRQ时，你的IRQ Handler很可能会通过一个'pusha'或者'pushad'来储存一些寄存器。很好，这就清楚了，你可能使用'popa'或'popad'以相反的顺序重新得到这些寄存器。大约可以像这样：
<pre lang="asm">_irq0:
cli    ;Disable interrupts
 push 0 ;Push IRQ number
 push 0 ;Push dummy error code
 jmp IRQ_CommonStub

.. ;Other IRQS are here, similiar to above

IRQ_CommonStub:
 pusha          ;Push all standard registers
 push ds        ;Push segment d
 push es        ;Push segmetn e
 push fs        ; ''
 push gs        ; ''

 mov eax, 0x10  ;Get kernel data segment
 mov ds, eax    ;Put it in the data segment registers
 mov es, eax
 mov fs, eax
 mov gs, eax

 push esp       ;Push pointer to all the stuff we just pushed
 call _IRQ_Handler ;Call C code

 pop gs         ;Put the data segments back
 pop fs
 pop es
 pop ds

 popa           ;Put the standard registers back

 add esp, 8     ;Take the error code and IRQ number off the stack

 iret           ;Interrupt-Return</pre>
考你个问题：这些pop可以将当前栈中的数据装回CPU，如果在这个C函数调用时将栈换掉又会怎样？哈哈~到点子上了。如果这样做，整个CPU的状态就切换了。把上面的代码稍微改下：
<pre lang="asm">_irq0:
 ;Notice there is no IRQ number or error code - we don't need them

 pusha          ;Push all standard registers
 push ds        ;Push segment d
 push es        ;Push segmetn e
 push fs        ; ''
 push gs        ; ''

 mov eax, 0x10  ;Get kernel data segment
 mov ds, eax    ;Put it in the data segment registers
 mov es, eax
 mov fs, eax
 mov gs, eax

 push esp       ;Push pointer to all the stuff we just pushed
 call _TaskSwitch ;Call C code

 mov esp, eax   ;Replace the stack with what the C code gave us

 mov al, 0x20   ;Port number AND command number to Acknowledge IRQ
 out al, al     ;Acknowledge IRQ, so we keep getting interrupts

 pop gs         ;Put the data segments back
 pop fs
 pop es
 pop ds

 popa           ;Put the standard registers back

 ;We didn't push an error code or IRQ number, so we don't have to edit esp now

 iret           ;Interrupt-Return</pre>

提醒下，你的C代码若返回一个unsigned int，gcc会把它放到eax寄存器里 - 简单漂亮。好，接下来做什么？一半了，是的！

接下来考虑创建新任务。这就意味着我们需要申请内存，并令它的堆栈看起来像是已经push了所有寄存器的状态（这一来在切换栈之后，才能有东西pop）。恩，x86体系结构下栈是向下增长的，你的push也就相当于设置esp指向的dword（双字），随后esp减去4。我们需要在C里模拟出来——万幸，这很简单——搞个unsigned int的指针指向栈顶就行了，然后每次push后都使用 -- 运算符下移。好，就这么创建一个任务：

<pre lang="c">
//This will create a task
//It will make a stack that looks like it has all
//of the stuff of an IRQ handler 'pushed' on it
void CreateTask(int id, void (*thread)()){
  unsigned int *stack;
  Threads[id].esp0 = AllocPage() + 4096; //This allocates 4kb of memory, then puts the pointer at the end of it

  stack = (unsigned int*)Threads[id].esp0; //This makes a pointer to the stack for us

  //First, this stuff is pushed by the processor
  *--stack = 0x0202; //This is EFLAGS
  *--stack = 0x08;   //This is CS, our code segment
  *--stack = (UINT)thread; //This is EIP

  //Next, the stuff pushed by 'pusha'
  *--stack = 0; //EDI
  *--stack = 0; //ESI
  *--stack = 0; //EBP
  *--stack = 0; //Just an offset, no value
  *--stack = 0; //EBX
  *--stack = 0; //EDX
  *--stack = 0; //ECX
  *--stack = 0; //EAX

  //Now these are the data segments pushed by the IRQ handler
  *--stack = 0x10; //DS
  *--stack = 0x10; //ES
  *--stack = 0x10; //FS
  *--stack = 0x10; //GS
  Threads[id].esp0 = (UINT)stack; //Update the stack pointer
}
</pre>

好，已经设好了任务，接着切换它们。不过怎样？恩，还记得Round Robin吧，你已经知道了！我们只有两个任务，所以在PIT IRQ被触发时只需要知道在执行的是哪个人物，把栈切换成另一个的。再保存当前ESP到旧任务的结构体里。如下：

<pre lang="c">
//Switch between our two tasks
//Notice how we get the old esp from the ASM code
//It's not a pointer, but we actually get the ESP value
//That way we can save it in our task structure
unsigned int TaskSwitch(unsigned int OldEsp){
if(CurrentTask != -1){ //Were we even running a task?
 Threads[CurrenTask].esp0 = OldEsp; //Save the new esp for the thread

 //Now switch what task we're on
 if(CurrentTask == 0)CurrentTask = 1;
 else CurrentTask = 0;
} else{
 CurrentTask = 0; //We just started multi-tasking, start with task 0
}
 return Threads[CurrentTask].esp0; //Return new stack pointer to ASM
}</pre>
随后我们需要由PIT来触发切换任务的asm stub，我假定你已经按照常规将IRQ映射到32-47。如果没，就设置一个IRQ0对应的handler。

<pre lang="c">extern void irq0(); //Our ASM stub
//This is a very simple function
//All it does is put us in the IDT
void StartMultitasking(){
IdtSetGate(0, (UINT)irq0); //Install us in IDT. We multitask NOW!
}</pre>

万事俱备，只剩调用这些函数！方便看执行效果起见，在kernel的主文件（可能是main.c）里面添上这两个函数。

<pre lang="c">//These are just two simple functions that act as
//'threads' to test our multi-tasker on
//I won't try to explain how they work
//Only that they change colors on two letters of the screen

//Also - they must NEVER return - just make an infinite loop

void ThreadTest1(){
unsigned char* VidMemChar = (unsigned char*)0xB8001;
for(;;)*VidMemChar++;
}

void ThreadTest2(){
unsigned char* VidMemChar = (unsigned char*)0xB8003;
for(;;)*VidMemChar++;
}</pre>

以上代码的执行效果是显而易见的。不错，加上执行多任务的代码！把几句放到你的主函数（其它函数也行，能执行就好）里面。

<pre lang="c">CreateTask(0, ThreadTest1); //Install the first task
CreateTask(1, ThreadTest2); //Install the second task
StartMultitasking(); //Start your multitasking OS!
//We're now multitasking. Celebrate!</pre>

差不多，已经可以把你的OS带出单任务的DOS时代了...进入多任务多线程的新时代吧！

你可以接着搞搞...这只是入门而已。简单的Round-Robin算法在很多情况下的效果并不好，不妨看下其他的调度算法，比如Mega Tokyo's OS FAQ中列出来的那些。

还有件事情你可能感兴趣，那就是如何将分页整合进来。大多数操作系统都给每个应用程序一个0开始的虚拟地址空间，并一个用户栈。这一来，你就需要给每个任务添加一个页表目录，随后添加一些push和pop以适应CR3中值的改变。

有问题？有建议？Email me at service@cjmovie.net!
