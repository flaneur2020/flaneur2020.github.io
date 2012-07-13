---
layout: post
title: "简介协程"
tags: 
- coroutine
- lua
- "翻译"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

原文：<a href="http://lua-users.org/wiki/CoroutinesTutorial">Coroutines Tutorial</a>
翻译：ssword

<h3>
What are coroutines?
何为协程？
</h3><br />

协程允许我们同时执行多个任务。将它们分派到不同的子程序(routine)中，在每个子程序表示完成时将控制权转移，并可以回到上次完成的位置继续执行，如之重复，从而实现并发。

在参考手册的<a href="http://www.lua.org/manual/5.1/manual.html#2.11">2.11节</a>和<a href="http://www.lua.org/manual/5.1/manual.html#5.2">5.2节中</a>都有对协程的讲解。

<h3>
Multi-threading
多线程
</h3><br />

让每个任务独立地运行在一个线程中，同时执行多个任务就叫做多线程(multi-threading)。使用多线程的应用就叫做多线程应用(multi-threaded)。

多线程的实现方式多种多样，一些系统是为每个线程申请固定的时间，在时间结束时转移控制权到下一个线程。这叫做抢占式(pre-emptive)多线程。这种调度方式中，每个线程不必关心自己占据的时间，而更关注于自身的功能。

还有别的系统，线程知道自己占据的时间，也知道应自己在何时转让控制权给别的线程，来执行各自的功能。这叫做联合式(cooperative)或协作式(collaborative)多线程。应用程序中的所有线程都是协作在一起，这也正是lua协程使用的多任务方式。

Lua的协程既非操作系统线程，也非进程。它是在lua中创建的一块代码，与线程一样有自己的控制流程，不过在同一时刻只能运行一个协程。而且只有新协程被激活或有yield(返回到执行它的那个协程)，才会转移控制权。协程就是表示协作式线程的一种简单方式，不过没有并行(execute in parallel)，也就无法得到多核心CPU的性能优势。但是，由于协程在切换起来要比操作系统线程快得多，也不需要复杂甚至代价昂贵的锁机制，使用协程通常都要比等价的操作系统线程轻快一些。

<h3>
Yielding
</h3><br />

要让多个协程共同执行，就必须停止当前协程的执行(在执行一些操作之后)，并转移控制权到另一个协程，这种操作就叫做yielding。协程可以直接调用个一个lua函数，coroutine.yield()，它与函数的return类似。使用yield退出函数的位置可以被记住，在稍后可以回到该位置接着刚才的上下文继续执行。不过若使用return退出，函数的整个上下文就被销毁了，我们也就无法回到该位置。

<pre lang="lua">
> function foo(x)
>>  if x>3 then return true end  -- we can exit the function before the end if need be
>>  return false                 -- return a value at the end of the function (optional)
>> end
> = foo(1)
false
> = foo(100)                     -- different exit point
true
</pre>

<h3>
Simple usage
简单的用法
</h3><br />

要创建一个协程，得先有个表示它的函数。

<pre lang="lua">
> function foo()
>>   print("foo", 1)
>>   coroutine.yield()
>>   print("foo", 2)
>> end
>
</pre>

使用coroutine.create(fn)函数可以创建一个协程，它的参数是个lua函数。它返回的类型为thread:
<pre lang="lua">
> co = coroutine.create(foo) -- create a coroutine with foo as the entry
> = type(co)                 -- display the type of object "co"
thread
</pre>

我们可以用coroutine.status()函数来检查线程的状态。

<pre lang="lua">
> = coroutine.status(co)
suspended
</pre>

状态suspended表示这个线程是可用的，而且如你所想，它还什么也没做。注意，在我们创建线程时，它不会立即执行。要执行它，我们使用corotine.resume()函数。Lua会进入这个线程，并在出现yield时离开。

<pre lang="lua">
> = coroutine.resume(co)
foo     1
true
</pre>

corotine.resume函数返回了resume调用的错误状态。这输出表示了我们进入的是foo函数，退出时没发生错误。有趣的地方就在这里。单靠一个函数，我们不可能回到离开时的上下文继续执行，而协程则允许我们一次次地resume：

<pre lang="lua">
> = coroutine.resume(co)
foo     2
true
</pre>

可以看出，这行代码回到了foo中上次yield的位置执行并返回，没有错误发生。不过如果看下它的状态，就可以看出我们退出了foo函数，协程也结束了。

<pre lang="lua">
> = coroutine.status(co)
dead
</pre>

如果试图再次resume，就会返回两个值，一个错误标记和一条错误信息：

<pre lang="lua">
> = coroutine.resume(co)
false   cannot resume dead coroutine
</pre>
一旦协程退出，或是像函数那样返回，它就无法执行resume了。

<h3>
More details
</h3><br />

下面是个复杂些的例子，展示协程的几个性质：
<pre lang="lua">
> function odd(x)
>>   print('A: odd', x)
>>   coroutine.yield(x)
>>   print('B: odd', x)
>> end
>
> function even(x)
>>   print('C: even', x)
>>   if x==2 then return x end
>>   print('D: even ', x)
>> end
>
> co = coroutine.create(
>>   function (x)
>>     for i=1,x do
>>       if i==3 then coroutine.yield(-1) end
>>       if i % 2 == 0 then even(i) else odd(i) end
>>     end
>>   end)
>
> count = 1
> while coroutine.status(co) ~= 'dead' do
>>   print('----', count) ; count = count+1
>>   errorfree, value = coroutine.resume(co, 5)
>>   print('E: errorfree, value, status', errorfree, value, coroutine.status(co))
>> end
----    1
A: odd  1
E: errorfree, value, status     true    1       suspended
----    2
B: odd  1
C: even 2
E: errorfree, value, status     true    -1      suspended
----    3
A: odd  3
E: errorfree, value, status     true    3       suspended
----    4
B: odd  3
C: even 4
D: even         4
A: odd  5
E: errorfree, value, status     true    5       suspended
----    5
B: odd  5
E: errorfree, value, status     true    nil     dead
>
</pre>

我们有个for循环，它调用到两个函数：如果它是个奇数，就调用odd()；是偶数，则调用even()。它的输出可能有点难看，所以我们就研究下由count计数的外部循环。已经加上了注释。

<pre lang="lua">
----    1
A: odd  1       -- yield from odd()
E: errorfree, value, status     true    1       suspended
</pre>

在循环中，我们使用coroutine.resume(co,5)来调用这个协程。第一次调用是在进入协程函数的for循环中。注意下这个odd函数，它由我们协程函数中的yield调用。协程函数中不一定非得yield，这点很重要。使用yield，我们返回1。

<pre lang="lua">
----    2
B: odd  1       -- resume in odd with the values we left on the yield
C: even 2       -- call even and exit prematurely
E: errorfree, value, status     true    -1      suspended  -- yield in for loop
</pre>

在第二个循环中，主循环yield并暂停了这个协程。这里的要点就是，我们可以在任何位置执行yield。我们不必纠结在协程中的一点执行yield。使用yield，我们返回-1。
<pre lang="lua">
----    3
A: odd  3       -- odd() yields again after resuming in for loop
E: errorfree, value, status     true    3       suspended
We resume the coroutine in the for loop and when odd() is called it yields again.
</pre>
在for循环中，我们resume这个协程，在调用odd()时，它就再执行次yield。
<pre lang="lua">
----    4
B: odd  3       -- resume in odd(), variable values retained
C: even 4       -- even called()
D: even 4       -- no return in even() this time
A: odd  5       -- odd() called and a yield
E: errorfree, value, status     true    5       suspended
</pre>
在第四个循环中，我们在离开时resume了odd()。注意下其中的变量都保留了，odd()函数的上下文在协程暂停时依然保留。even()函数执行到最后，我们到达了它的末尾。若使用coroutine.yield()以外的其他方式退出函数，函数的上下文及变量一律被销毁。只有使用yield才可以返回。
<pre lang="lua">
----    5
B: odd  5       -- odd called again
E: errorfree, value, status     true    nil     dead  -- for loop terminates
>
</pre>
再次回到odd()。这次主循环到了5，也就是协程的极限。5以及for循环的状态在协程的整个执行过程中都有保留。每个协程都有自己的栈和状态，而我们一旦退出协程函数，它就销毁了。
