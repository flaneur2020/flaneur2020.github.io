---
layout: post
title: "内核如何管理内存"
tags: 
- Kernel
- Linux
- MM
- "翻译"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

作者：Gustavo Duarte
翻译：ssword
原文：http://duartes.org/gustavo/blog/post/how-the-kernel-manages-your-memory
<!--more-->


<hr />

我们已经对类型的虚拟地址空间布局有了一定了解，接下来我们进入内核，了解其内存管理机制。再拿gonzo的图示出来：

<img class="alignnone size-full wp-image-645562" title="mm_struct" src="http://static.duartes.org/img/blogPosts/mm_struct.png" alt="mm_struct" />

Linux内核使用进程描述符task_struct的实例来表示进程。task_struct中的mm字段指向内存描述符mm_struct，它储存了内存中各个段的起始位置（如上图）、进程中使用物理页的数量（rss是Resident Set Size的缩写）、虚拟地址区域的大小以及其他一点细节。在内存描述符中，我们还可以发现其两大心腹：虚拟内存区域和页表。Gonzo的内存区域表示起来如下：

<img class="alignnone size-full wp-image-645565" title="memorydescriptorandmemoryareas" src="http://static.duartes.org/img/blogPosts/memoryDescriptorAndMemoryAreas.png" alt="memorydescriptorandmemoryareas" />

每块虚拟内存区域（VMA）都是一段连续的虚拟地址，其间没有重叠。vm_area_struct的实例就是对一段内存区域的描述，其中包含了内存区域的起始地址、描述行为和访问控制的标志以及表示文件映射的vm_field字段，没有文件映射的VMA就是匿名的（anoymous）。除去内存映射段，如上的每个段（如堆、栈）都单独与一个VMA相关联。x86的机器大都如此组织内存，不过并非如此不可--VMA本身不关心自己位于哪个段。

对程序员而言，VMA在其内存描述符，既有mmap字段里顺序的一段链表，又有mm_rb字段的一棵红黑树。这棵红黑树使得内核得以快速按照给出的虚拟地址来找出相应的内存区域。像读取文件/proc/pid_of_process/maps的内容，内核就只是简单遍历这个VMA的链表再将其输出。Windows下的EPROCESS差不多就是task_struct和mm_struct的混合体，而其相对于VMA的等价物就是虚拟地址描述符（Virtual Address Descriptor），简称VAD，储存在一颗AVL树中。你说Windows和Linux最有趣的东西是啥？就是那点小差异。

4GB的虚拟地址空间被划分为页。32位的x86处理器允许将页划分为三种大小：4kb、2mb或4mb。Linux和Windows的用户虚拟地址空间的页大小都是4kb。地址中0-4085字节是第一页，4096-8191是第二页，以此类推。VMA的大小必为页大小的整数倍。如下就是按4kb分页的3gb用户空间：

<img class="alignnone size-full wp-image-645567" title="pagedvirtualspace" src="http://static.duartes.org/img/blogPosts/pagedVirtualSpace.png" alt="pagedvirtualspace" />

处理器依据页表将虚拟地址转换为物理地址，不同进程的页表各不相同。因此在进程切换时，用户空间的页表也随之切换。Linux将指向进程页表的指针储存于内存描述符的pgd字段中。每个虚页都与一个页表入口（PTE）相关联，在一般的x86分页机制下，它就是一个4字节长的记录：

<img class="alignnone size-full wp-image-645569" title="x86pagetableentry4kb" src="http://static.duartes.org/img/blogPosts/x86PageTableEntry4KB.png" alt="x86pagetableentry4kb" />

Linux有专门的函数来读写PTE的属性标志。P位表示次页表是否位于物理内存。若清零，读取这块内存就会触发一个页异常。牢记，就算该位清零，内核依然可以修改其他属性域。R/W表示读/写(read/write)；若清零，该页只读。U/S表示用户/超级用户(user/supervisor)，若清零，该页只允许内核访问。有了这些标志，才可以实现我们前面所见的只读内存和内核空间保护。

D位与A位分别表示dirty和accessed。若一个页曾被写过，它就被标记为dirty；若曾有过读或写，它就会被标记为accessed。这两个标志都挺难搞：进程只管设置它们，清零却只允许内核来做。PTE保存与此页相关联的起始地址，以4kb对齐。这个貌似简单的属性域有一点不足，那就是限制了物理内存最大只能是4GB。物理地址扩展相关的PTE属性域改天再讲。

虚拟页是内存保护的基本单元，其中的所有字节共享同一U/S和R/W标志。不同的虚拟页可能映射自同一物理页，但其保护标志并不一定相同。注意下，PTE中并没有包含执行权的标志。这带来的后果就是，早期的x86系列CPU可能将堆栈段上的代码执行，因而易于遭到堆栈缓冲区溢出的攻击（如今使用return-to-libc或其他技巧，依然可以对堆栈中不可执行的代码搞溢出）。更深一层，PTE中执行权限标志的缺失使得VMA中的保护标志不一定能够应用到硬件的保护机制。内核已尽其全力，然而体系结构的限制尤在。

虚拟内存也不是什么都存。它只是将一个程序的地址空间映射到实在的物理内存，即物理地址空间。虽然在某种意义上有些内存操作也需要经过总线，但我们在此大可忽略之，从而将物理地址看作是从0开头，以字节为单位递增的一段地址。内核将物理地址空间划分为n个页框。处理器对页框的存在并不关心，不过对内核而言页框至关重要，因为页框就是管理物理内存的基本单元。32位的Linux和Windows都是用4kb的页框，如下是个2GB内存机器的例子：

<img src="http://static.duartes.org/img/blogPosts/physicalAddressSpace.png" alt="" />

Linux的每个页框都带有一个描述符以及n个属性标志。这些描述符管理了电脑的所有物理内存，使得每个页框的状态都可以明确。物理内存通过伙伴内存分配机制进行管理，对伙伴系统而言，一个页框若可以分配，那它就是free的。分配来的页框可以是匿名，以储存程序数据；也可以是页缓存，以包含来自文件或设备的数据。也有其他类型的页框，在这里先略过就是。Windows下有个类似的页框号（Page Frame Number,PFN）数据库来跟踪物理内存。

现在把虚拟内存区域、页表、页框放到一起，看看其整体是如何工作。如下是个用户堆的例子：

<img class="alignnone size-full wp-image-645564" title="heapmapped" src="http://static.duartes.org/img/blogPosts/heapAllocation.png" alt="heapmapped" width="549" height="176" />

蓝色方形表示了VMA中包含的页，箭头表示页经页表与页框形成的映射关系。某些虚拟页上并没有箭头，因为它们PTE中Present标志都是清零的。这些页可能是从未使用，也可能是已被换出。不过无论怎样，访问这些页都会导致页异常，即便这些页在VMA中也是如此。VMA和页表不能一一对应，可能难以理解，不过确实是经常发生的事情。

VMA就像是程序与内核间的通信员。你先下个什么请求（内存分配，文件映射等等），内核说“行”，它就创建或更新一个合适的VMA。但它并不会马上将其一步到位，而是等到有了页异常再来。内核是个懒惰且狡诈的混蛋，这就是虚拟内存的基本作风。熟悉与否，这一思想随处可见。VMA保存上面分配来的内容，而PTE决定其具体的行为。这两个数据结构共同管理着程序的内存，遇到页异常、释放内存、换页等操作的时候，都有它俩的份。看个这个内存分配的简单例子：

<img class="alignnone size-full wp-image-645563" title="heapallocation" src="http://static.duartes.org/img/blogPosts/heapAllocation.png" alt="heapallocation" width="678" height="402" />

程序使用brk()系统调用来申请更多内存，内核就简单更新下堆的VMA，说“行了”。不过这时并没有页框真正分配出来，页也不在物理内存中。程序一旦要访问这些页，处理器就会触发页异常，并执行do_page_fault()。它使用find_vma()找出异常发生地址对应的VMA，若找到，检查VMA的权限标志以防恶意访问（读或写）；如果没有合适的VMA，就不再管这个内存访问而交给处理器触发一个段异常。

找到相应VMA之后，内核必须检查PTE的内容以及VMA的类型，以处理这个异常。在我们的例子里，PTE显示出这个页不在物理内存中。这里我们PTE为空（全为零），Linux中就表示这段虚拟页从未被映射过。这是个匿名的VMA，因此只能用do_anoymous_page()处理。它会分配一个页框，修改PTE将那发生异常的虚拟页映射到新分配的页框上。

也有例外。例如页已经换出，那它PTE中Present标志就是0，里面则保存了页内容在硬盘上的地址，它使用do_swap_page()读取硬盘并将其装载置页框。

我们内核内存管理之旅大约已行至一半。在下篇post里，我们会讨论上文件及性能的因素，以了解内存管理的整体。

<hr />

囧，译到四分之三的时候发现已经有人译过了：<a href="http://blog.csdn.net/drshenlei/archive/2009/07/15/4350928.aspx">http://blog.csdn.net/drshenlei/archive/2009/07/15/4350928.aspx</a>
