---
layout: page
title: 基于 Bochs 的操作系统内核实现
---

# 基于 Bochs 的操作系统内核实现

[[fleuria](http://fleurer-lee.com)] 2012

## 简介

### Bochs 简介

Bochs(读音Box)是一个开源的模拟器(Emulator)，它可以完全模拟x86/x64的硬件以及一些外围设备。与VirtualBox / VMware等虚拟机(Virtual Machine)产品不同，它的设计目标在于模拟一台真正的硬件，并不追求执行速度的高效，而追求模拟环境的真实，同时带有强大的调试功能，比如观察寄存器、对实地址/虚拟地址下断点、装载符号表等等。对于操作系统内核的开发者而言，是一只不可多得的强力工具，通过简单的设置，即可大大地降低内核开发与调试的困难。

作为开源软件，我们可以很方便地获取它：

+ 主页：http://bochs.sourceforge.net/getcurrent.html
+ 参考文档: http://wiki.osdev.org/Bochs

#### 安装

在Ubuntu操作系统下，可以通过apt-get来安装：

    sudo apt-get install bochs

若要利用Bochs的调试功能，则需要自己编译安装：

    wget http://sourceforge.net/projects/bochs/files/bochs/2.5.1/bochs-2.5.1.tar.gz/download -O bochs.tar.gz
    tar -xvfz bochs.tar.gz
    cd bochs-2.5.1
    ./configure --enable-debugger --enable-debugger-gui --enable-disasm --with-x --with-term
    make
    sudo cp ./bochs /usr/bin/bochs-dbg

#### 配置

Bochs 提供了许多配置选项，在项目中，我们可以灵活的选择/设置自己所需的功能，比如模拟器的内存大小、软/硬盘镜像以及引导方式等等。而这些配置选项都统一在一个.bochsrc文件中，样例如下：

.bochsrc:

    # BIOS与VGA镜像
    romimage: file=/usr/share/bochs/BIOS-bochs-latest
    vgaromimage: file=/usr/share/bochs/VGABIOS-lgpl-latest
    # 内存大小
    megs: 128
    # 软盘镜像
    floppya: 1_44=bin/kernel.img, status=inserted
    # 硬盘镜像
    ata0-master: type=disk, path="bin/rootfs.img", mode=flat, cylinders=2, heads=16, spt=63
    # 引导方式(软盘)
    boot: a
    # 日志输出 
    log: .bochsout
    panic: action=ask
    error: action=report
    info: action=report
    debug: action=ignore
    # 杂项
    vga_update_interval: 300000
    keyboard_serial_delay: 250
    keyboard_paste_delay: 100000
    mouse: enabled=0
    private_colormap: enabled=0
    fullscreen: enabled=0
    screenmode: name="sample"
    keyboard_mapping: enabled=0, map=
    keyboard_type: at
    # 符号表(调试用)
    debug_symbols: file=main.sym
    # 键盘类型
    keyboard_type: at

在启动bochs时，使用命令：

    bochs -q -f .bochsrc

#### 内置调试器

bochs内置了强大且方便的调试功能。主要命令如下：

+ `b`,`vb`,`lb` 分别为物理地址、虚拟地址、逻辑地址设置断点
+ `c` 持续执行，直到遇到断点或者错误
+ `n` 下一步执行
+ `step` 单步执行
+ `r` 显示当前寄存器的值
+ `sreg` 显示当前的段寄存器的值
+ `info gdt`, `info idt`, `info tss`, `info tab` 分别显示当前的GDT、IDT、TSS、页表信息
+ `print-stack` 打印当前栈顶的值
+ `help` 显示帮助

### fleurix 简介

fleurix 是一个简单的单内核(Monolithic Kernel)操作系统实现，它的功能精简但不失完整，代码简短(七千行C，二百多行汇编)且易于阅读，可作为操作系统课程教学中的样例系统。在设计时选择采用了类UNIX的系统调用接口，因此在开发过程中可以获取丰富的文档供参考，也可以作为学习UNIX操作系统实现的一个参考材料。

fleurix 在编写时尽量使用最简单的方案。它假定CPU为单核心、内存固定为128mb，前者可以简化内核同步机制的实现，后者可以简化内存管理的实现。从技术角度来看，这些假定并不合理，但可以有效地降低刚开始开发时的复杂度。待开发进入轨道，也不难回头解决。 此外，你也可以在源码中发现许多穷举算法——在数据量较小的前提下，穷举并不是太遭的解决方案。

+ 主页： http://github.com/fleurer/fleurix
+ 开发环境： Ubuntu
+ 平台：x86
+ 依赖： `bochs`, `rake`, `binutils`, `nasm`, `mkfs.minix`

#### 特性

+ minix v1的文件系统。原理简单，而且可以利用linux下的mkfs.minix，fsck.minix等工具。
+ `fork()`/`exec()`/`exit()`等系统。可执行文件格式为a.out，实现了写时复制与请求调页。
+ 信号。
+ 一个纯分页的内存管理系统，每个进程4gb的地址空间，共享128mb的内核地址空间。至少比Linux0.11中的段页式内存管理方式更加灵活。
+ 一个简单的`kmalloc()`。
+ 一个简单的终端。

#### 编译运行

    git clone git@github.com:Fleurer/fleurix.git
    cd fleurix
    rake

#### 调试

    # 需要自行编译安装带调试功能的bochs-dbg，安装步骤参见前文。
    cd fleurix
    rake debug


---------------------------------------------------------------------

## 设计与实现

### 编译与链接

fleurix的内核镜像为裸的二进制文件，结构大体如下：

(补图)

#### Rakefile

对于项目中的一些日常性质操作，比如：

+ 编译bootloader，生成引导镜像
+ 编译并链接内核，生成内核镜像
+ 生成符号表
+ 初始化根文件系统，生成硬盘镜像
+ 编译整个项目，并运行bochs进行调试

它们需要的命令比较多，而且存在依赖关系，此任务必须在确保彼任务执行完毕并成功之后才可以执行。对此，比较通用的解决方案便是make，它可以自动分析任务之间的依赖关系再依次执行，从而简化日常操作的脚本编写。但是make的语法比较晦涩，对于没有任何基础的初学者来讲，上手起来并不容易。为此fleurix选择了rake，它相当于make的ruby实现，可以使用ruby语言的语法来编写make脚本，好处是易于上手，而代价是不如make的语法简洁。

fleurix中常用的rake命令有：

+ `rake`或者`rake bochs`，构建整个项目并运行bochs
+ `rake build`，构建整个项目到`/bin`目录
+ `rake debug`，构建整个项目并运行bochs的调试器
+ `rake clean`，将`/bin`目录清空
+ `rake nm`，生成符号表
+ `rake todo`，列出代码中遗留的待解决事项
+ `rake werr`，打开gcc的-Werror选项进行编译，方便排除代码中的warning
+ `rake rootfs`，构建根文件系统
+ `rake fsck`，对根文件系统执行fsck，检查结构是否正确

#### ldscript

内核开发与应用程序开发的不同之一便在于开发者需要对二进制镜像的结构有所了解，在必要时必须进行一些重定位。比如内核的入口为0x100000，为此需要将入口的代码(`bin/entry.o`)安排到内核镜像的最前方。而这便可以通过ldscript来完成，如下：

tool/main.ld:

    ENTRY(kmain)
    SECTIONS {
        __bios__ = 0xa0000; # 绑定BIOS保留内存的地址到__bios__ 
        vgamem = 0xb8000; # 绑定vga缓冲区的地址到符号vgamem
        .text 0x100000 : { # 内核二进制镜像中的.text段(Section)，从0x100000开始
            __kbegin__ = .; # 内核镜像的开始地址
            __code__ = .;
            bin/entry.o(.text) bin/main.o(.text) *(.text); # 将bin/entry.o中的.text段安排到内核镜像的最前方
            . = ALIGN(4096); # .text段按4kb对齐
        }
        .data : { 
            __data__ = .;
            *(.rodata);
            *(.data);
            . = ALIGN(4096);
        }
        .bss : {
            __bss__ = .;
            *(.bss);
            . = ALIGN(4096);
        }
        __kend__ = .; # 内核镜像的结束地址
    }

Rakefile中的相关命令如下，在链接时选择tool/main.ld作为链接脚本：

    sh "ld #{ofiles * ' '} -o bin/main.elf -e c -T tool/main.ld"

### bootloader

bootloader是一段小程序，负责执行一些初始化操作，并将内核装载到内存，是内核执行的入口，也是内核开发的第一步。

x86体系结构的CPU在设计中为了保持向前兼容，在PC机电源打开之后，x86平台的CPU会先进入实模式(Real Mode)，并从0xFFF0开始执行BIOS的一些初始化操作。随后，BIOS将依次检测启动设备(软盘或者硬盘)的第一个扇区(512字节)，如果它的第510字节处的值为0xAA55，则认为它是一个引导扇区，将它装载到物理地址0x7C00，并跳转到0x7C00处开始执行。这便是bootloader的入口地址。

实模式中默认可用的地址总线为20位，可以寻址1mb的内存，但寄存器只有16位。为此英特尔公司做出的设计是，在实模式的寻址模式中，令物理地址为16位段寄存器左移4位加16位逻辑地址的偏移所得的20位地址。若要访问1mb之后的内存，则必须开启A20 Line开关，将32位地址总线打开，并进入保护模式(Protect Mode)才可以。

在实模式中，0~4kb为中断向量表保留，640kb~1mb为显存与BIOS保留，实际可用的内存只有636kb。考虑到日后内核镜像的体积有超过1mb的可能，所以将其装载到物理地址1mb(0x100000)之后连续的一块内存中可能会更好。但实模式中并不可以访问1mb以后的内存，若要装载内核到物理地址1mb，一个解决方案便是在实模式中暂时将其装载到一个临时位置，待进入保护模式之后再移动它。

由上总结可知，bootloader所需要做的工作便依次为：

+ 装载内核镜像到一个临时的地址；
+ 进入保护模式；
+ 移动内核镜像；
+ 跳转到内核的入口。

相关代码可见于 `src/boot/boot.S` 。

#### 保护模式与GDT

x86的保护模式是对段寻址的增强，除去可以访问32位的地址空间(4Gb)之外，更有了对保护级别(即ring0/ring1/ring2/ring3)的划分、对内存区域的限制、以及访问控制。为实现这些功能，x86的做法是引入了GDT(Global Descriptor Table)。将每个段(Segments)的属性对应为GDT中的一项段描述符(Segment Descriptor)，并通过段寄存器(如`cs`、`ds`、`ss`)中指明的选择符进行选择。GDT是驻留于内存中的一个表，通过`lgdt`指令装载到CPU。

在bootloader中进入保护模式的目的仅仅是为了访问1mb以后的内存，而且bootloader在完成引导系统之后即被视为废弃，因此这里的GDT只能做临时使用。其中含有两个段描述符，它们的选择符分别为0x08与0x10，分别用于内核态代码与数据的访问。

进入内核之后，fleurix会在`gdt_init()`中重新设置GDT(见`scr/kern/seg.c`)。

fleurix是一个纯分页的系统，虽然并不需要段式的内存管理，但依然需要一个GDT，只采用它的内存保护功能，而绕过它的分段功能。在fleurix最终的GDT中，将只保留四个段描述符，它们的内存区域皆为0~4Gb，选择符分别为`KERN_CS`、`KERN_DS`、`USER_CS`与`USER_DS`——前两者的权限为ring0，用于内核态代码与数据的访问；后两者的权限为ring3，分别用于用户态代码与数据的访问——从而实现内核态与用户态的分离，使后者受到更多限制，将系统“保护”起来。

需要留意的是，除四个段描述符之外，fleurix的GDT中也带有一个TSS描述符，其选择符为(`_TSS`)。英特尔公司引入TSS机制的动机为实现硬件的任务切换，每个任务拥有一个TSS，在进程切换时，将当前进程的**所有**上下文保存在TSS中。比起软件的任务切换，硬件任务切换的开销相对比较大，而且没有调试与优化的余地。fleurix采用了软件的任务切换机制，并无用到TSS的任务切换功能，但依然保留一个TSS是为了保存中断处理时ss0与esp0两个寄存器的值，在CPU通过中断门或者自陷门转移控制权时，据此获取内核栈的位置。

#### 装载内核

在早期开发中为方便装载，fleurix内核的二进制镜像被放置在软盘镜像中，自第二个扇区开始，大约为50kb。

在实模式中，可以通过调用13h号中断来读取软盘扇区，将内核镜像临时读取到物理地址0x10000处。在设置临时的GDT之后，通过jmp指令进入保护模式，并将内核拷贝至物理地址0x100000(1mb)处。

### 内核初始化

待bootloader执行完毕之后，内核会首先进入`kmain()`(见`src/kern/main.c`)，执行一些初始化操作。这些操作依次为：

* 清理屏幕(`cls()`，见`src/chr/vga.c`)，初始化`puts()`与`printk()`等函数供调试与输出使用。
* 重新设置GDT(`gdt_init()`，见`src/kern/seg.c`)。
* 初始化IDT(`idt_init()`，见`src/kern/trap.c`)。
* 初始化内存管理(`mm_init()`，见`src/kern/pm.c`)。
* 初始化进程0(`proc0_init()`，见`src/kern/proc.c`)。
* 初始化高速缓冲(`buf_init()`，见`src/blk/buf.c`)。
* 初始化tty(`tty_init()`，见`src/chr/tty.c`)。
* 初始化硬盘驱动(`hd_init()`，见`src/blk/hd.c`)。
* 初始化内核定时器(`timer_init()`，见`src/kern/timer.c`)
* 初始化键盘驱动(`keybd_init()`，见`src/chr/keybd.c`)。
* 开启中断(`sti()`，见`src/inc/asm.h`)。
* 初始化进程1(`kspawn(&init)`)，通过`do_exec()`(见`src/kern/exec.c`)即进入用户态。

### 中断处理

中断是CPU中打断当前程序的控制流以处理外部事件、报告错误或者处理异常的一种机制。若详细分类，仍可将中断分为三种：

+ 中断(Interrupt)：由CPU外部产生，CPU处于被动的位置，多用于CPU与外部设备的交互。
+ 自陷(Trap)：在CPU本身的执行过程中产生。一般由专门的指令有意产生，比如`int $0x80`，因此又被称作"软件中断"。
+ 异常(Exception)：因CPU执行某指令失败而产生，如除0、缺页等等。与自陷的不同在于，CPU会在处理例程结束之后重新执行产生异常的指令。

(注：即，自陷发生时，入栈的返回地址为下一条指令的地址；而异常发生时，入栈的返回地址为当前指令的地址)

在保护模式的x86平台中，中断通过中断门(Interrupt Gate)转移控制权，自陷与异常通过自陷门(Trap Gate)转移控制权。

每个中断对应一个中断号，系统开发者可以将自己的中断处理例程绑定到相应的中断号，表示中断号与中断处理例程之间映射关系的结构被称作中断向量表(Interupt Vector Table)。在保护模式中的x86平台，这一结构的实现为IDT(Interrupt Descriptor Table)。与GDT类似，IDT也是一个驻留于内存中的结构，通过`lidt`指令装载到CPU。每个中断处理例程对应一个门描述符(Gate Descriptor)，。在fleurix中初始化IDT的代码位于`idt_init()`(见`src/trap.c`)。

在中断发生时，CPU会先执行一些权限检查，若正常，则依据特权级别从TSS中取出相应的ss与esp切换栈到内核栈，并将当前的eflags、cs、eip寄存器压栈(某些中断还会额外压一个error code入栈)，随后依据门描述符中指定的段选择符(Segment Selector)与目标地址跳转到中断处理例程。 保存当前程序的上下文则属于中断处理例程的工作。在fleurix中，保存中断上下文的操作由`_hwint_common_stub`(见`src/kern/entry.S.rb`)负责执行，它会将中断上下文保存到栈上的`struct trap`结构(见`src/inc/idt.h`)。

在这里有三个地方值得留意：

+ 只有部分中断会压入error code，这会导致栈结构的不一致。为了简化中断处理例程的接口，fleurix采用的方法是通过代码生成，在中断处理例程之初为不带有error code的中断统一压一个双字入栈，值为0，占据error code在`struct trap`中的位置。并将中断调用号压栈，以方便程序的编写与调试。
+ fleurix中的中断处理例程都经过汇编例程`_hwint_common_stub`，它在保存中断上下文之后，会调用`hwint_common()`(见`src/kern/trap.c`)函数。`hwint_common()`函数将依据中断号，再查询`hwint_routines`数组找到并调用相应的处理例程。
+ 中断的发生往往就意味着CPU特权级别的转换，因此，可以将陷入(或称"软件中断")作为用户态进入内核态的入口，从而实现系统调用。在fleurix中系统调用对应的中断号为0x80，与linux相同。

### 系统调用



### 分页

fleurix应用x86平台的分页机制，实现了纯页式的内存管理。与段式内存管理(如DOS)或者段页式混合的内存管理(如linux0.11)相比，纯页式内存管理(以下简称"页式内存管理")中可用的地址空间更大，也更加灵活。比如写时复制与请求调页这样的机制，在段式内存管理中则属于不可能实现的。

按照x86平台的分页机制，内存被划分为4kb或者4mb大小的物理页(又称"页框")，由页表来表示虚拟页到物理页的映射关系。为节约页表本身所占用的内存，x86采用了二级页表。每个页表占4kb，含有1024条页表项，可以映射4mb的地址空间；页目录也同样4kb，含有1024项，可以映射4gb的地址空间。在进行地址翻译时，将先查询页目录，找到虚拟地址对应的页表，再在页表中查询得出相应的物理页，外加页内的偏移，最终得到物理地址。其中有个例外，便是4mb的大页，页目录中的表项可以不指向一个页表，而是仅仅表示一个4mb大页的地址映射，它的便利之处在于映射大块连续的地址空间，可以做到既方便又高效。

对于CPU来说，每次地址翻译都到内存中查询页表是不可容忍的高开销，为此，支持分页的CPU往往都提供了TLB(Translation Lookaside Buffer，俗称"快表")作为页表的缓存。在这里开发者需要留意的是，只要更新了页表，便需要留意保持TLB的同步，不然就会有一些难于调试的问题出现。在bochs的内嵌调试器中，可以通过`info tab`命令来检查当前的页面映射。

(注: 因为内存局部性原理，TLB一般只需要很小(比如64项)即可达到不错的效果。)

页面可以被标记为只读(Readonly)或者不存在(Non-Present)，也可以设置页面的保护级别。这一来在读写内存时，如果发生不合法的内存读写，就会产生一个页面错误(Page Fault)，触发中断处理例程`do_pgfault()`(见`src/mm/pgfault.c`)。这时产生页面错误的地址，将被保存在cr2寄存器中，同时产生一个error code，表示页面错误的类型。待页面错误处理完成，被打断的程序可以恢复执行，也有可能因为严重的错误而中止(收到信号`SIGSEGV`)。

在fleurix中，每个进程拥有一个独立的页目录，从而实现进程地址空间的隔离；通过4mb的大页，实现虚拟地址与物理地址的一对一映射直到128mb为止，作为内核地址空间；并过页表项的保护级别，限制用户态应用程序对内核地址空间的读写；通过将页面标记为只读或者不存在，实现写时复制(Copy On Write)与请求调页(Demand Paging)。
。

对于x86平台，值得留意的地方有：

+ cr0寄存器中的Paging位表示分页机制的开关(`mmu_enable()`, 见`src/inc/asm.h`)；
+ cr4寄存器中的PSE位表示4mb大页的开关(在一些较旧的CPU上并没有PSE的支持)；
+ 页目录的地址装载于cr3寄存器(`lpgd()`，见`src/inc/asm.h`)；
+ 页面错误中的error code可能会有三种flag，即`PFE_P`、`PFE_W`与`PFE_U`(定义于`src/inc/mmu.h`)，分别表示页面不存在、页面只读及权限不足。
+ 只要重新装载页目录，即为刷新TLB(`flmmu()`，见`src/mm/pte.c`)。

### 内存分配

fleurix假定用户的物理内存为128mb，并将内核永远地映射于每个地址空间的低端(0~128mb)，使得内核地址空间中的虚拟地址与物理地址做到一对一的映射。这一来只要分配了物理页面，内核就可以直接读写它的内容或将它映射到用户进程。需要留意的是，从技术角度这一假设并不合理：若用户的物理内存若小于128mb，内核就会崩溃；若用户的物理内存大于128mb，则无法利用128mb以上的内存。但它可以有效地降低项目开发之初的复杂度，待项目进入轨道，则应优先解决这一问题。

`pgalloc()`与`pgfree()`为内核内存分配的基础例程，分别用于申请/释放一个物理页。一个物理页面可能会被多个进程映射到，因此一个引用计数是必须的；物理页面可能会比较多，使用穷举式的分配效率不高。对此，fleurix实现了一个`struct page`结构(定义于`src/inc/page.h`)，并在内核初始化时，初始化一个数组`struct page coremap[NPAGE]`与一个队列`struct page pgfreelist`(见于`src/mm/pm.c`中的`pm_init()`)，前者作为物理页是否可用的标记，数组的每一项对应一个物理页，物理页面的地址就等于`数组下标 * 4kb`，若对应的`struct page`结构中的引用计数为0，则表示物理页是可用的；后者则将所有可用的物理页组织到一个链表之中，这一来即可将分配/释放物理页的操作的时间复杂度降到O(1)。

#### kmalloc()

### 进程

进程即运行中的程序实体。每个进程拥有独立的地址空间以及一些资源，相互并发执行。在fleurix中，进程为代码执行与资源管理的基本单位。

终其一生，进程可能有五种状态：

+ `SSLEEP`： 睡眠且不可被信号唤醒，等待一个高优先级的事件；
+ `SWAIT`： 睡眠，可被信号唤醒，等待一个低优先级的事件；
+ `SRUN`： 正常执行；
+ `SZOMB`： 僵尸进程，是在进程因为某种原因退出执行(主动调用`_exit()`或者被信号杀死)、在被父进程回收之前的进程状态。
+ `SSTOP`： 停止中，在进程创建之初以及进程回收时的进程状态。

在fleurix中，表示进程的结构为`struct proc`，它含有进程的pid(`p_pid`)、状态(`p_stat`)、父进程id(`p_ppid`)、进程组(`p_pgid`)、用户id(`p_uid`)、组id(`p_gid`)、地址空间(`p_vm`)、上下文(`p_contxt`)、打开的文件(`p_ofile`)、信号处理例程(`p_sigact`)、可执行文件的inode(`p_inode`)等诸多信息，正是fleurix中最为复杂的结构。

`struct proc`与这一进程的内核栈同处一个物理页，前者位于低端固定，后者位于高端向下增长。在这里不难发现，内核栈的可用空间非常小(小于4kb)，因此在内核开发中，应尤其注意不要在栈上放置较大的对象，抑或进行较深的递归，不然内核栈若溢出，绝不会像用户态中那样出现`Segmentation Fault`的提示，而会默默地搞乱内核中的数据结构，出现一些难于调试的问题。

为方便对进程结构的引用，fleurix设置了一个数组即`struct proc *proc[NPROC]`，数组的下标即进程的pid，`NPROC`则为系统中进程数量的上限；以及一个指针`struct proc *cu`，永远指向当前的进程结构。

#### 进程创建

进程只能通过`fork()`系统调用创建，它会复制当前进程的地址空间与资源，生成一个一模一样的子进程。不过，`fork()`会在父进程中返回0，在子进程中则返回子进程的pid作为区别，如下：

    int pid;
    if ((pid = fork()) == 0) {
        printf("I'm the parent process\n");
    }
    else {
        printf("I'm the child process\n");
    }

在`fork()`时，直接复制整个进程地址空间的操作是昂贵的，而且大多数子进程都会在执行之初调用`exec()`覆盖掉当前地址空间，之前的复制也就没有意义了。对此，类UNIX系统大多基于CPU的分页机制，提供了写时复制的实现：在复制进程地址空间时，并不直接拷贝地址空间中页面的内容，而是仅仅复制父进程的页表，使得父子进程共享相同的物理页，并将二者的虚拟页面皆设置为只读。随后若二者任一方试图修改内存，则申请一个新的物理页并复制旧页的内容。这里需要留意的是，为控制物理页的共享，每个物理页都需要维护一个引用计数，当`fork()`时引用计数增1，当进程杀死或者发生写时复制时减1，并在引用计数为0时释放这个物理页。

在fleurix中，通过`vm_clone()`(见于`src/mm/vm.c`)实现进程地址空间的复制。

fleurix在开始运行之初会初始化一个0号进程(`proc0_init()`，见于`src/kern/fork.c`)，其后的所有进程皆由它fork而来。

#### 程序执行

`exec()`是fleurix中行为最为复杂的系统调用之一。就表面的行为而言，它会取一个可执行文件的地址与相关参数(`argv`)，并执行它。然而在内部，它所做的工作却远比表面上复杂：

+ 读取文件的第一个块，检查Magic Number(`NMAGIC`)是否正确
+ 保存参数(`argv`)到临时分配的几个物理页，其中的每个字符串单独一页
+ 清空旧的进程地址空间(`vm_clear()`，见于`src/mm/vm.c`)，并结合可执行文件的header，初始化新的进程地址空间(`vm_renew()`，见于`src/mm/vm.c`)
+ 将`argv`与`argc`压入新地址空间中的栈
+ 释放临时存放参数的几个物理页
+ 关闭带有`FD_CLOEXEC`标识的文件描述符
+ 清理信号处理例程
+ 通过`_retu()`返回用户态

这里值得留意的是，之所以将`argv`保存到临时分配的几个页面，是因为`argv`中的字符串与这个数组本身都是来自旧的地址空间，而旧的地址空间会被销毁，`argv`所指向的内存区域，也自然就无法访问了。

与写时复制的实现相似，`exec()`在执行时，并不会立即将可执行文件完全读入内存。而是通过`vm_renew()`，将当前进程的虚拟页面统统设置为不存在，待进入用户态开始执行时，每发生一次页面不存在的错误，便读取一页可执行文件的内容并映射。这样的机制被称作请求调页(Demand Paging)，好处是可以加速程序的启动，不必等待可执行文件完全读入内存即可开始程序的执行，在某种意义上，也可以节约内存的使用。缺点是如果程序的体积较小，就不如一次性将可执行文件全部读入内存的方式高效。

为简单起见，fleurix只支持a.out格式作为可执行文件格式，对应可执行文件中不同的区段(section)，进程的地址空间也分为不同的内存区(VMA，Virutal Memory Area)，如正文区(`.text`)、数据区(`.data`)、bss区(.bss)、堆区(`.heap`)与栈区(`.stack`)。它们的性质各不相同：正文区与数据区内容都来自可执行文件，然而正文区是只读的，数据区可读可写；bss区、堆区与栈区的内存皆来自动态分配，都可读可写，不过bss区的内存都默认为0，堆区可以通过`brk()`系统调用来调整它的长度，而栈区可以自动向下增长。对于这些不同需求，fleurix提供了一个结构`struct vma`，它可以绑定一个inode，并在必要时依据相关的几个标志(即`VMA_RDONLY`、`VMA_STACK`、`VMA_ZERO`、`VMA_MMAP`、`VMA_PRIVATE`)执行不同的操作。具体可见于`src/mm/pgfault.c`文件中`do_no_page()`的相关代码。

#### 进程切换

fleurix采用软件的进程切换，一切进程切换都发生在内核态。进程的上下文有：

+ 内核栈的顶，供中断处理例程使用；
+ 页目录，也就是地址空间；
+ `eip`与`esp`；
+ callee-saved registers(`ebp`、`ebx`、`esi`、`edi`)；

因为进程切换都发生在内核态，因此无需保存`cs`等段寄存器；依据gcc的调用约定，`eax`、`ecx`与`edx`为`caller-saved registers`，也同样不需要保存。

fleurix将上下文相关的寄存器保存在一个`struct jmp_buf`结构中，它与C标准库中的`jmp_buf`基本相同，甚至可以这样想：进程切换就是，在切换地址空间之后，为当前进程的上下文执行`setjmp()`记录下来，同时通过`longjmp()`跳转到目标进程的上下文。

负责进程切换的函数为`swtch_to()`，可见于`src/kern/sched.c`。内容如下：

    void swtch_to(struct proc *to){
        struct proc *from;
        tss.esp0 = (uint)to + PAGE; 
        from = cu;
        cu = to;
        lpgd(to->p_vm.vm_pgd);
        _do_swtch(&(from->p_contxt), &(to->p_contxt));
    }

`_do_swtch()`是一段汇编例程，它负责将当前的上下文保存到`from->p_contxt`，同时将`to->p_contxt`中保存的上下文恢复出来。


#### 进程调度

fleurix采用带优先级的Round Robbin调度算法。

负责进程调度与主动进程切换的函数为`swtch()`，它会找出当前优先级最高的进程并切换。

fleurix的内核是非抢占的，一切进程抢占都发生在内核态返回用户态的那一刻。可见于`src/kern/trap.c`中`hwint_common()的结尾处`：

    setpri(cu);
    if ((tf->cs & 3)==RING3) {
        swtch();
    }

它首先尝试调整当前进程的优先级，再通过中断上下文中保存的`cs`寄存器判断当前的中断上下文是否是来自用户态。只有确定是来自用户态，才尝试执行任务切换，这样可以保证内核态中不会发生抢占。

#### 进程同步

### 输入输出

### 文件系统

## 遇到的问题

## 总结

## 参考文献

+ 《Linux内核完全注释》，赵炯 著
+ 《莱昂氏UNIX源码分析》
+ 《UNIX操作系统设计》
+ 《UNIX Internals》
+ 《4.4 BSD操作系统的设计与实现》
+ 《Bran's Kernel Development Toturial》，Brandon Friesen 著

