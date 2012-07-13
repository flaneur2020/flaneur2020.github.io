---
layout: post
title: "Fleurer’s Stack VM"
tags: 
- C
- fsvm
- VM
- "备忘"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

<a href="http://github.com/Fleurer/fsvm">http://github.com/Fleurer/fsvm</a>
尝试用C写的堆栈机，好像烂尾了 - -!

只是个运行时，无视了语法分析。随便写的东西也没什么规划，编写的时候就郁闷不知道哪部分该归分析器那部分该归vm，也不知道现有的部分能不能真用到解释器上，于是华丽地烂尾～只实现了20来条指令，可以递归可以闭包，不过不能算乘法...囧，很简陋啦~ 

好像是第一次写C，不会make就先凑合了rake - -！ 对C不熟悉，满地的Segmentation Fault对我们只会用printf调试的菜鸟太残酷了...用了那个保守式的gc库Boehm GC，面对满地的malloc而无free不清楚泄漏起来会怎样...囧

本来是对C++那套OO无爱，想单用struct和函数也可以OO么。于是用了C，然后就后悔了：我不想重新实现hashmap之类的东西，C++那stl多好...T_T。倒也找到了个C的泛型库khash，不过宏终究不如模板来的好看...现在想来，信息学奥赛acm中用C的那些同学做题的时候都是自己实现一遍各个数据结构么？

拿段伪代码：
<pre lang="python">
def main:
     sum(10)

def sum(i):
     if (i==0) : return 0;
     else: return(i+sum(i-1)); 
</pre>

放到fsvm下大约是这样：
<pre lang="c">
int test_rec(){
    Op op_main[]={
        OP_PUSH_NUM, 10, 
        OP_PUSH_CONST, 0, //"sum"  
        OP_PUSH_VAR, 
        OP_CALL, 1,
        OP_RET
    };
    Op op_sum[]={
        OP_PUSH_CONST, 0, //"i"
        OP_PUSH_VAR, 
        OP_POP_TMP, 0, //store i
        OP_PUSH_TMP, 0, //push i
        OP_PUSH_NUM, 0, 
        OP_EQ,
        OP_NOT, // i!=0?
        OP_BRANCH, 3, 
            OP_PUSH_NUM, 0,
            OP_RET,
        //else
            OP_PUSH_TMP, 0, 
            OP_PUSH_NUM, 1, //1
            OP_SUB, 
            OP_PUSH_CONST, 1, //"sum"
            OP_PUSH_VAR, 
            OP_CALL, 1,   //sum(tmp[0]-1)
            OP_PUSH_TMP, 0, 
            OP_PRINT_STACK, 
            OP_ADD, //tmp[0]+sum(tmp[0]-1)
        OP_RET
    };
    Env *env=fnew_env(NULL); 
    
    Proto *p_main = fnew_proto(op_main, 0);
    fset_const(p_main, 0, fstr("sum"));
    Func* f_main=fnew_func(p_main, env);
    Obj o_main=ffunc(f_main);
 
    Proto* p_sum=fnew_proto(op_sum, 1);
    fset_const(p_sum, 0, fstr("i"));
    fset_const(p_sum, 1, fstr("sum"));
    Func* f_sum=fnew_func(p_sum, env);
    Obj o_sum=ffunc(f_sum);
 
    fbind_var(env, "sum", o_sum);
    
    fio_puts(fcall(0, f_main));
    return 0;
}
</pre>

创建函数的那几个函数我自己也看着别扭...不过写C还是老实点好 >_<
