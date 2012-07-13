---
layout: post
title: "理解Y组合子"
tags: 
- FP
- lambda
- Y-combinator
- "笔记"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

众所周知，lambda演算通过递归就可以图灵完备。好，用纯lambda演算写个递归吧。

等等，要递归必须得有名字，而lambda演算里赋予名字的唯一方式就是传递参数。像lisp那样define是不行的，只能这样绕个圈子：

<pre lang="lisp">
(\f.\x.
    (if  (= x 1)
        1
        (* x (f f  (- x 1)))))
</pre>

这里把函数本身作为第一个参数传递给自己，从而实现的递归。要调用这个递归函数，还得套一个let（当然，换成lambda形式）:

<pre lang="lisp">
((\fac. (fac fac 5))
        (\f.\x.
              (if (= x 1)
                    1
                    (* x (f f (- x 1))))))

</pre>

很难看吧，每次递归都得把自己当作参数传递一遍，也很机械，重复性的活不应由人类做。想下，如果将递归函数里的(f f (- x 1))换成(f (- x 1))，或许还可以接受…好，Y组合子应运而生，现在你可以这样自然地递归了：

<pre lang="lisp">
Y (\f.\x
     (if (= x 1)
          1
          (* x (f (- x 1))))) 5
</pre>

数学家在追求美感上可是不遗余力啊。不过Y是如何做到的？

想想，Y组合子又叫不动点函数。什么是不动点？x=f(x)=f(f(x))…，这个x就是不动点：不管套多少层函数调用，在不动点上的值总是相等。Y f = f (Y f)=f (f (Y f))，这个Y f就是个不动点，高阶函数的不动点。什么是组合子？很简单，可以柯里化、没有自由变量的函数就是组合子 :)

便于理解，我们给(\f.\x (if (= x 1) 1 (* x (f (- x 1)))))这个lambda一个名字fac，看看一步步的递归是怎么来的：

<pre lang="lisp">
Y fac 3
> fac (Y fac) 3         	//transform, Y!
> 3 * ((Y fac) 2)        	//3 !=1 so recurse
> 3 * (fac (Y fac) 2)    	//Y transform again
> 3 * (2 * ((Y fac) 1))       //2 !=1 so recurse
> 3 * (2 * (fac (Y fac) 1))   	//Y transform again
> 3 * (2 * 1)             	//1=1, so recursion ends.
> 6
</pre>

就是这样了。里面有柯里化，也有惰性求值（缺一不可！）。一环套一环，然后就递归了。

Y组合子的定义：Y = \y. (\x.y (x x)) (\x.y (x x))，天知道大神（大神的名字叫做Haskell Curry! -v-）是怎么想出来的 =v=

不妨自己在纸上推倒一下(也只能在纸上推倒，这东西在实际的编程中貌似是没有应用的 :D)
