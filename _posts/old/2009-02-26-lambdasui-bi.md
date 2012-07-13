---
layout: post
title: "lambda随笔"
tags: 
- FP
- lambda
- lisp
- "笔记"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

作为人类历史上第二门编程语言, lisp的语法是公认的简单了, 函数套函数, 括号套括号.  不谈应用, 学lisp就是为了娱乐的, 最有意思的东西莫过于lambda, 简单至极又无处不在, 而且几乎无所不能.

<span> </span>

读过sicp的同学们都知道定义函数的define语句

<span> </span>(define (func p1 p2) (...))<span> </span>

是(define func (lambda (p1 p2) (...)))的语法糖, 在lisp中, 函数就是个值, 跟普通变量一样可以传来传去. 

在当前作用域声明变量的define语句

 (define var1 val1)

又是let语句的缩写

<span> </span>(let ((var1 val1)

<span> </span>  (var2 val2))

<span> </span>(...))

而sicp中貌似没说, let语句实际上也就是lambda的语法糖

((lambda (var1 var2)

<span> </span>(...) val1 val2)

可以说, lisp中所谓的变量都是参数的变形罢了.

 

如果高兴, 你可以用lambda来表示pair

<span> </span>(define (cons x y) (lambda (f) (f x y)))

<span> </span>(define (car pair) (pair (lambda (x y) x)))

<span> </span>(define (cdr pair) (pair (lambda (x y) y)))

 

函数式编程的优势高阶函数与惰性编程可都是lambda带来的大礼. lambda返回一个函数而其中的语句并不会立即运行, 通过这一特性可以轻而易举地创造出无限长度的List以及很多不同的玩法, 具体内容请见sicp中关于流的那章.

用cons,car和cdr就可以构造出逻辑判断, 像这样:

(define new-true car)

(define new-false cdr)

(define (new-if condition exp1 exp2)

    (condition (cons exp1 exp2)))
