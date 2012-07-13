---
layout: post
title: note on ruby
tags: 
- ruby
- trick
- "备忘"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

Ruby和python这两门语言貌似不应该有可比性。C\C++出身的程序员们都喜欢Python带来的生产力，而ruby被发明则是为了取代perl----做为脚本语言。脚本语言么，效率不必太高，要容易书写，容易读貌似倒不是很重要…瞧瞧bash和perl吧。而Ruby强大的表达能力就弥补了可读性的不足，再加上<em><strong>已经</strong></em>足够多的第三方库（gems），可以算是一门理想的语言了。可惜就这门语言而言（不谈rails），在国内（国外github上ruby可是第一语言啊）一直没能很成气候呢（CpyUG这个社区太强大了，ruby在国内就没有对应物）。貌似大家都在搞rails的敏捷开发，没人搞ruby？

呃，跑题了。这两天用ruby写点小玩具，记几个小tricks就是了。玩具而已。玩具而已。

判断类型，可以这样obj.is_a? Array 。如果不喜欢问号，可以Array === obj，貌似要好看点？注意的地方就是类型对象要放在前面。

单元测试是好东西，一般源码会位于src文件夹，而测试文件会放在test文件夹。测试文件需要require源码的文件，这就要修改require的路径，要不然会乱套。也就是这样：

<pre lang="ruby">
$:.unshift "../src"
</pre>
update：$:其实是个全局数组。require的路径皆来自于此。

正则表达式，模式匹配。好吧，玩haskell后遗症。
<pre lang="ruby">
test=
    case content
    when /^abc(.*)/
         $1
    when /^abcd(.*)/
         $1
     end
</pre>
ruby里所有的语句都是表达式。正则表达式匹配的时候会修改几个全局变量，
$& 整个匹配
$1 第一个匹配
$2 第二个匹配
$` 位于匹配前的字符串
$’ 位于匹配后的字符串

很明显，都是继承自perl。Perl那堆带美元符号的变量可是饱受扣病啊，在《the ruby way》这书里作者貌似刻意地回避这种全局变量的使用。不过自己看着办就好了。

谈到模式匹配，haskell可以用(x:xs)这样的模式切开一个list得到首元素。Ruby可以这样：

<pre lang="ruby">
x, *xs=[1,2,3,4]
</pre>

在变量名前面加*的含义貌似就是将它里面的元素看作一个整体：
<pre lang="ruby">
def add(a,b)
a+b
end
paras=[1,2]
add(*paras)
</pre>
在处理可变长度参数时候也是如此：

<pre lang="ruby">
def add(*paras)
paras.inject{|acc,i| acc+=i}
end
</pre>
呃，inject就是python的reduce，haskell的fold。

block在ruby中应该是无所不在了，也算是函数式吧，不过搞的比较自然，谁也不会想到那儿去。（话说，貌似有不少同学都是在python的那个lambda关键字才了解到函数式编程的…ruby倒也有个lambda，不过只是个Proc类的实例，貌似没谁用 - -！）。在带block的函数递归的时候一定要记得传递这个block。

<pre lang="ruby">
def travel(arr, &block)
  x,*xs=arr
  block.call(x)
  travel(xs, &block)
end
</pre>
