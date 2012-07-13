---
layout: post
title: "eval, class_eval, instance_eval和binding"
tags: 
- ruby
- trick
- "笔记"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

前些天写html生成器的时候用到了erb，在生成html的时候是这么一句：

<pre lang="erb">
html=tpl.result(binding)
</pre>

binding这个变量（Kernel的一个方法 T_T）有点古怪，就搜了下。它表示了ruby的当前作用域，没有任何对外可见的成员函数，唯一的用途就是传递给eval作第二个参数。因而可以这样：

<pre lang="ruby">
def test_binding
    magic='brother Chun is PURE MAN'
    return binding
end
eval "puts magic", test_binding
</pre>

这样就穿越了一个作用域。

有时可以见到这样的构造函数：
<pre lang="ruby">
a=Baby.new {
    name "Makr"
    father "Mike"
    age 0.2
}
a.cry
</pre>
好处就是好看。实现起来其实也很容易，用instance_eval：

<pre lang="ruby">
class Baby
    def initialize(&blc)
        instance_eval(&blc) #here
    end

    def name(str=nil)
        @name=str if str
        @name
    end
    def age(num=nil)
        @age=num if num
        @age
    end
    def father(str=nil)
        @father=str if str
        @father
    end
    def cry
        puts "#{name} is only #{age.to_s} year old, he wanna milk! Brother Chun is PURE MAN!"
    end
end
</pre>

有重复代码？用class_eval缩短之，有点像宏了：

<pre lang="ruby">
class Baby
    def initialize(&blc)
        instance_eval(&blc)
    end

    def Baby.my_attr(*names)
        names.each{|n|
            class_eval %{
                def #{n}(x=nil)
                    @#{n}=x if x
                    @#{n}
                end
            }
        }
    end

    my_attr :name, :father, :age

    def cry
        puts "#{name} is only #{age.to_s} year old, he wanna milk! Brother Chun is PURE MAN!"
    end
end

a=Baby.new {
    name "Makr"
    father "Mike"
    age 0.2
}
a.cry
</pre>
这里class_eval穿越到了类的作用域，实现了动态添加函数。instance_eval也是，穿越到了实例的作用域，实现修改其内部数据。明白了它们的穿越关系，我们可以实现自己的class_eval和instance_eval——从合适的地方搞到binding就行了。

<pre lang="ruby">
class Baby
    def my_instance_eval(code)
        eval code, binding
    end
    def Baby.my_class_eval(code='')
        eval code, binding
    end
end
</pre>
就这么简单。调用的时候就像这样：
<pre lang="ruby">
class Baby
    def initialize(code)
        my_instance_eval(code)
    end
    my_attr :name, :father, :age
end
a=Baby.new %{
    name "Test"
    father "Orz"
    age 0.2
}
</pre>
刚才省略了一点，那就是class_eval和instance_eval可以接受block代替字符串。搜了下，貌似没找到eval接受block的方法，所以这顶多算是只能eval字符串的山寨class_eval。

update: 想起来ruby中lambda和proc在作用域上的小区别，也就是binding的不同了。proc直接使用原先的binding，lambda继承原先作用域创建一个新的binding。
