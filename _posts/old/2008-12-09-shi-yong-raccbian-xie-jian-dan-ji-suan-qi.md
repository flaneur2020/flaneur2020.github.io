---
layout: post
title: "使用racc编写简单计算器"
tags: 
- racc
- ruby
- "笔记"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

简单试用了下racc，感觉很是不爽。没找到合适的词法分析工具，于是就先用正则过一遍，再交给racc再过一遍，此不爽之一；由于ruby中$1,$2都是变量，为了避免冲突在这里改成了val[0],val[1]，$$变成了result，多打不少字先不说，可读性是非常的差，此不爽之二；vim的代码高亮对它不起作用，嵌入代码会别扭的很，此不爽之三。

牢骚完毕，步入正题。racc作为ruby下的语法分析并代码生成工具，可以使用类似yacc的语法来生成文本分析代码。它需要取一组token，按照bnf范式读出其中的内容，用你自己内嵌的代码来分析它。从取token开始，在yacc貌似是通过yylex()来取得token，而这个yylex函数可以用lex按照正则表达式自动生成。很遗憾，没找到ruby下对应的词法分析工具。不过还好，还有正则表达式可以使用，将token先取出来放到一个array中再让racc一次取一个，也就意味着词法分析和语法分析得分步进行，而不能像yacc+lex那般紧密。

按照<a href="http://i.loveruby.net/en/projects/racc/doc/parser.html">这个文档</a>里的说法，racc会生成一个继承自Parser的类，这个类中含有一个next_token的抽象方法，你可以继承这个类以提供它的实现，作为词法分析和语法分析的接口，与yacc的yylex函数类似。

如下是语法文件代码，参考自http://bbs.chinaunix.net/viewthread.php?tid=879956 ：
<pre lang="ruby">
class MyParser
token #声明token
  NUMBER #这里只有一个token，在词法分析器使用符号表示，即:NUMBER

prechigh #运算符优先级
  nonassoc UMINUS #负号与减号相同,这里用另一个符号注明
  left     '*' '/' #左结合，表示1+2+3这个表达式应该这样算：(1+2)+3，即先取左边后取右边
  left     '+' '-'
preclow

rule #描述语法，BNF范式
  exp: '(' exp ')' { result = val[1] }
    | exp '+' exp { result = val[0].to_f + val[2].to_f }
    | exp '-' exp { result = val[0].to_f - val[2].to_f }
    | '-' exp = UMINUS { result = -val[1].to_i }
    | exp '*' exp { result = val[0].to_f * val[2].to_f }
    | exp '/' exp { result = val[0].to_f / val[2].to_f }
    | NUMBER
end</pre>

保存为parser.y, 执行racc parser.y，然后racc就会生成一个名为parser.tab.rb的文件，里面有一个名为MyParser的类，你可以继承或者修改这个类。

感谢ruby的面向对象机制，你可以在另个文件中直接修改这个类

<pre lang="ruby">require 'parser.tab.rb'
class MyParser
    def parse(text)
       @tokens = get_tokens text
       do_parse
    end
    def next_token #词法分析和语法分析的接口
       @tokens.shift
    end
    def get_tokens(text) #词法分析
        reg=/+|*|-|/|d+|(|)/ #正则表达式
        tokens=[]
        text.scan(reg) do |t|
           tokens < < case t
               when /d+/ #如果是数字，这个token就是:NUMBER
                     [:NUMBER,t]
               else
                     [t,t] #我也不知道为什么非得这样
               end
         end
     return tokens
     end
end</pre>


测试代码如下：
</pre><pre lang="ruby">
parser = MyParser.new
p parser.parse(ARGV[0])
</pre>

保存为test.rb

如下是运行效果：
<blockquote>
$ ruby test.rb "12/(1+9)-3"
-1.8



<em>update@dec13th</em>孤陋寡闻了，其实过一遍文本是完全可以的，见<a href="http://dev.csdn.net/Develop/article/28/77129.shtm">这里</a>，使用了$`，不过感觉怪怪的~看来要对正则加深理解才行！</blockquote>
