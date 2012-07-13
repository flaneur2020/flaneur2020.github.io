---
layout: post
title: "在cygwin下编译了ruby和racc"
tags: 
- cygwin
- ruby
- "备忘"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

前段时间由于没找到传说中的《<a href="http://www.douban.com/subject/1134994/?i=0">龙之书</a>》，草草翻了下《<a href="http://www.douban.com/subject/1088057/">编译原理及实战</a>》中对yacc和lex用法的简单介绍，的确是很神奇的东西。

由于是在win32下，而且对C不甚熟悉，心想在ruby中应该也有类似yacc的工具吧，google了一下，找到了<a href="http://i.loveruby.net/en/projects/racc/">racc</a>，于是天真地下了来，里面有个setup.rb，运行之，提示在system("nmake")处出现错误。翻了下它的目录，里面有一堆.c文件（为啥没有纯ruby的cc，囧），然后到网上找到了nmake.exe，扔进windows文件夹之后又提示缺少cl.exe，找到cl.exe后又弹出了个alert,提示缺少MSPDB41.DLL，折腾。根据shiningray老师的说法，这几个文件貌似都是windows platform sdk那一套，而且据说在windows下编译成功的概率会很低，得，装cygwin。

在gougou上可以搜到不少cygwin的下载地址，但若是处于编译软件目的话，还是推荐下载那个600m的cygwin，里面的devel包要全。下载完毕，解压，安装。先设置一个root目录，在cygwin下这就是/，c d e f 盘则在/cygdrives之下，然后选择包的目录（解压文件夹即可），一路next。

貌似这个包的ruby版本比较低？到ruby-lang.org下载个ruby源代码，解压,./configure make make install一路搞定，回到racc的源代码目录，执行

<blockquote>
ruby setup.rb config
ruby setup.rb setup
ruby setup.rb install</blockquote>

quote:在运行ruby时可能会有<blockquote>/usr/bin/ruby: no such file to load -- ubygems (LoadError)</blockquote>的错误提示,执行<blockquote>unset RUBYOPT</blockquote>

即可
好的，racc安装成功

<blockquote>Administrator@NEWS-4643EE5FFC /cygdrive/e/test
$ racc
racc: no grammar file given
Usage: racc [options] <grammar file>
Options:
  -g,--debug                  output parser for user level debugging
  -o,--output-file <outfile>  file name of output [<fname>.tab.rb]
  -e,--executable <rubypath>  insert #! line in output ('ruby' to default)
  -E,--embedded               output file which don't need runtime
  -l,--no-line-convert        never convert line numbers (for ruby< =1.4.3)
  -c,--line-convert-all       convert line numbers also header and footer
  -a,--no-omit-actions        never omit actions
  -v,--verbose                create <filename>.output file
  -O,--log-file <fname>       file name of verbose output [</fname><fname>.output]
  -C,--check-only             syntax check only
  -S,--output-status          output status time to time
  --version                   print version and quit
  --runtime-version           print runtime version and quit
  --copyright                 print copyright and quit</fname></rubypath></fname></outfile></grammar></blockquote>

racc的文档不多，不过无非也就是yacc在ruby下的翻版，参考一下<a href="http://i.loveruby.net/en/projects/racc/">这个文档</a>再加一点yacc方面的资料应该就足够了
