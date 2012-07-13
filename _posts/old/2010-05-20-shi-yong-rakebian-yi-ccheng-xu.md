---
layout: post
title: "使用rake编译C程序"
tags: 
- C
- rake
- ruby
- "备忘"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

每次修改代码一般只会涉及一部分文件，大部分代码都是不必重复编译的。而且编译明显是分步骤也有依赖关系，比如要测试程序就得先链接出一个可执行文件，要链接就得先编译成.o。。。所以就有了make，自动分析任务的依赖关系，只对有变更的文件执行编译，省心省时间。

不过make的语法晦涩啊...就有了rake

rake提供了file函数可以指明文件的依赖关系，比如：

<pre lang="ruby">
file 'fdict.o' => ['src/fdict.c', 'src/fdict.h'] do
  sh 'gcc -Wall -c src/fdict.c'
end
file 'test.o' => ['src/test.c'] do 
  sh 'gcc -Wall -c src/test.c'
end
</pre>

file就是个ruby的函数调用，把重复的东西去掉很简单

<pre lang="ruby">
CFlags = '-Wall'

[
  ['src/fdict.c', 'src/fdict.h'],
  ['src/test.c']
].each do |fn_c, *_|
  fn_o = File.basename(fn_c).ext('o')
  file fn_o => [fn_c, *_] do
    sh "gcc #{CFlags} -c #{fn_c}"
  end
end
</pre>

再就是链接和执行

<pre lang="ruby">
OFiles = %w{fdict.o test.o}

task :run => [:link] do 
  sh "./test"
end

task :link => OFiles do
  sh "gcc #{CFlags} #{OFiles.join(' ')} -o test"
end

</pre>

每修改两行C程序到控制台下边一个rake run就可以立即执行，这一来写c就有点像写脚本了。
