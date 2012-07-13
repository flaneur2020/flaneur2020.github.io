---
layout: post
title: "Haskell趣学指南！"
tags: 
- FP
- haskell
- "翻译"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

译言前几天被杯具了，还好当初嗅到苗头全都抓了来 -v-

新地址见<a href="http://fleurer-lee.com/lyah">http://fleurer-lee.com/lyah</a>。

用自己写的<a href="http://code.google.com/p/fdoc/">小工具</a>生成的html，parser的代码不到200行，代码写的非常之perl，不过显然还算够用。
代码高亮用的一个jquery插件<a href="http://noteslog.com/chili/">Chili</a>，很容易扩展，随便改两个正则就自制了个<a href="http://fleurer-lee.com/lyah/js/chili/code.js">haskell高亮</a>。

顺便校对了下翻译。
<ul>
	<li>ChinaUnix上的朋友对“柯里函数”等译法的意见比较大，不过在校对中没有做修改。关于人名构成的术语，例如“Fourier transform”还是“傅立叶变换”，译者认为后者更好看。</li>
	<li>“Function Application”直译作“函数应用”，在这里译为“函数调用”。相应的“partial application”直译作“部分应用”，在这里译为“不全调用”。</li>
	<li>“predicate”在这里译为“限制条件”，因为译者认为“断言”这个词有点吓人。</li>
	<li>保留了List，Tuple，List Comprehension等名词，不过将Triple之类译为“三元组”，“function with two parameters”译为“二元函数”。</li>
	<li>把原先译文中“在处理数字时是非常有用的”类的句式改为“在处理数字时非常有用”，“的”真的是很别扭的。</li>
	<li>有一节的标题“Texas Range”译为“德州区间”，因为译者老家在德州...囧</li>
</ul>
修改的比较仓促，bug依然还有很多。呵呵，欢迎提意见！ ^_^

update: 可以svn checkout它的源码：https://lyah-cn.googlecode.com/svn/trunk/
