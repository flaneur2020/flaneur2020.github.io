---
layout: post
title: "[索引]haskell趣学指南"
tags: 
- FP
- haskell
- "haskell趣学指南"
- lyah
- "翻译"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

<del datetime="2010-02-03T10:39:58+00:00">update: 译言杯具掉了，转移阵地至<a href="http://swdpress.cn/lyah">http://swdpress.cn/lyah</a></del>
update：上面的地址也杯具了，当前可用<a href="http://fleurer-lee.com/lyah">http://fleurer-lee.com/lyah</a>

从决定装逼开始学习haskell以来也有一段时间了，从以前的一头雾水到稍显明朗，也算是进步吧。

很不幸的是haskell的中文内容少之又少，而且内容多为大牛所作，比较艰深，难以理解；幸运的是发现了这个<a href="http://learnyouahaskell.com">学习指南</a>，原作者以浅显诙谐的英文，漂亮的图画描述了haskell的基础概念，让我对haskell终于能够有了一个初步的了解。

决定翻译这个指南是在前个星期（动机很单纯，杀时间），手头还没有电脑，于是先将它打印出来带上电子词典再写到笔记本上，晚自习时倒是有了打发时间的好方法:-) 。现在还在翻译第一章，一方面英文烂，另一方面术语不通，翻译的过程很是困难，难免'有译不了就编"的情况出现，希望大家能够指正。

但愿能够完成吧，能够给使用中文的haskell learner一点帮助会是个不小的动力 :)

拉门<em>[update@09.7.6 我改信意大利飞天面条教了，应该说“拉门”，而不是“阿门”]</em>
<ul>
	<li>第一章 <a href="http://www.yeeyan.com/articles/view/ssword/18894">简介</a></li>
	<li>第二章 入门
<ul>
	<li><a href="http://www.yeeyan.com/articles/view/ssword/18923">各就各位，预备!</a></li>
	<li><a href="http://www.yeeyan.com/articles/view/ssword/19036">启蒙：第一个函数</a></li>
	<li><a href="http://www.yeeyan.com/articles/view/ssword/19212">list入门</a></li>
	<li><a href="http://www.yeeyan.com/articles/view/ssword/19330">Texas Ranges</a></li>
	<li><a href="http://www.yeeyan.com/articles/view/ssword/19624">我是list comprehension</a></li>
	<li><a href="http://www.yeeyan.com/articles/view/ssword/19762">Tuple</a></li>
</ul>
</li>
	<li>第三章 类型与类型类
<ul>
	<li><a href="http://www.yeeyan.com/articles/view/ssword/19807">相信类型</a></li>
	<li><a href="http://www.yeeyan.com/articles/view/ssword/19948">类型变量</a></li>
	<li><a href="http://www.yeeyan.com/articles/view/ssword/20230">类型类101</a></li>
</ul>
</li>
	<li>第四章 函数的语法
<ul>
	<li><a href="http://www.yeeyan.com/articles/view/ssword/50458">模式匹配</a></li>
	<li><a href="http://www.yeeyan.com/articles/view/ssword/50485">注意，Guard！</a></li>
	<li><a href="http://www.yeeyan.com/articles/view/ssword/50543">Where?</a></li>
	<li><a href="http://www.yeeyan.com/articles/view/ssword/50565">Let it be</a></li>
	<li><a href="http://www.yeeyan.com/articles/view/ssword/50619">case表达式</a></li>
</ul>
</li>

<li>第五章 递归
<ul>
	<li><a href="http://www.yeeyan.com/articles/view/ssword/19756">你好，递归！</a></li>
	<li><a href="http://www.yeeyan.com/articles/view/ssword/21165">麦克西米不可思议</a></li>
	<li><a href="http://www.yeeyan.com/articles/view/ssword/24551">几个其他的递归函数</a></li>
	<li><a href="http://www.yeeyan.com/articles/view/ssword/25938">排序,要快！</a></li>
	<li><a href="http://www.yeeyan.com/articles/view/ssword/26460">递归地思考</a></li>
</ul>
</li>

<li>第六章 高阶函数
<ul>
      <li><a href="http://www.yeeyan.com/articles/view/ssword/31762">柯里函数</a></li>
      <li><a href="http://www.yeeyan.com/articles/view/ssword/32390">是时候了，来点高阶函数！</a></li>
      <li><a href="http://www.yeeyan.com/articles/view/ssword/33070">map和filter</a></li>
      <li><a href="http://www.yeeyan.com/articles/view/ssword/50262">lambda</a></li>
      <li><a href="http://www.yeeyan.com/articles/view/ssword/50295">折叠纸鹤(fold)</a></li>
      <li><a href="http://www.yeeyan.com/articles/view/ssword/50366">带$的函数调用</a></li>
      <li><a href="http://www.yeeyan.com/articles/view/ssword/50439">函数组合</a></li>
</ul>
</li>

<li>第七章 模块
<ul>
      <li><a href="http://www.yeeyan.com/articles/view/ssword/27030">装载模块</a></li>
      <li><a href="http://www.yeeyan.com/articles/view/ssword/27851">Data.List</a></li>
      <li><a href="http://www.yeeyan.com/articles/view/ssword/28075">Data.Char</a></li>
      <li><a href="http://www.yeeyan.com/articles/view/ssword/28953">Data.Map</a></li>
      <li><a href="http://www.yeeyan.com/articles/view/ssword/29294">Data.Set</a></li>
      <li><a href="http://www.yeeyan.com/articles/view/ssword/29321">构造你自己的模块</a></li>
</ul>
</li>

<li>第八章 构造自己的类型和类型类
<ul>
       <li><a href="http://www.yeeyan.com/articles/view/ssword/51139">数据类型入门</a></li>
       <li><a href="http://www.yeeyan.com/articles/view/ssword/51217">Record syntax</a></li>
       <li><a href="http://www.yeeyan.com/articles/view/ssword/51750">类型参数</a></li>
       <li><a href="http://www.yeeyan.com/articles/view/ssword/51837">派生实例</a></li>
       <li><a href="http://www.yeeyan.com/articles/view/ssword/53151">类型别名</a></li>
       <li>递归型结构</li>
       <li>类型类 102</li>
       <li>一个yes-no类</li>
       <li>Functor类型类</li>
       <li>Kind和几个类型戏法</li>
</ul>
</li>

<li>第九章 输入输出
<ul>
       <li>hello,world!</li>
       <li>文件和流</li>
       <li>命令行参数</li>
       <li>随机数</li>
       <li>Bytestring</li>
       <li>异常</li>
</ul>
</li>

</ul>
<em>update @ Dec4th</em>：第一章已译完，有趣的东西还在后面:&gt;
<em>update @ Dec12th</em>：前三章已译完，有趣的东西开始了:) 联系上了同时翻译该学习指南的<a href="http://ironflower.blogbus.com/">Ironflower</a>，他已经进行至第四章，目前在探讨如何协作该任务:&gt;
<em>update @ Dec20th</em>：跳过第四章，第五章已经译完。接下来进入第七章:&gt;
<em>update @ Dec27th</em>：纠结马克思的原因，上个星期的翻译暂停了一下。接下来回到正轨:&gt;
<em>update @ Jan12th</em>：纠结冷笑话的原因，前个星期的翻译暂停了一下。第七章完毕。接下来回头翻第六章:&gt;
<em>update @ jun14th</em>：前九章已经低调地译完了...暑假再发布...囧
<em>update @ jul15th</em>：暑假到了...整理中...
