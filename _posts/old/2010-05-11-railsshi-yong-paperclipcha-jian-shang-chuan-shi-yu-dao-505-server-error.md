---
layout: post
title: "rails使用paperclip插件上传时遇到500 Internal Server Error"
tags: 
- Rails
- ruby
- "备忘"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

有可能是imageMagick没装利索。找不到imagemagick，显示错误信息的时候试图把上传的文件对象序列化，就505了。

<pre lang="shell">sudo apt-get install imagemagick --fix-missing</pre>

使用三方库就得做好不可预料事件的准备呢。

关于paperclip的使用，这两篇简介好像不错：

- <a href="http://jimneath.org/2008/04/17/paperclip-attaching-files-in-rails/">http://jimneath.org/2008/04/17/paperclip-attaching-files-in-rails/</a>
- <a href="http://thewebfellas.com/blog/2008/11/2/goodbye-attachment_fu-hello-paperclip">http://thewebfellas.com/blog/2008/11/2/goodbye-attachment_fu-hello-paperclip</a>

ps: 又遇到了个没预料的问题，上传validates_attachment_content_type指定的类型之外的文件时候同样会遇到个500 internal server error，未解中。
