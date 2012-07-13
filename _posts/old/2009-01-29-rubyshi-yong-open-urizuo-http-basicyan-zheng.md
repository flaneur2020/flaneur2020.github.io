---
layout: post
title: "ruby使用open-uri做http basic验证"
tags: 
- http-basic
- open-uri
- ruby
- "备忘"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

如今的web api基本上都是用http basic作为验证方式, 在浏览器中的话只要将用户名和密码用和一个@放到url的前面即可,如http://ssword:password@somedomain.com , 使用curl也是简单无比, 如饭否api的示例, 它可以验证你的身份并获得你好友的信息, 若密码错误, 就得到一个401的错误

curl -u loginname:password http://api.fanfou.com/statuses/friends_timeline.rss

在ruby中的解决方法有很多, 只是我不喜欢封装的库函数, 记不住那东西. http basic验证只是http协议的一部分嘛, 简单地加个http头不就行了? 再使用open-uri的话就更简单不过了, 它允许你像打开本地文件那样打开uri的内容, 并可以添加http头, 十分方便, 如下:

require 'open-uri'
require 'base64'
f = open(url,"Authorization"=>"Basic #{Base64.b64encode("myaccount:mypassword")}")
puts f.read
