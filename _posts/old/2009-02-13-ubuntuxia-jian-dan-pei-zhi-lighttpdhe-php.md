---
layout: post
title: "ubuntu下简单配置lighttpd和php"
tags: 
- lighttpd
- php
- ubuntu
- "备忘"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

弄个调试环境而已, 简单整下就是了..

先安装lighttpd和php, 有apt-get, 简单无比
sudo apt-get install lighttpd
sudo apt-get install php5-cgi

接着配置.
1.为了方便在windows下修改, 就把文件夹新建在d:盘
cd /media/DISK_VOL2/
mkdir server
2.创建lighttpd配置文件lighty.conf, 可以在<a href="http://redmine.lighttpd.net/attachments/659/lighttpd.conf">这里</a>下载一份基本的配置文件, 然后自己按需修改.
打开gvim把24行左右"mod_cgi"的注释删掉, 再把server.document-root  ,server.errorlog, accesslog.filename  等属性的值修改为刚才的目录,
3.server.port的默认值是80, 而80端口必须得是root权限才能打开. 如果仅仅搭调试环境的话设成3000就是了, 启动时省个sudo.
4.把fast_cgi那块的注释全都去掉, 修改如下
<pre LANGUAGE="php" line="1">
fastcgi.server = ( ".php" =>
                     ( "localhost" = (
                           "host" => "127.0.0.1",
                           "port" => 521
                           )
                      )
                  )
</pre>
好的, 可爱的lighty.conf配置基本完毕.

打开php-cgi
php5-cgi -b 127.0.0.1:521
打开lighty:
lighttpd -D -f lighttpd.conf

嗯, phpinfo测试一下:
echo '' &gt; test.php

打开浏览器, http://localhost:3000/test.php

如果无错误提示, 那貌似就可以用了
