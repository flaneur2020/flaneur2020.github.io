---
layout: post
title: "git使用入门"
tags: 
- "备忘"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

自己写小东西什么的还是git最省心 ^_^

版本控制工具里的命令什么还真够多的，所幸一个人要求不高，平时用的功能也就这么几个了。所以.....忘掉svn吧！ =V=

<b>开始</b>

<pre lang="sh">
git init 
git add .
git commit -a -m "commit message"
</pre>

git init就是创建当前目录的版本库，所有信息都在.git这一个文件夹里面。比起svn的每个目录下边都一个.svn可要清爽多了 :)
git add . 将目录中的所有文件加入跟踪，新建文件时候别忘了这个。
git commit 就是提交啦~注意下这个commit只是提交到本地。

git是跟踪代码变动的工具，而不是上传工具。commit时候需要注意的一点就是不要把二进制什么的文件交进去。记得在根目录下边加个.gitignore文件，内容大约可以这样：

<pre lang="sh">
*.o
*.so
*.a
*.exe
*~
</pre>

<b>代码杯具了怎么办~</b>

<pre lang="sh">
git checkout -- blah.c
</pre>

这样就把blah.c恢复到了commit时的状态，存档复活~

要恢复到之前的版本就用git reset了，git在每次提交的时候都有个hash作为标记，知道这个hash就行了（例如git log或者直接翻github的history之类）

<pre lang="">
git reset --hard 4df38hdf29f
</pre>

（reset好像有三种？初学者不用刨根究底的 =v=）

<b>分支</b>

<pre lang="sh">
git branch blah 
git checkout blah
</pre>

就创建出了一个名为blah的分支了~
git checkout 的功能就是将当前工作目录转向blah~

合并分支...

<pre lang="sh">
git branch master
git merge blah
</pre>

blah就合并进master啦~ 

<b>github</b>

ssh密钥好像比较绕...github使用ssh协议传输东东也做了身份验证嘛，所以需要一个双向的密匙。具体的操作步骤忘了囧，不过照着提示来也就一次性的工作啦~

<pre lang="sh">
git config --global user.name 'Fleurer'
git config --global user.email me.ssword@gmail.com

git remote add origin git@github.com:（你的名字，这里是Fleurer~）/项目名.git  
git push origin master
</pre>
