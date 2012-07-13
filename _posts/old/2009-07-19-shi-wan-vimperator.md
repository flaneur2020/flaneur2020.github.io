---
layout: post
title: "试玩vimperator"
tags: 
- firefox
- trick
- vim
- vimperator
- "备忘"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

以前貌似在豆瓣看到有人提起过这vim键操作firefox的插件，当时的第一反应就是蛋疼：有人用导线连了一台cpu，有人只用emacs上网，有人用《最后的晚餐》里的面包谱曲子，有人的妈妈找他回家吃饭...拉门，我确实都想到一块去了。

好吧，暑假在家无所事事，我又蛋疼了，装vimperator。简单熟悉一下后发现这东西还是很成熟的，除去猎奇者，拿它日常使用的也大有人在。装上之后ff的导航栏和书签栏都消失了，习惯chrome的同学们可能会找到一点熟悉的感觉，反正清爽多了。再就是一些快捷键绑定，ctrl+c ctrl+v等键全变了，唔，这不爽。还好，可以自定义.vimperatorrc修改键绑定。同vim一样，vimperator也内置了强大的帮助，即:help，该有的options里面貌似都有了。

进入vimperator的ff之后，执行命令:mkv将当前vimperator的配置保存到$HOME/.vimperatorrc里，然后就可以diy了。参考<a href="http://pchu.blogbus.com/logs/36870556.html">大牛的配置</a>，修改了个简化版如下：
<pre lang="vim">
" 先把麻烦的东西拿掉
map A <nop>
map <c-q> <nop>
map <c-o> <nop>
map <c-i> <nop>
map <c-z> <nop>
map <c-p> <nop>

" show toolbar & scrollbar
set guioptions=Tr

" 前进后退
noremap q :back<cr>
noremap w :forward<cr>

" search
noremap <c-f> /

" close tab
noremap c :q<cr>

" F5
noremap <f5> :reload<cr>

" 解决全选、复制、粘帖、剪切和撤销与vimperator冲突的问题
noremap <c-V> <c-v>
noremap <c-Z> <c-z>
noremap <c-c> <c-v><c-c>
noremap <c-a> <c-v><c-a>
cnoremap <c-c> <c-v><c-c>
cnoremap <c-v> <c-v><c-v>
cnoremap <c-x> <c-v><c-x>
inoremap <c-a> <c-v><c-a>
inoremap <c-c> <c-v><c-c>
inoremap <c-v> <c-v><c-v>
inoremap <c-x> <c-v><c-x>
inoremap <c-z> <c-v><c-z>
inoremap <c-y> <c-v><c-y>


set titlestring=Mozilla Firefox

" 齐全的next和previous……
set nextpattern=\s*下一页|下一张|下一篇|下一????下页|后页\s*,^\bnext\b,\bnext\b,\bsuivant\b,^>$,^(>>|??????|??)$,^(>|??),(>|??)$,\bmore\b
set previouspattern=\s*上一页|上一张|上一篇|上一????上页|前页\s*,^\bprev|previous\b, \bprev|previous\b,\bprécédent\b,^<$,^(<<|??????|??)$,^(<|??),(<|??)$

" PassThrough gmail and greader
autocmd LocationChange .* js modes.passAllKeys = /.*(mail\.google\.com|www\.google\.com\/reader).*/.test(buffer.URL)

" Commands
" noimg可以减少流量，nojs用于调控一些不听话的网页
command noimg set! permissions.default.image=2
command ysimg set! permissions.default.image=1
command nojs set! javascript.enabled=false
command ysjs set! javascript.enabled=true
</pre>

虽然痛恨配置，不过确实也是一劳永逸的东西。只要快捷键不要冲突，vimperator里设置的这些键还是很顺手的。看文档说貌似还可以用javascript扩展vimperator命令，想起来mozilla官方搞的那个<a href="http://labs.mozilla.com/projects/ubiquity/">Ubiquity</a>，感觉两个东西貌似有点像！或许vimperator还要更强大些？ :p
