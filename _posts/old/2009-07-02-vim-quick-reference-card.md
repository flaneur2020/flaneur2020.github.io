---
layout: post
title: vim quick reference card
tags: 
- trick
- vim
- "翻译"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

原地址：<a href="http://tnerual.eriogerg.free.fr/vim.html">http://tnerual.eriogerg.free.fr/vim.html</a>
翻译：ssword

这里的排版弄的挺难看的，大家如果觉得有用，就将就下吧。

<strong>Basic movement
基本移动
</strong>
<table border="1">
<tr><td>h l k j</td><td>左移、右移一个字符；上一行，下一行；</td></tr>
<tr><td>b w</td><td>左移、右移一个词元</td></tr>
<tr><td>ge e</td><td>左移、右移到一个词元末尾</td></tr>
<tr><td>{  }</td><td>前、后移动一个段落</td></tr>
<tr><td>( )</td><td>前、后移动一个句子</td></tr>
<tr><td>0 ^  $</td><td>句子的开头、首字符、最后一个字符</td></tr>
<tr><td>nG ngg</td><td>第n行的句首、句尾</td></tr>
<tr><td>n%</td><td>文件的n%处（n不可省略）</td></tr>
<tr><td>n|</td><td>当前行的第n列</td></tr>
<tr><td>%</td><td>匹配下一个括号，中括号，大括号，注释或#define</td></tr>
<tr><td>nH nL</td><td>当前窗口从顶数、从底数第n行</td></tr>
<tr><td>M</td><td>当前窗口的中间行</td></tr>

</table><br />

<strong>Insertion & replace→ insert mode
插入&替换 ->插入模式</strong>

<table border="1">
<tr><td>i a</td><td>在光标前、后插入</td></tr>
<tr><td>I A</td><td>在当前行前、后插入</td></tr>
<tr><td>gI</td><td>在当前行的第一列插入</td></tr>
<tr><td>o O</td><td>在当前行的上、下插一新行</td></tr>
<tr><td>rc</td><td>把光标下的字符替换为c</td></tr>
<tr><td>grc</td><td>同rc相似，不过不影响布局</td></tr>
<tr><td>R</td><td>替换从光标往后的所有字符</td></tr>
<tr><td>gR</td><td>同R相似，不过不影响布局</td></tr>
<tr><td>cm</td><td>修改到移动命令m指向的位置</td></tr>
<tr><td>cc or S</td><td>修改当前行</td></tr>
<tr><td>C</td><td>修改到行尾</td></tr>
<tr><td>s</td><td>修改一个字符，并进入插入模式</td></tr>
<tr><td>~</td><td>转换当前字符的大小写，并右移光标</td></tr>
<tr><td>g~m</td><td>转换字符大小写到移动命令m指向的位置</td></tr>
<tr><td>gum gUm</td><td>将移动命令m中间的字符转换为大写、小写</td></tr>
<tr><td><m>m</m></td><td>按照移动命令m，左移、右移缩进</td></tr>
<tr><td>n< < n>></td><td>左移、右移n个缩进</td></tr>

</table><br />

<strong>Deletion
删除</strong>

<table border="1">
<tr><td>x X</td><td>删除当前字符、前一个字符</td></tr>
<tr><td>dm</td><td>删除到移动命令m指向的位置</td></tr>
<tr><td>dd D</td><td>删除当前行</td></tr>
<tr><td>J gJ</td><td>将当前行与下一行合并</td></tr>
<tr><td>:rd↵</td><td>删除区间内的文本</td></tr>
<tr><td>:rdx↵</td><td>删除区间内的文本，并将其存入寄存器x</td></tr>

</table><br />

<strong>Insert mode
插入模式</strong>

<table border="1">
<tr><td>^Vc ^Vn</td><td>逐字插入字符c、数值n</td></tr>
<tr><td>^A</td><td>在当前输入的文本前插入</td></tr>
<tr><td>^@</td><td>同^A,并退出到命令模式</td></tr>
<tr><td>^Rx ^R^Rx</td><td>逐字)插入x寄存器的内容</td></tr>
<tr><td>^N ^P</td><td>在光标前、光标后自动完成</td></tr>
<tr><td>^W</td><td>删除光标前一个单词</td></tr>
<tr><td>^U</td><td>删除前面输入的文本</td></tr>
<tr><td>^D ^T</td><td>左移、右移一个缩进</td></tr>
<tr><td>^Kc1c2 or c1←c2</td><td>插入一个digraph</td></tr>
<tr><td>^Oc</td><td>进入临时命令模式执行c</td></tr>
<tr><td>^X^E ^X^Y</td><td>向上、向下滚动</td></tr>
<tr><td><esc></esc></td><td>回到命令模式</td></tr>
</table><br />

<strong>Copying
复制</strong>

<table border="1">
<tr><td>"x</td><td>将下个用于删除、复制、粘贴的寄存器设置为x</td></tr>
<tr><td>:reg↵</td><td>显示所有寄存器的内容</td></tr>
<tr><td>:reg x↵</td><td>显示x寄存器的内容</td></tr>
<tr><td>ym</td><td>复制移动命令m之间的文本</td></tr>
<tr><td>yy or Y</td><td>复制当前行</td></tr>
<tr><td>p P</td><td>将寄存器中的文本粘贴到光标前、后</td></tr>
<tr><td>]p [p</td><td>同p P，并自动处理缩进</td></tr>
<tr><td>gp gP</td><td>同p P，并保留光标。</td></tr>
</table><br />

<strong>Advanced insertion
高级插入</strong>

<table border="1">
<tr><td>g?m</td><td>对移动命令m中间的文本进行rot13加密</td></tr>
<tr><td>n^A n^X</td><td>对移动命令m中间的文本进行rot13加密</td></tr>
<tr><td>gqm</td><td>将移动命令m中间的多行文本格式化为同一宽度</td></tr>
<tr><td>:rce w↵</td><td>将区间r中的内容以w为宽度居中</td></tr>
<tr><td>:rle i↵</td><td>将区间r中的内容以i个缩进左对齐</td></tr>
<tr><td>:rri w↵</td><td>将区间r中的内容以w为宽度右对齐</td></tr>
<tr><td>!mc↵</td><td>以c命令过滤处理移动命令m中间的文本</td></tr>
<tr><td>n!!c↵</td><td>以c命令过滤处理n行文本</td></tr>
<tr><td>:r!c↵</td><td>以c命令处理区间r中间的文本</td></tr>
</table><br />

<strong>Visual mode
可视模式</strong>

<table border="1">
<tr><td>v V ^V</td><td>开始、退出选取文本</td></tr>
<tr><td>o</td><td>将光标移动到选取文本的开始</td></tr>
<tr><td>gv</td><td>回到上一个选取的文本</td></tr>
<tr><td>aw as ap</td><td>选取一个单词、句子、段落</td></tr>
<tr><td>ab aB</td><td>选取一个()、{}块</td></tr>
</table><br />

<strong>Undoing, repeating & registers
撤销，重复和寄存器</strong>

<table border="1">
<tr><td>u U</td><td>撤销上一个命令，返回上一个修改的行</td></tr>
<tr><td>.  ^R</td><td>重复上一个修改，重复上一个撤销</td></tr>
<tr><td>n. </td><td>重复上一个修改n次</td></tr>
<tr><td>qc qC</td><td>记录、追加输入的字符到寄存器c</td></tr>
<tr><td>q</td><td>停止记录</td></tr>
<tr><td>@c</td><td>执行寄存器c的内容</td></tr>
<tr><td>@@</td><td>重复执行上一个@命令</td></tr>
<tr><td>:@c↵</td><td>把寄存器c作为Ex命令执行</td></tr>
<tr><td>:rg/p/c↵</td><td>对区间r中匹配p的文本执行Ex命令c</td></tr>
</table><br />

<strong>Complex movement
高级移动</strong>

<table border="1">
<tr><td>- +</td><td>上移，下移到行首</td></tr>
<tr><td>B W</td><td>按空格左移、右移一个词元</td></tr>
<tr><td>gE E</td><td>按空格左移、右移到一个词元的末尾</td></tr>
<tr><td>n</td><td></td><td>下移n-1行到行首</td></tr>
<tr><td>g0 gm</td><td>当前行首、行中央</td></tr>
<tr><td>g^  g$</td><td>当前行的首字符、尾字符</td></tr>
<tr><td>gk gj</td><td>上移、下移</td></tr>
<tr><td>fc Fc</td><td>下一个、前一个字符c</td></tr>
<tr><td>tc Tc</td><td>下一个、前一个字符c的前面</td></tr>
<tr><td>; ,</td><td>重复上一个fFtT操作，反方向</td></tr>
<tr><td>[[ ]]</td><td>上一个、下一个节开头 </td></tr>
<tr><td>[] ][</td><td>上一个、下一个节结尾</td></tr>
<tr><td>[( ])</td><td>上一个、下一个未关闭的括号</td></tr>
<tr><td>[{  ]}</td><td>上一个、下一个未关闭的大括号</td></tr>
<tr><td>[m ]m</td><td>上一个、下一个java方法的开头</td></tr>
<tr><td>[# ]#</td><td>上一个、下一个未关闭的#if #else #endif</td></tr>
<tr><td>[* ]*</td><td>上一个、下一个/* */的开头和结尾</td></tr>
</table><br />

<strong>Search & substitution
搜索 &替换</strong>

<table border="1">
<tr><td>/s↵  ?s↵</td><td>向前、向后搜索s </td></tr>
<tr><td>/s/o↵  ?s?o↵</td><td>按o个偏移向前、向后搜索s</td></tr>
<tr><td>n or /↵</td><td>向前重复上一个搜索</td></tr>
<tr><td>N or ?↵</td><td>向后重复上一个搜索</td></tr>
<tr><td># *</td><td>向前、向后搜索当前词元</td></tr>
<tr><td>g# g*</td><td>同上，额外匹配不完整的词元</td></tr>
<tr><td>gd gD</td><td>当前符号的局部、全局定义</td></tr>
<tr><td>:rs/f/t/x↵</td><td>将区间r中匹配f的文本替换为t </td></tr>
<tr><td>:rs x↵</td><td>以新的区间和x重复替换</td></tr>
</table><br />

<strong>Special characters in search patterns
模式匹配中的特殊字符</strong>

<table border="1">
<tr><td>.   ^  $</td><td>任一字符，行的首字符，尾字符</td></tr>
<tr><td>\< \></td><td>单词的开头，结尾</td></tr>
<tr><td>[c1-c2]</td><td>在c1..c2之间的任一字符</td></tr>
<tr><td>[^c1-c2]</td><td>不在c1..c2之间的任一字符</td></tr>
<tr><td>\i \k \I \K</td><td>标志符，关键字；字母，数字</td></tr>
<tr><td>\f \p \F \P</td><td>文件名；可打印字符；字母；数字</td></tr>
<tr><td>\s \S</td><td>空格；非空字符</td></tr>
<tr><td>\e \t \r \b</td><td>←>键<esc>, <tab>, < ?>, < ←> </tab></esc></td></tr>
<tr><td>\= * \+</td><td>匹配0个或一个、0个或多个、一个或多个模式   </td></tr>
<tr><td>\|</td><td>两个选择</td></tr>
<tr><td>\( \)</td><td>将一组模式组合成一个</td></tr>
<tr><td>\& \n</td><td>匹配全部、匹配第n个括号中的内容 *</td></tr>
<tr><td>\u \l</td><td>匹配下一个大写、小写字母</td></tr>
<tr><td>\c \C</td><td>忽略、匹配下一个模式</td></tr>
</table><br />

<strong>Offsets in search commands
偏移</strong>

<table border="1">
<tr><td>n or +n</td><td>下n行的第1列</td></tr>
<tr><td>-n</td><td>上n行的第1列</td></tr>
<tr><td>e+n e-n</td><td>匹配文本结尾右边、左边的第n个字符</td></tr>
<tr><td>s+n s-n</td><td>匹配文本右边开头右边、左边的第n个字符</td></tr>
<tr><td>;sc</td><td>向下执行搜索命令sc</td></tr>
</table><br />

<strong>Marks and motions
标记 &跳转 </strong>

<table border="1">
<tr><td>mc,c∈[a..Z]</td><td>把当前位置标记为c，c∈[a..Z]</td></tr>
<tr><td>`c `C</td><td>跳到当前文件、任意文件的c标记</td></tr>
<tr><td>`0..9</td><td>跳到上一个位置</td></tr>
<tr><td>`` `"</td><td>跳到上一个位置，上一次编辑的位置</td></tr>
<tr><td>`[ `]</td><td>跳到上一个修改段落的开头、结尾</td></tr>
<tr><td>:marks?</td><td>输出可用的标记列表</td></tr>
<tr><td>:jumps?</td><td>输出跳转列表</td></tr>
<tr><td>n^O</td><td>跳到跳转列表的前一个位置</td></tr>
<tr><td>n^I</td><td>跳到跳转列表的后一个位置</td></tr>

</table><br />

<strong>Key mapping & abbreviations
键映射 &缩写</strong>

<table border="1">
<tr><td>:map c e↵</td><td>在普通模式和可见模式中将c映射为e</td></tr>
<tr><td>:map!  c e↵</td><td>在插入模式和命令模式中将c映射为e</td></tr>
<tr><td>:unmap c↵  :unmap!  c↵</td><td>移除映射c</td></tr>
<tr><td>:mk f↵</td><td>将当前的映射和设置写入到文件f</td></tr>
<tr><td>:ab c e↵</td><td>把e设置为c的别名</td></tr>
<tr><td>:ab c↵</td><td>显示c开头的所有别名</td></tr>
<tr><td>:una c↵</td><td>移除别名c</td></tr>
</table><br />

<strong>Tags
标签</strong>

<table border="1">
<tr><td>:ta t↵</td><td>跳到t匹配的tag</td></tr>
<tr><td>:nta↵</td><td>跳到列表中后面第n个tag</td></tr>
<tr><td>^] ^T</td><td>跳到光标指向的tag，从tag返回</td></tr>
<tr><td>:ts t↵</td><td>列出匹配的tag供选择跳转</td></tr>
<tr><td>:tj t↵</td><td>跳到标签t，如果有多个匹配则提示选择</td></tr>
<tr><td>:tags↵</td><td>显示tag列表</td></tr>
<tr><td>:npo↵  :n^T↵</td><td>向前返回、跳至第n个tag </td></tr>
<tr><td>:tl↵</td><td>跳到最后一个匹配的tag</td></tr>
<tr><td>^W}  :pt t↵</td><td>当前光标指向的前一个tag、 t匹配的tag</td></tr>
<tr><td>^W]</td><td>分割窗口，显示当前光标指向的tag </td></tr>
<tr><td>^Wz or :pc↵</td><td>关闭显示tag的窗口</td></tr>

</table><br />

<strong>Scrolling & multi-windowing
滚动 &多窗口</strong>

<table border="1">
<tr><td>^E ^Y</td><td>向上、向下滚动一行</td></tr>
<tr><td>^D ^U</td><td>向上、向下滚动半页</td></tr>
<tr><td>^F ^B</td><td>向上、向下滚动一页</td></tr>
<tr><td>zt or z↵</td><td>将当前行滚动到窗口顶部</td></tr>
<tr><td>zz or z. </td><td>将当前行滚动到窗口中央</td></tr>
<tr><td>zb or z-</td><td>将当前行滚动到窗口底部</td></tr>
<tr><td>zh zl</td><td>向右、向左滚动一个字符</td></tr>
<tr><td>zH zL</td><td>向右、向左移动半屏</td></tr>
<tr><td>^Ws or :split↵</td><td>将窗口分割成两个</td></tr>
<tr><td>^Wn or :new↵</td><td>创建一个新窗口</td></tr>
<tr><td>^Wo or :on↵</td><td>关掉当前窗口以外的其他窗口</td></tr>
<tr><td>^Wj ^Wk</td><td>移动到下一个、上一个窗口</td></tr>
<tr><td>^Ww ^W^W</td><td>移动到下一个、上一个窗口 (wrap)*</td></tr>

</table><br />

<strong>Ex commands
Ex命令</strong>

<table border="1">
<tr><td>:e f</td><td>编辑文件f，如果没有更改</td></tr>
<tr><td>:e!  f</td><td>强制编辑文件（默认覆盖原先的修改）</td></tr>
<tr><td>:wn :wN</td><td>写入文件，并编辑下一个、前一个文件</td></tr>
<tr><td>:n :N</td><td>编辑文件列表中的下一个、前一个文件</td></tr>
<tr><td>:rw</td><td>把区间r写入到当前文件</td></tr>
<tr><td>:rw f</td><td>把区间r写入到文件f</td></tr>
<tr><td>:rw>>f</td><td>把如见r追加到文件f</td></tr>
<tr><td>:q :q!</td><td>退出并确认，强制退出忽略修改</td></tr>
<tr><td>:wq or :x or ZZ</td><td>写入到当前文件并退出</td></tr>
<tr><td>:r f</td><td>将光标下的内容插入文件f</td></tr>
<tr><td>:r!  c</td><td>将光标下的内容经命令c处理的结果插入</td></tr>
<tr><td>:args</td><td>显示参数列表</td></tr>
<tr><td>:rc  a :rm  a</td><td>复制、移动区间r到a行下</td></tr>

</table><br />

<strong>Ex ranges
Ex区间 </strong>

<table border="1">
<tr><td>.   $</td><td>文件的当前行，最后一行</td></tr>
<tr><td>% *</td><td>整个文件、可视部分</td></tr>
<tr><td>'t</td><td>标记t指向的位置</td></tr>
<tr><td>/p/ ?p?</td><td>匹配的下一个、前一个位置</td></tr>
<tr><td>+n -n</td><td>当前行的前n-1行、后n-1行</td></tr>
</table><br />

<strong>Folding
折叠</strong>

<table border="1">
<tr><td>zfm</td><td>创建折叠到移动命令m指向的位置</td></tr>
<tr><td>:rfo</td><td>为区间r创建折叠</td></tr>
<tr><td>zd zE</td><td>移除当前折叠、当前窗口的所有折叠</td></tr>
<tr><td>zo zc zO zC</td><td>打开、关闭一个折叠；递归地执行</td></tr>
<tr><td>[z ]z</td><td>移动到当前打开折叠的开头、结尾</td></tr>
<tr><td>zj zk</td><td>向下、向上移动到下一个折叠的开头、结尾</td></tr>
</table><br />

<strong>Miscellaneous
杂</strong>

<table border="1">
<tr><td>:sh↵  :!c↵</td><td>运行shell，执行shell命令c</td></tr>
<tr><td>K</td><td>在man中搜索当前关键字</td></tr>
<tr><td>:make↵</td><td>执行make，读取错误并跳转到首个错误</td></tr>
<tr><td>:cn↵  :cp↵</td><td>显示下一个、前一个错误</td></tr>
<tr><td>:cl↵  :cf↵</td><td>显示所有错误、从文件中读取错误</td></tr>
<tr><td>^L ^G</td><td>重绘屏幕，显示文件名及位置</td></tr>
<tr><td>g^G</td><td>显示光标所在行、列及字符位置</td></tr>
<tr><td>ga</td><td>显示当前字符的ascii值</td></tr>
<tr><td>gf</td><td>打开当前光标下的文件名</td></tr>
<tr><td>:redir>f↵</td><td>将输出重定向到文件f</td></tr>
<tr><td>:mkview ↵</td><td>保存view配置[到文件f]</td></tr>
<tr><td>:loadview ↵</td><td>装载view配置[从文件f]</td></tr>

</table><br />

ps:高数59分华丽地挂掉鸟～
