---
layout: post
title: "备份下vimrc"
tags: 
- vim
- "备忘"
status: publish
type: post
published: true
meta: 
  _edit_last: "2"
---

vim嘛，理念就是找到你在输入文本时累人的操作，用简单快捷的操作代替之，没必要多找麻烦。

<pre lang="bash">
set nocompatible
source $VIMRUNTIME/vimrc_example.vim
source $VIMRUNTIME/mswin.vim
behave mswin

syntax on 
filetype plugin on

"about tab
set autoindent
set smartindent
set smarttab 					
set expandtab 					

set shiftwidth=4 
set tabstop=4 

" encoding . utf-8 rules!
" let $LANG="zh_CN.UTF-8" " locales
set encoding=utf-8 " ability
" set fileencoding=utf-8 " prefer
set ambiwidth=double
set fileencodings=utf-8,gb2312,gbk,gb18030
set termencoding=utf-8
set encoding=utf-8

" misc
set nu
set wildmenu
set ruler
set tags=tags
"set autochdir

" ^c^V i don't know how these about
" set laststatus=2
set lbr
set fo+=mB 
set sm 
set cin
set cino=:0g0t0(sus 
" set guifont=Courier_New:h10:cANSI 
set hls 
set backspace=indent,eol,start 
set noswapfile
set whichwrap=b,s,<,>,[,] 
set bsdir=buffer
set smartcase

set nowrap 
set autochdir
set autoread
set autowrite

set nobackup
set nowritebackup
            
" toolbar sucks
set guioptions-=T


color slate

" key bindings

noremap <C-[>   <C-T>

" tabs
noremap <C-Right> :tabn<CR>
noremap <C-left> :tabprevious<CR>


" windows
noremap <C-Up> <C-W>k<C-W>_
inoremap <C-Up> <C-o><C-W>k<C-W>_
noremap <C-Down> <C-W>j<C-W>_
inoremap <C-Down> <C-o><C-W>j<C-W>_

" v
noremap <UP> gk
noremap <Down> gj
noremap <LEFT> h
noremap <Right> l
noremap <ESC> v<ESC>

"
source $VIMRUNTIME/delmenu.vim
source $VIMRUNTIME/menu.vim

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" on plugins
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

" Vimwiki
let g:vimwiki_use_mouse = 1
let g:vimwiki_camel_case = 0
let g:vimwiki_CJK_length = 1
let g:vimwiki_list = [{'path': '~/code/wiki/src/',
                    \ 'path_html': '~/code/wiki/',
                    \ 'html_header': '~/code/wiki/src/tpl/head.tpl',
                    \ 'html_footer': '~/code/wiki/src/tpl/foot.tpl'}
                    \ ]

" au BufLeave             *.wiki		Vimwiki2HTML
au BufNewFile,BufRead   *           set nowrap
au BufNewFile,BufRead   *.wiki      set wrap
au BufNewFile,BufRead   *.wiki      color darkblue

" FuzzyFinder
let g:fuf_previewHeight=0 

" noremap ff :FufFile<cr>
" noremap fb :FufBuffer<cr>
" noremap fd :FufDir<cr>
" noremap ft :FufTag<cr>
noremap <Leader>ft :FufTag<cr>
noremap <Leader>fb :FufBuffer<cr>
noremap <Leader>ff :FufFile<cr>
</pre>
